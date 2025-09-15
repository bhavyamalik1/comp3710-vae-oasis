# train_vae.py
import os, glob, argparse, math, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

# Optional deps
try:
    import nibabel as nib
except Exception:
    nib = None
try:
    import cv2
except Exception:
    cv2 = None
try:
    import umap
except Exception:
    umap = None

def _minmax(x):
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    if hi - lo < 1e-6: hi = lo + 1e-6
    x = np.clip((x - lo) / (hi - lo), 0., 1.)
    return x

class Oasis2D(Dataset):
    """Loads PNG/JPG slices or slices NIfTI volumes on the fly."""
    def __init__(self, root, size=128, split="train", train_ratio=0.8,
                 slice_stride=2, use_middle=True, seed=42):
        self.size = size
        self.slice_stride = slice_stride
        self.use_middle = use_middle
        rng = np.random.default_rng(seed)

        files = []
        for pat in ("**/*.png", "**/*.jpg", "**/*.jpeg", "**/*.nii", "**/*.nii.gz"):
            files += glob.glob(os.path.join(root, pat), recursive=True)
        files = sorted(files)
        if not files:
            raise RuntimeError(f"No images/NIfTIs found under: {root}")

        rng.shuffle(files)
        ntrain = int(len(files) * train_ratio)
        self.files = files[:ntrain] if split == "train" else files[ntrain:]

    def __len__(self):
        return max(1, len(self.files) * 30)  # approx slices per NIfTI

    def _read_png(self, path):
        if cv2 is None:
            raise RuntimeError("opencv-python not installed")
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None: raise RuntimeError(f"Failed to read: {path}")
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA).astype(np.float32)/255.0
        return img

    def _read_nii_slice(self, path, index_hint):
        if nib is None:
            raise RuntimeError("nibabel not installed")
        vol = nib.load(path).get_fdata()
        vol = np.nan_to_num(vol).astype(np.float32)
        depth = vol.shape[2]
        z0, z1 = (int(0.2*depth), int(0.8*depth)) if self.use_middle else (0, depth)
        zs = list(range(z0, z1, self.slice_stride)) or [depth//2]
        z = zs[index_hint % len(zs)]
        sl = vol[:, :, z]
        return sl

    def __getitem__(self, i):
        path = self.files[i % len(self.files)]
        if path.endswith((".nii", ".nii.gz")):
            sl = self._read_nii_slice(path, index_hint=i)
            sl = _minmax(sl)
            if cv2 is not None:
                sl = cv2.resize(sl, (self.size, self.size), interpolation=cv2.INTER_AREA)
            else:
                h, w = sl.shape
                ys = np.linspace(0, h-1, self.size).astype(int)
                xs = np.linspace(0, w-1, self.size).astype(int)
                sl = sl[ys][:, xs]
            img = sl
        else:
            img = _minmax(self._read_png(path))
        img = img * 2.0 - 1.0   # [-1,1] for Tanh output
        return torch.from_numpy(img).unsqueeze(0)  # 1xHxW

class Encoder(nn.Module):
    def __init__(self, z=16, ch=32):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, ch, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(ch, ch*2, 4, 2, 1), nn.BatchNorm2d(ch*2), nn.ReLU(True),
            nn.Conv2d(ch*2, ch*4, 4, 2, 1), nn.BatchNorm2d(ch*4), nn.ReLU(True),
            nn.Conv2d(ch*4, ch*8, 4, 2, 1), nn.BatchNorm2d(ch*8), nn.ReLU(True),
        )
        self.mu = nn.Linear(ch*8*8*8, z)
        self.logvar = nn.Linear(ch*8*8*8, z)
    def forward(self, x):
        h = self.body(x)
        h = torch.flatten(h, 1)
        return self.mu(h), self.logvar(h)

class Decoder(nn.Module):
    def __init__(self, z=16, ch=32):
        super().__init__()
        self.fc = nn.Linear(z, ch*8*8*8)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(ch*8, ch*4, 4, 2, 1), nn.BatchNorm2d(ch*4), nn.ReLU(True),
            nn.ConvTranspose2d(ch*4, ch*2, 4, 2, 1), nn.BatchNorm2d(ch*2), nn.ReLU(True),
            nn.ConvTranspose2d(ch*2, ch,   4, 2, 1), nn.BatchNorm2d(ch),   nn.ReLU(True),
            nn.ConvTranspose2d(ch, 1,      4, 2, 1), nn.Tanh()
        )
    def forward(self, z):
        h = self.fc(z).view(z.size(0), 256, 8, 8)
        return self.deconv(h)

class VAE(nn.Module):
    def __init__(self, z=16):
        super().__init__()
        self.enc, self.dec = Encoder(z), Decoder(z)
    def reparam(self, mu, logvar):
        std = torch.exp(0.5*logvar); eps = torch.randn_like(std)
        return mu + eps * std
    def forward(self, x):
        mu, logvar = self.enc(x)
        z = self.reparam(mu, logvar)
        xhat = self.dec(z)
        return xhat, mu, logvar

def vae_loss(x, xhat, mu, logvar, recon="mse", beta=1.0):
    if recon == "bce":
        loss_rec = F.binary_cross_entropy((xhat+1)/2, (x+1)/2, reduction="sum")
    else:
        loss_rec = F.mse_loss(xhat, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return loss_rec + beta*kl, loss_rec, kl

def save_grid(tensor, path, nrow=8, title=None):
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    grid = make_grid(((tensor.clamp(-1,1)+1)/2), nrow=nrow)
    plt.figure(figsize=(8,8)); plt.axis('off')
    if title: plt.title(title)
    plt.imshow(grid.permute(1,2,0).cpu().numpy())
    plt.savefig(path, bbox_inches='tight'); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--zdim", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=str, default="out")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--recon", type=str, default="mse", choices=["mse","bce"])
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_ds = Oasis2D(args.data_root, size=args.size, split="train", seed=args.seed)
    val_ds   = Oasis2D(args.data_root, size=args.size, split="val",   seed=args.seed)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

    model = VAE(z=args.zdim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best = math.inf
    for ep in range(1, args.epochs+1):
        model.train(); tot=0.0
        for x in train_dl:
            x = x.to(device, non_blocking=True)
            xhat, mu, logvar = model(x)
            loss, _, _ = vae_loss(x, xhat, mu, logvar, recon=args.recon, beta=args.beta)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        print(f"[ep {ep}] train_loss/sample={tot/len(train_dl.dataset):.4f}")

        model.eval(); vtot=0.0; first_in=None; first_out=None
        with torch.no_grad():
            for i, x in enumerate(val_dl):
                x = x.to(device, non_blocking=True)
                xhat, mu, logvar = model(x)
                loss, _, _ = vae_loss(x, xhat, mu, logvar, recon=args.recon, beta=args.beta)
                vtot += loss.item()
                if i == 0:
                    first_in  = x[:64].detach().cpu()
                    first_out = xhat[:64].detach().cpu()
        print(f"[ep {ep}] val_loss/sample={vtot/len(val_dl.dataset):.4f}")

        if first_in is not None:
            save_grid(first_in,  f"{args.out}/ep{ep:03d}_val_input.png", nrow=8, title="Val inputs")
            save_grid(first_out, f"{args.out}/ep{ep:03d}_val_recon.png", nrow=8, title="Recon")
            z = torch.randn(64, args.zdim, device=device)
            x_samp = model.dec(z).detach().cpu()
            save_grid(x_samp, f"{args.out}/ep{ep:03d}_samples.png", nrow=8, title="Random samples")

        if vtot < best:
            best = vtot
            os.makedirs(args.out, exist_ok=True)
            torch.save({"ep": ep, "state_dict": model.state_dict(), "args": vars(args)},
                       f"{args.out}/best.pt")

    print("Computing latent manifold...")
    zs = []
    with torch.no_grad():
        for x in DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2):
            x = x.to(device, non_blocking=True)
            mu, logvar = model.enc(x)
            zs.append(mu.cpu().numpy())
    Z = np.concatenate(zs, axis=0)
    if Z.shape[1] == 2:
        Z2, title = Z, "Latent (zdim=2)"
    else:
        if umap is not None:
            reducer = umap.UMAP(n_components=2, random_state=args.seed)
            Z2, title = reducer.fit_transform(Z), "UMAP of latent"
        else:
            Zc = Z - Z.mean(0, keepdims=True)
            U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
            Z2, title = Zc @ Vt[:2].T, "PCA of latent"
    plt.figure(figsize=(6,6))
    plt.scatter(Z2[:,0], Z2[:,1], s=2, alpha=0.5)
    plt.title(title); plt.tight_layout()
    os.makedirs(args.out, exist_ok=True)
    plt.savefig(f"{args.out}/latent_manifold.png"); plt.close()
    print("Saved:", f"{args.out}/latent_manifold.png")

if __name__ == "__main__":
    main()
