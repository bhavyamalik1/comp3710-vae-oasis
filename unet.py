import os, math, random, argparse, sys, numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============ Utils ============
def set_seed(seed: int = 123):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def natkey(s: str):
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def list_pngs(p: Path) -> List[str]:
    return sorted([f for f in os.listdir(p) if f.lower().endswith(".png")], key=natkey)

def stem_noext(name: str) -> str:
    return os.path.splitext(name)[0]

def normalize_key(stem: str) -> tuple[str, str]:
    s = stem.lower()
    for tok in [
        "_seg","_mask","_labels","_label","_gt",
        "seg_","mask_","labels_","label_","gt_",
        "seg","mask","labels","label","groundtruth","gt","ground_truth"
    ]:
        s = s.replace(tok, "")
    import re
    alnum = re.sub(r'[^a-z0-9]+', '', s)
    digits = re.sub(r'[^0-9]+', '', s)
    return alnum, digits

# ============ Dataset ============
class OasisSegDataset(Dataset):
    """
    Pairs images and masks from two folders with robust filename matching.
    - Images normalized to [-1, 1]
    - Masks are integer class IDs (HxW, long)
    """
    def __init__(self, img_dir: str, seg_dir: str, size: int):
        self.img_dir = Path(img_dir); self.seg_dir = Path(seg_dir)
        if not self.img_dir.is_dir() or not self.seg_dir.is_dir():
            raise RuntimeError(f"Dirs not found: {img_dir}, {seg_dir}")

        imgs = list_pngs(self.img_dir)
        segs = list_pngs(self.seg_dir)

        masks_by_exact: Dict[str, str] = {s: s for s in segs}
        masks_by_norm: Dict[str, str] = {}
        masks_by_digits: Dict[str, str] = {}
        for m in segs:
            st = stem_noext(m)
            akey, dkey = normalize_key(st)
            if akey and akey not in masks_by_norm: masks_by_norm[akey] = m
            if dkey and dkey not in masks_by_digits: masks_by_digits[dkey] = m

        pairs: List[Tuple[str, str]] = []
        missing = []

        for img in imgs:
            st_img = stem_noext(img)
            if img in masks_by_exact:
                pairs.append((img, img)); continue
            candidates = [
                st_img + "_seg.png", st_img + "_mask.png",
                "seg_" + st_img + ".png", "mask_" + st_img + ".png",
                st_img.replace("_seg", "").replace("_mask", "") + ".png"
            ]
            found = None
            for c in candidates:
                if c in masks_by_exact:
                    found = c; break
            if found:
                pairs.append((img, found)); continue
            akey, dkey = normalize_key(st_img)
            if akey and akey in masks_by_norm:
                pairs.append((img, masks_by_norm[akey])); continue
            if dkey and dkey in masks_by_digits:
                pairs.append((img, masks_by_digits[dkey])); continue

            missing.append(img)

        if not pairs:
            ex_i = imgs[:5]; ex_s = segs[:5]
            raise RuntimeError(f"""No matching image/mask filenames.
            Sample images: {ex_i}
            Sample masks:  {ex_s}
            Hint: ensure img_dir=keras_png_slices_* and seg_dir=keras_png_slices_seg_*""")

        if missing:
            print(f"[WARN] Unmatched images (showing up to 10): {missing[:10]} ... ({len(missing)} total)")

        self.pairs = pairs
        self.tx_img = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((size, size), interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.size = size
        print(f"[PAIRING] {len(self.pairs)} pairs from {self.img_dir.name} ↔ {self.seg_dir.name}")

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        img_name, seg_name = self.pairs[idx]

        # load
        img = Image.open(self.img_dir / img_name)
        seg = Image.open(self.seg_dir / seg_name)

        # image → [-1,1] (already handled by self.tx_img)
        img_t = self.tx_img(img)

        # mask → grayscale → NEAREST resize → binary {0,1}
        seg = seg.convert("L").resize((self.size, self.size), resample=Image.NEAREST)
        seg_np = np.array(seg, dtype=np.uint8)
        seg_np = (seg_np > 0).astype(np.int64)   # foreground=1, background=0

        # to tensor (long for CE loss)
        seg_t = torch.from_numpy(seg_np).long()

        return img_t, seg_t, img_name

# ============ UNet ============
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_ch=1, n_classes=4, base=32):
        super().__init__()
        self.down1 = DoubleConv(in_ch, base);   self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(base, base*2);  self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(base*2, base*4);self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(base*4, base*8);self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base*8, base*16)
        self.up4 = nn.ConvTranspose2d(base*16, base*8, 2, 2); self.conv4 = DoubleConv(base*16, base*8)
        self.up3 = nn.ConvTranspose2d(base*8,  base*4, 2, 2); self.conv3 = DoubleConv(base*8,  base*4)
        self.up2 = nn.ConvTranspose2d(base*4,  base*2, 2, 2); self.conv2 = DoubleConv(base*4,  base*2)
        self.up1 = nn.ConvTranspose2d(base*2,  base,   2, 2); self.conv1 = DoubleConv(base*2,  base)
        self.head = nn.Conv2d(base, n_classes, 1)
    def forward(self, x):
        d1 = self.down1(x); p1 = self.pool1(d1)
        d2 = self.down2(p1); p2 = self.pool2(d2)
        d3 = self.down3(p2); p3 = self.pool3(d3)
        d4 = self.down4(p3); p4 = self.pool4(d4)
        bn = self.bottleneck(p4)
        u4 = self.up4(bn); c4 = self.conv4(torch.cat([u4, d4], dim=1))
        u3 = self.up3(c4); c3 = self.conv3(torch.cat([u3, d3], dim=1))
        u2 = self.up2(c3); c2 = self.conv2(torch.cat([u2, d2], dim=1))
        u1 = self.up1(c2); c1 = self.conv1(torch.cat([u1, d1], dim=1))
        return self.head(c1)

# ============ Losses & metrics ============
def one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(labels, num_classes=num_classes).permute(0,3,1,2).float()

def dice_coeff(pred_probs: torch.Tensor, target_1h: torch.Tensor, eps=1e-6):
    dims = (0,2,3)
    inter = (pred_probs * target_1h).sum(dims)
    denom = pred_probs.sum(dims) + target_1h.sum(dims)
    return (2*inter + eps) / (denom + eps)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6): super().__init__(); self.smooth = smooth
    def forward(self, logits, targets, num_classes):
        probs = F.softmax(logits, dim=1)
        tgt_1h = one_hot(targets, num_classes)
        dice_c = dice_coeff(probs, tgt_1h, eps=self.smooth)
        return 1 - dice_c.mean()

# ============ Eval & Viz ============
def infer_num_classes(ds: Dataset, samples: int = 50) -> int:
    mx = 0; step = max(1, len(ds)//samples)
    for i in range(0, len(ds), step):
        _, seg, _ = ds[i]; mx = max(mx, int(seg.max()))
    return mx + 1

def colorize_mask(mask: np.ndarray, num_classes: int) -> np.ndarray:
    import colorsys
    palette=[]; h,w = mask.shape
    for i in range(num_classes):
        h_ = (i/max(1,num_classes))%1.0
        r,g,b = colorsys.hsv_to_rgb(h_,0.6,1.0)
        palette.append((int(r*255),int(g*255),int(b*255)))
    rgb = np.zeros((h,w,3), dtype=np.uint8)
    for c,(r,g,b) in enumerate(palette): rgb[mask==c]=(r,g,b)
    return rgb

def show_batch(model, loader, num_classes: int, out_dir: str, tag: str, n: int = 3, device="cpu"):
    model.eval(); shown=0
    with torch.no_grad():
        for imgs, segs, names in loader:
            imgs = imgs.to(device)
            preds = torch.argmax(model(imgs), dim=1).cpu().numpy()
            for i in range(min(n, imgs.size(0))):
                img = imgs[i].cpu().permute(1,2,0).numpy()
                img = (img*0.5+0.5).clip(0,1)
                gt  = segs[i].cpu().numpy(); pr  = preds[i]
                plt.figure(figsize=(12,4))
                plt.subplot(1,3,1); plt.title("Image"); plt.axis('off'); plt.imshow(img.squeeze(), cmap='gray')
                plt.subplot(1,3,2); plt.title("Ground Truth"); plt.axis('off'); plt.imshow(colorize_mask(gt, num_classes))
                plt.subplot(1,3,3); plt.title("Prediction"); plt.axis('off'); plt.imshow(colorize_mask(pr, num_classes))
                os.makedirs(out_dir, exist_ok=True)
                plt.savefig(os.path.join(out_dir, f"viz_{tag}_{shown:03d}.png"), bbox_inches='tight'); plt.close()
                shown+=1
                if shown>=n: return

def evaluate(model, loader, num_classes: int, device="cpu"):
    model.eval(); dice_sums = torch.zeros(num_classes, device=device); n=0
    with torch.no_grad():
        for imgs, segs, _ in loader:
            imgs = imgs.to(device); segs = segs.to(device)
            logits = model(imgs); probs = F.softmax(logits, dim=1)
            tgt_1h = one_hot(segs, num_classes).to(device)
            dice_sums += dice_coeff(probs, tgt_1h); n += 1
    return (dice_sums / n).detach().cpu().numpy()

# ============ Argparse ============
def build_argparser():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--data_root", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--out", type=str, default="unet_oasis_out")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--train_imgs", type=str, default="keras_png_slices_train")
    ap.add_argument("--val_imgs",   type=str, default="keras_png_slices_validate")
    ap.add_argument("--test_imgs",  type=str, default="keras_png_slices_test")
    ap.add_argument("--train_segs", type=str, default="keras_png_slices_seg_train")
    ap.add_argument("--val_segs",   type=str, default="keras_png_slices_seg_validate")
    ap.add_argument("--test_segs",  type=str, default="keras_png_slices_seg_test")
    return ap

def _has_expected_subdirs(root: Path) -> bool:
    needed = [
        "keras_png_slices_train","keras_png_slices_validate",
        "keras_png_slices_seg_train","keras_png_slices_seg_validate"
    ]
    return all((root/d).is_dir() for d in needed)

def resolve_data_root(dr_arg: str | None) -> str:
    from pathlib import Path

    def ok(root: Path) -> bool:
        need = [
            "keras_png_slices_train","keras_png_slices_validate",
            "keras_png_slices_seg_train","keras_png_slices_seg_validate",
        ]
        return all((root/d).is_dir() for d in need)

    # 1) Try the CLI arg exactly
    if dr_arg:
        p = Path(dr_arg.strip())
        if p.is_dir() and ok(p):
            return str(p.resolve())
        # common nesting: <arg>/keras_png_slices_data
        q = p / "keras_png_slices_data"
        if q.is_dir() and ok(q):
            return str(q.resolve())

    # 2) Try $DATA_ROOT
    env = os.environ.get("DATA_ROOT", "").strip()
    if env:
        p = Path(env)
        if p.is_dir() and ok(p):
            return str(p.resolve())
        q = p / "keras_png_slices_data"
        if q.is_dir() and ok(q):
            return str(q.resolve())

    # 3) Colab-friendly defaults
    for base in ["/content/keras_png_slices_data",
                 "/content/drive/MyDrive/keras_png_slices_data"]:
        p = Path(base)
        if p.is_dir() and ok(p):
            return str(p.resolve())
        q = p / "keras_png_slices_data"
        if q.is_dir() and ok(q):
            return str(q.resolve())



# ============ Main ============
def main_cli(ns=None):
    if ns is None:
        ap = build_argparser(); ns, _unknown = ap.parse_known_args()
    set_seed(ns.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    data_root = resolve_data_root(ns.data_root)
    root = Path(data_root)
    print("Using data_root =", root)

    train_ds = OasisSegDataset(root/ns.train_imgs, root/ns.train_segs, ns.img_size)
    val_ds   = OasisSegDataset(root/ns.val_imgs,   root/ns.val_segs,   ns.img_size)
    test_ds  = OasisSegDataset(root/ns.test_imgs,  root/ns.test_segs,  ns.img_size)

    train_dl = DataLoader(train_ds, batch_size=ns.batch, shuffle=True,  num_workers=0, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=ns.batch, shuffle=False, num_workers=0, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=ns.batch, shuffle=False, num_workers=0, pin_memory=True)


    num_classes = infer_num_classes(train_ds, samples=50)
    print("Detected NUM_CLASSES =", num_classes)

    model = UNet(in_ch=1, n_classes=num_classes, base=ns.base).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    ce_loss = nn.CrossEntropyLoss(); dice_loss_fn = DiceLoss()

    best_mean = -1.0; os.makedirs(ns.out, exist_ok=True)

    for ep in range(1, ns.epochs+1):
        model.train(); run=0.0
        for imgs, segs, _ in train_dl:
            imgs = imgs.to(device); segs = segs.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(imgs)
            loss = ce_loss(logits, segs) + dice_loss_fn(logits, segs, num_classes)
            loss.backward(); opt.step(); run += loss.item()*imgs.size(0)
        train_loss = run/len(train_dl.dataset)
        per_class = evaluate(model, val_dl, num_classes, device=device)
        mean_dice = float(per_class.mean())
        print(f"[ep {ep:03d}] train_loss={train_loss:.4f}  val_mean_DSC={mean_dice:.4f}  per-class={np.round(per_class,4)}")
        if mean_dice > best_mean:
            best_mean = mean_dice
            torch.save({"state_dict": model.state_dict(),
                        "num_classes": num_classes,
                        "img_size": ns.img_size,
                        "base": ns.base},
                       os.path.join(ns.out, "best_unet.pt"))
            print("  ✓ saved best_unet.pt")
            show_batch(model, val_dl, num_classes, ns.out, tag=f"val_ep{ep:03d}", n=3, device=device)

    ckpt = torch.load(os.path.join(ns.out, "best_unet.pt"), map_location=device)
    model.load_state_dict(ckpt["state_dict"]); model.eval()
    per_class_test = evaluate(model, test_dl, num_classes, device=device)
    print("Test per-class DSC:", np.round(per_class_test,4))
    print("Test mean DSC:", float(per_class_test.mean()))
    show_batch(model, test_dl, num_classes, ns.out, tag="test", n=3, device=device)
    print("Saved visualisations and checkpoint to:", ns.out)

if __name__ == "__main__":
    main_cli()
