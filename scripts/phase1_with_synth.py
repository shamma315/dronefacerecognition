"""phase1_only_synth — Phase 1 on VGGFace2 (~480 train ids) + 30 synthetic
identities (570 classes total).

Same Phase 1 hyperparameters as phase1_original.py (Adam, batch 256, 30 epochs,
backbone lr=1e-5, head/ArcFace lr=1e-4, ArcFace s=64 m=0.5, CosineAnnealing).
Saves an intermediate checkpoint every 10 epochs (so an interruption doesn't
lose everything) and the best-by-val-Rank-1 model to
checkpoints/checkpoints/best_model_synth.pth.
"""
import os, sys, time, csv, random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms as transforms

ROOT = Path("/home/buthaina.almulla/Documents/CV7502")
sys.path.insert(0, str(ROOT / "src"))
from facenet_pytorch import InceptionResnetV1
from model import Embeddinghead, ArcFaceLoss

# ---- config ---------------------------------------------------------------
VGG_TRAIN = ROOT / "datasets/vggface2/train"
VGG_VAL = ROOT / "datasets/vggface2/val"
SYNTH = ROOT / "synthetic_identities"
CKPT_DIR = ROOT / "checkpoints" / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
CURVES = ROOT / "results" / "training_curves"; CURVES.mkdir(parents=True, exist_ok=True)
METRICS_CSV = CURVES / "phase1_synth_per_epoch_metrics.csv"

BATCH_SIZE = 256
EPOCHS = 30
LR_BACKBONE = 1e-5
LR_HEAD = 1e-4
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- transforms (mirror VGGFaceDataset in dataset.py) ---------------------
TRAIN_T = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomResizedCrop(112, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(25),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
])
VAL_T = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])


class CombinedFaceDataset(Dataset):
    """VGGFace2 train + VGGFace2 val + synthetic_identities, all under one label space."""

    def __init__(self, augment=True):
        self.augment = augment
        # collect identities in canonical order
        vgg_train_ids = sorted(os.listdir(VGG_TRAIN))
        vgg_val_ids = sorted(os.listdir(VGG_VAL))
        # union of vgg ids (matches notebook's union); 60 vgg-val ids may overlap with train
        vgg_ids = sorted(set(vgg_train_ids) | set(vgg_val_ids))
        synth_ids = sorted(os.listdir(SYNTH)) if SYNTH.exists() else []

        self.id_to_label = {ident: i for i, ident in enumerate(vgg_ids)}
        offset = len(vgg_ids)
        for j, sid in enumerate(synth_ids):
            self.id_to_label[sid] = offset + j

        self.num_classes = len(self.id_to_label)
        self.image_paths, self.labels = [], []

        for ident in vgg_train_ids:
            folder = VGG_TRAIN / ident
            for fn in os.listdir(folder):
                if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.image_paths.append(str(folder / fn))
                    self.labels.append(self.id_to_label[ident])
        for ident in vgg_val_ids:
            folder = VGG_VAL / ident
            for fn in os.listdir(folder):
                if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.image_paths.append(str(folder / fn))
                    self.labels.append(self.id_to_label[ident])
        for sid in synth_ids:
            folder = SYNTH / sid
            for fn in os.listdir(folder):
                if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.image_paths.append(str(folder / fn))
                    self.labels.append(self.id_to_label[sid])

        self._n_vgg = len(vgg_ids)
        self._n_synth = len(synth_ids)
        self._n_synth_imgs = sum(1 for p in self.image_paths if "/synthetic_identities/" in p)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, i):
        img = Image.open(self.image_paths[i]).convert("RGB")
        t = TRAIN_T if self.augment else VAL_T
        return t(img), self.labels[i]


def main():
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"device={DEVICE} | torch={torch.__version__}", flush=True)
    print("loading combined dataset…", flush=True)
    full = CombinedFaceDataset(augment=True)
    print(
        f"  classes: {full.num_classes} ({full._n_vgg} VGGFace2 + {full._n_synth} synthetic)\n"
        f"  total images: {len(full)} ({full._n_synth_imgs} synthetic)",
        flush=True,
    )
    if full._n_synth == 0:
        print("ERROR: no synthetic identities found at", SYNTH, flush=True)
        sys.exit(1)

    n_train = int(0.9 * len(full))
    n_val = len(full) - n_train
    train_subset, val_subset = random_split(
        full, [n_train, n_val], generator=torch.Generator().manual_seed(SEED)
    )
    # Build a non-augmented twin so val measures clean embeddings
    full_noaug = CombinedFaceDataset(augment=False)
    val_subset_noaug = torch.utils.data.Subset(full_noaug, val_subset.indices)
    print(f"split: {n_train} train / {n_val} val", flush=True)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_subset_noaug, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=8, pin_memory=True, persistent_workers=True)

    backbone = InceptionResnetV1(pretrained="vggface2").to(DEVICE)
    for p in backbone.parameters():
        p.requires_grad = False
    for p in backbone.block8.parameters(): p.requires_grad = True
    for p in backbone.avgpool_1a.parameters(): p.requires_grad = True

    head = Embeddinghead(input_dim=512, Embedding_dim=256).to(DEVICE)
    arcface = ArcFaceLoss(in_features=256, num_classes=full.num_classes).to(DEVICE)

    optimizer = torch.optim.Adam([
        {"params": [p for p in backbone.parameters() if p.requires_grad], "lr": LR_BACKBONE},
        {"params": head.parameters(),    "lr": LR_HEAD},
        {"params": arcface.parameters(), "lr": LR_HEAD},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    metrics = []
    best_val_acc = 0.0
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        t_ep = time.time()
        backbone.train(); head.train(); arcface.train()
        total_loss, n_batches = 0.0, 0
        for imgs, lbls in train_loader:
            imgs = imgs.to(DEVICE, non_blocking=True); lbls = lbls.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            proj = head(backbone(imgs))
            logits = arcface(proj, lbls)
            loss = F.cross_entropy(logits, lbls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(backbone.parameters()) + list(head.parameters()) + list(arcface.parameters()),
                max_norm=5.0,
            )
            optimizer.step()
            total_loss += float(loss.item()); n_batches += 1
        scheduler.step()
        train_loss = total_loss / max(1, n_batches)

        # gallery + val Rank-1 (matches the protocol in the original Phase 1)
        backbone.eval(); head.eval()
        gal_emb, gal_lbl = [], []
        with torch.no_grad():
            for imgs, lbls in train_loader:
                e = head(backbone(imgs.to(DEVICE)))
                gal_emb.append(e.cpu()); gal_lbl.append(lbls)
        gal_emb = F.normalize(torch.cat(gal_emb), p=2, dim=1)
        gal_lbl = torch.cat(gal_lbl)

        val_emb, val_lbl = [], []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                e = head(backbone(imgs.to(DEVICE)))
                val_emb.append(e.cpu()); val_lbl.append(lbls)
        val_emb = F.normalize(torch.cat(val_emb), p=2, dim=1)
        val_lbl = torch.cat(val_lbl)
        # nearest-neighbour Rank-1 (gallery-instance, not centroid — matches original)
        sims = val_emb @ gal_emb.T
        pred = gal_lbl[sims.argmax(dim=1)]
        rank1 = (pred == val_lbl).float().mean().item() * 100.0

        flag = ""
        if rank1 > best_val_acc:
            best_val_acc = rank1
            torch.save({
                "backbone": backbone.state_dict(),
                "head": head.state_dict(),
                "arcface": arcface.state_dict(),
                "epoch": epoch,
                "val_rank1": rank1,
                "num_classes": full.num_classes,
            }, str(CKPT_DIR / "best_model_synth.pth"))
            flag = " (best, saved)"

        # Periodic 10-epoch checkpoint for resumability
        if epoch % 10 == 0 or epoch == EPOCHS:
            torch.save({
                "backbone": backbone.state_dict(),
                "head": head.state_dict(),
                "arcface": arcface.state_dict(),
                "epoch": epoch,
                "val_rank1": rank1,
                "num_classes": full.num_classes,
            }, str(CKPT_DIR / f"phase1_synth_epoch_{epoch:02d}.pth"))

        metrics.append({"epoch": epoch, "train_loss": train_loss, "val_rank1": rank1})

        print(
            f"epoch {epoch:>2}/{EPOCHS} | loss={train_loss:.4f} | val_R1={rank1:.2f}% | "
            f"[{time.time()-t_ep:.1f}s, total {(time.time()-t0)/60:.1f}min]{flag}",
            flush=True,
        )

    with open(METRICS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_rank1"])
        w.writeheader()
        for r in metrics: w.writerow(r)
    print(f"wrote {METRICS_CSV}", flush=True)
    print(f"done | best val R1 {best_val_acc:.2f}% | total {(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
