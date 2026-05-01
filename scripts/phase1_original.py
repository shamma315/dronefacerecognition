"""phase1_only_original — adapt the InceptionResnetV1 backbone to drone-style
imagery on VGGFace2, without synthetic identities.

Hyperparameters:
  Adam, batch 256, backbone lr=1e-5, head/ArcFace lr=1e-4
  ArcFace s=64 m=0.5 (standard for ~480-class identity-rich pretraining)
  30 epochs, cosine annealed to 1e-6
  Unfreezes block8 + avgpool_1a (2 layers)
  90/10 train/val split of VGGFace2/train

Saves best-by-val-Rank-1 to checkpoints/checkpoints/best_model.pth.

Idempotent: skip if output exists unless --force is passed.

Phase 2 (any of phase2_*_original.py / phase2_from_synth.py / both_phases_original.py)
uses s=32 m=0.3 — smaller margins are appropriate for the 8-identity DroneFace
head. Per-phase ArcFace asymmetry is intentional and documented in the
methodology note.
"""
import os, sys, time, csv, argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms

ROOT = Path("/home/buthaina.almulla/Documents/CV7502")
sys.path.insert(0, str(ROOT / "src"))
from facenet_pytorch import InceptionResnetV1
from dataset import VGGFaceDataset
from model import Embeddinghead, ArcFaceLoss

VGG_TRAIN = ROOT / "datasets/vggface2/train"
VGG_VAL = ROOT / "datasets/vggface2/val"
CKPT_DIR = ROOT / "checkpoints" / "checkpoints"
OUT_CKPT = CKPT_DIR / "best_model.pth"
CURVES = ROOT / "results" / "training_curves"; CURVES.mkdir(parents=True, exist_ok=True)
METRICS_CSV = CURVES / "phase1_original_per_epoch_metrics.csv"

BATCH_SIZE = 256
EPOCHS = 30
LR_BACKBONE = 1e-5
LR_HEAD = 1e-4
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true",
                    help="Re-run even if output checkpoint already exists.")
    args = ap.parse_args()

    if OUT_CKPT.exists() and not args.force:
        print(f"[skip] {OUT_CKPT} already exists. Pass --force to retrain.", flush=True)
        return 0

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED); np.random.seed(SEED)

    train_ids = sorted(os.listdir(VGG_TRAIN))
    val_ids   = sorted(os.listdir(VGG_VAL))
    all_ids   = sorted(set(train_ids) | set(val_ids))
    id_map = {name: i for i, name in enumerate(all_ids)}
    n_classes = len(all_ids)

    train_id_map = {n: id_map[n] for n in train_ids}
    full_train = VGGFaceDataset(str(VGG_TRAIN), train_ids, train_id_map, augment=True)
    print(f"VGGFace2: {len(train_ids)} train ids + {len(val_ids)} val ids = {n_classes} classes "
          f"({len(full_train)} train images)", flush=True)

    train_size = int(0.9 * len(full_train))
    val_size = len(full_train) - train_size
    g = torch.Generator().manual_seed(SEED)
    train_subset, val_subset = random_split(full_train, [train_size, val_size], generator=g)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=8, pin_memory=True)
    val_loader   = DataLoader(val_subset,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=8, pin_memory=True)

    backbone = InceptionResnetV1(pretrained="vggface2").to(DEVICE)
    for p in backbone.parameters(): p.requires_grad = False
    for blk in [backbone.block8, backbone.avgpool_1a]:
        for p in blk.parameters(): p.requires_grad = True
    head = Embeddinghead(input_dim=512, Embedding_dim=256).to(DEVICE)
    # ArcFace defaults s=64 m=0.5 (standard for ~480-class identity-rich pretraining).
    arcface = ArcFaceLoss(in_features=256, num_classes=n_classes).to(DEVICE)

    optimizer = torch.optim.Adam([
        {"params": [p for p in backbone.parameters() if p.requires_grad], "lr": LR_BACKBONE},
        {"params": head.parameters(),    "lr": LR_HEAD},
        {"params": arcface.parameters(), "lr": LR_HEAD},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    metrics = []
    best_val_r1 = -1.0
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        t_ep = time.time()
        backbone.train(); head.train(); arcface.train()
        loss_sum, nb = 0.0, 0
        for imgs, lbls in train_loader:
            imgs = imgs.to(DEVICE, non_blocking=True); lbls = lbls.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = arcface(head(backbone(imgs)), lbls)
            loss = F.cross_entropy(logits, lbls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(backbone.parameters()) + list(head.parameters()) + list(arcface.parameters()),
                max_norm=5.0,
            )
            optimizer.step()
            loss_sum += float(loss.item()); nb += 1
        scheduler.step()
        train_loss = loss_sum / max(1, nb)

        # validation: build gallery from train_subset, query from val_subset; rank-1 by cosine
        backbone.eval(); head.eval()
        gal_e, gal_l = [], []
        with torch.no_grad():
            for imgs, lbls in train_loader:
                gal_e.append(head(backbone(imgs.to(DEVICE))).cpu()); gal_l.append(lbls)
            gal_e = F.normalize(torch.cat(gal_e), p=2, dim=1); gal_l = torch.cat(gal_l)
            val_e, val_l = [], []
            for imgs, lbls in val_loader:
                val_e.append(head(backbone(imgs.to(DEVICE))).cpu()); val_l.append(lbls)
            val_e = F.normalize(torch.cat(val_e), p=2, dim=1); val_l = torch.cat(val_l)
            preds = gal_l[torch.mm(val_e, gal_e.T).argmax(dim=1)]
            r1 = (preds == val_l).float().mean().item() * 100

        metrics.append({"epoch": epoch, "train_loss": train_loss, "val_rank1": r1})

        flag = ""
        if r1 > best_val_r1:
            best_val_r1 = r1
            torch.save({
                "backbone": backbone.state_dict(),
                "head": head.state_dict(),
                "arcface": arcface.state_dict(),
                "epoch": epoch,
                "val_rank1": r1,
                "num_classes": n_classes,
            }, str(OUT_CKPT))
            flag = " (best)"
        print(f"epoch {epoch:>2}/{EPOCHS} | loss={train_loss:.4f} | val_R1={r1:.2f}% | "
              f"[{time.time()-t_ep:.1f}s]{flag}", flush=True)

    with open(METRICS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_rank1"])
        w.writeheader()
        for r in metrics: w.writerow(r)
    print(f"wrote {METRICS_CSV}", flush=True)
    print(f"done | best val R1 {best_val_r1:.2f}% | total {(time.time()-t0)/60:.1f}min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
