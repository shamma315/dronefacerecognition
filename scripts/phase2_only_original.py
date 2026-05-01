"""Phase 2 only — fine-tune from raw VGGFace2 backbone (no Phase 1 pretrain),
on DroneFace 8 train identities (A-H), no synthetic identities.

Same hyperparameters as the v2 Phase 2 in training_v2.ipynb so the ablation
isolates the variable of interest (Phase 1 pretrain on/off):
  Adam, batch 32, backbone lr=1e-6, head/ArcFace lr=1e-4
  ArcFace s=32 m=0.3, 50 epochs, cosine annealed to 1e-7
  Unfreezes block8 + avgpool_1a + last_linear + last_bn

Saves best-by-val-Rank-1 to checkpoints/checkpoints/best_drone_model_phase2only.pth.

Idempotent: skip if output exists unless --force is passed.
"""
import os, sys, time, csv, argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image

ROOT = Path("/home/buthaina.almulla/Documents/CV7502")
sys.path.insert(0, str(ROOT / "src"))
from facenet_pytorch import InceptionResnetV1
from dataset import DroneFaceDataset
from model import Embeddinghead, ArcFaceLoss

SPLIT = ROOT / "datasets/droneface/split"
CKPT_DIR = ROOT / "checkpoints" / "checkpoints"
OUT_CKPT = CKPT_DIR / "best_drone_model_phase2only.pth"
RESULTS = ROOT / "results"; RESULTS.mkdir(exist_ok=True)
CURVES = RESULTS / "training_curves"; CURVES.mkdir(exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 50
LR_BACKBONE = 1e-6
LR_HEAD = 1e-4
ARCFACE_SCALE = 32.0
ARCFACE_MARGIN = 0.3
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_IDS = ["A","B","C","D","E","F","G","H"]
VAL_IDS = ["I"]
HELDOUT_IDS = ["J","K"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true",
                    help="Re-run even if output checkpoint already exists.")
    args = ap.parse_args()

    if OUT_CKPT.exists() and not args.force:
        print(f"[skip] {OUT_CKPT} already exists. Pass --force to retrain.", flush=True)
        return 0

    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED); np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_set = DroneFaceDataset(str(SPLIT / "train"), augment=True)
    n_classes = len(train_set.classes)
    assert train_set.classes == TRAIN_IDS, f"unexpected classes {train_set.classes}"

    # Raw VGGFace2 backbone — no Phase 1 weights loaded.
    backbone = InceptionResnetV1(pretrained="vggface2").to(DEVICE)
    for p in backbone.parameters(): p.requires_grad = False
    for blk in [backbone.block8, backbone.avgpool_1a, backbone.last_linear, backbone.last_bn]:
        for p in blk.parameters(): p.requires_grad = True
    head = Embeddinghead(input_dim=512, Embedding_dim=256).to(DEVICE)
    arcface = ArcFaceLoss(in_features=256, num_classes=n_classes,
                          scale=ARCFACE_SCALE, margin=ARCFACE_MARGIN).to(DEVICE)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)

    optimizer = torch.optim.Adam([
        {"params": [p for p in backbone.parameters() if p.requires_grad], "lr": LR_BACKBONE},
        {"params": head.parameters(),    "lr": LR_HEAD},
        {"params": arcface.parameters(), "lr": LR_HEAD},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)

    eval_t = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    def list_imgs(folder):
        return sorted([f for f in os.listdir(folder)
                       if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    eval_items = []
    for split, ids in [("train", TRAIN_IDS), ("validation", VAL_IDS), ("test", HELDOUT_IDS)]:
        for ident in ids:
            folder = SPLIT / split / ident
            for fn in list_imgs(folder):
                eval_items.append((str(folder / fn), ident))
    eval_paths = [p for p, _ in eval_items]
    eval_lbls = np.array([l for _, l in eval_items])

    @torch.no_grad()
    def extract():
        backbone.eval(); head.eval()
        out = []
        for s in range(0, len(eval_paths), 64):
            chunk = eval_paths[s:s+64]
            x = torch.stack([eval_t(Image.open(p).convert("RGB")) for p in chunk]).to(DEVICE)
            out.append(head(backbone(x)).cpu())
        return torch.cat(out, dim=0)

    def loo_rank1(emb, lbls):
        unique = np.unique(lbls)
        counts = {i: int((lbls == i).sum()) for i in unique}
        sums = {i: emb[lbls == i].sum(dim=0) for i in unique}
        q = F.normalize(emb, p=2, dim=1)
        cents = torch.stack([sums[i] / counts[i] for i in unique])
        cent_lbls = np.array(unique)
        correct = {i: 0 for i in unique}; total = {i: 0 for i in unique}
        for i in range(emb.shape[0]):
            ident = lbls[i]
            if counts[ident] <= 1:
                total[ident] += 1; continue
            adj = (sums[ident] - emb[i]) / (counts[ident] - 1)
            c = cents.clone()
            c[np.where(cent_lbls == ident)[0][0]] = adj
            cn = F.normalize(c, p=2, dim=1)
            pred = cent_lbls[int((cn @ q[i]).argmax().item())]
            if pred == ident: correct[ident] += 1
            total[ident] += 1
        return {i: (correct[i], total[i]) for i in unique}

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

        emb = extract()
        per_id = loo_rank1(emb, eval_lbls)

        def slc(ids):
            c = sum(per_id[i][0] for i in ids); t = sum(per_id[i][1] for i in ids)
            return 100.0 * c / max(1, t)
        tr = slc(TRAIN_IDS); va = slc(VAL_IDS); ho = slc(HELDOUT_IDS)
        metrics.append({"epoch": epoch, "train_loss": train_loss,
                        "train_rank1": tr, "val_rank1": va, "heldout_rank1": ho})

        flag = ""
        if va > best_val_r1:
            best_val_r1 = va
            torch.save({
                "backbone": backbone.state_dict(),
                "head": head.state_dict(),
                "arcface": arcface.state_dict(),
                "epoch": epoch,
                "val_rank1": va,
            }, str(OUT_CKPT))
            flag = " (best)"
        print(f"epoch {epoch:>2}/{EPOCHS} | loss={train_loss:.4f} | "
              f"tr_R1={tr:.1f}% va_R1={va:.1f}% ho_R1={ho:.1f}% | "
              f"[{time.time()-t_ep:.1f}s]{flag}", flush=True)

    csv_path = CURVES / "phase2only_original_per_epoch_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch","train_loss","train_rank1","val_rank1","heldout_rank1"])
        w.writeheader()
        for r in metrics: w.writerow(r)
    print(f"wrote {csv_path}", flush=True)
    print(f"done | best val R1 {best_val_r1:.2f}% | total {(time.time()-t0)/60:.1f}min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
