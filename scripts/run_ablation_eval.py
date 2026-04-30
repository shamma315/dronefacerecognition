"""Run loo_eval over three configurations and write results/two_phase_ablation_v2.csv:

  raw_baseline               — InceptionResnetV1(pretrained='vggface2'), no head, no Normalize
  phase1_phase2_original     — checkpoints/checkpoints/best_drone_model.pth (existing run, no synth)
  phase1_phase2_with_synth   — checkpoints/checkpoints/best_drone_model_synth.pth (new run, with synth)

Uses the same protocol as the LOO eval in training_fixed.ipynb cell 20:
mean-then-normalise centroid, normalise query at match time, 160×160 inputs,
Normalize([0.5]*3, [0.5]*3) for trained models, no Normalize for raw backbone.
"""
import os, sys, csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

ROOT = Path("/home/buthaina.almulla/Documents/CV7502")
sys.path.insert(0, str(ROOT))
from facenet_pytorch import InceptionResnetV1
from model import Embeddinghead, ArcFaceLoss

SPLIT = ROOT / "datasets/droneface/split"
CKPT_DIR = ROOT / "checkpoints" / "checkpoints"
RESULTS = ROOT / "results"; RESULTS.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_IDS = ["A","B","C","D","E","F","G","H"]
VAL_IDS = ["I"]
HELDOUT_IDS = ["J","K"]


def list_imgs(folder):
    return sorted([f for f in os.listdir(folder)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))])


def collect():
    items = []
    for split, ids in [("train", TRAIN_IDS), ("validation", VAL_IDS), ("test", HELDOUT_IDS)]:
        for ident in ids:
            folder = SPLIT / split / ident
            for fn in list_imgs(folder):
                items.append((str(folder / fn), ident))
    return items


@torch.no_grad()
def embed(items, transform, backbone, head=None):
    backbone.eval()
    if head is not None: head.eval()
    out, lbls = [], []
    paths = [p for p, _ in items]
    labels = [l for _, l in items]
    for s in range(0, len(paths), 64):
        chunk = paths[s:s+64]
        x = torch.stack([transform(Image.open(p).convert("RGB")) for p in chunk]).to(DEVICE)
        feat = backbone(x)
        if head is not None: feat = head(feat)
        out.append(feat.cpu())
    return torch.cat(out, dim=0), np.array(labels)


def loo_rank1(emb, lbls):
    """Mean-then-normalise centroid LOO; returns per-identity (correct,total) and overall."""
    unique = np.unique(lbls)
    counts = {i: int((lbls == i).sum()) for i in unique}
    sums = {i: emb[lbls == i].sum(dim=0) for i in unique}
    q = F.normalize(emb, p=2, dim=1)
    cents = torch.stack([sums[i] / counts[i] for i in unique])
    cent_lbls = np.array(unique)
    correct = {i: 0 for i in unique}; total = {i: 0 for i in unique}
    rank5_total, rank5_correct = 0, 0
    for i in range(emb.shape[0]):
        ident = lbls[i]
        if counts[ident] <= 1:
            total[ident] += 1; continue
        adj = (sums[ident] - emb[i]) / (counts[ident] - 1)
        c = cents.clone()
        c[np.where(cent_lbls == ident)[0][0]] = adj
        cn = F.normalize(c, p=2, dim=1)
        sims = (cn @ q[i]).cpu().numpy()
        order = np.argsort(-sims)
        if cent_lbls[order[0]] == ident: correct[ident] += 1
        if ident in cent_lbls[order[:5]]: rank5_correct += 1
        rank5_total += 1
        total[ident] += 1
    rank1 = sum(correct.values()) / max(1, sum(total.values())) * 100.0
    rank5 = 100.0 * rank5_correct / max(1, rank5_total)
    return correct, total, rank1, rank5


def split_acc(correct, total, ids):
    c = sum(correct[i] for i in ids); t = sum(total[i] for i in ids)
    return 100.0 * c / max(1, t)


def evaluate(label, backbone, head, transform):
    items = collect()
    emb, lbls = embed(items, transform, backbone, head)
    correct, total, r1, r5 = loo_rank1(emb, lbls)
    return {
        "config": label,
        "rank1_overall": r1,
        "rank5_overall": r5,
        "rank1_train_AH": split_acc(correct, total, TRAIN_IDS),
        "rank1_val_I":    split_acc(correct, total, VAL_IDS),
        "rank1_heldout_JK": split_acc(correct, total, HELDOUT_IDS),
    }


def main():
    rows = []

    # --- raw baseline (no Normalize, no head) ---
    raw_t = transforms.Compose([transforms.Resize((160, 160)), transforms.ToTensor()])
    print("evaluating raw_baseline…", flush=True)
    raw = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)
    rows.append(evaluate("raw_baseline", raw, None, raw_t))
    del raw; torch.cuda.empty_cache()

    # --- trained model template ---
    trained_t = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    def load_trained(ckpt_path):
        backbone = InceptionResnetV1(pretrained="vggface2").to(DEVICE)
        head = Embeddinghead(input_dim=512, Embedding_dim=256).to(DEVICE)
        ckpt = torch.load(str(ckpt_path), map_location=DEVICE)
        backbone.load_state_dict(ckpt["backbone"])
        head.load_state_dict(ckpt["head"])
        return backbone, head

    print("evaluating phase1_phase2_original (best_drone_model.pth)…", flush=True)
    bb, hd = load_trained(CKPT_DIR / "best_drone_model.pth")
    rows.append(evaluate("phase1_phase2_original", bb, hd, trained_t))
    del bb, hd; torch.cuda.empty_cache()

    print("evaluating phase1_phase2_with_synth (best_drone_model_synth.pth)…", flush=True)
    bb, hd = load_trained(CKPT_DIR / "best_drone_model_synth.pth")
    rows.append(evaluate("phase1_phase2_with_synth", bb, hd, trained_t))
    del bb, hd; torch.cuda.empty_cache()

    csv_path = RESULTS / "two_phase_ablation_v2.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "config","rank1_overall","rank5_overall",
            "rank1_train_AH","rank1_val_I","rank1_heldout_JK",
        ])
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"\nwrote {csv_path}\n", flush=True)
    for r in rows:
        print(
            f"{r['config']:<28}  "
            f"R1={r['rank1_overall']:5.2f}%  R5={r['rank5_overall']:5.2f}%  |  "
            f"train(A-H)={r['rank1_train_AH']:5.1f}%  "
            f"val(I)={r['rank1_val_I']:5.1f}%  "
            f"heldout(J,K)={r['rank1_heldout_JK']:5.1f}%",
            flush=True,
        )


if __name__ == "__main__":
    main()
