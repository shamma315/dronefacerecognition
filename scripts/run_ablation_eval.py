"""Run loo_eval over the explicit 2x3 ablation matrix and write
results/two_phase_ablation.csv.

Configurations (all Phase 2 runs share the v2 hyperparams: bb lr 1e-6,
head/ArcFace lr 1e-4, ArcFace s=32 m=0.3, 4 layers unfrozen — so the
"original vs synth" axis and "phase1_only / phase2_only / both_phases"
axis are isolated):

  raw_baseline           — InceptionResnetV1(pretrained='vggface2'), no head
  phase1_only_original   — best_model.pth                            (Phase 1 only, no synth; ArcFace s=64 m=0.5 — historical Phase 1 hyperparams)
  phase2_only_original   — best_drone_model_phase2only.pth           (Phase 2 from raw VGGFace2, no Phase 1 pretrain, no synth)
  both_phases_original   — best_drone_model_v2.pth                   (Phase 1 + Phase 2)
  phase1_only_synth      — best_model_synth.pth                      (Phase 1 only on VGGFace2 + 30 synth ids)
  phase2_only_synth      — best_drone_model_phase2only_synth.pth     (Phase 2 from raw VGGFace2 on DroneFace + 30 synth ids as classes)
  both_phases_synth      — best_drone_model_synth_v2.pth             (Phase 1 with synth + Phase 2)

LOO protocol (mean-then-normalise centroid, normalise query at match time):
mean-then-normalise centroid, normalise query at match time, 160x160 inputs,
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
sys.path.insert(0, str(ROOT / "src"))
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
    out = []
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


def save_artifacts(label, emb, lbls):
    """For the 'headline' config, refresh the inputs that analysis_extensions.ipynb
    consumes (embeddings, labels, confusion matrix). This keeps the per-class /
    ROC / failure-cases figures in sync with whichever checkpoint we treat as
    the headline two-phase model."""
    emb_dir = RESULTS / "embeddings"; emb_dir.mkdir(exist_ok=True)
    np.save(emb_dir / "two_phase_droneface.npy", emb.numpy())
    np.save(emb_dir / "two_phase_droneface_labels.npy", lbls)

    unique = sorted(np.unique(lbls).tolist())
    id_to_idx = {i: k for k, i in enumerate(unique)}
    n = len(unique)
    sums = {i: emb[lbls == i].sum(dim=0) for i in unique}
    counts = {i: int((lbls == i).sum()) for i in unique}
    cm = np.zeros((n, n), dtype=np.int64)
    q = F.normalize(emb, p=2, dim=1)
    cents = torch.stack([sums[i] / counts[i] for i in unique])
    cent_lbls = np.array(unique)
    for i in range(emb.shape[0]):
        ident = lbls[i]
        if counts[ident] <= 1:
            cm[id_to_idx[ident], id_to_idx[ident]] += 1
            continue
        adj = (sums[ident] - emb[i]) / (counts[ident] - 1)
        c = cents.clone()
        c[np.where(cent_lbls == ident)[0][0]] = adj
        cn = F.normalize(c, p=2, dim=1)
        sims = (cn @ q[i]).cpu().numpy()
        pred = cent_lbls[int(np.argmax(sims))]
        cm[id_to_idx[ident], id_to_idx[pred]] += 1
    np.save(RESULTS / "confusion_matrix.npy", cm)
    print(f"[{label}] refreshed embeddings + confusion_matrix.npy", flush=True)


def evaluate(label, backbone, head, transform, save=False):
    items = collect()
    emb, lbls = embed(items, transform, backbone, head)
    correct, total, r1, r5 = loo_rank1(emb, lbls)
    if save:
        save_artifacts(label, emb, lbls)
    return {
        "config": label,
        "rank1_overall": round(r1, 2),
        "rank5_overall": round(r5, 2),
        "rank1_train_AH": round(split_acc(correct, total, TRAIN_IDS), 2),
        "rank1_val_I": round(split_acc(correct, total, VAL_IDS), 2),
        "rank1_heldout_JK": round(split_acc(correct, total, HELDOUT_IDS), 2),
    }


# (config_name, checkpoint_filename) — checkpoint_filename=None means raw baseline
ABLATION_CONFIGS = [
    ("raw_baseline",         None),
    ("phase1_only_original", "best_model.pth"),
    ("phase2_only_original", "best_drone_model_phase2only.pth"),
    ("both_phases_original", "best_drone_model_v2.pth"),
    ("phase1_only_synth",    "best_model_synth.pth"),
    ("phase2_only_synth",    "best_drone_model_phase2only_synth.pth"),
    ("both_phases_synth",    "best_drone_model_synth_v2.pth"),
]


def load_trained(ckpt_path):
    backbone = InceptionResnetV1(pretrained="vggface2").to(DEVICE)
    head = Embeddinghead(input_dim=512, Embedding_dim=256).to(DEVICE)
    ckpt = torch.load(str(ckpt_path), map_location=DEVICE)
    backbone.load_state_dict(ckpt["backbone"])
    head.load_state_dict(ckpt["head"])
    return backbone, head


def main():
    rows = []

    raw_t = transforms.Compose([transforms.Resize((160, 160)), transforms.ToTensor()])
    trained_t = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    for label, ckpt_name in ABLATION_CONFIGS:
        if ckpt_name is None:
            print(f"evaluating {label}...", flush=True)
            bb = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)
            rows.append(evaluate(label, bb, None, raw_t))
            del bb
        else:
            ckpt_path = CKPT_DIR / ckpt_name
            if not ckpt_path.exists():
                print(f"SKIP {label}: missing checkpoint {ckpt_path}", flush=True)
                rows.append({
                    "config": label,
                    "rank1_overall": "",
                    "rank5_overall": "",
                    "rank1_train_AH": "",
                    "rank1_val_I": "",
                    "rank1_heldout_JK": "",
                })
                continue
            print(f"evaluating {label} ({ckpt_name})...", flush=True)
            bb, hd = load_trained(ckpt_path)
            # both_phases_original is the headline two-phase model — also refresh
            # the artifacts that analysis_extensions.ipynb consumes.
            rows.append(evaluate(label, bb, hd, trained_t, save=(label == "both_phases_original")))
            del bb, hd
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    csv_path = RESULTS / "two_phase_ablation.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "config", "rank1_overall", "rank5_overall",
            "rank1_train_AH", "rank1_val_I", "rank1_heldout_JK",
        ])
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"\nwrote {csv_path}\n", flush=True)
    for r in rows:
        r1 = r["rank1_overall"]; r5 = r["rank5_overall"]
        if r1 == "":
            print(f"{r['config']:<22}  (skipped — checkpoint missing)", flush=True)
        else:
            print(
                f"{r['config']:<22}  R1={r1:5.2f}%  R5={r5:5.2f}%  |  "
                f"train(A-H)={r['rank1_train_AH']:5.1f}%  "
                f"val(I)={r['rank1_val_I']:5.1f}%  "
                f"heldout(J,K)={r['rank1_heldout_JK']:5.1f}%",
                flush=True,
            )


if __name__ == "__main__":
    main()
