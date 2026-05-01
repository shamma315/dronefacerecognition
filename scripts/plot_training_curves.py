"""Regenerate training-curve PNGs from the per-epoch metrics CSVs in
results/training_curves/.

Outputs:
  results/training_curves/phase1_curves.png            Figure 1 — Phase 1 (no synth) train loss + val Rank-1 over 30 epochs
  results/training_curves/phase2_loss_curves.png       Figure 2a — train vs val loss for both_phases_original
  results/training_curves/phase2_accuracy_curves.png   Figure 2b — per-split Rank-1 over training for both_phases_original
  results/training_curves/phase2_ablation_comparison.png   Comparison plot — all 4 Phase 2 ablation configs

Idempotent: always overwrites. Safe to re-run after every retrain.
"""
import csv
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path("/home/buthaina.almulla/Documents/CV7502")
CURVES = ROOT / "results" / "training_curves"


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def fcol(rows, key):
    return [float(r[key]) for r in rows]


def plot_phase1_curves():
    """Phase 1 (no synth) train loss + in-training val Rank-1 over 30 epochs.
    The val Rank-1 is gallery-vs-query within the VGGFace2 90/10 split — an
    in-training metric, not LOO over DroneFace."""
    csv_path = CURVES / "phase1_original_per_epoch_metrics.csv"
    if not csv_path.exists():
        print(f"[skip] {csv_path} missing")
        return
    rows = load_csv(csv_path)
    epochs = fcol(rows, "epoch")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(epochs, fcol(rows, "train_loss"), color="#d62728")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("ArcFace cross-entropy loss")
    axes[0].set_title("Phase 1 — train loss"); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, fcol(rows, "val_rank1"), color="#1f77b4")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("Rank-1 (%)")
    axes[1].set_title("Phase 1 — val Rank-1 (VGGFace2 90/10 split)")
    axes[1].grid(True, alpha=0.3); axes[1].set_ylim(0, 50)

    fig.suptitle("Phase 1 — VGGFace2 backbone adaptation (30 epochs)", y=1.02)
    fig.tight_layout()
    out = CURVES / "phase1_curves.png"
    fig.savefig(str(out), dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")


def plot_headline_loss():
    """Phase 2 train + val loss for the headline both_phases_original run."""
    csv_path = CURVES / "phase2_per_epoch_metrics.csv"
    if not csv_path.exists():
        print(f"[skip] {csv_path} missing — re-run scripts/both_phases_original.py")
        return
    rows = load_csv(csv_path)
    epochs = fcol(rows, "epoch")
    has_val = "val_loss" in rows[0]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, fcol(rows, "train_loss"), label="train ArcFace loss", color="#d62728")
    if has_val:
        ax.plot(epochs, fcol(rows, "val_loss"),
                label="val ArcFace loss (10% holdout from A–H)", color="#1f77b4")
    ax.set_xlabel("epoch"); ax.set_ylabel("cross-entropy on ArcFace logits")
    ax.set_title("Phase 2 — train vs. validation loss (both_phases_original)")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    out = CURVES / "phase2_loss_curves.png"
    fig.savefig(str(out), dpi=200); plt.close(fig)
    print(f"wrote {out}")


def plot_headline_accuracy():
    """Phase 2 per-split Rank-1 over training for the headline run."""
    csv_path = CURVES / "phase2_per_epoch_metrics.csv"
    if not csv_path.exists():
        print(f"[skip] {csv_path} missing")
        return
    rows = load_csv(csv_path)
    epochs = fcol(rows, "epoch")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, fcol(rows, "train_rank1"),   label="train Rank-1 (LOO over A–H)", color="#d62728")
    ax.plot(epochs, fcol(rows, "val_rank1"),     label="val Rank-1 (LOO incl. I)",     color="#1f77b4")
    ax.plot(epochs, fcol(rows, "heldout_rank1"), label="held-out Rank-1 (LOO over J,K)", color="#2ca02c")
    ax.set_xlabel("epoch"); ax.set_ylabel("Rank-1 (%)"); ax.set_ylim(0, 100)
    ax.set_title("Phase 2 — per-split Rank-1 over training (both_phases_original)")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    out = CURVES / "phase2_accuracy_curves.png"
    fig.savefig(str(out), dpi=200); plt.close(fig)
    print(f"wrote {out}")


def plot_ablation_comparison():
    """All 4 Phase 2 ablation configs on one plot, per-split Rank-1 over training."""
    configs = [
        ("phase2_only_original", CURVES / "phase2only_original_per_epoch_metrics.csv", "#1f77b4"),
        ("phase2_only_synth",    CURVES / "phase2only_synth_per_epoch_metrics.csv",    "#2ca02c"),
        ("both_phases_original", CURVES / "phase2_per_epoch_metrics.csv",              "#d62728"),
        ("both_phases_synth",    CURVES / "phase2_synth_per_epoch_metrics.csv",        "#9467bd"),
    ]
    have = [(name, p, c) for name, p, c in configs if p.exists()]
    if not have:
        print("[skip] no ablation per_epoch_metrics CSVs found")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    titles = ["train (A–H, LOO)", "val (I, LOO)", "held-out (J,K, LOO)"]
    keys = ["train_rank1", "val_rank1", "heldout_rank1"]
    for ax, title, key in zip(axes, titles, keys):
        for name, path, color in have:
            rows = load_csv(path)
            ax.plot(fcol(rows, "epoch"), fcol(rows, key),
                    label=name, color=color, linewidth=1.4)
        ax.set_xlabel("epoch"); ax.set_title(title); ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Rank-1 (%)")
    axes[-1].legend(loc="lower right", fontsize=9)
    fig.suptitle("Phase 2 ablation — per-split Rank-1 across configs", y=1.02)
    fig.tight_layout()
    out = CURVES / "phase2_ablation_comparison.png"
    fig.savefig(str(out), dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")


def main():
    plot_phase1_curves()
    plot_headline_loss()
    plot_headline_accuracy()
    plot_ablation_comparison()


if __name__ == "__main__":
    main()
