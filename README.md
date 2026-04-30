# Drone Face Recognition

Face recognition for drone-captured imagery using a two-phase training pipeline on top of a VGGFace2-pretrained InceptionResnetV1 backbone, an ArcFace classification head, and an optional synthetic-identity augmentation step.

## Overview

The model is trained in two phases:

1. **Phase 1** — Fine-tune the backbone + projection head on VGGFace2 (and optionally a set of synthetic identities generated via Stable Diffusion / IP-Adapter FaceID) using ArcFace loss.
2. **Phase 2** — Further fine-tune on the DroneFace dataset with stronger drone-specific augmentations (motion blur, random crops, occlusion).

Evaluation uses leave-one-out cosine-centroid matching: each query image is compared to the L2-normalised mean embedding of every other image of each identity, with both centroid and query L2-normalised at match time.

## Repository structure

```
.
├── src/                    Reusable modules
│   ├── dataset.py            Dataset classes + identity-level splits
│   └── model.py              EmbeddingHead and ArcFaceLoss
├── notebooks/              Exploratory + training notebooks
│   ├── model.ipynb           Dataset preprocessing + initial model dev
│   ├── training_fixed.ipynb  Two-phase training + LOO evaluation
│   └── training_v2.ipynb     Phase 2 with synthetic-identity augmentation
├── scripts/                Standalone training / eval scripts
│   ├── generate_synthetic_identities.py        SD 1.5 text-to-image
│   ├── generate_synthetic_identities_faceid.py SD + IP-Adapter FaceID
│   ├── phase1_with_synth.py                    Phase 1 retrain w/ synth ids
│   ├── phase2_from_synth.py                    Phase 2 fine-tune
│   └── run_ablation_eval.py                    LOO eval across configs
├── models/                 Trained checkpoints (best_model.pth)
├── results/                Metrics, figures, embeddings, caches
│   ├── figures/              Confusion matrix, ROC, training curves
│   ├── training_curves/      Per-epoch loss / accuracy CSVs + PNGs
│   ├── embeddings/           Saved embeddings + filename/label arrays
│   ├── cache/                Cached train/val embeddings (recomputable)
│   ├── *.csv                 Ablation, cross-dataset, per-attribute tables
│   └── classification_report.txt
├── requirements.txt
└── .gitignore
```

Datasets (`datasets/`), checkpoints other than `models/best_model.pth`, the virtual environment (`venv/`), and synthetic identity image dumps (`synthetic_identities/`) are gitignored — they live locally only.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins CUDA 12.1 PyTorch wheels. Adjust the `--extra-index-url` for a different CUDA / CPU build.

You also need:

- The **DroneFace** dataset, prepared into `datasets/droneface/split/{train,validation,test}/<identity>/...`
- (Optional) A **VGGFace2** subset under `datasets/vggface2/{train,val}/...` for Phase 1
- (Optional) Generated synthetic identities under `synthetic_identities/identity_<NNN>/img_<NNN>.jpg`

## Usage

### Run the canonical pipeline (notebooks)

```bash
jupyter lab
# open notebooks/training_fixed.ipynb and run top-to-bottom for the
# two-phase pipeline + LOO evaluation
```

### Run individual stages (scripts)

```bash
# generate 30 synthetic identities × 40 images
python scripts/generate_synthetic_identities_faceid.py

# phase 1 retrain (VGGFace2 + synthetic identities)
python scripts/phase1_with_synth.py

# phase 2 fine-tune from the phase-1 checkpoint
python scripts/phase2_from_synth.py

# ablation eval across raw / phase1+2 / phase1+2-with-synth
python scripts/run_ablation_eval.py
```

Scripts hard-code the project root (`/home/buthaina.almulla/Documents/CV7502`) — adjust the `ROOT` constant if running from a different machine.

## Results

Headline numbers and per-attribute / cross-dataset breakdowns live in `results/*.csv`. Training curves are in `results/training_curves/`. Confusion matrix and ROC are in `results/figures/`.

## Notes

- The LOO protocol in `training_fixed.ipynb` corrects a train/test preprocessing mismatch present in earlier iterations: trained models are evaluated with `Normalize([0.5]*3, [0.5]*3)`, the raw VGGFace2 baseline is evaluated without normalization, and L2 normalization is applied only at match time (not on per-image embeddings before centroid averaging).
- Phase 2 uses heavier augmentations (random rotation up to 25°, motion blur, random erasing) to simulate drone capture conditions.
