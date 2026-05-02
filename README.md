# Drone Face Recognition

Face recognition for drone-captured imagery using a two-phase training pipeline on top of a VGGFace2-pretrained InceptionResnetV1 backbone, an ArcFace head, and an optional synthetic-identity augmentation step.

**Headline result:** the proposed two-phase pipeline reaches **82.48% Rank-1 / 97.58% Rank-5** on DroneFace under leave-one-out cosine-centroid matching, beating the raw VGGFace2 backbone (78.37% / 96.19%) by **+4.11 pp R-1**.

A self-contained demo with sample inputs and outputs lives at `notebooks/demo.ipynb` — open it for an inline visualization of the model's predictions on 8 hand-curated DroneFace queries.

## Pipeline

| stage | data | what it does |
|---|---|---|
| **Phase 1** | VGGFace2 (~480 train ids; optionally + 30 synthetic ids) | Adapt the InceptionResnetV1 backbone + 256-d projection head under ArcFace, with UAV-style augmentations (rotation, blur, low-res, RandomErasing). 30 epochs, batch 256, ArcFace `s=64 m=0.5`. |
| **Phase 2** | DroneFace (8 train identities A–H, optionally + 30 synthetic ids) | Fine-tune Phase 1's backbone + head on DroneFace. 50 epochs, batch 32, ArcFace `s=32 m=0.3` (smaller scale appropriate for the 8-id head). Backbone lr `1e-6`, head/ArcFace lr `1e-4`. Unfreezes `block8`, `avgpool_1a`, `last_linear`, `last_bn`. |
| **Eval** | DroneFace 11 identities × 124 images | Leave-one-out cosine-centroid Rank-1/Rank-5. Centroid = L2-normalize(mean of remaining same-identity embeddings); query L2-normalized at match time. |

The 7-row ablation matrix is documented in `results/two_phase_ablation.csv` and discussed in `notebooks/training_v2.ipynb`.

## Repository structure

```
.
├── src/                              Reusable modules
│   ├── dataset.py                      DroneFace + VGGFace2 dataset classes
│   └── model.py                        Embeddinghead + ArcFaceLoss
├── notebooks/
│   ├── training_v2.ipynb               Reproducible orchestrator — calls every script in order
│   ├── analysis_extensions.ipynb       t-SNE / per-identity / failure-case figures
│   └── demo.ipynb                      Sample-input + sample-output demo (run end-to-end in ~3 s)
├── scripts/                          Standalone training + eval scripts (idempotent: skip if output exists, --force to rerun)
│   ├── generate_synthetic_identities_faceid.py    30 synthetic identities × 40 imgs (IP-Adapter FaceID)
│   ├── phase1_original.py                         Phase 1 on VGGFace2, no synth          → best_model.pth
│   ├── phase1_with_synth.py                       Phase 1 on VGGFace2 + synth ids        → best_model_synth.pth
│   ├── phase2_only_original.py                    Phase 2 from raw, DroneFace only       → best_drone_model_phase2only.pth
│   ├── phase2_only_synth.py                       Phase 2 from raw, DroneFace + synth    → best_drone_model_phase2only_synth.pth
│   ├── both_phases_original.py                    Phase 2 from Phase 1 (no synth)        → best_drone_model_v2.pth         [headline]
│   ├── phase2_from_synth.py                       Phase 2 from Phase 1 with synth        → best_drone_model_synth_v2.pth
│   ├── run_ablation_eval.py                       LOO eval over all 7 configs            → results/two_phase_ablation.csv
│   ├── benchmark_recognizers.py                   Reference recognizers (LBPH, SFace, MobileFaceNet, FaceNet ×2, ArcFace ×2)
│   └── run_all_benchmarks.py                      Sweep harness for benchmark_recognizers.py
├── checkpoints/checkpoints/          Trained checkpoints (.pth, gitignored)
├── results/
│   ├── two_phase_ablation.csv          7-row ablation matrix (headline + supporting)
│   ├── confusion_matrix.npy            from best_drone_model_v2.pth
│   ├── roc_data.npy                    pairwise cosine similarities for ROC
│   ├── embeddings/                     two_phase_droneface{,_labels}.npy
│   ├── cache/                          raw VGGFace2 baseline embeddings (for t-SNE comparison)
│   ├── training_curves/                Per-epoch metrics + PNGs (Figure 2 source)
│   ├── analysis/                       t-SNE / per-identity / failure-case figures (Figs 3, 5, 6)
│   ├── benchmarks/                     Reference recognizer comparison tables (Tables 2, 4–6)
│   ├── demo_predictions.png            Demo figure — 8 sample inputs with top-3 predictions
│   └── demo_predictions.txt            Demo text summary (true / predicted / ✓ ✗ for each sample)
├── synthetic_identities/             30 IP-Adapter FaceID identities (gitignored)
├── datasets/                         DroneFace + VGGFace2 (gitignored)
├── requirements.txt
└── .gitignore
```

Datasets, checkpoints, the venv, and synthetic identity image dumps are gitignored.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins CUDA 12.1 PyTorch wheels. Adjust the `--extra-index-url` for a different CUDA/CPU build.

Required directories (gitignored):

- `datasets/droneface/split/{train,validation,test}/<identity>/...` — DroneFace identity-disjoint split (A–H train, I val, J,K test)
- `datasets/vggface2/{train,val}/<identity>/...` — VGGFace2 subset for Phase 1
- (auto-generated) `synthetic_identities/identity_NNN/{anchor.jpg, img_001..040.jpg, anchor_embed.npy}`

## Usage

### Demo (sample inputs and outputs)

```bash
jupyter lab
# open notebooks/demo.ipynb and run all cells (~3 s)
```

Loads the headline `best_drone_model_v2.pth` checkpoint, runs inference on 8 hand-curated DroneFace queries spanning easy/hard identities, the validation identity (I), the held-out identities (J, K), and the hardest altitudes/distances. Produces:

- inline image-with-prediction visualizations for each sample
- `results/demo_predictions.png` — composite figure
- `results/demo_predictions.txt` — text summary (`true=X  pred=Y  CORRECT/INCORRECT  top-3: ...`)

### Reproducible end-to-end run

```bash
jupyter lab
# open notebooks/training_v2.ipynb and run top-to-bottom
```

The orchestrator subprocess-calls every script in dependency order. Each script skips if its output checkpoint already exists. Pass `--force` to retrain.

### Individual scripts

```bash
# 1. generate 30 synthetic identities × 40 images (~30 min)
python scripts/generate_synthetic_identities_faceid.py

# 2. Phase 1 (~3-4 hr each)
python scripts/phase1_original.py
python scripts/phase1_with_synth.py

# 3. Phase 2 — the four ablation runs (~30 min each)
python scripts/phase2_only_original.py
python scripts/phase2_only_synth.py
python scripts/both_phases_original.py
python scripts/phase2_from_synth.py

# 4. Eval all 7 configs → results/two_phase_ablation.csv
python scripts/run_ablation_eval.py

# 5. Reference recognizers (LBPH, SFace, MobileFaceNet, FaceNet ×2, ArcFace ×2)
python scripts/run_all_benchmarks.py \
  --droneface-root datasets/droneface/open_data_set \
  --vggface2-root datasets/vggface2
```

Scripts hard-code the project root via the `ROOT` constant — adjust if running on a different machine.

## Results

| where | what |
|---|---|
| `results/two_phase_ablation.csv` | 7-row ablation matrix |
| `results/benchmarks/{droneface_main,cross_dataset,per_distance,per_gender,per_altitude}.csv` | Reference-recognizer comparison + Ours rows |
| `results/training_curves/phase2_per_epoch_metrics.csv` + PNGs | Phase 2 per-epoch curves (Figure 2 source) |
| `results/analysis/*.png` | t-SNE, confusion matrix, ROC, per-identity bars, top-10 failures |
| `results/demo_predictions.{png,txt}` | Demo outputs (sample inputs + model predictions) — see `notebooks/demo.ipynb` |

## Hyperparameters at a glance

| param | Phase 1 | Phase 2 |
|---|---|---|
| epochs | 30 | 50 |
| batch size | 256 | 32 |
| optimizer | Adam | Adam |
| backbone lr | 1e-5 | 1e-6 |
| head/ArcFace lr | 1e-4 | 1e-4 |
| LR schedule | cosine → 1e-6 | cosine → 1e-7 |
| ArcFace scale `s` | 64.0 | 32.0 |
| ArcFace margin `m` | 0.5 | 0.3 |
| backbone unfrozen | `block8`, `avgpool_1a` | `block8`, `avgpool_1a`, `last_linear`, `last_bn` |
| input size | 112×112 | 112×112 (train) / 160×160 (eval) |

ArcFace s/m differ by phase intentionally: Phase 1 trains on ~480 classes (large scale + larger margin is standard ArcFace practice for identity-rich pretraining); Phase 2 fine-tunes on 8 classes (smaller margin avoids over-tightening).

## Notes

- **LOO eval protocol:** trained models use `Normalize([0.5]*3, [0.5]*3)` (matches training-time preprocessing); the raw VGGFace2 baseline uses no Normalize (matches `facenet-pytorch` convention). L2 normalization is applied at match time on centroids and queries — not on per-image embeddings before centroid averaging.
- **Identity-disjoint split:** A–H train (8 ids), I validation (1 id), J,K held-out test (2 ids). The model never sees images of I/J/K during gradient steps.
- **Phase 2 augmentations** (lighter than Phase 1): RandomResizedCrop(112, 0.8–1.0), HorizontalFlip, RandomRotation(20°), GaussianBlur(3), ColorJitter(0.2, 0.2). No RandomErasing or Grayscale.

## Dataset Download Links: 
- VGGFace2: 
https://www.kaggle.com/datasets/hearfool/vggface2

- DroneFace:
https://www.dropbox.com/s/c9odbl7eckavten/DnHFaces.zip?dl=1
