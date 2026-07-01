# MoCaT — Anonymous Code for Double-Blind Review

Code accompanying the submission on **MoCaT** (a Motion-aware Cardiac Teacher for
ECG–CMR representation learning). It is released **for double-blind review** so that
reviewers can verify the method and evaluation protocol. Because the underlying data
and pretrained weights are access-restricted, the code is **not intended to run
end-to-end**; it documents the exact architecture, training recipe, and metrics used
in the paper.

## Contents

| Directory | What it covers |
|---|---|
| `teacher/` | The MoCaT CMR teacher — model (weight-shared Hiera backbone + geometry-aware SAX↔LAX cross-attention with the `−λd²` bias + region/view/phase tokens + multi-task heads), 3-D token-coordinate computation from DICOM geometry, geometry-synchronised augmentation, multi-task phenotype-regression training, and read-out. |
| `cross_modal_alignment/` | ECG↔CMR alignment for Table I — symmetric InfoNCE training of a ViT-MAE ECG student against a frozen CMR teacher, plus the frozen linear-probe / fine-tune evaluation protocol and per-phenotype metrics. |
| `external_validation/` | External validation on EchoNext (Table II) — ECG preprocessing, embedding extraction, linear probing of the covered targets, and the clinical read-out figures (Bland–Altman, ROC). |

Each subdirectory has its own `README.md` with details and the environment variables
it expects (data paths, teacher embeddings, released baseline weights).

## Configuration ↔ paper

All hyperparameters live in the per-section configs and **match the paper's
Implementation Details**:
- `teacher/config.yaml` — teacher pretraining (optimizer, cosine schedule + warmup,
  epochs, backbone freezing, batch sizes, geometric-bias initialisation, model
  selection criterion).
- `cross_modal_alignment/configs/hyperparameters.yaml` — alignment (InfoNCE
  temperature, learning rate, batch size, epochs, number of seeds).

## Data & weights

- **UK Biobank** (teacher training / Table I): access-restricted; obtain through the
  UK Biobank application process and point the path variables at your local copy.
- **EchoNext** (Table II): available from PhysioNet under its data-use agreement.
- **Trained weights** (teacher and aligned ECG encoders): **released upon acceptance.**
- **Baseline encoders** (ViT-MAE ECG, CineMA, CMR-T, DiffAE, ViTa, …) are obtained
  from their official repositories/checkpoints and are **not** redistributed here.

## Requirements

Python 3.10 + PyTorch; see `requirements.txt`. Section-specific extras (the
Hiera/SAM2 backbone, the ViT-MAE ECG encoder, etc.) follow their official
repositories — see each subdirectory's README.

_No author, institution, or other identifying information is included, in accordance
with double-blind review._
