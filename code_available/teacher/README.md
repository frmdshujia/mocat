# MoCaT — Motion- and geometry-aware CMR teacher (code for review)

Code for the **proposed CMR teacher (MoCaT)**: a multi-view, geometry-aware cardiac
MRI encoder trained by multi-task phenotype regression, whose representation is later
distilled into an ECG encoder (cross-modal alignment is provided by a separate module).

> **Purpose.** Double-blind review — to inspect the method and protocol. Raw data is
> access-controlled, so this is **not** intended to run end-to-end; the priority is
> clarity and exact consistency with the paper's Implementation Details.

## Files → paper sections

| File | Paper section | Content |
|---|---|---|
| `model.py` | §Method | MoCaT model: shared Hiera backbone → P×P region tokens (+region/view/phase embeddings) → **geometry-aware SAX↔LAX cross-attention** with additive `−λ·d²` bias → 2-layer spatio-temporal transformer → per-task heads. Key classes: `CMREncoderV3`, `GeoCorrespondenceFusion`. |
| `geometry.py` | §Method (token 3D coordinates) | Compute each token's **3D physical coordinate** (DICOM patient space, mm) from `ImageOrientationPatient`/`ImagePositionPatient`/`PixelSpacing` → `token_xyz`. |
| `augment.py` | §Method (geometry-synchronised augmentation) | Flip/rotate applied **jointly** to image and `token_xyz`; rotation in centroid-centred coordinates. |
| `train.py` | §Implementation Details | Teacher training: multi-task phenotype regression (z-standardised MSE), freeze→unfreeze, cosine schedule + warmup, mixed precision. |
| `eval.py` | §Experiments | Held-out test evaluation: per-phenotype R²/MAE + disease AUROC. |
| `heads.py`, `labels.py`, `paths.py` | — | Task heads; target column definitions; machine-agnostic paths (env-overridable). |
| `config.yaml` | §Implementation Details | **Complete hyperparameters** (matches the paper item-by-item). |

## The five required components (where to look)

1. **Model structure** (backbone + geo `−λd²` fusion + region/view/phase tokens + multi-task heads) → `model.py` (`CMREncoderV3`, `GeoCorrespondenceFusion`).
2. **Token 3D coordinates from DICOM** → `geometry.py`.
3. **Geometry-synchronised augmentation** → `augment.py`.
4. **Teacher training** (multi-task phenotype regression) → `train.py`.
5. **Full hyperparameters** → `config.yaml` (learned geo scale σ = 12.1 mm; enc lr 1e-5 / head lr 1e-4; 25 epochs; freeze 2; batch 24/12; fp16).

## Dependencies

`torch>=2.1`, `torchvision`, `numpy`, `pandas`, `scikit-learn`, `pydicom` (for `geometry.py`),
and the **Hiera / SAM2** backbone (`pip install` the official `sam2` package; init from the
**public** SAM2.1 Hiera weights `sam2.1_hiera_tiny.pt` — download from the official SAM2 repo).

## Data (not included)

The model consumes preprocessed UK Biobank CMR: per subject a `(16,224,224) float16 [0,1]`
`.npy` (15 used frames), a `teacher_geom/{eid}.npz` (`token_xyz` from `geometry.py`), and a
subject-level split `csv` with the 22 phenotype columns. **UK Biobank is access-controlled;
raw images/labels and trained weights are not distributed here** (weights released on
acceptance). The expected formats are documented in the file docstrings.

## Reproducing the ablations

The ablation table is produced by toggling CLI flags in `train.py`
(`--views {sax,lax,saxlax}`, `--geo_fusion`, `--no_global_attn`, `--spatial_pool P`);
each row = one flag combination. See `config.yaml` for the shared recipe.

## Anonymity note

All machine names, absolute paths, account/user names and institutional identifiers have
been removed; paths are placeholders overridable via environment variables (see `paths.py`).
