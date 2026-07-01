# Cross-Modal Alignment (ECG ↔ CMR)

Code for the cross-modal alignment and evaluation of Table I. It trains an ECG
*student* encoder to match embeddings from a *frozen* CMR *teacher*, then probes
the aligned ECG encoder on paired-cohort phenotypes and disease endpoints.

This directory contains **only the proposed method and its evaluation protocol**.
It is provided for double-blind review to verify method and protocol; it is not
meant to run end-to-end (data and pretrained weights are access-restricted).

## Method

Given paired ECG–CMR subjects, a frozen CMR teacher produces one embedding per
subject. The ECG student is trained with a **symmetric InfoNCE** loss so its
projected embedding matches the teacher's projected embedding for the same
subject (positive) against the rest of the batch (negatives):

- per-modality projection heads `Linear(d, 512) → GELU → Linear(512, 128)`;
- a single **learnable temperature** (logit-scale, initialized to `log(1/0.07)`);
- AdamW, constant LR `3e-4`, batch 64; checkpoint selected at **minimum
  validation InfoNCE loss**; 3 training seeds.

The engine is teacher-agnostic — it consumes only precomputed teacher embeddings,
so the same recipe aligns any CMR teacher (our MoCaT teacher or a baseline
foundation model). The ECG student is a ViT-MAE ECG encoder.

## Evaluation protocol (tuning parity)

Every row of Table I is scored with the **identical frozen linear probe** — same
estimator, same hyper-parameters, same `StandardScaler` (fit on train only),
same `fit = crossmodal_train` / `eval = crossmodal_test` (n = 5886):

- phenotypes → `RidgeCV(alphas=[0.1,1,10,100,1000])` → R²;
- disease endpoints → `LogisticRegression(class_weight=balanced)` → AUROC.

No per-method model selection is performed. Upstream, every CMR teacher is fed
the **identical 15-frame / 5-view CMR input** and reduced to a frozen pooled
embedding; only the teacher then changes. This isolates *teacher quality* and is
the fairness protocol described in the paper. Error bars: std over 3 training
seeds (aligned rows) or a 1000× test-set bootstrap 95% CI (single-checkpoint
rows).

## Files

| file | purpose (paper section) |
|---|---|
| `align_infonce.py`    | ECG↔CMR InfoNCE alignment engine (Cross-Modal Alignment) |
| `extract_ecg_vitmae.py` | frozen features from the aligned ViT-MAE ECG encoder |
| `probe.py`            | frozen linear-probe evaluation → Table I scorecard (Eval Protocol) |
| `metrics.py`          | shared ridge/logistic scoring + bootstrap CI |
| `targets.py`          | phenotype / disease / demographic column groups |
| `dataio.py`           | feature & label IO |
| `configs/hyperparameters.yaml` | all hyper-parameters (match Implementation Details) |

Typical flow: `align_infonce.py` (produces `best_valloss.ckpt` with the aligned
ECG state in `ck['ecg']`) → `extract_ecg_vitmae.py` (train + test features) →
`probe.py` (scorecard).

## External components (not redistributed)

- **ViT-MAE ECG encoder** — Selivanov et al., *Global and Local Contrastive
  Learning for Joint Representations from Cardiac MRI and ECG*, MICCAI 2025.
  Requires their layerscale-enabled `transformers` fork; use the released model.
- **Baseline methods and foundation-model teachers** are **not** included; see
  their official repositories.

## Notes

- All paths in the scripts are placeholders (`./data`, `./features`, `./weights`,
  …) and/or read from environment variables (`SPLIT_DIR`, `ECG_BASE`, `EMB_DIR`,
  `TEACHER`, `OUT`, `VITMAE_W`). Point them at your local checkout.
- No UK Biobank / external data, and no trained weights, are included (to be
  released upon acceptance).
