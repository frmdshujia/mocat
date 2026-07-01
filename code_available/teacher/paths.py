# [Paper] Centralised, machine-agnostic paths (override via environment variables).
"""paths.py — all locations are environment-overridable and default to placeholders
relative to the repo root. Raw data (UK Biobank CMR) and backbone/teacher checkpoints
are NOT included (controlled access / released on acceptance); set the env vars below
to your local copies to run."""
from __future__ import annotations
import os
import pathlib

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = pathlib.Path(os.environ.get('PROJECT_ROOT', str(_HERE.parent.parent)))

# UK Biobank preprocessed CMR: (16,224,224) float16 in [0,1]  (NOT included)
DATA_DIR = os.environ.get('CMR_V3_DIR', str(_ROOT / 'data' / 'preprocessed_cmr_v3'))
# Per-token 3D geometry from geometry.py: teacher_geom/{eid}.npz (token_xyz)
GEOM_DIR = os.environ.get('CMR_GEOM_DIR', str(_ROOT / 'data' / 'teacher_geom'))
# Subject-level leakage-free splits (csv w/ phenotype + disease columns)
SPLIT_DIR = pathlib.Path(os.environ.get('CMR_SPLIT_DIR', str(_ROOT / 'data' / 'teacher')))
TRAIN_CSV = str(SPLIT_DIR / 'teacher_train.csv')
VAL_CSV   = str(SPLIT_DIR / 'teacher_val.csv')
TEST_CSV  = str(SPLIT_DIR / 'teacher_test.csv')
# Backbone init: public SAM2.1 Hiera weights (download from the official SAM2 repo)
SAM2_CKPT_DIR = pathlib.Path(os.environ.get('SAM2_CKPT_DIR', str(_ROOT / 'weights' / 'sam2')))
BACKBONE_CKPTS = {
    'sam2_tiny':  str(SAM2_CKPT_DIR / 'sam2.1_hiera_tiny.pt'),
    'sam2_small': str(SAM2_CKPT_DIR / 'sam2.1_hiera_small.pt'),
}
