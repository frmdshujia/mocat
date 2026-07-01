"""Feature / label IO for the paired ECG-CMR cohort.

Paper section: Evaluation Protocol (data plumbing).

`read_emb` loads pre-extracted, frozen ECG features (see extract_ecg_*.py);
`labels_for` aligns the paired-cohort label table to a given eid order. Paths are
placeholders — point EMB_DIR / SPLIT_DIR at your local checkout.
"""
import os, numpy as np, pandas as pd
from targets import ALL_TARGETS

SPLIT_DIR = os.environ.get("SPLIT_DIR", "./data/crossmodal")   # crossmodal_{train,val,test}.csv
EMB_DIR   = os.environ.get("EMB_DIR",   "./features")          # {model}_{fit,eval}_emb.npy / _eids.npy
RESULT_DIR = os.environ.get("RESULT_DIR", "./scorecards")


def load_split(split):
    df = pd.read_csv(f"{SPLIT_DIR}/crossmodal_{split}.csv")
    return df.drop_duplicates("eid").set_index("eid")


def labels_for(eids, split):
    """Rows = eids (order preserved), cols = ALL_TARGETS present in the table; missing -> NaN."""
    df = load_split(split)
    cols = [c for c in ALL_TARGETS if c in df.columns]
    return df.reindex([int(e) for e in eids])[cols]


def read_emb(model, kind):
    """kind in {fit, eval} -> (features (N, d) float32, eids (N,) int64)."""
    emb = np.load(f"{EMB_DIR}/{model}_{kind}_emb.npy")
    eids = np.load(f"{EMB_DIR}/{model}_{kind}_eids.npy")
    return emb.astype(np.float32), eids.astype(np.int64)
