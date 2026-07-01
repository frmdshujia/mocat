"""Paper Sec. V (external-evaluation table): frozen 5-fold linear probe on EchoNext.

Complements `clinical_figures.py` (the transfer-based Fig. 4/5). Here we quantify --
directly on the external cohort -- how much structure/function information the frozen
ECG embeddings carry, via a subject-level 5-fold cross-validated linear probe. We
report mean +/- std over folds for the proposed aligned encoder and its no-alignment
ablation.

Targets mirror what the cross-modal alignment learns:
  function  : LVEF (R^2),  RV systolic dysfunction (AUROC)
  structure : IVS, LVPW wall thickness (R^2)

The protocol is identical for every encoder (ours and baselines); only baseline model
code is not part of this repository. Embeddings come from `extract_embeddings.py`
(keys "aligned" / "unaligned"). Data are access-restricted (PhysioNet EchoNext);
this file is provided for protocol review.
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import r2_score, roc_auc_score
import config

# extends the config echo-label set with wall-thickness + RV-function targets
IVS_COL = "ivs_measurement"          # interventricular septum thickness (cm)
LVPW_COL = "lvpw_measurement"        # LV posterior wall thickness (cm)
RV_DYSF_COL = "rv_systolic_dysfunction_moderate_or_greater_flag"  # RV systolic dysfunction (binary)

REG = {"LVEF": config.LVEF_COL, "IVS": IVS_COL, "LVPW": LVPW_COL}   # R^2
CLF = {"RV_dysf": RV_DYSF_COL}                                       # AUROC
ALPHAS = np.logspace(-1, 3, 5)
N_FOLDS, SEED = 5, 0
ENCODERS = ("unaligned", "aligned")   # ablation vs proposed; identical protocol


def reg_cv(X, y):
    m = np.isfinite(y); idx = np.where(m)[0]; Xv, yv = X[m], y[m]
    oof = np.full(len(y), np.nan); scores = []
    for tr, te in KFold(N_FOLDS, shuffle=True, random_state=SEED).split(Xv):
        sc = StandardScaler().fit(Xv[tr])
        mdl = RidgeCV(alphas=ALPHAS).fit(sc.transform(Xv[tr]), yv[tr])
        p = mdl.predict(sc.transform(Xv[te])); oof[idx[te]] = p
        scores.append(r2_score(yv[te], p))
    return float(np.mean(scores)), float(np.std(scores)), oof


def clf_cv(X, y):
    m = np.isfinite(y); idx = np.where(m)[0]; Xv, yv = X[m], y[m].astype(int)
    oof = np.full(len(y), np.nan); scores = []
    for tr, te in StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(Xv, yv):
        sc = StandardScaler().fit(Xv[tr])
        mdl = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(Xv[tr]), yv[tr])
        p = mdl.predict_proba(sc.transform(Xv[te]))[:, 1]; oof[idx[te]] = p
        scores.append(roc_auc_score(yv[te], p))
    return float(np.mean(scores)), float(np.std(scores)), oof


def main():
    meta = pd.read_csv(os.path.join(config.ECHONEXT_DIR, "echonext_metadata_100k.csv"))
    te = meta[meta[config.SPLIT_COL] == "test"].reset_index(drop=True)
    Y = {k: pd.to_numeric(te[c], errors="coerce").to_numpy(float) for k, c in {**REG, **CLF}.items()}

    rows, oof_store = [], {}
    for key in ENCODERS:
        X = np.load(os.path.join(config.EMB_DIR, f"echonext_{key}.npy")).astype(np.float64)
        assert X.shape[0] == len(te)
        rec = {"encoder": key}
        for k in REG:
            mu, sd, oof = reg_cv(X, Y[k]); rec[k], rec[f"{k}_sd"] = mu, sd; oof_store[f"{key}|{k}"] = oof
        for k in CLF:
            mu, sd, oof = clf_cv(X, Y[k]); rec[k], rec[f"{k}_sd"] = mu, sd; oof_store[f"{key}|{k}"] = oof
        rows.append(rec)
        print(f"{key:10s} " + "  ".join(f"{k}={rec[k]:.3f}+/-{rec[f'{k}_sd']:.3f}" for k in {**REG, **CLF}))

    os.makedirs(config.EMB_DIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(config.EMB_DIR, "echonext_probe_table.csv"), index=False)
    np.savez_compressed(os.path.join(config.EMB_DIR, "echonext_oof.npz"),
                        **oof_store, **{f"Y|{k}": Y[k] for k in {**REG, **CLF}})
    print("[saved] echonext_probe_table.csv + echonext_oof.npz")


if __name__ == "__main__":
    main()
