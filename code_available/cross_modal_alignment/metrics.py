"""Per-target scoring for the frozen linear probe.

Paper section: Evaluation Protocol (how R^2 / AUROC are computed).

`_reg` fits ridge regression (leave-one-out alpha selection) on the train split
and reports test R^2. `_clf` fits class-balanced logistic regression and reports
test AUROC. Both add a 1000x test-set bootstrap 95% CI (fixed seed) used for the
single-checkpoint rows of Table I. Non-finite labels are masked per target.

The regressor / classifier and their hyper-parameters are identical for every
method (ours and all baselines) — this is the tuning-parity guarantee stated in
the paper: no per-method model selection is performed.
"""
import numpy as np
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score

ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]


def _reg(Xf, yf, Xt, yt, B=1000, seed=0):
    """Ridge probe -> test R^2 (+ bootstrap 95% CI). Returns None if too few labels."""
    rng = np.random.RandomState(seed)
    mf, mt = np.isfinite(yf), np.isfinite(yt)
    if mf.sum() < 50 or mt.sum() < 20:
        return None
    model = RidgeCV(alphas=ALPHAS).fit(Xf[mf], yf[mf])
    p, y = model.predict(Xt[mt]), yt[mt]
    base = float(r2_score(y, p)); n = len(y); bs = []
    for _ in range(B):
        idx = rng.randint(0, n, n)
        if np.std(y[idx]) > 0:
            bs.append(r2_score(y[idx], p[idx]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"r2": round(base, 4), "std": round(float(np.std(bs)), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)], "n": int(n)}


def _clf(Xf, yf, Xt, yt, B=1000, seed=0):
    """Balanced logistic probe -> test AUROC (+ bootstrap 95% CI). Returns None if degenerate."""
    rng = np.random.RandomState(seed)
    mf, mt = np.isfinite(yf), np.isfinite(yt)
    yf2, yt2 = yf[mf].astype(int), yt[mt].astype(int)
    if mf.sum() < 50 or mt.sum() < 20 or len(np.unique(yf2)) < 2 or len(np.unique(yt2)) < 2:
        return None
    model = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xf[mf], yf2)
    s = model.predict_proba(Xt[mt])[:, 1]
    base = float(roc_auc_score(yt2, s)); n = len(yt2); bs = []
    for _ in range(B):
        idx = rng.randint(0, n, n)
        if len(np.unique(yt2[idx])) == 2:
            bs.append(roc_auc_score(yt2[idx], s[idx]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"auroc": round(base, 4), "std": round(float(np.std(bs)), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)], "pos": int(yt2.sum()), "n": int(n)}
