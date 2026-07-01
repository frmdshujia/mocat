"""Frozen linear-probe evaluation of an aligned ECG encoder (Table I).

Paper section: Evaluation Protocol / Table I.

Given pre-extracted, frozen ECG features for a model, fit a standardizer on the
train split and score every target with the shared ridge / logistic probe
(metrics.py): functional & structural phenotypes -> R^2, disease endpoints ->
AUROC. Features are read from EMB_DIR; results are written as one JSON scorecard.

The exact same probe (same estimator, same alphas, same fit / eval split) is
applied to *every* method in Table I — ours and all baselines — with no
per-method tuning. Baselines are made comparable upstream by feeding each CMR
teacher the identical 15-frame / 5-view input and taking frozen pooled features;
only the teacher then differs. This is the tuning-parity protocol described in
the paper.

  python probe.py --model <feature_tag>
"""
import os, json, argparse, numpy as np
from sklearn.preprocessing import StandardScaler

import targets as T
import dataio as D
from metrics import _reg, _clf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="feature tag (prefix of {tag}_{fit,eval}_emb.npy)")
    a = ap.parse_args()

    Xf, ef = D.read_emb(a.model, "fit")
    Xt, et = D.read_emb(a.model, "eval")
    Lf = D.labels_for(ef, "train")
    Lt = D.labels_for(et, "test")

    sc = StandardScaler().fit(Xf)                       # standardizer fit on train only
    Xf, Xt = sc.transform(Xf), sc.transform(Xt)

    out = {"model": a.model, "n_fit": int(len(ef)), "n_eval": int(len(et)),
           "functional": {"per": {}}, "structural": {"per": {}},
           "disease": {"per": {}}, "demo": {"per": {}}}

    for block, cols in [("functional", T.FUNCTIONAL), ("structural", T.STRUCTURAL)]:
        for c in cols:
            if c in Lf.columns:
                r = _reg(Xf, Lf[c].values.astype(float), Xt, Lt[c].values.astype(float))
                if r:
                    out[block]["per"][c] = r
        vals = [v["r2"] for v in out[block]["per"].values()]
        out[block]["mean_r2"] = round(float(np.mean(vals)), 4) if vals else None

    for c in T.DISEASE:
        if c in Lf.columns:
            r = _clf(Xf, Lf[c].values.astype(float), Xt, Lt[c].values.astype(float))
            if r:
                out["disease"]["per"][c] = r
    aucs = [v["auroc"] for v in out["disease"]["per"].values()]
    out["disease"]["mean_auroc"] = round(float(np.mean(aucs)), 4) if aucs else None

    for c in T.DEMO_CONT:
        if c in Lf.columns:
            r = _reg(Xf, Lf[c].values.astype(float), Xt, Lt[c].values.astype(float))
            if r:
                out["demo"]["per"][c] = r
    for c in T.DEMO_BIN:
        if c in Lf.columns:
            r = _clf(Xf, Lf[c].values.astype(float), Xt, Lt[c].values.astype(float))
            if r:
                out["demo"]["per"][c] = r

    os.makedirs(D.RESULT_DIR, exist_ok=True)
    path = f"{D.RESULT_DIR}/{a.model}_probe.json"
    json.dump(out, open(path, "w"), indent=2)
    print(f"[{a.model}] n_fit={len(ef)} n_eval={len(et)}  "
          f"FUNC={out['functional']['mean_r2']} STRUCT={out['structural']['mean_r2']} "
          f"DISEASE={out['disease']['mean_auroc']} -> {path}")


if __name__ == "__main__":
    main()
