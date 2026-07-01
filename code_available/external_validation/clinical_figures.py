"""Paper Fig. 4-5 (external validation): clinical agreement & discrimination figures.

From frozen-encoder embeddings on the internal (UKB) and external (EchoNext) cohorts, produce
the publication figures:
  - Bland-Altman agreement for predicted vs. true LVEF (external);
  - ROC for detecting  LVEF<40% (HFrEF),  LVEF<50%,  and  40-50% (HFmrEF) vs >=50%,
    comparing the proposed aligned encoder against the no-alignment ablation, with bootstrap 95% CI.

Protocol (matches Implementation Details):
  - Predicted LVEF: RidgeCV linear probe fit on INTERNAL frozen embeddings and applied, frozen,
    to EXTERNAL embeddings (strict external transfer, clinical units).
  - AUROC 95% CI: 1000x bootstrap. Operating point: Youden's J on the predicted score
    (recalibrated, because compressed predictions make a fixed clinical LVEF cutoff mis-calibrated
    on an external cohort with different case mix).
"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_curve, roc_auc_score
import config

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.linewidth": 0.9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "savefig.dpi": 350})                     # full box (all four spines) = journal style
OURS, BASE, ACC = "#2A9D8F", "#B0AFA8", "#C44E52"


def fit_internal_lvef_probe(internal_emb, internal_lvef):
    """RidgeCV probe on internal frozen embeddings -> (scaler, probe)."""
    m = np.isfinite(internal_lvef)
    sc = StandardScaler().fit(internal_emb[m].astype(np.float64))
    pr = RidgeCV(alphas=np.logspace(-1, 3, 10)).fit(sc.transform(internal_emb[m].astype(np.float64)), internal_lvef[m])
    return sc, pr


def transfer_predict(scaler, probe, external_emb):
    """Apply the frozen internal probe to external embeddings -> predicted LVEF (%)."""
    return probe.predict(scaler.transform(external_emb.astype(np.float64)))


def auc_ci(label, score, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed); idx = np.arange(len(label)); vals = []
    for _ in range(n_boot):
        b = rng.choice(idx, len(idx), replace=True)
        if 0 < label[b].sum() < len(b):
            vals.append(roc_auc_score(label[b], score[b]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def roc_figure(y_true, pred_base, pred_ours, mask, positive, title_note, out_png):
    """One ROC panel: positive = boolean label array; mask selects the evaluated subgroup."""
    yy = positive[mask].astype(int)
    fig, ax = plt.subplots(figsize=(3.5, 3.1))
    for name, pred, c in [("Baseline", pred_base, BASE), ("Ours", pred_ours, OURS)]:
        s = -pred[mask]                                     # lower predicted LVEF -> higher risk
        fpr, tpr, _ = roc_curve(yy, s); auc = roc_auc_score(yy, s); lo, hi = auc_ci(yy, s)
        ax.plot(fpr, tpr, color=c, lw=2.3, label=f"{name}  {auc:.3f} ({lo:.2f}-{hi:.2f})", solid_capstyle="round")
        j = np.argmax(tpr - fpr)                            # Youden recalibrated operating point
        ax.scatter([fpr[j]], [tpr[j]], color=c, s=28, zorder=5, edgecolor="white", lw=1.0)
    ax.plot([0, 1], [0, 1], color="#cccccc", ls="--", lw=0.9)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("1 - specificity"); ax.set_ylabel("Sensitivity")
    ax.legend(loc="lower right", frameon=False)             # legend lower-right (does not occlude curves)
    fig.tight_layout(); fig.savefig(out_png, bbox_inches="tight"); plt.close(fig)


def bland_altman(y_true, pred_ours, mask, out_png):
    y = y_true[mask]; p = pred_ours[mask]; diff = p - y; bias = diff.mean(); sd = diff.std()
    fig, ax = plt.subplots(figsize=(3.5, 3.1))
    ax.hexbin((p + y) / 2, diff, gridsize=32, cmap="Blues", mincnt=1, linewidths=0)
    ax.axhline(bias, color=ACC, lw=1.4)
    ax.axhline(bias - 1.96 * sd, color=ACC, ls="--", lw=1.1); ax.axhline(bias + 1.96 * sd, color=ACC, ls="--", lw=1.1)
    ax.axhline(0, color="#888888", lw=0.6)
    ax.text(0.97, 0.96, f"bias {bias:+.1f}%\n95% LoA +/-{1.96 * sd:.1f}%", transform=ax.transAxes,
            fontsize=7.5, va="top", ha="right", color=ACC,
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="none", boxstyle="round,pad=0.3"))
    ax.set_xlabel("Mean of predicted & true LVEF (%)"); ax.set_ylabel("Predicted - true LVEF (%)")
    fig.tight_layout(); fig.savefig(out_png, bbox_inches="tight"); plt.close(fig)


def main():
    os.makedirs(config.FIG_DIR, exist_ok=True)
    # internal embeddings + LVEF (to fit the transfer probe) -- restricted, not included
    Ei = np.load(os.path.join(config.INTERNAL_DIR, "internal_{key}_emb.npy"))       # placeholder layout
    yi = pd.read_csv(os.path.join(config.INTERNAL_DIR, "internal_labels.csv"))[config.LVEF_COL].to_numpy(float)
    # external labels
    meta = pd.read_csv(os.path.join(config.ECHONEXT_DIR, "echonext_metadata_100k.csv"))
    te = meta[meta[config.SPLIT_COL] == "test"].reset_index(drop=True)
    y = pd.to_numeric(te[config.LVEF_COL], errors="coerce").to_numpy(float)
    m = np.isfinite(y); y = y[m]

    preds = {}
    for key in ("unaligned", "aligned"):
        Ei_k = np.load(os.path.join(config.INTERNAL_DIR, f"internal_{key}_emb.npy"))
        sc, pr = fit_internal_lvef_probe(Ei_k, yi)
        Ee = np.load(os.path.join(config.EMB_DIR, f"echonext_{key}.npy"))
        preds[key] = transfer_predict(sc, pr, Ee)[m]
    pb, po = preds["unaligned"], preds["aligned"]

    allm = np.ones(len(y), bool); ge40 = y >= 40
    bland_altman(y, po, allm, os.path.join(config.FIG_DIR, "fig_ba_lvef.pdf"))
    roc_figure(y, pb, po, allm, y < 40, "HFrEF", os.path.join(config.FIG_DIR, "fig_roc_lvef40.pdf"))
    roc_figure(y, pb, po, allm, y < 50, "LVEF<50", os.path.join(config.FIG_DIR, "fig_roc_lvef50.pdf"))
    roc_figure(y, pb, po, ge40, y < 50, "HFmrEF vs preserved", os.path.join(config.FIG_DIR, "fig_roc_hfmref.pdf"))
    print("wrote Fig. 4-5 panels to", config.FIG_DIR)


if __name__ == "__main__":
    main()
