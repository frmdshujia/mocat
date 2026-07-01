"""ECG->CMR cross-modal alignment via symmetric InfoNCE.

Paper section: "Cross-Modal Alignment" (Method) and Implementation Details.

An ECG *student* encoder is trained so that its per-record embedding matches the
embedding a *frozen* CMR *teacher* produces for the same subject. Alignment uses a
symmetric InfoNCE loss with a learnable temperature and per-modality projection
heads. The teacher is any frozen CMR encoder (our MoCaT teacher or a baseline
foundation model); only its pre-computed embeddings are consumed here, so this
script is agnostic to the teacher architecture.

The ECG student is a ViT-MAE ECG encoder; it consumes center-cropped raw ECG.

Checkpointing: every epoch we store the checkpoint with the lowest InfoNCE
validation loss (`best_valloss.ckpt`, used for all reported numbers) and, when
the in-loop probe is enabled, the checkpoint at the peak validation functional
R^2 (`best_peak.ckpt`). `ck['ecg']` holds the aligned ECG-encoder state dict.

Configuration is read from environment variables (see configs/hyperparameters.yaml):
  TEACHER (teacher-embedding prefix), OUT (output dir),
  MAXEP, BS, LR, SEED, PROJ_DIM, TEMP, PROBE_N, PROBE_PEAK, SAVE_EVERY,
  SPLIT_DIR / ECG_BASE / VITMAE_W (ViT-MAE ECG weights placeholder).

Paths below are placeholders; point them at your local checkout. No raw data or
pretrained weights are distributed with this repository.
"""
import os, json, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from multiprocessing import Pool
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# --- paths (placeholders; override via environment) ---
SPLIT_DIR = os.environ.get("SPLIT_DIR", "./data/crossmodal")        # crossmodal_{train,val,test}.csv
ECG_BASE  = os.environ.get("ECG_BASE",  "./data/ecg")               # root that ecg_npy paths are relative to
TEACHER   = os.environ["TEACHER"]        # teacher-embedding prefix; loads {TEACHER}_{split}_emb.npy / _eids.npy
OUT       = os.environ["OUT"]; os.makedirs(OUT, exist_ok=True)

MAXEP     = int(os.environ.get("MAXEP", "20"))
BS        = int(os.environ.get("BS", "64"))
LR        = float(os.environ.get("LR", "3e-4"))
SEED      = int(os.environ.get("SEED", "0"))
PROJ_DIM  = int(os.environ.get("PROJ_DIM", "128"))
TEMP      = float(os.environ.get("TEMP", "0.07"))                   # initial InfoNCE temperature
PROBE_N   = int(os.environ.get("PROBE_N", "8000"))
PROBE_PEAK = int(os.environ.get("PROBE_PEAK", "0"))                 # 1 = compute val functional R^2 each epoch
SAVE_EVERY = int(os.environ.get("SAVE_EVERY", "1"))                 # also dump per-epoch ckpt for post-hoc selection
dev = "cuda"; torch.manual_seed(SEED); np.random.seed(SEED)
FUNC = ["LVEF", "RVEF", "GLS", "GCS", "GRS", "LAEF"]                # functional targets for the in-loop probe

# ---------------------------------------------------------------------------
# ECG preprocessing: center-crop raw 12-lead ECG to 2500 samples (no normalization)
# ---------------------------------------------------------------------------
_CROP = 2500
def ecg_prep(x):
    x = np.asarray(x, np.float32); T = x.shape[1]
    if T < _CROP:
        return None
    st = max(0, (T - _CROP) // 2)
    return np.ascontiguousarray(x[:, st:st + _CROP])

def _load(path):
    try:
        return ecg_prep(np.load(path))
    except Exception:
        return None


def build_split(split):
    """Load paired (ECG, frozen-teacher embedding) tensors for one split."""
    df = pd.read_csv(f"{SPLIT_DIR}/crossmodal_{split}.csv").drop_duplicates("eid").set_index("eid")
    te = np.load(f"{TEACHER}_{split}_emb.npy"); tid = np.load(f"{TEACHER}_{split}_eids.npy").astype(int)
    tmap = {int(e): i for i, e in enumerate(tid)}
    rows = [(int(e), os.path.join(ECG_BASE, df.loc[e, "ecg_npy"])) for e in df.index if int(e) in tmap]
    eids = [e for e, _ in rows]; paths = [p for _, p in rows]
    with Pool(16) as pl:
        res = list(pl.imap(_load, paths, chunksize=32))
    ok = [i for i, r in enumerate(res) if r is not None]
    ecg = np.stack([res[i] for i in ok], 0)
    keep = np.array([eids[i] for i in ok], dtype=int)
    teach = te[[tmap[e] for e in keep]].astype(np.float32)
    lab = df.reindex(keep)[FUNC].values.astype(np.float32)
    print(f"[{split}] N={len(ok)} ecg{ecg.shape} teach{teach.shape}", flush=True)
    return ecg, teach, keep, lab


print(f"building data | teacher={os.path.basename(TEACHER)}", flush=True)
ECG, TEA, EIDS, LAB = {}, {}, {}, {}
for s in ("train", "val"):
    ECG[s], TEA[s], EIDS[s], LAB[s] = build_split(s)

# ---------------------------------------------------------------------------
# ECG student encoder: ViT-MAE ECG (Selivanov et al.); requires their
# layerscale-enabled transformers fork. VITMAE_W is the released HF model dir.
# ---------------------------------------------------------------------------
from transformers import ViTMAEModel
enc = ViTMAEModel.from_pretrained(os.environ.get("VITMAE_W", "./weights/vitmae_ecg"))
enc.config.mask_ratio = 0.0
assert any("layerscale" in k for k in enc.state_dict()), "install the ViT-MAE ECG transformers fork"
enc = enc.to(dev); H = enc.config.hidden_size
def ecg_raw(t):
    return enc(pixel_values=t.unsqueeze(1)).last_hidden_state[:, 1:].mean(1)


def mk_proj(din):
    """Per-modality projection head: Linear(d, 512) -> GELU -> Linear(512, PROJ_DIM)."""
    return nn.Sequential(nn.Linear(din, 512), nn.GELU(), nn.Linear(512, PROJ_DIM)).to(dev)


TEACH_DIM = int(TEA["train"].shape[1])
proj_e = mk_proj(H); proj_c = mk_proj(TEACH_DIM)
logit_scale = nn.Parameter((torch.ones([]) * np.log(1 / TEMP)).to(dev))   # learnable temperature
opt = torch.optim.AdamW(list(enc.parameters()) + list(proj_e.parameters())
                        + list(proj_c.parameters()) + [logit_scale], lr=LR)


def infonce(za, zb):
    """Symmetric InfoNCE over an L2-normalized batch (ECG<->CMR both directions)."""
    za = F.normalize(za, dim=-1); zb = F.normalize(zb, dim=-1)
    lg = za @ zb.T * logit_scale.exp().clamp(max=100)
    lab = torch.arange(len(za), device=za.device)
    return (F.cross_entropy(lg, lab) + F.cross_entropy(lg.T, lab)) / 2


@torch.no_grad()
def embed(arr, idx):
    enc.eval(); out = []
    for i in range(0, len(idx), BS):
        b = idx[i:i + BS]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(ecg_raw(torch.from_numpy(arr[b]).float().to(dev)).float().cpu().numpy())
    enc.train(); return np.concatenate(out, 0)


_probe_idx = np.random.RandomState(123).choice(len(ECG["train"]),
                                               min(PROBE_N, len(ECG["train"])), replace=False)


def func_r2():
    """Mean validation functional R^2 via a frozen linear probe (optional in-loop monitor)."""
    Xtr = embed(ECG["train"], _probe_idx); Xv = embed(ECG["val"], np.arange(len(ECG["val"])))
    sc = StandardScaler().fit(Xtr); Xtr = sc.transform(Xtr); Xv = sc.transform(Xv)
    ytr = LAB["train"][_probe_idx]; yv = LAB["val"]; r2s = []
    for j in range(len(FUNC)):
        mt = np.isfinite(ytr[:, j]); mv = np.isfinite(yv[:, j])
        if mt.sum() < 50 or mv.sum() < 20:
            continue
        m = RidgeCV(alphas=[0.1, 1, 10, 100, 1000]).fit(Xtr[mt], ytr[mt, j])
        r2s.append(r2_score(yv[mv, j], m.predict(Xv[mv])))
    return float(np.mean(r2s)) if r2s else float("nan")


@torch.no_grad()
def val_loss():
    enc.eval(); tot = 0.; nb = 0; N = len(ECG["val"])
    for i in range(0, N - BS + 1, BS):
        idx = np.arange(i, i + BS)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ze = proj_e(ecg_raw(torch.from_numpy(ECG["val"][idx]).float().to(dev)))
            zc = proj_c(torch.from_numpy(TEA["val"][idx]).float().to(dev))
            tot += float(infonce(ze.float(), zc.float())); nb += 1
    enc.train(); return tot / max(nb, 1)


CKPT = f"{OUT}/last.ckpt"; PEAK = f"{OUT}/best_peak.ckpt"; BVL = f"{OUT}/best_valloss.ckpt"
traj = []; best_func = -1e9; best_func_ep = 0; best_vl = 1e9; best_vl_ep = 0; start_ep = 1


def save(path, ep, vl, fr):
    torch.save({"epoch": ep, "valloss": vl, "func_r2": fr, "encoder": "vitmae",
                "ecg": enc.state_dict(), "pe": proj_e.state_dict(), "pc": proj_c.state_dict(),
                "lsc": logit_scale.detach().cpu()}, path)


if os.path.exists(CKPT):                                            # resume
    ck = torch.load(CKPT, map_location=dev)
    enc.load_state_dict(ck["ecg"]); proj_e.load_state_dict(ck["pe"]); proj_c.load_state_dict(ck["pc"])
    logit_scale.data.copy_(ck["lsc"].to(dev))
    traj = ck["traj"]; best_func = ck["best_func"]; best_func_ep = ck["best_func_ep"]
    best_vl = ck["best_vl"]; best_vl_ep = ck["best_vl_ep"]; start_ep = ck["epoch"] + 1
    print(f"== RESUME ep{ck['epoch']} best_func={best_func:.4f}@{best_func_ep} ==", flush=True)
else:
    fr0 = func_r2() if PROBE_PEAK else float("nan"); vl0 = val_loss()
    print(f"ep0 valloss={vl0:.4f} func={fr0:.4f}", flush=True)
    best_func = fr0 if PROBE_PEAK else -1e9
    if PROBE_PEAK:
        save(PEAK, 0, vl0, fr0)
    best_vl = vl0; save(BVL, 0, vl0, fr0)

N = len(ECG["train"])
for ep in range(start_ep, MAXEP + 1):
    enc.train(); perm = np.random.RandomState(1000 + ep).permutation(N); tot = 0.; nb = 0; t0 = time.time()
    for i in range(0, N - BS + 1, BS):
        idx = perm[i:i + BS]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ze = proj_e(ecg_raw(torch.from_numpy(ECG["train"][idx]).float().to(dev)))
            zc = proj_c(torch.from_numpy(TEA["train"][idx]).float().to(dev))
            loss = infonce(ze.float(), zc.float())
        opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss.detach()); nb += 1
    vl = val_loss(); fr = func_r2() if PROBE_PEAK else float("nan")
    traj.append({"ep": ep, "trloss": round(tot / max(nb, 1), 4), "valloss": round(vl, 4),
                 "func_r2": (round(fr, 4) if PROBE_PEAK else None)})
    json.dump(traj, open(f"{OUT}/traj.json", "w"), indent=2)
    peak = PROBE_PEAK and fr > best_func
    if peak:
        best_func = fr; best_func_ep = ep; save(PEAK, ep, vl, fr)
    if vl < best_vl:                                                # checkpoint selection = min val InfoNCE
        best_vl = vl; best_vl_ep = ep; save(BVL, ep, vl, fr)
    if SAVE_EVERY and ep % SAVE_EVERY == 0:
        torch.save({"epoch": ep, "valloss": vl, "encoder": "vitmae", "ecg": enc.state_dict()}, f"{OUT}/ep{ep}.ckpt")
    torch.save({"epoch": ep, "ecg": enc.state_dict(), "pe": proj_e.state_dict(), "pc": proj_c.state_dict(),
                "lsc": logit_scale.detach().cpu(), "traj": traj, "best_func": best_func,
                "best_func_ep": best_func_ep, "best_vl": best_vl, "best_vl_ep": best_vl_ep}, CKPT)
    print(f"ep{ep} trloss={tot/max(nb,1):.4f} valloss={vl:.4f} func={fr:.4f} "
          f"{'*PEAK' if peak else ''} (bvl={best_vl:.4f}@{best_vl_ep}) {time.time()-t0:.0f}s", flush=True)
print(f"DONE peak_func={best_func:.4f}@ep{best_func_ep} | best_valloss@ep{best_vl_ep}", flush=True)
