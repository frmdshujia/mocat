"""Extract frozen features from an aligned ViT-MAE ECG encoder.

Paper section: Evaluation Protocol (feature extraction for the probe).

Rebuilds the ViT-MAE ECG student, loads the aligned weights (`ck['ecg']` from
align_infonce.py), and writes mean-pooled patch features for one split. ECG is
center-cropped to 2500 samples with no normalization, matching the alignment-time
preprocessing. Requires the ViT-MAE ECG transformers fork (layerscale) on the
path; VITMAE_W points at the released HF model directory (Selivanov et al.).

  python extract_ecg_vitmae.py --ecg_state <run>/best_valloss.ckpt --out_tag <tag> --split test
"""
import os, argparse, numpy as np, torch
from multiprocessing import Pool
import pandas as pd

SPLIT_DIR = os.environ.get("SPLIT_DIR", "./data/crossmodal")
ECG_BASE  = os.environ.get("ECG_BASE",  "./data/ecg")
EMB_DIR   = os.environ.get("EMB_DIR",   "./features")
VITMAE_W  = os.environ.get("VITMAE_W",  "./weights/vitmae_ecg")
_CROP = 2500


def _winit():
    import torch as _t; _t.set_num_threads(1)


def _load(path):
    try:
        x = np.load(path).astype(np.float32)
        if x.shape != (12, 5000):
            return None
        st = max(0, (x.shape[1] - _CROP) // 2)
        return np.ascontiguousarray(x[:, st:st + _CROP])
    except Exception:
        return None


def build(ecg_state):
    from transformers import ViTMAEModel
    enc = ViTMAEModel.from_pretrained(VITMAE_W); enc.config.mask_ratio = 0.0
    ck = torch.load(ecg_state, map_location="cpu", weights_only=False)
    sd = ck["ecg"] if isinstance(ck, dict) and "ecg" in ck else ck
    msg = enc.load_state_dict(sd, strict=False)
    assert any("layerscale" in k for k in enc.state_dict()), "install the ViT-MAE ECG transformers fork"
    print(f"loaded aligned encoder: miss={len(msg.missing_keys)} unexp={len(msg.unexpected_keys)}", flush=True)
    return enc.eval()


@torch.no_grad()
def encode(model, arr, dev, bs=128):
    out = []
    for s in range(0, len(arr), bs):
        t = torch.from_numpy(arr[s:s + bs]).float().to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f = model(pixel_values=t.unsqueeze(1)).last_hidden_state[:, 1:].mean(1)
        out.append(f.float().cpu().numpy())
    return np.concatenate(out, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ecg_state", required=True)
    ap.add_argument("--out_tag", required=True)
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--nworkers", type=int, default=12)
    a = ap.parse_args()
    kind = {"train": "fit", "test": "eval", "val": "val"}[a.split]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(f"{SPLIT_DIR}/crossmodal_{a.split}.csv").drop_duplicates("eid")
    eids = df["eid"].astype(int).values
    paths = [os.path.join(ECG_BASE, p) for p in df["ecg_npy"].values]
    with Pool(a.nworkers, initializer=_winit) as p:
        res = list(p.imap(_load, paths, chunksize=32))
    ok = [i for i, r in enumerate(res) if r is not None]
    arr = np.stack([res[i] for i in ok], 0).astype(np.float32); keep = eids[ok].astype(np.int64)

    emb = encode(build(a.ecg_state).to(dev), arr, dev)
    os.makedirs(EMB_DIR, exist_ok=True)
    np.save(f"{EMB_DIR}/{a.out_tag}_{kind}_emb.npy", emb)
    np.save(f"{EMB_DIR}/{a.out_tag}_{kind}_eids.npy", keep)
    print(f"[{a.out_tag}_{kind}] emb {emb.shape} finite={np.isfinite(emb).all()}", flush=True)


if __name__ == "__main__":
    main()
