# [Paper §Experiments] Held-out test evaluation: per-phenotype R^2 / MAE and disease AUROC.

"""
eval_ablation.py — score a trained ablation variant on teacher_test.

Reconstructs the model from the checkpoint's stored config, loads encoder+head,
computes MAE/R² (26 regression, spotlight 5 focus functional) + AUROC (7 cls) on
the held-out teacher_test split. Writes metrics.json next to the checkpoint.

  CUDA_VISIBLE_DEVICES=0 python eval_ablation.py --ckpt ../../results/ablation/run01_base/best.pth
"""
from __future__ import annotations
import argparse, json, pathlib, sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from models_cmr_v3 import CMREncoderV3, select_frames
from heads import TaskHead, GatedTaskHead
from labels import REG_COLS, CLS_COLS, FOCUS_REG
from paths import DATA_DIR

BIBM = HERE.parent.parent
TEST_CSV = BIBM / 'data' / 'teacher' / 'teacher_test.csv'
LABEL_STATS = HERE / 'label_stats.json'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--split_csv', default=str(TEST_CSV))
    ap.add_argument('--batch_size', type=int, default=32)
    args = ap.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    stats = json.load(open(LABEL_STATS))
    reg_mean = np.array([stats['reg_norm'][c]['mean'] for c in REG_COLS])
    reg_std = np.array([stats['reg_norm'][c]['std'] for c in REG_COLS])

    ck = torch.load(args.ckpt, map_location=dev)
    cfg = ck['config']
    frames = select_frames(cfg['views'], cfg['use_t1'])
    enc = CMREncoderV3(proj_dim=256, embed_dim=768, spatial_pool=cfg['spatial_pool'],
                       transformer_heads=8, transformer_depth=cfg.get('transformer_depth', 2),
                       backbone=cfg['backbone'], backbone_ckpt=None, freeze_backbone=True,
                       frame_indices=frames, use_delta=cfg['use_delta'],
                       use_view_emb=cfg['view_emb'], use_time_emb=cfg['time_emb'],
                       sax_lax_fusion=cfg.get('sax_lax_fusion', False),
                       deform_token=cfg.get('deform_token', False),
                       geo_fusion=cfg.get('geo_fusion', False),
                       geo_variant=cfg.get('geo_variant', 'iso'),
                       geo_posenc=cfg.get('geo_posenc', False),
                       no_global_attn=cfg.get('no_global_attn', False)).to(dev).eval()
    r = enc.load_state_dict(ck['encoder'], strict=False)
    moe = cfg.get('view_moe', False)
    geo = cfg.get('geo_fusion', False) or cfg.get('geo_posenc', False)
    import os
    geom_dir = os.environ.get('CMR_GEOM_DIR', str(pathlib.Path(DATA_DIR).parent / 'teacher_geom')) if geo else None
    GP2 = cfg['spatial_pool'] ** 2
    head = (GatedTaskHead if moe else TaskHead)(in_dim=768, n_cls=len(CLS_COLS), n_reg=len(REG_COLS)).to(dev).eval()
    head.load_state_dict(ck['head'])
    print(f"[ckpt] {args.ckpt} epoch={ck.get('epoch')} miss={len(r.missing_keys)} unexp={len(r.unexpected_keys)} frames={len(frames)}")

    df = pd.read_csv(args.split_csv)
    df['path'] = df['eid'].apply(lambda e: f"{DATA_DIR}/{e}.npy")
    df = df[df['path'].apply(lambda p: pathlib.Path(p).exists())].reset_index(drop=True)

    REG_N, CLS_N = len(REG_COLS), len(CLS_COLS)
    rp = [[] for _ in range(REG_N)]; cp = [[] for _ in range(CLS_N)]
    paths = df['path'].tolist(); eids = df['eid'].tolist()
    with torch.no_grad():
        for i in range(0, len(paths), args.batch_size):
            xs = []; gs = []
            for k2, p in enumerate(paths[i:i + args.batch_size]):
                a = np.load(p).astype(np.float32)[frames]
                t = torch.from_numpy(a).unsqueeze(1)
                for k in range(t.shape[0]):
                    lo, hi = t[k].min(), t[k].max()
                    if hi > lo: t[k] = (t[k] - lo) / (hi - lo)
                xs.append(t.squeeze(1).unsqueeze(0))
                if geo:
                    try: g = np.load(f"{geom_dir}/{eids[i + k2]}.npz")['token_xyz'].astype(np.float32)[frames]
                    except Exception: g = np.zeros((len(frames), GP2, 3), np.float32)
                    gs.append(torch.from_numpy(g).unsqueeze(0))
            x = torch.cat(xs, 0).to(dev)
            geom = torch.cat(gs, 0).to(dev) if geo else None
            with torch.cuda.amp.autocast():
                if moe:
                    _, z, z_lax, z_sax = enc(x, geom=geom, return_views=True)
                else:
                    _, z = enc(x, geom=geom)
            cls_out, reg_out = head(z.float(), z_lax.float(), z_sax.float()) if moe else head(z.float())
            for j in range(REG_N): rp[j].append(reg_out[j].float().cpu().numpy())
            for j in range(CLS_N): cp[j].append(torch.sigmoid(cls_out[j]).float().cpu().numpy())

    metrics = {}
    for j, c in enumerate(REG_COLS):
        pred = np.concatenate(rp[j]) * reg_std[j] + reg_mean[j]
        y = pd.to_numeric(df[c], errors='coerce').values
        m = ~np.isnan(y)
        if m.sum() < 2:
            metrics[c] = {'mae': None, 'r2': None, 'n': int(m.sum())}; continue
        p, g = pred[m], y[m]
        ss_res = np.sum((g - p) ** 2); ss_tot = np.sum((g - g.mean()) ** 2)
        metrics[c] = {'mae': float(np.abs(p - g).mean()),
                      'r2': float(1 - ss_res / ss_tot) if ss_tot > 0 else None,
                      'n': int(m.sum())}
    cls_metrics = {}
    for j, c in enumerate(CLS_COLS):
        s = np.concatenate(cp[j]); y = pd.to_numeric(df[c], errors='coerce').fillna(-1).values
        v = y >= 0
        try:
            cls_metrics[c] = float(roc_auc_score(y[v], s[v])) if v.sum() >= 2 and len(set(y[v].tolist())) > 1 else None
        except Exception:
            cls_metrics[c] = None
    focus = {c: metrics[c] for c in FOCUS_REG}
    focus_r2 = [metrics[c]['r2'] for c in FOCUS_REG if metrics[c]['r2'] is not None]
    # physics-EF: EF computed from PREDICTED volumes (EDV-ESV)/EDV vs the direct EF head
    def _pred(c):
        j = REG_COLS.index(c); return np.concatenate(rp[j]) * reg_std[j] + reg_mean[j]
    def _r2(p, y):
        mm = ~np.isnan(y) & ~np.isnan(p)
        if mm.sum() < 2: return None
        pp, gg = p[mm], y[mm]; st = np.sum((gg - gg.mean()) ** 2)
        return float(1 - np.sum((gg - pp) ** 2) / st) if st > 0 else None
    physics_ef = {}
    for edv, esv, ef in [('LVEDV', 'LVESV', 'LVEF'), ('RVEDV', 'RVESV', 'RVEF')]:
        EDV, ESV = _pred(edv), _pred(esv)
        ef_phys = np.where(EDV > 1e-6, (EDV - ESV) / EDV * 100.0, np.nan)
        yt = pd.to_numeric(df[ef], errors='coerce').values
        physics_ef[ef] = {'direct_r2': metrics[ef]['r2'], 'physics_r2': _r2(ef_phys, yt)}
    out = {'ckpt': args.ckpt, 'epoch': ck.get('epoch'), 'n_test': len(df),
           'config': cfg, 'focus_r2_mean': float(np.mean(focus_r2)) if focus_r2 else None,
           'focus': focus, 'regression': metrics, 'classification_auroc': cls_metrics,
           'physics_ef': physics_ef}
    outp = pathlib.Path(args.ckpt).parent / 'metrics.json'
    json.dump(out, open(outp, 'w'), indent=2)
    print(f"[saved] {outp}")
    print(f"  focus_r2_mean = {out['focus_r2_mean']}")
    for c in FOCUS_REG:
        print(f"    {c:<6} MAE={metrics[c]['mae']:.3f}  R²={metrics[c]['r2']:.4f}  (n={metrics[c]['n']})")
    for ef, v in physics_ef.items():
        if v['direct_r2'] is not None and v['physics_r2'] is not None:
            print(f"  [physics-EF] {ef}: direct={v['direct_r2']:.4f} → physics={v['physics_r2']:.4f}  (Δ{v['physics_r2']-v['direct_r2']:+.4f})")


if __name__ == '__main__':
    main()
