# [Paper §Implementation Details] MoCaT teacher training: multi-task phenotype regression (z-standardised MSE), freeze->unfreeze backbone, cosine schedule + warmup, mixed precision. CLI flags (views/geo_fusion/no_global_attn/spatial_pool) reproduce the ablation table.

"""
train_ablation.py — single-GPU multi-task trainer for the BIBM2026 G3 ablation study.

Trains CMREncoderV3 (config-driven) + TaskHead on the SAME multi-task label set
(26 regression + 7 classification) from data/teacher/, for every ablation variant.
Reports MAE/R² (regression) — model selection = val mean R² over the 5 focus
functional metrics (LVEF, RVEF, GLS, GCS, GRS).

Reuses the canonical train_cmr_v3.py recipe (augment, freeze→unfreeze, cosine,
AMP, grad-clip) but single-GPU (a storage server, ≤2 concurrent) and the new label set.

Ablation knobs (all CLI):
  --backbone {sam2_tiny,medsam2_heart,...}   per-frame backbone
  --views {saxlax,sax,lax}  --use_t1/--no_t1  frame subset (study = --no_t1)
  --use_delta/--no_delta                      ED→ES difference tokens
  --view_emb/--no_view_emb  --time_emb/--no_time_emb   positional encodings
  --spatial_pool {1,2,4,8}                    region-token granularity

Example (BASE = P1-B1 = P2-D0):
  CUDA_VISIBLE_DEVICES=0 python train_ablation.py --backbone sam2_tiny \
    --views saxlax --no_t1 --no_delta --view_emb --time_emb --spatial_pool 4 \
    --out_dir ../../results/ablation/run01_base --epochs 25
"""
from __future__ import annotations
import argparse, json, math, pathlib, random, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF
from sklearn.metrics import roc_auc_score

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from models_cmr_v3 import CMREncoderV3, select_frames
from heads import TaskHead, GatedTaskHead
from labels import REG_COLS, CLS_COLS, FOCUS_REG, STRUCT_REG, FUNC_REG
TRAIN_REG_IDX = None
TRAIN_CLS_IDX = None
from paths import DATA_DIR, BACKBONE_CKPTS

BIBM = HERE.parent.parent
TEACHER_DIR = BIBM / 'data' / 'teacher'
TRAIN_CSV = TEACHER_DIR / 'teacher_train.csv'
VAL_CSV   = TEACHER_DIR / 'teacher_val.csv'
TEST_CSV  = TEACHER_DIR / 'teacher_test.csv'
LABEL_STATS = HERE / 'label_stats.json'


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def cmr_augment(cmr, spatial=True, geom=None, pool=4):
    # cmr: (F,1,H,W). Spatially-consistent aug — single batched flip/rotate (all frames same params).
    # geom: (F, P*P, 3) per-eid token xyz — tracked through the SAME flip+rotate so the geometry stays
    #   aligned with the (augmented) token grid. Returns (cmr, geom) when geom is given, else cmr.
    #   Lets ② geo runs use spatial aug too (was previously disabled to avoid image↔geom desync).
    if spatial:
        do_h = random.random() < 0.5
        do_v = random.random() < 0.5
        angle = random.uniform(-30, 30)
        if geom is not None:
            P, Fn = pool, geom.shape[0]
            cen = geom.reshape(-1, 3).mean(0)                              # ONE global centroid → preserves
            g = (geom - cen).reshape(Fn, P, P, 3).permute(0, 3, 1, 2)      #   cross-frame (LAX↔SAX) geometry
        if do_h:
            cmr = cmr.flip(-1)
            if geom is not None: g = g.flip(-1)                            # width axis = token col (exact)
        if do_v:
            cmr = cmr.flip(-2)
            if geom is not None: g = g.flip(-2)                            # height axis = token row (exact)
        cmr = TF.rotate(cmr, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0.0)
        if geom is not None:
            # same TF.rotate (identical convention); centered so fill=0 means "centroid" (not origin →
            # no pull-to-origin pollution); un-center afterwards so rotated-out cells become the centroid.
            g = TF.rotate(g, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0.0)
            geom = g.permute(0, 2, 3, 1).reshape(Fn, P * P, 3) + cen
    cmr = (cmr * random.uniform(0.85, 1.15)).clamp(0, 1)
    if random.random() < 0.5:
        cmr = (cmr + torch.randn_like(cmr) * 0.02).clamp(0, 1)
    return (cmr, geom) if geom is not None else cmr


class TeacherDataset(Dataset):
    """teacher_*.csv → (selected frames, 1, H, W) raw[0,1] + reg/cls targets."""
    def __init__(self, csv_path, frame_indices, is_train=True, geom_dir=None, spatial_pool=4, no_spatial_aug=False):
        self.geom_dir = geom_dir; self.geom_p2 = spatial_pool * spatial_pool   # ② per-eid token xyz
        self.spatial_pool = spatial_pool       # for syncing geom through the same flip/rotate as the image
        self.no_spatial_aug = no_spatial_aug   # disable flip/rotate (fair no-aug control, e.g. run33)
        df = pd.read_csv(csv_path)
        df['path'] = df['eid'].apply(lambda e: f"{DATA_DIR}/{e}.npy")
        # teacher splits are pre-cleaned (47 bad npy already removed) → cheap stat-only
        # filter; __getitem__ resamples on any residual unreadable file.
        df = df[df['path'].apply(_exists_ok)].reset_index(drop=True)
        self.rows = df.to_dict('records')
        self.frame_indices = list(frame_indices)
        self.is_train = is_train
        print(f"[dataset] {len(self.rows)} samples ({'train' if is_train else 'eval'})  "
              f"frames={len(self.frame_indices)}")

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        try:
            npy = np.load(r['path']).astype(np.float32)      # (16,224,224)
            assert npy.shape[0] == 16
        except Exception:
            return self.__getitem__((idx + 1) % len(self.rows))   # skip rare corrupt file
        npy = npy[self.frame_indices]                        # select frames
        cmr = torch.from_numpy(npy).unsqueeze(1)             # (F,1,H,W)
        lo = cmr.amin(dim=(1, 2, 3), keepdim=True)           # per-frame min-max → [0,1]
        hi = cmr.amax(dim=(1, 2, 3), keepdim=True)
        cmr = torch.where(hi > lo, (cmr - lo) / (hi - lo).clamp_min(1e-6), cmr)
        geom = None
        if self.geom_dir is not None:                          # ② per-eid token xyz (F,P²,3); load BEFORE aug
            try:
                g = np.load(f"{self.geom_dir}/{r['eid']}.npz")['token_xyz'].astype(np.float32)[self.frame_indices]
            except Exception:
                g = np.zeros((len(self.frame_indices), self.geom_p2, 3), np.float32)  # missing → neutral bias
            geom = torch.from_numpy(g)
        if self.is_train:                                      # flip/rotate; geom tracks the SAME transform
            if geom is not None:
                cmr, geom = cmr_augment(cmr, spatial=not self.no_spatial_aug, geom=geom, pool=self.spatial_pool)
            else:
                cmr = cmr_augment(cmr, spatial=not self.no_spatial_aug)
        reg_vals, reg_mask = [], []
        for c in REG_COLS:
            v = r.get(c, float('nan'))
            if v is None or (isinstance(v, float) and np.isnan(v)):
                reg_vals.append(0.0); reg_mask.append(0)
            else:
                reg_vals.append(float(v)); reg_mask.append(1)
        cls_vals = [float(r.get(c, -1)) if not (isinstance(r.get(c), float) and np.isnan(r.get(c)))
                    else -1.0 for c in CLS_COLS]
        out = {'cmr': cmr,
               'reg': torch.tensor(reg_vals, dtype=torch.float32),
               'reg_mask': torch.tensor(reg_mask, dtype=torch.bool),
               'cls': torch.tensor(cls_vals, dtype=torch.float32)}
        if geom is not None:                                   # already loaded + (maybe) augmented above
            out['geom'] = geom
        return out


def _exists_ok(p):
    """Fast stat-only check (size > 1MB). Shape/readability handled in __getitem__."""
    try:
        return pathlib.Path(p).stat().st_size > 1024 * 1024
    except OSError:
        return False


def get_scheduler(opt, warmup, total, min_ratio=0.01):
    def f(ep):
        if ep < warmup: return (ep + 1) / max(warmup, 1)
        prog = (ep - warmup) / max(total - warmup, 1)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def task_loss(cls_out, reg_out, cls_gt, reg_gt, reg_mask, pos_w, reg_mean, reg_std, device):
    cls_losses = []
    for i, pred in enumerate(cls_out):
        if TRAIN_CLS_IDX is not None and i not in TRAIN_CLS_IDX: continue
        valid = cls_gt[:, i] >= 0
        if not valid.any(): continue
        pw = torch.tensor([pos_w[i]], dtype=torch.float32, device=device)
        cls_losses.append(nn.BCEWithLogitsLoss(pos_weight=pw)(pred[valid], cls_gt[valid, i]))
    l_cls = torch.stack(cls_losses).mean() if cls_losses else torch.tensor(0., device=device)
    reg_losses = []
    for i, pred in enumerate(reg_out):
        if TRAIN_REG_IDX is not None and i not in TRAIN_REG_IDX: continue
        valid = reg_mask[:, i]
        if not valid.any(): continue
        t = (reg_gt[:, i] - reg_mean[i]) / reg_std[i]
        reg_losses.append(F.mse_loss(pred[valid], t[valid]))
    l_reg = torch.stack(reg_losses).mean() if reg_losses else torch.tensor(0., device=device)
    return l_cls + l_reg


@torch.no_grad()
def evaluate(encoder, head, loader, device, reg_mean, reg_std):
    encoder.eval(); head.eval()
    REG_N, CLS_N = len(REG_COLS), len(CLS_COLS)
    rp = [[] for _ in range(REG_N)]; rg = [[] for _ in range(REG_N)]; rm = [[] for _ in range(REG_N)]
    cp = [[] for _ in range(CLS_N)]; cg = [[] for _ in range(CLS_N)]
    moe = isinstance(head, GatedTaskHead)
    for b in loader:
        cmr = b['cmr'].to(device)
        geom = b['geom'].to(device) if 'geom' in b else None
        with torch.cuda.amp.autocast():
            if moe:
                _, z, z_lax, z_sax = encoder(cmr, geom=geom, return_views=True)
            else:
                _, z = encoder(cmr, geom=geom)
        cls_out, reg_out = head(z.float(), z_lax.float(), z_sax.float()) if moe else head(z.float())
        for i in range(REG_N):
            rp[i].append(reg_out[i].float().cpu()); rg[i].append(b['reg'][:, i]); rm[i].append(b['reg_mask'][:, i])
        for i in range(CLS_N):
            cp[i].append(torch.sigmoid(cls_out[i]).float().cpu()); cg[i].append(b['cls'][:, i])
    metrics = {}
    for i, c in enumerate(REG_COLS):
        pred = torch.cat(rp[i]).numpy() * reg_std[i] + reg_mean[i]
        gt = torch.cat(rg[i]).numpy(); mask = torch.cat(rm[i]).numpy().astype(bool)
        if mask.sum() < 2:
            metrics[f'mae/{c}'] = float('nan'); metrics[f'r2/{c}'] = float('nan'); continue
        p, g = pred[mask], gt[mask]
        metrics[f'mae/{c}'] = float(np.abs(p - g).mean())
        ss_res = np.sum((g - p) ** 2); ss_tot = np.sum((g - g.mean()) ** 2)
        metrics[f'r2/{c}'] = float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan')
    for i, c in enumerate(CLS_COLS):
        s = torch.cat(cp[i]).numpy(); y = torch.cat(cg[i]).numpy()
        valid = y >= 0
        try:
            metrics[f'auc/{c}'] = float(roc_auc_score(y[valid], s[valid])) if valid.sum() >= 2 and len(set(y[valid].tolist())) > 1 else float('nan')
        except Exception:
            metrics[f'auc/{c}'] = float('nan')
    focus_r2 = [metrics[f'r2/{c}'] for c in FOCUS_REG if not np.isnan(metrics.get(f'r2/{c}', float('nan')))]
    metrics['focus_r2_mean'] = float(np.mean(focus_r2)) if focus_r2 else float('nan')
    metrics['focus_r2'] = {c: metrics[f'r2/{c}'] for c in FOCUS_REG}
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--backbone', default='sam2_tiny')
    ap.add_argument('--backbone_ckpt', default=None,
                    help='explicit backbone init ckpt; if unset, defaults to the SAM2-pretrained '
                         'weights for --backbone (matches the canonical G3 recipe)')
    ap.add_argument('--scratch', action='store_true',
                    help='force random backbone init instead of SAM2-pretrained')
    ap.add_argument('--views', default='saxlax', choices=['saxlax', 'sax', 'lax'])
    ap.add_argument('--use_t1', dest='use_t1', action='store_true')
    ap.add_argument('--no_t1', dest='use_t1', action='store_false')
    ap.set_defaults(use_t1=False)                                   # study default: no T1
    ap.add_argument('--use_delta', dest='use_delta', action='store_true')
    ap.add_argument('--no_delta', dest='use_delta', action='store_false')
    ap.set_defaults(use_delta=False)                               # BASE default: no delta
    ap.add_argument('--view_emb', dest='view_emb', action='store_true')
    ap.add_argument('--no_view_emb', dest='view_emb', action='store_false')
    ap.set_defaults(view_emb=True)
    ap.add_argument('--time_emb', dest='time_emb', action='store_true')
    ap.add_argument('--no_time_emb', dest='time_emb', action='store_false')
    ap.set_defaults(time_emb=True)
    ap.add_argument('--spatial_pool', type=int, default=4)
    ap.add_argument('--transformer_depth', type=int, default=2)
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--unfreeze_batch_size', type=int, default=8)
    ap.add_argument('--enc_lr', type=float, default=1e-5)
    ap.add_argument('--head_lr', type=float, default=1e-4)
    ap.add_argument('--weight_decay', type=float, default=0.1)
    ap.add_argument('--warmup_epochs', type=int, default=2)
    ap.add_argument('--freeze_epochs', type=int, default=2)
    ap.add_argument('--patience', type=int, default=6)
    ap.add_argument('--frame_dropout', type=float, default=0.15)
    ap.add_argument('--attn_dropout', type=float, default=0.1)
    ap.add_argument('--num_workers', type=int, default=6)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--smoke_test', action='store_true')
    ap.add_argument('--train_set', default='full', choices=['full','cardiac','focus'])
    ap.add_argument('--sax_lax_fusion', action='store_true')
    ap.add_argument('--deform_token', action='store_true')        # ③ deformation token
    ap.add_argument('--view_moe', action='store_true')            # ④ target-gated view experts
    ap.add_argument('--consistency_loss', action='store_true')    # ⑤ cross-view consistency
    ap.add_argument('--consist_w', type=float, default=0.3)
    ap.add_argument('--geo_fusion', action='store_true')          # ② geometry correspondence
    ap.add_argument('--geo_variant', default='iso')               # ② geo bias mode: iso|perhead|aniso|mlpdist|adaptive|relvec
    ap.add_argument('--geo_posenc', action='store_true')          # ② Plan A: geometry positional encoding
    ap.add_argument('--no_global_attn', action='store_true')      # drop spatio-temporal transformer (pool+concat)
    ap.add_argument('--no_spatial_aug', action='store_true')      # disable flip/rotate (fair control for geom runs)
    ap.add_argument('--ef_strain_consist', action='store_true')   # EF↔strain physics consistency
    ap.add_argument('--ef_consist_w', type=float, default=0.3)
    ap.add_argument('--dp', action='store_true')                  # multi-GPU DataParallel (encoder forward)
    args = ap.parse_args()
    global TRAIN_REG_IDX, TRAIN_CLS_IDX
    if args.train_set == 'cardiac':
        TRAIN_REG_IDX = [REG_COLS.index(c) for c in STRUCT_REG + FUNC_REG]; TRAIN_CLS_IDX = []
    elif args.train_set == 'focus':
        TRAIN_REG_IDX = [REG_COLS.index(c) for c in FOCUS_REG]; TRAIN_CLS_IDX = []
    print(f'[train_set] {args.train_set}: train_reg={len(TRAIN_REG_IDX) if TRAIN_REG_IDX else "all"} train_cls={0 if TRAIN_CLS_IDX==[] else "all"}')

    # default backbone init = SAM2-pretrained (matches canonical G3); --scratch forces random
    if not args.scratch and args.backbone_ckpt is None:
        args.backbone_ckpt = BACKBONE_CKPTS.get(args.backbone)

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = pathlib.Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(out_dir / 'config.json', 'w'), indent=2)

    stats = json.load(open(LABEL_STATS))
    reg_mean = [stats['reg_norm'][c]['mean'] for c in REG_COLS]
    reg_std  = [stats['reg_norm'][c]['std'] for c in REG_COLS]
    pos_w    = [stats['pos_weights'][c]['pos_weight'] for c in CLS_COLS]

    frame_indices = select_frames(args.views, args.use_t1)
    print(f"[config] backbone={args.backbone} views={args.views} t1={args.use_t1} "
          f"delta={args.use_delta} view_emb={args.view_emb} time_emb={args.time_emb} "
          f"pool={args.spatial_pool} | frames={frame_indices}")

    print(f'[fusion] sax_lax={args.sax_lax_fusion} deform={args.deform_token} '
          f'view_moe={args.view_moe} consist={args.consistency_loss}(w={args.consist_w}) geo={args.geo_fusion} geo_posenc={args.geo_posenc}')
    geom_dir = None
    if args.geo_fusion or args.geo_posenc:
        import os
        geom_dir = os.environ.get('CMR_GEOM_DIR', str(pathlib.Path(DATA_DIR).parent / 'teacher_geom'))
        print(f'[geo] geom_dir={geom_dir}')
    train_ds = TeacherDataset(TRAIN_CSV, frame_indices, is_train=True,  geom_dir=geom_dir, spatial_pool=args.spatial_pool, no_spatial_aug=args.no_spatial_aug)
    val_ds   = TeacherDataset(VAL_CSV,   frame_indices, is_train=False, geom_dir=geom_dir, spatial_pool=args.spatial_pool, no_spatial_aug=args.no_spatial_aug)
    if args.smoke_test:
        train_ds.rows = train_ds.rows[:48]; val_ds.rows = val_ds.rows[:32]

    def loaders(bs):
        return (DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=args.num_workers,
                           pin_memory=True, drop_last=True, persistent_workers=args.num_workers > 0),
                DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=max(2, args.num_workers // 2),
                           pin_memory=True))
    train_loader, val_loader = loaders(args.batch_size)

    enc_raw = CMREncoderV3(
        proj_dim=256, embed_dim=768, spatial_pool=args.spatial_pool,
        transformer_heads=8, transformer_depth=args.transformer_depth,
        backbone=args.backbone, backbone_ckpt=args.backbone_ckpt,
        attn_dropout=args.attn_dropout, freeze_backbone=False,
        frame_indices=frame_indices, use_delta=args.use_delta,
        use_view_emb=args.view_emb, use_time_emb=args.time_emb,
        sax_lax_fusion=args.sax_lax_fusion, deform_token=args.deform_token,
        geo_fusion=args.geo_fusion, geo_variant=args.geo_variant, geo_posenc=args.geo_posenc, no_global_attn=args.no_global_attn,
    ).to(device)
    n_gpu = torch.cuda.device_count()
    use_dp = args.dp and n_gpu > 1
    # parallelize the (bottleneck) encoder forward; head + loss stay on cuda:0.
    # save/load/eval use enc_raw so checkpoints are identical-format to 1-GPU runs.
    encoder = nn.DataParallel(enc_raw) if use_dp else enc_raw
    if use_dp: print(f"[dp] DataParallel across {n_gpu} GPUs")
    if args.view_moe:
        head = GatedTaskHead(in_dim=768, n_cls=len(CLS_COLS), n_reg=len(REG_COLS)).to(device)
    else:
        head = TaskHead(in_dim=768, n_cls=len(CLS_COLS), n_reg=len(REG_COLS)).to(device)
    # ⑤ integrative reg targets both views can estimate (LV/RV volume+mass+EF+CO); excl view-specific strains
    INTEGRATIVE_IDX = [0, 1, 2, 3, 5, 6, 7, 14, 15, 16]
    # EF↔strain consistency: map predicted strains (GLS/GCS/GRS) → EF (LVEF/RVEF) as a physics prior
    EF_IDX = [REG_COLS.index('LVEF'), REG_COLS.index('RVEF')]
    STRAIN_IDX = [REG_COLS.index('GLS'), REG_COLS.index('GCS'), REG_COLS.index('GRS')]
    strain2ef = nn.Sequential(nn.Linear(3, 32), nn.GELU(), nn.Linear(32, 2)).to(device) if args.ef_strain_consist else None

    if args.freeze_epochs > 0:
        for p in enc_raw.backbone.parameters(): p.requires_grad_(False)

    head_params = list(head.parameters()) + (list(strain2ef.parameters()) if strain2ef is not None else [])
    opt = torch.optim.AdamW([
        {'params': enc_raw.parameters(), 'lr': args.enc_lr},
        {'params': head_params, 'lr': args.head_lr}],
        weight_decay=args.weight_decay)
    sched = get_scheduler(opt, args.warmup_epochs, args.epochs)
    scaler = torch.cuda.amp.GradScaler()

    best, no_improve, start_ep, history = -1e9, 0, 0, []
    last_ckpt = out_dir / 'last.pth'
    if args.resume and last_ckpt.exists():
        ck = torch.load(last_ckpt, map_location=device)
        enc_raw.load_state_dict(ck['encoder']); head.load_state_dict(ck['head'])
        if strain2ef is not None and ck.get('strain2ef'): strain2ef.load_state_dict(ck['strain2ef'])
        opt.load_state_dict(ck['optimizer']); sched.load_state_dict(ck['scheduler'])
        scaler.load_state_dict(ck['scaler'])
        start_ep = ck['epoch'] + 1; best = ck.get('best', -1e9)
        no_improve = ck.get('no_improve', 0); history = ck.get('history', [])
        if start_ep >= args.freeze_epochs:
            for p in enc_raw.backbone.parameters(): p.requires_grad_(True)
            enc_raw.grad_checkpoint = True
            train_loader, val_loader = loaders(args.unfreeze_batch_size)
        print(f"[resume] ep{ck['epoch']} best_focus_r2={best:.4f}")

    n_tot = sum(p.numel() for p in enc_raw.parameters()) + sum(p.numel() for p in head.parameters())
    print(f"[params] {n_tot/1e6:.1f}M  cls={len(CLS_COLS)} reg={len(REG_COLS)}")

    for epoch in range(start_ep, args.epochs):
        if args.freeze_epochs > 0 and epoch == args.freeze_epochs:
            for p in enc_raw.backbone.parameters(): p.requires_grad_(True)
            enc_raw.grad_checkpoint = True
            train_loader, val_loader = loaders(args.unfreeze_batch_size)
            print(f"[unfreeze] ep{epoch} grad_ckpt=True bs→{args.unfreeze_batch_size}")
        encoder.train(); head.train()
        t0 = time.time(); tl = 0.0; n = 0
        for b in train_loader:
            cmr = b['cmr'].to(device, non_blocking=True)
            cls_gt = b['cls'].to(device); reg_gt = b['reg'].to(device); reg_mk = b['reg_mask'].to(device)
            geom = b['geom'].to(device, non_blocking=True) if 'geom' in b else None
            if args.frame_dropout > 0:
                keep = (torch.rand(cmr.shape[0], cmr.shape[1], 1, 1, 1, device=device) > args.frame_dropout).float()
                cmr = cmr * keep
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                need_views = args.view_moe or args.consistency_loss
                if need_views:
                    _, z, z_lax, z_sax = encoder(cmr, geom=geom, return_views=True)
                else:
                    _, z = encoder(cmr, geom=geom)
                if args.view_moe:
                    cls_out, reg_out = head(z, z_lax, z_sax)
                else:
                    cls_out, reg_out = head(z)
                loss = task_loss(cls_out, reg_out, cls_gt, reg_gt, reg_mk, pos_w, reg_mean, reg_std, device)
                if args.consistency_loss:                          # ⑤ reconcile per-view estimates
                    cls_l, reg_l = head(z_lax); cls_s, reg_s = head(z_sax)
                    cons = [F.mse_loss(reg_l[i][reg_mk[:, i]], reg_s[i][reg_mk[:, i]])
                            for i in INTEGRATIVE_IDX if reg_mk[:, i].any()]
                    consist = torch.stack(cons).mean() if cons else torch.tensor(0., device=device)
                    sup = (task_loss(cls_l, reg_l, cls_gt, reg_gt, reg_mk, pos_w, reg_mean, reg_std, device)
                           + task_loss(cls_s, reg_s, cls_gt, reg_gt, reg_mk, pos_w, reg_mean, reg_std, device))
                    loss = loss + args.consist_w * consist + 0.5 * sup
                if args.ef_strain_consist:                          # EF↔strain physics consistency
                    strains = torch.stack([reg_out[i] for i in STRAIN_IDX], dim=1)        # (B,3) z-scored GLS/GCS/GRS
                    ef_dir  = torch.stack([reg_out[i] for i in EF_IDX], dim=1)            # (B,2) z-scored LVEF/RVEF
                    ef_phys = strain2ef(strains)                                          # (B,2) strain-derived EF
                    ef_true = torch.stack([(reg_gt[:, i] - reg_mean[i]) / reg_std[i] for i in EF_IDX], dim=1).to(ef_phys.dtype)
                    em = reg_mk[:, EF_IDX].float()
                    L_s2ef = ((ef_phys - ef_true) ** 2 * em).sum() / em.sum().clamp_min(1)            # learn strain→EF
                    L_cons = ((ef_dir - ef_phys.detach()) ** 2 * em).sum() / em.sum().clamp_min(1)    # pull EF head→it
                    loss = loss + args.ef_consist_w * (L_s2ef + L_cons)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(list(enc_raw.parameters()) + head_params, 3.0)
            scaler.step(opt); scaler.update()
            tl += loss.item(); n += 1
        sched.step()
        tl /= max(n, 1)
        m = evaluate(enc_raw, head, val_loader, device, reg_mean, reg_std)
        sel = m['focus_r2_mean']
        mem = torch.cuda.max_memory_allocated(device) / 1e9
        row = {'epoch': epoch, 'train_loss': tl, 'lr': opt.param_groups[0]['lr'],
               'focus_r2_mean': sel, **{f'r2/{c}': m[f'r2/{c}'] for c in FOCUS_REG},
               **{f'mae/{c}': m[f'mae/{c}'] for c in FOCUS_REG}}
        history.append(row)
        print(f"Ep{epoch:3d} | {time.time()-t0:.0f}s | loss={tl:.4f} | focus_R²={sel:.4f} | "
              f"mem={mem:.1f}G | " + " ".join(f"{c}:{m[f'r2/{c}']:.3f}" for c in FOCUS_REG))
        torch.save({'epoch': epoch, 'encoder': enc_raw.state_dict(), 'head': head.state_dict(),
                    'strain2ef': strain2ef.state_dict() if strain2ef is not None else None,
                    'optimizer': opt.state_dict(), 'scheduler': sched.state_dict(),
                    'scaler': scaler.state_dict(), 'best': best, 'no_improve': no_improve,
                    'history': history, 'config': vars(args)}, last_ckpt)
        json.dump(history, open(out_dir / 'metrics_history.json', 'w'), indent=2)
        if sel > best:
            best = sel; no_improve = 0
            torch.save({'epoch': epoch, 'encoder': enc_raw.state_dict(), 'head': head.state_dict(),
                        'metrics': m, 'config': vars(args)}, out_dir / 'best.pth')
            print(f"  [saved] best focus_R²={best:.4f}")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"[early stop] patience {args.patience} exhausted"); break

    print(f"\nTraining complete. best_focus_R²={best:.4f}  out={out_dir}")


if __name__ == '__main__':
    main()
