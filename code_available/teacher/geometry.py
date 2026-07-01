# [Paper §Method: token 3D coordinates] Compute each token's 3D physical coordinate (DICOM patient space, mm) from ImageOrientationPatient/ImagePositionPatient/PixelSpacing -> token_xyz, the geometry that drives the -lambda*d^2 cross-attention bias.

"""
extract_cmr_geometry.py — ② per-eid CMR plane geometry for SAX↔LAX correspondence.

For each eid, replicate preprocess_cmr_v3.py's EXACT series selection + _heart_roi
crop, then instead of decoding pixels, compute the 3D PHYSICAL coordinate (DICOM
patient space, mm) of every P×P token-grid cell center for the 5 distinct planes
(LAX 2Ch, LAX 4Ch, SAX base/mid/apex). Tokens that are physically close across
views (i.e. near a LAX∩SAX intersection line) are the geometric correspondences.

Output per eid: teacher_geom/{eid}.npz
  token_xyz : (15, P², 3) float32  — physical xyz of each token, frame order matches
              the (16,224,224) npy: [0-2]2Ch [3-5]4Ch [6-8]SAXbase [9-11]SAXmid [12-14]SAXapex
  ok        : bool   — all 5 planes resolved (no fallback / missing)
  used_descs: the 3 SAX series descriptions (sanity vs {eid}.json)

Frame order, _heart_roi, series selection are copied VERBATIM from preprocess_cmr_v3.py
so token positions align with the actual npy pixels.

Usage:
  python extract_cmr_geometry.py --validate 5 --eid_file teacher_test.csv --cmr_dir ~/UKB/CMRI/downloaded --out_dir /tmp/geomtest
  python extract_cmr_geometry.py --eid_file teacher_train.csv --cmr_dir ~/UKB/CMRI/downloaded --out_dir ~/projects/BIBM2026/data/teacher_geom --num_workers 32
"""
import argparse, io, json, os, zipfile
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import pydicom

# ── constants (verbatim from preprocess_cmr_v3.py) ──
LAX_2CH = "CINE_segmented_LAX_2Ch"
LAX_4CH = "CINE_segmented_LAX_4Ch"
SAX_PREFIX = "CINE_segmented_SAX_b"
SAX_INLINE = "CINE_segmented_SAX_InlineVF"
ROI_SIZE = 160
IMG = 224


def _hdr(data):  return pydicom.dcmread(io.BytesIO(data), stop_before_pixels=True, force=True)
def _full(data): return pydicom.dcmread(io.BytesIO(data), force=True)


def _heart_roi(img, crop=ROI_SIZE):
    """VERBATIM from preprocess_cmr_v3.py — bright-blood centroid crop box."""
    H, W = img.shape
    margin_y, margin_x = H // 10, W // 10
    roi_img = img[margin_y:H - margin_y, margin_x:W - margin_x].astype(np.float32)
    thr = np.percentile(roi_img, 82)
    mask = roi_img > thr
    ys, xs = np.where(mask)
    if len(ys) < 50:
        cy, cx = H // 2, W // 2
    else:
        cy = int(ys.mean()) + margin_y
        cx = int(xs.mean()) + margin_x
    half = crop // 2
    y0 = max(0, min(cy - half, H - crop))
    x0 = max(0, min(cx - half, W - crop))
    return y0, y0 + crop, x0, x0 + crop


def _sort_frames(metas):
    if all(m['inst'] != 9999 for m in metas):
        return sorted(metas, key=lambda m: m['inst'])
    return sorted(metas, key=lambda m: m['tt'])


def _ed_name(metas):
    """ED frame = first timepoint (frac 0.0), same as _pick_timepoints(...)[0]."""
    return _sort_frames(metas)[0]['name']


def _token_xyz(ds, pixels, P):
    """(P², 3) physical xyz (mm) of each P×P token-grid cell center.

    token center (224px) -> crop(160) coords -> DICOM pixel (r=row,c=col) -> patient xyz:
      xyz = ipp + c·ps_col·Xdir + r·ps_row·Ydir   (DICOM PS3.3 C.7.6.2.1.1)
    Xdir=iop[0:3] (incr column), Ydir=iop[3:6] (incr row); ps=[row,col]; matches
    feat.permute(0,2,3,1).reshape(...) row-major token order  idx=gy*P+gx.
    """
    iop = np.asarray(ds.ImageOrientationPatient, dtype=np.float64)
    ipp = np.asarray(ds.ImagePositionPatient, dtype=np.float64)
    ps = np.asarray(ds.PixelSpacing, dtype=np.float64)   # [row_spacing, col_spacing]
    Xdir, Ydir = iop[0:3], iop[3:6]
    y0, _, x0, _ = _heart_roi(pixels)
    s = ROI_SIZE / IMG                                    # 224 -> 160 crop scale
    out = np.zeros((P * P, 3), np.float32)
    for gy in range(P):
        for gx in range(P):
            cy224 = (gy + 0.5) * IMG / P                  # token center in 224px
            cx224 = (gx + 0.5) * IMG / P
            r = cy224 * s + y0                            # DICOM row index
            c = cx224 * s + x0                            # DICOM col index
            out[gy * P + gx] = ipp + c * ps[1] * Xdir + r * ps[0] * Ydir
    return out


def _read_series(zip_path, want):
    """Return {desc: (ed_dataset_full, slice_location)} for series matching `want`
    (a predicate on SeriesDescription). Reads headers to find/sort, full-reads ED only."""
    metas, raw, locs = {}, {}, {}
    with zipfile.ZipFile(zip_path) as z:
        for name in (n for n in z.namelist() if n.endswith('.dcm')):
            data = z.read(name); raw[name] = data
            try:
                d = _hdr(data); desc = str(getattr(d, 'SeriesDescription', ''))
                if not want(desc):
                    continue
                metas.setdefault(desc, []).append({
                    'name': name,
                    'inst': int(getattr(d, 'InstanceNumber', 9999) or 9999),
                    'tt': float(getattr(d, 'TriggerTime', 9e9) or 9e9)})
                if desc not in locs:
                    locs[desc] = float(getattr(d, 'SliceLocation', 0.0) or 0.0)
            except Exception:
                pass
    out = {}
    for desc, ms in metas.items():
        out[desc] = (_full(raw[_ed_name(ms)]), locs[desc])
    return out


def process_one(eid, cmr_dir, out_dir, P=4):
    op = os.path.join(out_dir, f'{eid}.npz')
    if os.path.exists(op):
        return {'eid': eid, 'status': 'skip'}
    lax_zip = os.path.join(cmr_dir, f'{eid}_20208_2_0.zip')
    sax_zip = os.path.join(cmr_dir, f'{eid}_20209_2_0.zip')
    if not (os.path.exists(lax_zip) and os.path.exists(sax_zip)):
        return {'eid': eid, 'status': 'missing'}
    try:
        lax = _read_series(lax_zip, lambda d: d in (LAX_2CH, LAX_4CH))
        sax = _read_series(sax_zip, lambda d: d.startswith(SAX_PREFIX) and d != SAX_INLINE)
        if LAX_4CH not in lax or not sax:
            return {'eid': eid, 'status': 'missing_views'}
        ok = (LAX_2CH in lax)
        ds2 = lax[LAX_2CH][0] if ok else lax[LAX_4CH][0]   # 2Ch fallback→4Ch (matches preprocess)
        ds4 = lax[LAX_4CH][0]
        # SAX: sort by SliceLocation, take 20/50/80% (verbatim dedup logic)
        sorted_descs = sorted(sax, key=lambda s: sax[s][1])
        n = len(sorted_descs)
        idxs = [min(round(n * f), n - 1) for f in (0.2, 0.5, 0.8)]
        if idxs[0] == idxs[1]: idxs[1] = min(idxs[0] + 1, n - 1)
        if idxs[1] == idxs[2]: idxs[2] = min(idxs[1] + 1, n - 1)
        sax_ds = [sax[sorted_descs[i]][0] for i in idxs]
        # per-plane token xyz (ED frame); phases of a plane share geometry
        xyz2 = _token_xyz(ds2, ds2.pixel_array, P)
        xyz4 = _token_xyz(ds4, ds4.pixel_array, P)
        xyzs = [_token_xyz(d, d.pixel_array, P) for d in sax_ds]   # [base, mid, apex]
        token_xyz = np.stack(
            [xyz2] * 3 + [xyz4] * 3 + [xyzs[0]] * 3 + [xyzs[1]] * 3 + [xyzs[2]] * 3, axis=0)
        np.savez_compressed(op, token_xyz=token_xyz.astype(np.float32), ok=ok,
                            used_descs=np.array([sorted_descs[i] for i in idxs]))
        return {'eid': eid, 'status': 'ok', 'full': ok}
    except Exception as e:
        return {'eid': eid, 'status': 'err', 'msg': f'{type(e).__name__}: {e}'}


def _sanity(eid, npz):
    """numeric check that the affine is geometrically sane."""
    xyz = npz['token_xyz']                                  # (15,16,3)
    def nrm(a, b, c):   # plane normal from 3 token corners
        v1, v2 = b - a, c - a; nv = np.cross(v1, v2); return nv / (np.linalg.norm(nv) + 1e-9)
    n2 = nrm(xyz[0, 0], xyz[0, 3], xyz[0, 12])              # 2Ch (frame0)
    nsb = nrm(xyz[6, 0], xyz[6, 3], xyz[6, 12])             # SAX base (frame6)
    nsm = nrm(xyz[9, 0], xyz[9, 3], xyz[9, 12])             # SAX mid (frame9)
    nsa = nrm(xyz[12, 0], xyz[12, 3], xyz[12, 12])          # SAX apex (frame12)
    lax_t = xyz[3].reshape(-1, 3)                           # 4Ch tokens (16,3)
    sax_t = xyz[9].reshape(-1, 3)                           # SAX mid tokens
    d = np.linalg.norm(lax_t[:, None, :] - sax_t[None, :, :], axis=-1)
    extent = np.linalg.norm(xyz[9].max(0) - xyz[9].min(0))
    print(f"  eid {eid}: SAX∥(base·mid={nsb@nsm:+.2f} mid·apex={nsm@nsa:+.2f}) "
          f"LAX⊥SAX(2ch·saxmid={abs(n2@nsm):.2f}) token-extent={extent:.0f}mm "
          f"min 4Ch↔SAXmid dist={d.min():.1f}mm  ok={bool(npz['ok'])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eid_file', required=True)
    ap.add_argument('--cmr_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--num_workers', type=int, default=32)
    ap.add_argument('--P', type=int, default=4)
    ap.add_argument('--validate', type=int, default=0)     # run N eids + sanity, no full launch
    ap.add_argument('--max_eids', type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cmr_dir = os.path.expanduser(args.cmr_dir)
    src = Path(args.eid_file)
    if src.suffix == '.csv':
        import pandas as pd
        eids = pd.read_csv(src)['eid'].astype(int).unique().tolist()
    else:
        eids = [int(x) for x in src.read_text().split()]
    if args.max_eids:
        eids = eids[:args.max_eids]

    if args.validate:
        for eid in eids[:args.validate]:
            r = process_one(eid, cmr_dir, args.out_dir, args.P)
            print(r)
            if r['status'] == 'ok':
                _sanity(eid, np.load(os.path.join(args.out_dir, f'{eid}.npz')))
        return

    print(f"[geom] {len(eids)} eids → {args.out_dir} ({args.num_workers} workers, P={args.P})")
    from functools import partial
    fn = partial(process_one, cmr_dir=cmr_dir, out_dir=args.out_dir, P=args.P)
    cnt = {}
    with Pool(args.num_workers) as pool:
        for i, r in enumerate(pool.imap_unordered(fn, eids, chunksize=8)):
            cnt[r['status']] = cnt.get(r['status'], 0) + 1
            if (i + 1) % 2000 == 0:
                print(f"  {i+1}/{len(eids)}  {cnt}", flush=True)
    print(f"[geom] DONE {cnt}")


if __name__ == '__main__':
    main()
