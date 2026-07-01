# [Paper §Method: geometry-synchronised augmentation] Flip/rotate applied jointly to
# the image AND the token 3D coordinates so image and geometry stay consistent.
"""augment.py — geometry-synchronised spatial augmentation for MoCaT.

Spatial augmentation (H/V flip, in-plane rotation) is applied IDENTICALLY to
(a) the multi-frame CMR image and (b) the per-token 3D coordinates (token_xyz),
so the geometry driving the -lambda*d^2 cross-attention bias stays aligned with the
augmented image. Rotation is done in centroid-centred coordinates (one global
centroid) so the zero-fill region maps to the anatomical centre, not the coordinate
origin (avoids pull-to-origin pollution); the single global centroid preserves the
cross-frame LAX<->SAX relative geometry. (Identical to the function used in train.py.)
"""
import random
import torch
import torchvision.transforms.functional as TF


def cmr_augment(cmr, spatial=True, geom=None, pool=4):
    """cmr:(F,1,H,W) in [0,1]; geom:(F,P*P,3) token xyz or None. All frames share one
    flip/rotate; geom tracked through the SAME transform. Returns (cmr,geom) or cmr."""
    if spatial:
        do_h = random.random() < 0.5
        do_v = random.random() < 0.5
        angle = random.uniform(-30, 30)
        if geom is not None:
            P, Fn = pool, geom.shape[0]
            cen = geom.reshape(-1, 3).mean(0)                          # ONE global centroid
            g = (geom - cen).reshape(Fn, P, P, 3).permute(0, 3, 1, 2)  # preserves LAX<->SAX geometry
        if do_h:
            cmr = cmr.flip(-1)
            if geom is not None: g = g.flip(-1)                        # width = token col (exact)
        if do_v:
            cmr = cmr.flip(-2)
            if geom is not None: g = g.flip(-2)                        # height = token row (exact)
        cmr = TF.rotate(cmr, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0.0)
        if geom is not None:
            g = TF.rotate(g, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0.0)
            geom = g.permute(0, 2, 3, 1).reshape(Fn, P * P, 3) + cen   # un-center
    cmr = (cmr * random.uniform(0.85, 1.15)).clamp(0, 1)              # intensity jitter
    if random.random() < 0.5:
        cmr = (cmr + torch.randn_like(cmr) * 0.02).clamp(0, 1)        # gaussian noise
    return (cmr, geom) if geom is not None else cmr
