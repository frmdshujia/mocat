"""Strict inference loaders for the three MoCaT release assets (seed 0).

No original pretrained weights are needed after loading the full aligned state.
External model implementations remain required; see MODEL_LOADING.md.
"""
from pathlib import Path
import importlib
import sys
import numpy as np
import torch


def _read(path, kind):
    ck = torch.load(path, map_location='cpu', weights_only=True)
    if ck.get('model_type') != kind:
        raise ValueError(f"Expected {kind}, found {ck.get('model_type')}")
    return ck


def load_teacher(path, device='cpu'):
    sys.path.insert(0, str(Path(__file__).resolve().parent/'teacher'))
    from model import CMREncoderV3
    ck = _read(path, 'teacher')
    cfg = ck['config']
    enc = CMREncoderV3(
        proj_dim=256, embed_dim=768, backbone=cfg['backbone'],
        backbone_ckpt=None, freeze_backbone=False,
        spatial_pool=cfg['spatial_pool'], transformer_depth=cfg['transformer_depth'],
        frame_indices=cfg['frame_indices'], use_delta=cfg['use_delta'],
        use_view_emb=cfg['view_emb'], use_time_emb=cfg['time_emb'],
        geo_fusion=cfg['geo_fusion'], attn_dropout=cfg.get('attn_dropout',0.1),
        sax_lax_fusion=cfg.get('sax_lax_fusion',False),
    )
    enc.load_state_dict(ck['encoder'], strict=True)
    return enc.to(device).eval()


def load_stmem(path, stmem_repo, device='cpu'):
    sys.path.insert(0, str(Path(stmem_repo).resolve()))
    implementation = importlib.import_module('models.encoder')
    ck = _read(path, 'stmem')
    cfg = ck['config']
    enc = implementation.st_mem_vit_base(
        num_leads=cfg['num_leads'], seq_len=cfg['seq_len'], patch_size=cfg['patch_size'])
    enc.load_state_dict(ck['ecg'], strict=True)
    return enc.to(device).eval()


def load_vitmae(path, device='cpu'):
    # Use the PTACL layerscale-enabled fork, not stock transformers.
    from transformers import ViTMAEConfig, ViTMAEModel
    ck = _read(path, 'vitmae')
    enc = ViTMAEModel(ViTMAEConfig(**ck['config']))
    if not any('layerscale' in k for k in enc.state_dict()):
        raise RuntimeError('Install PTACL external/transformers (layerscale fork).')
    enc.load_state_dict(ck['ecg'], strict=True)
    enc.config.mask_ratio = 0.0
    return enc.to(device).eval()


def prepare_vitmae(ecg_500hz):
    """Input (12,5000), original amplitudes; central 2500, no normalization."""
    x=np.asarray(ecg_500hz,dtype=np.float32)
    if x.shape != (12,5000):
        raise ValueError('Expected a 12-lead, 10-second ECG sampled at 500 Hz.')
    return np.ascontiguousarray(x[:,1250:3750])


def prepare_stmem(ecg_500hz, stmem_repo):
    """Exact original transform chain; returns (12,2250) float32."""
    sys.path.insert(0,str(Path(stmem_repo).resolve()))
    from util.transforms import Resample, CenterCrop, HighpassFilter, LowpassFilter, Standardize
    x=np.asarray(ecg_500hz,dtype=np.float64)
    if x.shape != (12,5000):
        raise ValueError('Expected a 12-lead, 10-second ECG sampled at 500 Hz.')
    x=Resample(target_fs=250)(x,fs=500)
    for transform in [CenterCrop(2250),HighpassFilter(fs=250,cutoff=0.67),
                      LowpassFilter(fs=250,cutoff=40),Standardize(axis=(-1,-2))]:
        x=transform(x)
    return np.ascontiguousarray(x,dtype=np.float32)


@torch.no_grad()
def encode_vitmae(model, preprocessed_batch):
    """(B,12,2500) -> (B,512); average patch tokens, exclude CLS."""
    return model(pixel_values=preprocessed_batch.unsqueeze(1)).last_hidden_state[:,1:].mean(1)
