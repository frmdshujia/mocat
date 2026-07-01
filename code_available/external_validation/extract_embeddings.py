"""Paper Fig. 4-5 (external validation): frozen ECG-encoder feature extraction.

Applies the FROZEN ECG encoder to preprocessed ECG to obtain EMB_DIM-d embeddings.
Run once for the proposed aligned encoder ("aligned") and once for the no-alignment
ablation ("unaligned", same backbone with pretrained weights) to quantify the alignment
benefit under an identical downstream protocol.

The encoder architecture is the ECG foundation model referenced in the cross-modal
alignment module / its original paper (not vendored here); checkpoints are released on acceptance.
"""
import os, sys, numpy as np, torch
import config

sys.path.insert(0, config.ENCODER_REPO)
from models.encoder import build_ecg_encoder  # official repo API (12-lead ViT, EMB_DIM-d pooled output)


def _load_state_dict(ckpt):
    """Aligned checkpoints store the ECG encoder under 'ecg'; pretrained store a raw/`model` state dict."""
    if isinstance(ckpt, dict) and "ecg" in ckpt:
        return ckpt["ecg"]
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    return ckpt


@torch.no_grad()
def extract(weights_key, prepped_ecg, device="cuda", bs=128):
    """weights_key in {'aligned','unaligned'}; prepped_ecg: (N, 12, SEQ_LEN) -> (N, EMB_DIM)."""
    enc = build_ecg_encoder()
    sd = _load_state_dict(torch.load(config.WEIGHTS[weights_key], map_location="cpu"))
    missing, unexpected = enc.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (len(missing), len(unexpected))
    enc = enc.to(device).eval()
    X = torch.tensor(np.asarray(prepped_ecg), dtype=torch.float32)
    out = [enc(X[i:i + bs].to(device)).float().cpu().numpy() for i in range(0, len(X), bs)]
    return np.concatenate(out)


if __name__ == "__main__":
    X = np.load(os.path.join(config.EMB_DIR, "echonext_test_prepped.npy"))
    for key in ("unaligned", "aligned"):
        E = extract(key, X)
        np.save(os.path.join(config.EMB_DIR, f"echonext_{key}.npy"), E)
        print(key, E.shape)
