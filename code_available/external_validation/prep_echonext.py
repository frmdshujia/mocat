"""Paper Fig. 4-5 (external validation): EchoNext ECG preprocessing with SAMPLING-RATE ALIGNMENT.

EchoNext waveforms are 12-lead, 10 s at 250 Hz, shape (N, 1, 2500, 12).
The proposed ECG encoder consumes 12-lead signals at 250 Hz over a fixed window,
band-pass filtered and per-record standardized. This module reconciles the EchoNext
sampling rate with the encoder's expected input so that external evaluation is
apples-to-apples with internal training.

Sampling-rate alignment (the key external-validation detail):
  - the internal cohort is acquired at 500 Hz and resampled to 250 Hz before cropping;
  - EchoNext is already 250 Hz, so it is only transposed / cropped / filtered / standardized
    (no resampling). For any cohort whose rate != FS_MODEL, resample to FS_MODEL first.
"""
import numpy as np
import config

# Signal transforms follow the ECG-encoder repo (config.ENCODER_REPO): resample, center-crop,
# high-pass, low-pass, per-record standardize. (Imported from the official repo; not vendored here.)
from ecg_transforms import Resample, CenterCrop, HighpassFilter, LowpassFilter, Standardize  # noqa: E402

_cc = CenterCrop(config.SEQ_LEN)
_hp = HighpassFilter(fs=config.FS_MODEL, cutoff=config.HP_HZ)
_lp = LowpassFilter(fs=config.FS_MODEL, cutoff=config.LP_HZ)
_st = Standardize(axis=(-1, -2))          # standardize over (lead, time) per record


def prep_record(x, fs_in):
    """One record -> (12, SEQ_LEN) float32, encoder-ready.

    x : (time, 12) or (12, time). fs_in : acquisition rate (Hz).
    """
    x = np.asarray(x, dtype=np.float64)
    if x.shape[-1] == 12:                 # (time, lead) -> (lead, time)
        x = x.T
    if fs_in != config.FS_MODEL:          # sampling-rate alignment
        x = Resample(target_fs=config.FS_MODEL)(x, fs=fs_in)
    x = _cc(x)                            # center-crop to SEQ_LEN
    x = _hp(x); x = _lp(x)                # band-pass
    x = _st(x)                            # per-record standardize
    return np.ascontiguousarray(x, dtype=np.float32)


def prep_echonext(waveforms, fs_in=250):
    """EchoNext waveforms (N, 1, 2500, 12) -> (N, 12, SEQ_LEN) float32."""
    return np.stack([prep_record(w[0], fs_in) for w in waveforms])


if __name__ == "__main__":
    import os
    w = np.load(os.path.join(config.ECHONEXT_DIR, "EchoNext_test_waveforms.npy"), mmap_mode="r")
    X = prep_echonext(w, fs_in=config.FS_MODEL)
    np.save(os.path.join(config.EMB_DIR, "echonext_test_prepped.npy"), X)
    print("prepped", X.shape)
