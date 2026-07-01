"""Paper Fig. 4-5 (external validation): shared path placeholders and label columns.

All paths are RELATIVE PLACEHOLDERS. Raw data and trained weights are NOT included
(released on acceptance / credentialed access); adapt these to your own layout.
"""

# ---- data (not included) ----
ECHONEXT_DIR = "./data/echonext"    # PhysioNet EchoNext v1.1.1 (credentialed): waveforms + metadata csv
INTERNAL_DIR = "./data/internal"    # internal cohort: cached embeddings + phenotype labels
EMB_DIR = "./embeddings"            # cached frozen-encoder embeddings
FIG_DIR = "./figures"               # output figures (Fig. 4-5)

# ---- ECG encoder (architecture referenced, weights not included) ----
ENCODER_REPO = "./ecg_encoder"      # official ECG foundation-model repo (see cross-modal module / original paper)
WEIGHTS = {
    "aligned":   "./weights/ecg_aligned.ckpt",    # proposed: cross-modally aligned ECG encoder
    "unaligned": "./weights/ecg_pretrained.pth",  # ablation: same backbone, no alignment
}

# ---- signal / model constants (Implementation Details) ----
FS_MODEL = 250        # encoder operating sampling rate (Hz)
SEQ_LEN = 2250        # encoder input length (samples) = 9 s @ 250 Hz
HP_HZ, LP_HZ = 0.67, 40   # band-pass cutoffs (Hz)
EMB_DIM = 768

# ---- EchoNext echo-derived clinical labels used here ----
LVEF_COL = "lvef_value"                       # continuous LVEF (%)
SHD_COL = "shd_moderate_or_greater_flag"      # structural heart disease composite (binary)
LVH_COL = "lvwt_gte_13_flag"                  # LV hypertrophy (binary)
SPLIT_COL = "split"
