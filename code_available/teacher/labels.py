# [Paper §Method] Target column definitions: structural + functional phenotypes (+ demographics/disease).

"""
labels.py — the single multi-task label set for the BIBM2026 ablation study.

Same labels across ALL variants (structure + function + disease + demographics),
sourced from data/teacher/teacher_*.csv. Reported metric = MAE/R² for regression
(spotlight FOCUS_REG), AUROC for classification (auxiliary).

Order is FIXED — head index i ↔ column. Do not reorder (checkpoints depend on it).
"""

# ── regression targets (report MAE/R²) ──────────────────────────────────────────
# structure (volumes / mass / wall thickness / atrial sizes)
STRUCT_REG = ['LVEDV', 'LVESV', 'LVSV', 'LVM', 'LV_WT',
              'RVEDV', 'RVESV', 'RVSV',
              'LA_max', 'LA_min', 'LASV', 'RA_max', 'RA_min', 'RASV']
# function (ejection fractions / cardiac output / strain)
FUNC_REG = ['LVEF', 'RVEF', 'LV_CO', 'GLS', 'GCS', 'GRS', 'LAEF', 'RAEF']
# demographics (auxiliary regression)
DEMO_REG = ['age', 'bmi', 'height', 'weight']

REG_COLS = STRUCT_REG + FUNC_REG + DEMO_REG          # 14 + 8 + 4 = 26

# the 5 focus functional metrics — drive model selection + main results table
FOCUS_REG = ['LVEF', 'RVEF', 'GLS', 'GCS', 'GRS']

# ── classification targets (report AUROC; auxiliary) ────────────────────────────
DISEASE_CLS = ['I21', 'I25', 'I48', 'I20', 'I50', 'ISCHEMIC']
DEMO_CLS = ['sex']
CLS_COLS = DISEASE_CLS + DEMO_CLS                    # 6 + 1 = 7
