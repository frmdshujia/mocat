"""Target column groups for the frozen-probe evaluation.

Paper section: Table I column definitions (Evaluation Protocol).

Every method in Table I is scored on the identical set of targets. The three
reported blocks are:
  * FUNCTIONAL  (R^2): ejection fractions and strain;
  * STRUCTURAL  (R^2): chamber volumes, mass, wall thickness;
  * DISEASE  (AUROC): ICD-derived cardiac endpoints.
Continuous / binary demographics are scored the same way as controls.

Column *names* are those of the paired-cohort label table (`crossmodal_*.csv`);
the probe scores whichever of these columns are present. The extended phenotype
set used for the supplementary per-metric (Manhattan) figure is handled by the
same code path — no metric is special-cased.
"""

FUNCTIONAL = ["LVEF", "RVEF", "GLS", "GCS", "GRS", "LAEF"]
STRUCTURAL = ["LVEDV", "LVESV", "LVM", "LV_WT", "RVEDV", "RVESV"]
PHENO = FUNCTIONAL + STRUCTURAL

# ICD-derived endpoints (display names): MI, CAD, Angina, AF, HF, Ischaemic.
DISEASE = ["I21", "I25", "I20", "I48", "I50", "ISCHEMIC"]

# Demographic controls (scored identically; extend to match your label table).
DEMO_CONT = ["age"]
DEMO_BIN = ["sex"]

ALL_TARGETS = PHENO + DISEASE + DEMO_CONT + DEMO_BIN
