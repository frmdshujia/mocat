# External Validation on EchoNext

Code for the external-validation experiments (paper **Sec. V** and **Fig. 4 / Fig. 5**).

The ECG encoder that was cross-modally aligned on the internal cohort is **frozen**
and evaluated on the independent **EchoNext** cohort (12-lead ECGs with
echocardiography-confirmed labels). The encoder is never trained or tuned on
EchoNext. Throughout, `aligned` = the proposed cross-modally aligned encoder and
`unaligned` = the same backbone without alignment (ablation).

Two complementary analyses:

* **External-evaluation table** (`linear_probe.py`) — quantifies, *directly on
  EchoNext*, how much structure/function information the frozen embeddings carry,
  via a subject-level **5-fold CV** linear probe (mean±std over folds). Targets
  mirror what the alignment learns: LVEF and RV systolic dysfunction (function),
  IVS/LVPW wall thickness (structure).
* **Clinical figures** (`clinical_figures.py`) — the transfer view: an LVEF probe
  is fit on the *internal* embeddings and applied, frozen, to *external*
  embeddings (strict transfer, clinical units), producing Bland–Altman agreement
  and ROC discrimination (LVEF<40 / <50 / HFmrEF) with bootstrap 95% CI.

## Files

| File | Paper part | What it does |
|---|---|---|
| `config.py` | Implementation Details | Shared relative paths, signal constants, label columns. |
| `prep_echonext.py` | Sec. V (sampling-rate alignment) | EchoNext → encoder-ready input; resamples any cohort to the model rate. |
| `extract_embeddings.py` | Sec. V | Frozen encoder → embeddings for `aligned` / `unaligned`. |
| `linear_probe.py` | Sec. V (external table) | 5-fold CV frozen linear probe; LVEF/IVS/LVPW (R²), RV dysf. (AUROC). |
| `clinical_figures.py` | Fig. 4 / Fig. 5 | Bland–Altman + ROC via internal→external LVEF transfer. |

## Pipeline

```
prep_echonext.py         # EchoNext waveforms -> prepped (sampling-rate aligned)
extract_embeddings.py    # -> echonext_aligned.npy, echonext_unaligned.npy
linear_probe.py          # -> echonext_probe_table.csv, echonext_oof.npz   (external table)
clinical_figures.py      # -> Fig. 4/5 panels                              (clinical transfer)
```

## Notes

* **Data are access-restricted** (PhysioNet EchoNext, credentialed use) and the
  internal cohort is not public. Raw waveforms/labels are **not** redistributed
  here. Trained encoders are available in Release v1.0.0; use the root
  `MODEL_LOADING.md` for verified representation extraction. This archived
  cohort pipeline requires private inputs and adaptation of its external API.
* **External encoders are not vendored.** The ECG backbone comes from its original
  release (see the paper's references / the cross-modal-alignment module); place it
  under the path in `config.ENCODER_REPO`.
* Only the *proposed* pipeline is included. Baselines are evaluated with the
  **identical** `linear_probe.py` protocol (which is what makes the comparison
  fair), but baseline model code is not part of this repository.
* All paths in `config.py` are relative placeholders — adapt to your own layout.
