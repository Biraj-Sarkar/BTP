# Experiments

Investigations kept separate from the pipeline itself. Everything here is
exploratory: outputs, one-off analyses, and findings. The pipeline that
produces them lives at the top level (`src/`, `serve/`, `scripts/`).

| Folder | What it is | Headline |
|---|---|---|
| `01_stage_traces/` | One contact sheet per capture showing every pipeline stage side by side, for both the main pipeline (`main_pipeline/`) and the CIELAB pipeline (`cielab_pipeline/`) | Learning and debugging aid; open a single sheet and compare each stage against the previous one |
| `02_extraction_check/` | Overlay sheets and per-image QC for the 52 cohort captures | Extraction lands on conjunctiva across the cohort; 49/52 pass QC |
| `03_colour_normalisation/` | Gray-world vs CIELAB measured against laboratory Hb | Gray-world correlates better with Hb (ρ +0.30 vs +0.02) and gives more consistent left/right eye readings |
| `04_lighting_stress/` | One eye under six coloured illuminants | Both pipelines segment correctly only under neutral light; extreme casts destroy the measurement physically, not just numerically |
| `05_representation_ablation/` | Four colour representations scored on Hb prediction, including the gray-world + a\*/b\* hybrid | **No representation beats the predict-the-mean baseline at channel-mean scale**; gray-world consistently beats its uncorrected counterpart |
| `07_pipeline_comparison/` | Both pipelines end-to-end and shared-mask, scored with linear models (OLS + ridge) — the floor a CNN must clear | Pipeline A wins consistently but narrowly; **every arm is at or below the predict-the-mean baseline**; 24 features on 25 points blows up to R² −118.9 |
| `06_metrics_evaluation/` | How the Hb number is produced, the full metric suite (MSE/RMSE/MAE/R²/bias/LoA, accuracy/precision/recall/F1/specificity/NPV) on synthetic and real data, plus diagnostic plots | Synthetic: R² 0.888, F1 0.986. **Real: R² −0.166, F1 below baseline — the current real-data model does not work** |

| `08_regularisation/` | OLS vs ridge vs lasso vs elastic net on the cohort, with coefficient-stability measurements | Features are near-collinear (max \|r\| = 0.993), which is why ridge is used; unregularised OLS wobbles ~300x more, and lasso at moderate penalty zeroes every coefficient |

| `09_train_test_protocol/` | In-sample vs out-of-sample performance, protocol agreement, and the instability of a single split at n=26 | Training works (in-sample R² **+0.103**) but does not generalise (out-of-sample **−0.166**); a single 60/20/20 split reports test R² anywhere from **−2.43 to +0.78** on identical data |

| `10_fitted_equation/` | The actual fitted equation in both standardised and raw form, bootstrap confidence intervals, and a training-size comparison | `Hb = 12.5311 + 0.0102·R − 0.0587·G + 0.0356·B`. **The coefficients are physiologically backwards** — red is the weakest term, so the model is reading a blue-minus-green contrast, not haemoglobin |

| `11_head_to_head/` | **The deployment decision.** Both pipelines trained and scored on identical patients: five feature representations, full metric suite, nested CV, and a paired test of the difference | A wins 9 of 11 metrics, but **p = 0.408 — the two are not statistically separable at n=26**. Illumination-invariant features (erythema index) materially improved both |

## Reproducing

Most folders contain their outputs directly. Two are regenerable:

```bash
python -m anemia stages --dir left_eye --out experiments/01_stage_traces/main_pipeline
python cielab_stages.py --dir left_eye --out experiments/01_stage_traces/cielab_pipeline
python experiments/05_representation_ablation/run_ablation.py
python experiments/06_metrics_evaluation/evaluate.py
python experiments/06_metrics_evaluation/plot_results.py
python experiments/07_pipeline_comparison/compare_pipelines.py
python experiments/08_regularisation/compare_regularisers.py
python experiments/09_train_test_protocol/protocol_comparison.py
python experiments/10_fitted_equation/fit_equation.py
python experiments/11_head_to_head/head_to_head.py
```

## Conventions

- Results on synthetic phantoms are always labelled as such and are never
  clinical claims.
- Every predictive result is reported against a predict-the-mean baseline.
- Negative results are kept, not discarded — `05` and `07` are the clearest
  examples, and `../RESEARCH_LOG.md` records every failed approach with its
  reason.
- Models here are **linear only** (OLS / ridge). They exist to establish the
  floor a neural network has to clear, not to be the final model.
