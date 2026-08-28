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
| `05_representation_ablation/` | Four colour representations scored on Hb prediction, including the gray-world + a\*/b\* hybrid | *Superseded in part.* No representation beats baseline **at channel-mean scale**; gray-world consistently beats its uncorrected counterpart. Illumination-invariant features later changed this — see `11` |
| `06_metrics_evaluation/` | How the Hb number is produced, the full metric suite on synthetic and real data, plus diagnostic plots | *Real-cohort figures superseded.* Synthetic R² 0.888, F1 0.986 (trivial by construction). The metric definitions and the mechanism write-up are current |
| `07_pipeline_comparison/` | Both pipelines end-to-end and shared-mask, scored with linear models (OLS + ridge) — the floor a CNN must clear | *Superseded in part.* Pipeline A wins consistently but narrowly; **24 features on 25 points blows up to R² −118.9**, the overfitting warning for the network stage |
| `08_regularisation/` | OLS vs ridge vs lasso vs elastic net, with coefficient-stability measurements | *Numbers superseded.* Features are near-collinear (max \|r\| = 0.993), which is why ridge is used; unregularised OLS wobbles ~300× more, and lasso at moderate penalty zeroes every coefficient |
| `09_train_test_protocol/` | In-sample vs out-of-sample performance, protocol agreement, and the instability of a single split at n=26 | *Numbers superseded; protocol findings stand.* A single 60/20/20 split reports test R² anywhere from **−2.43 to +0.78** on identical data — the strongest argument for getting the public datasets |
| `10_fitted_equation/` | The channel-mean equation in standardised and raw form, bootstrap confidence intervals, and a training-size comparison | *Superseded — not the deployed equation.* Its channel-mean coefficients are physiologically backwards; **the deployed erythema model is not**, with a dominant **+1.049 on log(R/G)**. Read §1–2 (protocol) as current, §3–4 as history |
| `11_head_to_head/` | **The deployment decision.** Both pipelines scored on identical patients: five feature representations, full metric suite, nested CV, a paired test, and the noise floor for each fitted R² | A leads B on every matched framing, but **p = 0.408 — the two are not separable at n=26**, and no fitted R² clears its own permutation null |
| `12_mask_placement/` | Where each pipeline's mask actually lands, measured by redness index inside the mask on all 52 captures | **A lands on tissue in 52 of 52; B is off-tissue in 46 of 52** and its placement is statistically indistinguishable from the sclera-selecting brightness baseline. The two masks overlap by a **median of 0.000** |

## The deployed model is not in this folder

These are studies. The model that actually ships is fitted by
`anemia fit-linear` and lives in `runs/linear_model.json`: the **erythema
index with the QC gate applied before fitting**, on all 26 patients.

```
Hb = 9.8308 + 15.101390·log(R/G) − 12.790747·log(R/B)     [g/dL]
```

| | in-sample | held-out (LOO) | baseline |
|---|---|---|---|
| MAE (g/dL) | **0.918** | 1.033 | 1.083 |
| R² | **+0.195** | **+0.007** | −0.082 |

It differs from experiment 11's figures because `fit-linear` applies the QC
gate and drops 3 blurred captures, which the head-to-head deliberately does not
— that comparison needs both pipelines to see identical patients. Applying the
gate is what lifted held-out R² above zero, making this the only arm in the
project that beats its baseline on unseen patients. It is still not significant
at n = 26: a permutation test gives **p = 0.067** held out, **p = 0.074**
in-sample.

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
python experiments/12_mask_placement/measure_placement.py
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
