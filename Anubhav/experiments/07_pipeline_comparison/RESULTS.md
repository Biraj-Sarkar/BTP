# Pipeline Comparison and the Linear Floor

Answers two questions with one experiment:

1. **Which pipeline performs better?** Both, end to end and with a shared mask.
2. **How far is a simple model from working?** Linear regression only — no
   neural network — to establish the floor a CNN has to clear.

Run: `python experiments/07_pipeline_comparison/compare_pipelines.py`.
Raw output: `comparison.json`.

## Method

26 patients, laboratory Hb, leave-one-patient-out. Two framings:

- **End-to-end** — each pipeline uses its own segmentation and its own colour
  representation, as written. Answers "which would do better if deployed".
- **Shared-mask** — both representations measured inside the same mask.
  Isolates colour treatment from segmentation.

Models are **ordinary least squares** and **ridge** (OLS plus an L2 penalty,
which shrinks coefficients and reduces variance when data is scarce). Feature
sets: 3–6 channel means, and a deliberately over-rich 24-feature vector
(mean, standard deviation, 25th and 75th percentiles for R, G, B, L\*, a\*, b\*).

Segmentation backends actually used: pipeline A took the refined extractor on
all 52 images; pipeline B took its colour heuristic on 51 and grabCut on 1.

## Results

| Arm | Features | MAE | RMSE | R² | Pearson r | F1 | Recall |
|---|---|---|---|---|---|---|---|
| **End-to-end A** — gray-world + refined seg, RGB | 3 | **1.073** | 1.479 | −0.166 | −0.238 | 0.688 | 0.846 |
| **End-to-end B** — CLAHE-L + own seg, a\*/b\* | 2 | 1.135 | 1.498 | −0.196 | −0.562 | 0.647 | 0.846 |
| Shared-mask A — RGB, ridge | 3 | **1.073** | 1.479 | −0.166 | −0.238 | 0.688 | 0.846 |
| Shared-mask B — a\*/b\*, ridge | 2 | 1.192 | 1.489 | −0.182 | −0.233 | 0.667 | 0.846 |
| Shared-mask A — RGB, plain OLS | 3 | 1.113 | 1.501 | −0.200 | −0.021 | 0.710 | 0.846 |
| Shared-mask A — 24 rich features, ridge | 24 | 1.102 | 1.532 | −0.251 | +0.063 | 0.593 | 0.615 |
| Shared-mask A — 24 rich features, plain OLS | 24 | **11.444** | 15.002 | **−118.937** | −0.194 | 0.444 | 0.462 |
| *BASELINE — predict the training mean* | — | *1.083* | *1.425* | *−0.082* | — | *0.722* | *1.000* |

## Reading it

### 1. Pipeline A wins, consistently but narrowly

End-to-end, A beats B on every metric: MAE 1.073 vs 1.135, R² −0.166 vs
−0.196, F1 0.688 vs 0.647. With a shared mask A still wins (1.073 vs 1.192),
so the advantage is not only better segmentation — the RGB representation
itself carries slightly more. This is the fourth independent measurement
pointing the same way, after the cohort correlation, the lighting stress test,
and the representation ablation.

The margin is small and n = 26. Treat it as a consistent direction, not a
settled result.

### 2. Nothing beats the baseline — the linear floor is unmet

Every arm has **negative R²**, meaning it explains less variance than a
constant. Every arm has **F1 at or below** the trivial baseline's 0.722. The
baseline achieves recall 1.000 simply by calling almost everyone anaemic, which
is exactly why F1 and R² must be read next to accuracy rather than instead of
it.

So the honest answer to "how far are we from the real thing" is: **at this
scale we have not started.** Mean colour statistics from 26 patients contain no
usable haemoglobin signal.

### 3. More features made it worse — the important finding

Going from 3 features to 24 did not help ridge (MAE 1.102 vs 1.073) and
destroyed plain OLS: **MAE 11.4 g/dL, R² −118.9**. With 24 features and 25
training points per fold, OLS has almost enough freedom to interpolate the
training set exactly, so it fits noise and predicts nonsense on the held-out
patient. Regularisation is the only thing standing between the model and that
outcome.

The problem is therefore **not** that the features are too crude. Adding
features makes it worse, because the constraint is data quantity, not feature
richness.

## What this means for the neural network

The 24-feature blow-up is the practical warning. ResNet-18 has ~11.2 million
parameters, roughly **8.4 million trainable** in the current configuration —
that is 400,000 times more freedom than the 24-feature model that produced
R² = −118.9 on this cohort.

Concretely, before a CNN result on this data can be believed:

1. **Do not train on 26 patients.** Pool Eyes-defy-anemia (218) and CP-AnemiC
   (710) first; ~900 images is the minimum realistic base.
2. **Train the segmenter first.** Both pipelines' extraction is heuristic
   today, and pipeline B's demonstrably lands off-tissue on most images.
3. **Freeze aggressively and regularise hard.** The current setup already
   freezes through `layer3` and applies dropout; on small data that matters
   more than architecture.
4. **Report against the same baselines used here.** A CNN that cannot beat
   predict-the-mean has learned nothing, however good its training curve looks.
5. **Use this file as the reference floor.** MAE 1.073, R² −0.166, F1 0.688 on
   26 patients with linear models. Any CNN claim should be compared against
   both that and the baseline.

## Limits

- n = 26; differences between arms are inside the noise.
- Leave-one-out on small samples gives nearly unbiased but high-variance error
  estimates.
- One cohort, one capture device, one operator.
- Ridge alpha fixed at 1.0, not tuned — tuning inside the loop would leak, and
  a separate tuning split is not affordable at this n.
