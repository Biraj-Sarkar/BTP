# Which Regulariser, and Why

Empirical answer to "why ridge rather than lasso, elastic net, or plain least
squares" on this cohort, rather than a textbook one.

Run: `python experiments/08_regularisation/compare_regularisers.py`

> **Numbers superseded, conclusion unchanged.** This compares regularisers over
> **six channel means** (R, G, B, L\*, a\*, b\*) with no quality gate, so its
> MAEs sit around the leave-one-out baseline of 1.083. The deployed model uses
> two erythema-index features with the QC gate and reaches held-out MAE 1.033.
> The finding this experiment exists for — the predictors are near-collinear
> (max \|r\| = 0.993), which is exactly where lasso becomes unstable and ridge
> does not — is a property of colour features in general and is unaffected.

## Setup

26 patients, 6 features (mean R, G, B, L\*, a\*, b\* inside the conjunctiva
mask), leave-one-patient-out. "Coefficient stability" is the mean standard
deviation of each fitted coefficient across the 26 refits — lower means the
model is telling a consistent story about which features matter.

## Result

| Model | LOO MAE | Features kept | Coefficient stability |
|---|---|---|---|
| OLS (no penalty) | 1.282 | 6.0 | **11.499** |
| Ridge, alpha = 1 | 1.102 | 6.0 | 0.042 |
| Ridge, alpha = 10 | **1.068** | 6.0 | **0.024** |
| Lasso, alpha = 0.1 | 1.123 | 2.9 | 0.034 |
| Lasso, alpha = 0.5 | 1.083 | **0.0** | 0.000 |
| Elastic Net (0.3, l1=0.5) | 1.109 | 3.4 | 0.026 |
| *baseline: predict the mean* | *1.083* | — | — |

**Feature correlation: max |r| = 0.993, mean |r| = 0.593.**

## Reading it

**The predictors are near-collinear.** Mean R, G and B rise and fall together
because a brighter capture raises all three. That single fact drives the whole
choice.

**Lasso is the wrong tool for collinear predictors.** Its L1 penalty drives
coefficients exactly to zero, so given a group of correlated features it keeps
one arbitrarily and discards the rest — and which one it keeps varies between
folds. Ridge spreads weight across the group instead, which is both steadier
and more truthful: no single channel mean "is" the signal.

**Unregularised least squares is unusable here.** Coefficient stability 11.499
versus ridge's 0.042 — nearly 300x the wobble. Its coefficients are fitting
noise, and its MAE (1.282) is the worst of any model tested, well below the
baseline.

**Lasso at alpha = 0.5 zeroes every coefficient** and its MAE becomes exactly
the baseline, 1.083. That is not a failure of the method — it is Lasso
concluding that no feature earns its keep. It confirms the project's central
finding from an independent direction.

**Ridge is the best available, and still barely at baseline.** alpha = 10
reaches 1.068 against a baseline of 1.083. The margin is inside the noise at
n = 26.

## Why alpha = 1 in the pipeline rather than 10

Tuning alpha on the same 26 patients used for evaluation would leak: we would
be selecting the value that flatters this particular sample. Doing it properly
needs a nested loop or a separate tuning split, neither affordable at this n.
alpha = 1 is therefore a principled default, and the alpha = 10 figure is
reported as an observation rather than adopted.

## Advantages and limitations of each regulariser

| Method | Advantages | Limitations |
|---|---|---|
| **OLS** | No hyperparameter; unbiased when assumptions hold; coefficients directly interpretable | Collapses with few samples or correlated features — here, coefficient wobble of 11.5 and the worst MAE |
| **Ridge** | Stable under collinearity; keeps all features; closed-form solution; one hyperparameter | Never produces exactly zero coefficients, so it cannot tell you which features are useless |
| **Lasso** | Automatic feature selection; yields sparse, readable models | Arbitrary among correlated predictors, and the choice shifts between folds; needs iterative solving |
| **Elastic Net** | Sparsity plus stability under collinearity | Two hyperparameters to set instead of one |

## When the other choices would be right

- **Lasso** — many features (say 200 texture descriptors) where knowing *which*
  few matter is itself the goal. Feature selection is its purpose; we have six
  features and no need to remove any.
- **Elastic Net** — many correlated features where sparsity is still wanted. It
  landed between the two here, as expected.
- **Plain OLS** — plenty of data relative to features, and low collinearity.
  Neither holds here.
