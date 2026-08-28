# Why R² Is Negative, and Why There Is No Validation Set

Two questions that look like bugs but are results.

Run: `python experiments/09_train_test_protocol/protocol_comparison.py`

> **The protocol conclusions hold; the specific numbers are superseded.**
> This experiment fits **channel means** (mean R, G, B) with no quality gate,
> so its in-sample/out-of-sample pair is **+0.103 / −0.166**. The deployed
> model uses the erythema index with the QC gate applied before fitting and
> reaches **+0.195 / +0.007** on the same 26 patients — so it no longer
> illustrates a *negative* held-out R², though the fitted-versus-held-out gap
> it exists to explain is unchanged and, if anything, larger.
>
> Everything in §2 (three protocols agree), §3 (a single 60/20/20 split at
> n = 26 reports which patients landed where) and §4 is about evaluation
> method, not about this model, and stands as written.

---

## 1. Yes, the model trains

Leave-one-out **is** training. It fits 26 separate models, each on 25 patients,
and tests each on the one patient it never saw. Training happens 26 times.

The way to prove training works is to compare in-sample against out-of-sample:

| Evaluation | R² | MAE (g/dL) |
|---|---|---|
| **In-sample** — fit on all 26, scored on the same 26 | **+0.103** | 0.940 |
| **Out-of-sample** — leave-one-out, scored only on unseen patients | **−0.166** | 1.073 |

In-sample R² is **positive**. The model does fit: given the answers, it finds a
relationship that explains about 10% of the variance. On patients it has never
seen, that relationship is worth less than nothing.

**The gap between those two rows is the overfitting**, and it is the whole
answer to "didn't you train first?". Training succeeded. Generalisation failed.

### Why R² can be negative at all

R² is bounded between 0 and 1 **only when the model is scored on the same data
it was fitted to**. Out-of-sample there is no such guarantee:

```
R² = 1 − Σ(predicted − true)² / Σ(true − mean)²
```

If the model's squared error exceeds the error of simply predicting the mean,
the fraction goes above 1 and R² goes below 0. Negative R² has a plain reading:
**worse than a model that ignores the image entirely.**

### A nuance worth noticing

In-sample R² is only +0.103, not +0.9. With three features and a ridge penalty
the model *cannot* memorise the training set — so this is not a case of wild
overfitting. It is something more sobering: even with full access to the
answers, mean conjunctival colour explains only about a tenth of the
haemoglobin variance in this cohort, and that tenth does not survive to new
patients.

---

## 2. It is not an artefact of leave-one-out

If negative R² were a quirk of the protocol, other protocols would disagree.
They do not:

| Protocol | R² | Spread across repeats |
|---|---|---|
| Leave-one-out | −0.166 | — |
| 5-fold CV, 20 repeats | −0.175 | −0.362 to +0.011 |
| 13-fold CV, 20 repeats | −0.168 | −0.230 to −0.109 |

Every protocol lands in the same place. The finding is about the data, not the
evaluation method.

---

## 3. Why there is no separate validation set

A 60/20/20 split of 26 patients gives roughly **15 train, 5 validation,
6 test**. Each number is then computed from a handful of people.

To measure how unstable that is, the same model and the same data were split
200 times with different random seeds:

| Statistic | Mean | Range across seeds |
|---|---|---|
| Test MAE | 1.101 g/dL | **0.316 to 2.263** |
| Test R² | −0.186 | **−2.433 to +0.782** |

**Read that last row carefully.** With nothing changing except which six
patients land in the test set, the same model reports a test R² anywhere from
−2.43 to **+0.78**.

An R² of +0.78 would look like a working system. It would be entirely luck. A
single train/validation/test split at this sample size does not measure
performance; it measures which patients happened to fall where.

### What a validation set is normally for

To choose hyperparameters — the ridge penalty, the number of features, a
learning rate — without touching the test set.

We deliberately tune **nothing**. The ridge penalty is fixed at α = 1 as a
principled default rather than a fitted value, precisely because there is no
validation budget to tune against. Choosing α by looking at the same 26
patients used for evaluation would leak: we would be picking the value that
flatters this particular sample, and the reported error would be optimistic.

So the protocol is:

```
26 patients
   └── leave-one-out: 26 × (train on 25, test on 1)
       every patient tested exactly once, on a model that never saw them
       no hyperparameter chosen from the data
```

### When a three-way split becomes the right choice

Once the public datasets are in and the corpus is ~900 images:

- enough data for a real held-out test set that is not dominated by luck
- enough for a validation set to tune against
- and the correct protocol for the CNN, where leave-one-out would mean
  training the network 900 times

At that point: grouped train/validation/test, or grouped k-fold with an
untouched final test set. Not before.

---

## 4. What this changes

Nothing about the conclusion — every protocol agrees the model does not
generalise. What it adds is the *reason*, and a defence against two reasonable
objections:

- "You didn't train properly" → in-sample R² is positive; training worked.
- "Negative R² must be a bug" → it is the definition, and three protocols agree.

It also produces the strongest single argument for getting the public datasets:
at n = 26, **the difference between a triumphant result and a catastrophic one
is which six patients you happen to test on.**
