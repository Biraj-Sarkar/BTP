# How to Add Things to This Codebase

Recipes for the extensions most likely to be wanted. Each is a small, local
change — the pipeline is deliberately built so that adding a component means
registering it, not rewiring anything.

**Read this first:** every stage sits behind a narrow contract. If a change
requires editing more than the file you are adding to, the contract is probably
being bypassed — that is worth a second look rather than a workaround.

Run the tests after any change:

```bash
.venv/bin/python -m pytest tests -q
```

---

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
export SSL_CERT_FILE=$(.venv/bin/python -c "import certifi;print(certifi.where())")
```

That last line is needed only when downloading pretrained weights — macOS
Python often lacks a usable certificate bundle.

The package lives in `src/`, so command-line use needs:

```bash
PYTHONPATH=src .venv/bin/python -m anemia --help
```

---

## 1. Add a new conjunctiva extractor

**The contract:** an extractor is any callable
`(rgb: np.ndarray) -> (mask: np.ndarray, name: str)`, where `mask` is uint8
with values in {0, 255} at the input resolution. Nothing else.

**Step 1** — write the mask function in `src/anemia/segment.py`:

```python
def my_extractor_mask(image_rgb: np.ndarray) -> Mask:
    """One sentence on the observation this exploits."""
    # ... your logic ...
    return smooth_mask(candidate.astype(np.uint8) * 255)
```

Useful helpers already there: `redness_index`, `lightness`, `smooth_mask`,
`largest_connected_component`, `_fill_holes`.

**Step 2** — register it at the bottom of the same file:

```python
EXTRACTORS = {
    ...
    "myextractor": _named(my_extractor_mask, "myextractor"),
}
```

`_named` wraps a bare mask function into the `(mask, name)` contract. If your
extractor needs its own fallback logic, write the full `(mask, name)` callable
instead and register it directly.

**Step 3** — add the name to the CLI choices in `src/anemia/cli.py`
(`fit_linear.add_argument("--extractor", choices=[...])`) and in
`predict_folder.py`.

**Then use it:**

```bash
PYTHONPATH=src .venv/bin/python -m anemia fit-linear --data . --extractor myextractor --out runs/mine.json
PYTHONPATH=src .venv/bin/python -m anemia stages --dir left_eye --out /tmp/trace   # look at the masks
```

**Important:** the extractor name is written into the model file, and
`predict_folder.py` reads it back. Features measured inside a different mask
are not comparable, so a model served with the wrong extractor is silently
wrong rather than obviously broken. Do not bypass this.

**Before believing it works:** run `python -m anemia debug --dir left_eye` and
*look at the overlay sheet*. Every extraction failure in this project's history
was invisible to the numbers and obvious in the overlays.

---

## 2. Add a new feature representation

Features are the numbers the model actually sees. Keep them few — with a small
cohort, more features reliably makes things worse.

**In `src/anemia/linear_model.py`:**

```python
REPRESENTATIONS = ("means", "chroma", "erythema", "lab", "a_only", "mine")

FEATURE_NAMES = {
    ...
    "mine": ["my first feature", "my second feature"],
}
```

then add a branch in `extract_features`:

```python
if representation == "mine":
    return np.array([something(R, G, B), something_else(R, G, B)])
```

Add the name to the `--representation` choices in `cli.py`.

**A note on what tends to work:** representations that are invariant to overall
illumination intensity — anything of the form ratio or log-ratio, like
`R/(R+G+B)` or `log(R/G)` — have outperformed raw channel means here, because
they cancel exposure differences between captures. Absolute channel values
carry lighting as well as physiology.

---

## 3. Add a new metric

`src/anemia/metrics.py` holds them. Regression metrics take
`(predictions, targets)` and return a dict; classification metrics take
`(predicted_flags, true_flags)`.

Add the key to the printing loops in
`experiments/06_metrics_evaluation/evaluate.py` and
`experiments/11_head_to_head/head_to_head.py`.

**House rule:** every metric is reported next to a baseline (predict-the-mean
for regression, majority class for classification). A metric quoted alone on a
small cohort is how a useless model gets published.

---

## 4. Add a new dataset

Datasets differ in layout, so each gets an adapter that emits the common
`Sample` type. Nothing downstream changes.

**In `src/anemia/data.py`:**

```python
def load_my_dataset(root: Path) -> List[Sample]:
    """Layout: describe it in one line."""
    samples = []
    for ...:
        samples.append(Sample(
            patient_id=f"mydata:{subject_id}",   # SAME id for all images of one subject
            source="my-dataset",
            image_path=path,
            hb=float(hb),
            age_years=age,
            sex=sex,
        ))
    return samples
```

Register it in `LOADERS`, and add a shape check to `load_dataset` so
`--data <folder>` finds it automatically.

**Two things that must be right:**

- **`patient_id` groups a subject, not an image.** Both eyes of one person, or
  repeat captures, must share it. Otherwise near-identical photos land on
  opposite sides of a train/test split and the model scores well by recognising
  the subject rather than reading pallor.
- **Use `_first_column` for column lookup**, not a bare substring test. It
  matches on word boundaries, because `"age"` is a substring of `"image id"` —
  that exact bug once assigned every patient the WHO threshold for their ID
  number instead of their age, and nothing crashed.

**Always run this before training on new data:**

```bash
PYTHONPATH=src .venv/bin/python -m anemia inspect --data path/to/dataset
```

If the parsed image count does not match what the dataset claims, the loader
did not match the layout. Fix that first.

---

## 5. Change quality-control rules

`src/anemia/preprocess.py`, function `assess_quality`. Thresholds live in
`QualityConfig` in `config.py`.

To add a check, compute the statistic and append a human-readable reason:

```python
if my_statistic > config.my_threshold:
    reasons.append(f"too something ({my_statistic:.2f})")
```

A non-empty `reasons` list means the capture is rejected, dropped from
training, and returned to the app as a retake prompt.

**One known gap worth filling:** there is no check on illuminant *colour*. A
strongly tinted photo currently passes. The ratio of brightest to dimmest
channel mean separates them cleanly — around 1.7 for acceptable captures versus
22–200 for unusable ones.

---

## 6. Change what the API returns

`src/anemia/predict.py` — the `Prediction` dataclass and its `to_dict`. Both
`AnemiaPredictor` (neural) and `LinearPredictor` (linear) return the same type,
so the app cannot tell them apart.

If you add a field, add it to `to_dict` and tell whoever builds the UI. Do not
remove `usable` — the app branches on it.

Any change here is a change to a published contract: `UI_INTEGRATION.md`
documents the response shape field by field and carries a mock server that
mirrors it. Update both, or the client is built against a contract the server
no longer honours.

---

## 7. Retrain the model

Only needed when **new labelled data** arrives. Prediction never retrains.

```bash
PYTHONPATH=src .venv/bin/python -m anemia fit-linear --data . --out runs/linear_model.json
```

Options: `--representation`, `--extractor`, `--alpha`, or `--segmenter <dir>`
for a trained segmentation model.

It prints in-sample and held-out metrics side by side. **In-sample describes
the fit; only held-out estimates a new patient.** Both are stored in the model
file so the distinction survives.

---

## Conventions worth keeping

- **Results on synthetic data are labelled as such.** They verify the plumbing
  and prove nothing clinical.
- **Everything predictive is reported against a baseline.**
- **Splits group by patient**, never by image.
- **Failed approaches go in `RESEARCH_LOG.md`** with the reason. A rejected
  approach with a measurement is a result; an untried one is a gap.
- **Look at images, not only numbers.** Three separate failures in this project
  were invisible to every metric and obvious in a contact sheet.

## Where things are

| Need | File |
|---|---|
| How the method works | `PIPELINE.md` |
| What each file does | `CODE_GUIDE.md` |
| What has been tried and failed | `RESEARCH_LOG.md` |
| Abbreviations | `GLOSSARY.md` |
| Experiment results | `experiments/README.md` |
| Weekly progress | `WORKLOG.md` |
| How to build the app against the API | `UI_INTEGRATION.md` |
