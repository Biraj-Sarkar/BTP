# Building the App Against This Pipeline

Step-by-step guide for the client (phone app or web UI). You do not need the
datasets, the training stack, a Python environment, or any of `experiments/` to
follow this — the contract below is enough to build the whole app.

**What you are building.** A thin client. The phone captures a photograph of
the everted lower eyelid plus the patient's age and sex, uploads it, and the
server returns an estimated haemoglobin value in g/dL and an anaemic /
not-anaemic verdict. All image processing and inference happen server-side.

---

## Step 1 — Read only these files

| File | What you need from it |
|---|---|
| `serve/app.py` | Routes, form field names, status codes, upload limit |
| `src/anemia/predict.py` | `Prediction.to_dict()` — the exact response shape |
| `src/anemia/data.py` → `anemia_threshold()` | Which demographics change the cutoff |
| `predict_folder.py` | CSV columns, if you build a bulk/clinic view |
| This file | The steps |

Everything else — `experiments/`, `papers/`, `src/anemia/train*.py`,
`cielab_pipeline.py`, `gemini_anemia_pipeline.py` — is research material and is
not needed to build the client.

---

## Step 2 — Learn the contract

### `POST /predict`

`multipart/form-data`:

| Field | Type | Required | Notes |
|---|---|---|---|
| `image` | file | yes | JPEG or PNG. Max **12 MB** |
| `age_years` | float | no | Strongly recommended — see step 4 |
| `sex` | string | no | `"M"` or `"F"` |
| `pregnant` | bool | no | Defaults to `false` |

Error responses: **400** empty upload, **413** larger than 12 MB, **400**
undecodable image.

### `GET /health`

Returns `200` with a status block once a model is loaded, or **503** with a
`detail` string when none is. Poll this on app start so you can show "service
unavailable" instead of failing the first capture.

### The two success shapes — both HTTP 200

A usable capture:

```json
{
  "hb_g_dl": 11.42,
  "anemic": true,
  "threshold_g_dl": 11.5,
  "usable": true,
  "quality": {
    "focus": 85.3, "exposure": 42.1, "clipped": 0.0021,
    "mask_ratio": 0.211, "passed": true, "reasons": []
  },
  "message": "Estimated haemoglobin 11.4 g/dL. Below the 11.5 g/dL screening threshold for this patient. This is a screening estimate, not a diagnosis — confirm with a lab test."
}
```

A capture that failed quality control:

```json
{
  "hb_g_dl": null,
  "anemic": null,
  "threshold_g_dl": 12.0,
  "usable": false,
  "quality": {
    "focus": 25.4, "exposure": 38.9, "clipped": 0.0009,
    "mask_ratio": 0.187, "passed": false,
    "reasons": ["blurred (focus 25.4 < 35.0)"]
  },
  "message": "Could not read this photo clearly: blurred (focus 25.4 < 35.0). Please retake with the lower eyelid pulled down and steady lighting."
}
```

### Field reference

| Field | Type | Meaning |
|---|---|---|
| `hb_g_dl` | float \| null | Estimated haemoglobin, 2 dp. `null` when `usable` is false |
| `anemic` | bool \| null | `hb_g_dl < threshold_g_dl`. `null` when `usable` is false |
| `threshold_g_dl` | float | The WHO cutoff applied to *this* patient. Always present |
| `usable` | bool | **Branch on this.** False means the capture was rejected |
| `quality.focus` | float | Variance of the Laplacian. Below 35 rejects the capture |
| `quality.mask_ratio` | float | Fraction of the frame identified as conjunctiva. Outside 0.015–0.85 rejects |
| `quality.clipped` | float | Fraction of blown-out or crushed pixels. Reported only — see step 5 |
| `quality.exposure` | float | Spread of the lightness channel. Reported only |
| `quality.passed` | bool | Mirrors `usable` |
| `quality.reasons` | string[] | Human-readable rejection reasons. Empty when passed |
| `message` | string | Ready-to-display text. Already contains the required disclaimer |

`reasons` currently takes one of three forms, which you can pattern-match if you
want tailored guidance per failure:

- `blurred (focus 25.4 < 35.0)` → "hold still, tap to focus"
- `conjunctiva too small (0.008)` → "move closer, pull the lid down further"
- `mask implausibly large (0.910)` → "move back, frame just the lower lid"

---

## Step 3 — Build against a mock now

The server needs a fitted model file, which is produced separately. You do not
have to wait for it: the contract above is settled, so build the whole client
against this stand-in and switch the base URL later.

Save as `mock_server.py`, then `pip install fastapi uvicorn python-multipart`
and `uvicorn mock_server:app --port 8000`:

```python
"""Stand-in for serve/app.py. Same routes, fields and status codes."""
import random
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

app = FastAPI()
MAX_UPLOAD_BYTES = 12 * 1024 * 1024


def threshold(age, sex, pregnant=False):
    """Mirrors anemia_threshold() in src/anemia/data.py."""
    if age is None:
        return 12.0
    if age < 0.5:
        return 13.5
    if age < 5:
        return 11.0
    if age < 12:
        return 11.5
    if age < 15:
        return 12.0
    initial = (sex or "").strip().upper()[:1]
    if initial == "M":
        return 13.0
    if initial == "F":
        return 11.0 if pregnant else 12.0
    return 12.0


@app.get("/health")
def health():
    return {"status": "ok", "backbone": "mock", "metrics": {}}


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    age_years: Optional[float] = Form(None),
    sex: Optional[str] = Form(None),
    pregnant: bool = Form(False),
):
    payload = await image.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Empty upload")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image too large")

    cutoff = threshold(age_years, sex, pregnant)

    # One capture in four is rejected, so the retake path gets exercised.
    if random.random() < 0.25:
        return {
            "hb_g_dl": None, "anemic": None, "threshold_g_dl": cutoff, "usable": False,
            "quality": {"focus": 25.4, "exposure": 38.9, "clipped": 0.0009,
                        "mask_ratio": 0.187, "passed": False,
                        "reasons": ["blurred (focus 25.4 < 35.0)"]},
            "message": "Could not read this photo clearly: blurred (focus 25.4 < 35.0). "
                       "Please retake with the lower eyelid pulled down and steady lighting.",
        }

    hb = round(random.uniform(8.1, 14.3), 2)
    anemic = hb < cutoff
    return {
        "hb_g_dl": hb, "anemic": anemic, "threshold_g_dl": cutoff, "usable": True,
        "quality": {"focus": 85.3, "exposure": 42.1, "clipped": 0.0021,
                    "mask_ratio": 0.211, "passed": True, "reasons": []},
        "message": f"Estimated haemoglobin {hb:.1f} g/dL. "
                   f"{'Below' if anemic else 'At or above'} the {cutoff:.1f} g/dL "
                   "screening threshold for this patient. This is a screening estimate, "
                   "not a diagnosis — confirm with a lab test.",
    }
```

The Hb range 8.1–14.3 g/dL is the real range in the reference cohort, so your
result screen gets realistic values to lay out.

---

## Step 4 — Build the three screens

### Capture

Collect **age and sex before or alongside the photo.** This is not optional
polish. The anaemia cutoff depends on both, and when they are missing the
server falls back to 12.0 g/dL:

| Patient | Cutoff (g/dL) |
|---|---|
| Under 6 months | 13.5 |
| 6 months – 4 years | 11.0 |
| 5 – 11 years | 11.5 |
| 12 – 14 years | 12.0 |
| Women 15+ | 12.0 (11.0 if pregnant) |
| Men 15+ | 13.0 |
| Unknown | 12.0 |

A 7-year-old scored against 12.0 instead of 11.5, or an adult man scored
against 12.0 instead of 13.0, is wrong in a way no server-side accuracy can
recover. Ask for `pregnant` only when sex is F and age is 15+.

On-screen capture guidance: pull the lower lid down so the moist inner surface
is exposed, fill the frame with the lid, hold steady, use even indoor light, no
direct flash.

### Result — `usable: true`

Show `hb_g_dl` to one decimal, the verdict from `anemic`, and
`threshold_g_dl` beside it so the number has context ("11.4 g/dL — below the
11.5 g/dL threshold for a 7-year-old"). Render `message` verbatim, or write your
own copy **and keep the "screening estimate, not a diagnosis — confirm with a
lab test" wording**. Give the anaemic result a clear next action: see a
clinician for a confirmatory blood test.

### Retake — `usable: false`

Show the reason and how to fix it. **Show no number at all** — not greyed out,
not "approximately", nothing. Offer a one-tap retake that keeps the age and sex
already entered.

---

## Step 5 — Five rules that are not negotiable

1. **Branch on `usable`, never on the HTTP status.** A rejected capture is a
   normal outcome and returns 200. Treating it as an error, or treating 200 as
   success, both produce the wrong screen.
2. **Never display a number when `usable` is false.** A confident-looking
   haemoglobin value derived from an unreadable photo is the most harmful
   output this system can produce.
3. **Always carry the disclaimer.** Screening estimate, not a diagnosis.
4. **Do not compute the verdict client-side.** Use the server's `anemic` and
   `threshold_g_dl`. If the threshold rules are ever revised, a client that
   re-derives them silently disagrees with the server.
5. **`usable: true` does not mean the photo was well lit.** `clipped` and
   `exposure` are measured and returned but do not currently gate a capture,
   and there is no illuminant-colour check yet, so a flash-blown or strongly
   tinted photo can pass. If you want to warn the user client-side, `clipped`
   above roughly 0.02 is a reasonable trigger. Treat this as advisory only.

---

## Step 6 — Show which pipeline produced the number

The pipeline is **not** a user-facing choice, and the client must not offer one.
Extraction method and colour representation are selected once when the model is
fitted and recorded inside the model file; measurements taken inside a
different mask are not comparable, so a mismatched selection produces a
plausible number that means nothing.

What the client *should* do is display provenance — the extractor and
representation the active model was fitted with — on an "about" or debug
screen, so any result on screen can be traced to the configuration that
produced it. Those fields are not exposed by `/health` yet; they are being
added. Until then, leave a placeholder in the layout rather than designing it
in later.

Two axes are recorded in every model file, for reference:

- **Extractor** — `refined`, `redness`, `brightness`, `grabcut`, `cielab`
- **Representation** — `means`, `chroma`, `erythema`, `lab`, `a_only`

---

## Step 7 — Switch to the real server

The service is started with the model path in the environment and read by
`serve/app.py`:

```bash
ANEMIA_CHECKPOINT=runs/hb/best.pt ANEMIA_SEGMENTER=runs/segmenter \
  uvicorn serve.app:app --host 0.0.0.0 --port 8000
```

Model files are produced by the training and fitting workflow and are not
carried in the repository, so ask for the current one rather than expecting a
clone to contain it. When you switch over:

1. Point the base URL at the real host.
2. Poll `GET /health` and confirm it returns 200 rather than 503.
3. Re-test both branches — a good capture and a deliberately blurred one.
4. Re-test the three error codes: empty body, a file over 12 MB, a non-image.

Nothing else in the client changes. The mock and the real service return the
same field names and the same status codes by construction.

---

## Step 8 — Bulk view (optional)

If you build a clinic-side screen that scores many patients at once, the batch
entry point is `predict_folder.py` and its CSV columns are:

```
patient, hb_g_dl, anemic, threshold_g_dl, usable, eyes_used, age_years, sex, note
```

`eyes_used` is 0, 1 or 2 — both eyes are averaged when both are present.
`note` carries the per-eye rejection reasons when a patient was skipped.

---

## Checklist before calling it done

- [ ] Age and sex collected on every capture; `pregnant` shown only when relevant
- [ ] `usable: false` renders a retake prompt and no number
- [ ] `usable: true` renders value, verdict, threshold and disclaimer
- [ ] Verdict and threshold come from the response, not from client-side logic
- [ ] 400, 413 and 503 each have a distinct, non-technical message
- [ ] Images compressed under 12 MB before upload
- [ ] Retake preserves already-entered demographics
- [ ] Provenance placeholder present on the about screen

---

## Questions worth asking before you build

- Does a result need to be stored, or is the app stateless per capture?
- One eye or both? The batch path averages two; `/predict` scores one image.
- Offline capture with deferred upload, or online only?
- Who reads the result — a health worker or the patient? That changes the
  wording of every screen far more than any of the above.
