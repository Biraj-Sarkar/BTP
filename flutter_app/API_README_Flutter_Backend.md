# Flutter ↔ FastAPI API Contract

This document is the hand-off specification for the backend. The Flutter app is being redesigned to make **one request per photo** and receive the haemoglobin screening result and all image previews in the same response.

> **Important:** The Flutter client in this project calls `POST /analyze`. The current backend `server.py` has separate `/anemia-detection` and `/extract` routes and does not yet implement `/analyze`. Please add the combined endpoint below, or tell the Flutter developer if you choose a different contract. The model loading and inference implementation remain server-side.

## 1. Server address used by Flutter

Default development URL in `App/anaemia_detector/lib/config/api_config.dart`:

```text
http://127.0.0.1:8000
```

Select the correct host for the device running Flutter:

- Flutter desktop / web on the same computer: `http://127.0.0.1:8000`
- Android Emulator: `http://10.0.2.2:8000`
- Physical phone: `http://<COMPUTER_LAN_IP>:8000` (phone and computer must be on the same Wi-Fi)
- iOS Simulator: usually `http://127.0.0.1:8000`

Run the app with an override when needed:

```bash
flutter run --dart-define=SERVER_BASE_URL=http://10.0.2.2:8000
```

Production URL retained for later; do not use during local development:

```dart
// https://btp-9obt.onrender.com
```

The FastAPI server must listen on `0.0.0.0` for a physical phone to reach it, for example:

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

For Android local HTTP development, ensure the Android app permits cleartext traffic in its debug configuration (or use HTTPS). Do not ship an insecure cleartext production endpoint.

## 2. Endpoint required by Flutter

| Item | Value |
|---|---|
| Method | `POST` |
| Path | `/analyze` |
| Content type | `multipart/form-data` |
| Authentication | None for local development (if added later, document the header) |
| Image size | Accept up to 10 MB; reject larger uploads clearly |
| Response | JSON |

### Multipart request fields

| Field | Type | Required? | Example | Meaning |
|---|---|---:|---|---|
| `image` | File | Yes | `eye_photo.jpg` | Original eye photograph. JPEG or PNG preferred. |
| `age_years` | Number | No | `24` | User age in years. Can be decimal for infants if supported. |
| `sex` | String | No | `female` | `male`, `female`, `other`; empty/omitted means unknown. |
| `pregnant` | Boolean string | No | `false` | `true` or `false`. Relevant only when sex is female. |

The client sends `pregnant=false` by default. Do not silently assume a missing age or sex. Use the project's agreed fallback threshold logic when demographic information is unavailable, and return the threshold actually used.

## 3. Successful response JSON

Return HTTP `200` when the request was processed. The recommended response shape is:

```json
{
  "status": "success",
  "usable": true,
  "message": "Image processed successfully.",
  "prediction": {
    "hb_g_dl": 11.8,
    "threshold_g_dl": 12.0,
    "anemic": true,
    "model": "resnet18"
  },
  "quality": {
    "passed": true,
    "mask_ratio": 0.18,
    "focus": 245.6,
    "reasons": []
  },
  "visuals": {
    "original": "<BASE64_ENCODED_JPEG_OR_PNG>",
    "mask": "<BASE64_ENCODED_PNG>",
    "masked": "<BASE64_ENCODED_JPEG_OR_PNG>",
    "roi": "<BASE64_ENCODED_JPEG_OR_PNG>"
  }
}
```

### Required response fields

| JSON path | Type | Required? | Notes |
|---|---|---:|---|
| `status` | String | Yes | `success` or `error`. |
| `usable` | Boolean | Yes | `true` only when segmentation and quality checks pass and prediction is valid. |
| `message` | String | Yes | Human-readable status or retake guidance. Empty string is allowed on success. |
| `prediction.hb_g_dl` | Number or null | Yes | Hb estimate in g/dL. Must be `null` when `usable=false`. |
| `prediction.threshold_g_dl` | Number or null | Recommended | Threshold used for screening; return the actual applied threshold. |
| `prediction.anemic` | Boolean or null | Yes | `true` means below the selected threshold; `false` means not below it. `null` if no valid prediction. |
| `prediction.model` | String | Recommended | Name/version of predictor used, e.g. `resnet18`. |
| `quality` | Object | Recommended | Quality information from the ROI pipeline. |
| `visuals.original` | Base64 string | Recommended | Original uploaded image. |
| `visuals.mask` | Base64 string | Recommended | Binary/segmentation mask image. |
| `visuals.masked` | Base64 string | Recommended | Original image with pixels outside the predicted mask removed. |
| `visuals.roi` | Base64 string | Recommended | Final crop actually passed to the Hb predictor. |

**Base64 rules:** Return raw Base64 only (no `data:image/...;base64,` prefix is necessary). JPEG is fine for photographic images; PNG is preferred for the mask so its edges remain crisp. The Flutter client can also decode a data-URI-prefixed string.

**Image semantics:** `mask` must be the mask itself, not a copy of the original. `masked` must show the original image restricted to the mask. `roi` must be the exact final crop used by the predictor (including the production colour handling and required dimensions).

## 4. Image rejected / retake response

If the upload is valid but the U-Net cannot find a usable conjunctiva, or the image fails the server's quality checks, return HTTP `200` with `usable=false`. Do not run the Hb predictor on an invalid crop and do not return a made-up Hb value.

Example:

```json
{
  "status": "success",
  "usable": false,
  "message": "We could not clearly see the lower eyelid. Please take another photo in soft, bright light and keep the camera still.",
  "prediction": {
    "hb_g_dl": null,
    "threshold_g_dl": null,
    "anemic": null,
    "model": "resnet18"
  },
  "quality": {
    "passed": false,
    "mask_ratio": 0.002,
    "focus": 32.1,
    "reasons": ["low_mask_area", "blurry"]
  },
  "visuals": {
    "original": "<BASE64_IMAGE>",
    "mask": "<BASE64_IMAGE>",
    "masked": "<BASE64_IMAGE>",
    "roi": null
  }
}
```

If a visual cannot be generated, use JSON `null` or omit that visual key. Do not put the text `"null"` in its place.

## 5. Error responses

Use FastAPI's normal error status codes and a JSON body with a useful `detail` string.

| HTTP status | When to use | Example `detail` |
|---:|---|---|
| `400` | Empty upload, unsupported/corrupt image | `The uploaded image is empty or invalid.` |
| `413` | File exceeds upload limit | `Image is larger than the 10 MB limit.` |
| `415` | Unsupported media type | `Please upload a JPG or PNG image.` |
| `422` | Invalid form values / invalid request | `age_years must be a positive number.` |
| `503` | Model weights unavailable or inference service unavailable | `The screening service is not ready. Please try again later.` |
| `500` | Unexpected server failure | `The server could not process this photo.` |

Example:

```json
{
  "detail": "The screening service is not ready. Please try again later."
}
```

Do not expose stack traces, local filesystem paths, model file paths, or secrets to the app.

## 6. Health check

The app does not currently block the user on a health-check call, but the server should retain:

```http
GET /health
```

Recommended response when ready:

```json
{
  "status": "ok",
  "models_loaded": true,
  "extractor": "unet",
  "model": "resnet18"
}
```

When models are not ready, return a non-ready status and `models_loaded: false`.

## 7. Processing order expected on the server

For one `/analyze` request, the backend should:

1. Validate and decode the uploaded image.
2. Keep the original image for the response visual.
3. Run the selected production conjunctiva extractor exactly once.
4. Apply the same preprocessing used during approved predictor training/inference.
5. Check segmentation and ROI quality.
6. If the ROI is rejected, return `usable=false`, retake message and any available debug visuals. Do not predict Hb.
7. If usable, pass the exact final ROI to the selected Hb regression model.
8. Apply the agreed demographic threshold logic and return the threshold used.
9. Return prediction, quality and all available visuals in one JSON response.

The model architecture, checkpoint selection, image normalization, colour handling, segmentation, and prediction are server responsibilities. The Flutter app only uploads the original photo and displays the returned values/images.

## 8. Existing routes in the current backend ZIP

The current `Backend/server.py` contains:

- `GET /`
- `GET /health`
- `POST /predict` and `POST /anemia-detection` — prediction only; currently require `age_years` and `sex`.
- `POST /extract` — extraction only; supports `extract_target` and `include_visuals`.

These existing routes are **not yet the combined `/analyze` contract** described above. Implement `/analyze` to avoid making Flutter upload the same photo twice and running the extractor twice. Keep existing routes if they are useful for testing.

## 9. Quick test with curl

After implementing `/analyze` and starting the server:

```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
  -F "image=@eye_photo.jpg" \
  -F "age_years=24" \
  -F "sex=female" \
  -F "pregnant=false"
```

Check the interactive API docs at `http://127.0.0.1:8000/docs`.

## 10. Backend hand-off checklist

- [ ] `POST /analyze` exists and accepts multipart `image`.
- [ ] Optional `age_years`, `sex`, and `pregnant` fields are accepted.
- [ ] One request runs extraction once and prediction once (only if ROI passes).
- [ ] Response always includes `status`, `usable`, and `message`.
- [ ] Successful prediction includes numeric `prediction.hb_g_dl` and boolean `prediction.anemic`.
- [ ] Rejected image returns `usable=false` and null Hb/anemia values.
- [ ] Actual applied screening threshold is returned when available.
- [ ] `visuals.original`, `visuals.mask`, `visuals.masked`, and `visuals.roi` are Base64 image strings when available.
- [ ] Mask, masked image and ROI have the meanings defined above.
- [ ] Invalid image and server/model errors return useful HTTP errors with a `detail` field.
- [ ] `/health` accurately reports whether models are loaded.
- [ ] Test with curl, then test from the actual Android emulator or phone.

---

**Privacy reminder:** Eye photographs and demographic information may be sensitive. Do not log uploaded image bytes or unnecessary patient details. Define retention and access controls before using real patient data.
