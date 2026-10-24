# Final model and ROI UI status

Date: 2026-10-22

## Production model selection

The backend now uses exactly one production pipeline:

1. `Models/UNet_extractor/unet.pt`
2. `predictor/runs/resnet/best.pt`

Verified SHA-256 hashes:

- U-Net: `35b07d3c8bf7f36603d3b38f96ec95bfee8aac367089111046cc163d6983865c`
- ResNet checkpoint: `73a06d7582f5d78fe6436438176603d9b8f820975703c8157cfc5930ef12f065`

The predictor bundle reports architecture `resnet`, variant `resnet18`,
ImageNet normalization, and the existing target normalization. The backend no
longer falls back to alternate or Hugging Face model artifacts, so the
experimental Attention U-Net and matched predictor checkpoints cannot be
silently selected in production.

## Backend changes

- `Backend/model_loader.py` resolves only the two approved local production
  checkpoints.
- `Backend/server.py` validates uploaded image size/type, keeps `/predict` and
  `/anemia-detection` compatible, and adds `POST /extract`.
- `Backend/inference/__init__.py` exposes extraction through the already-loaded
  production extractor; no duplicate model is loaded. It also includes the
  explicit `usable: true` field expected by the Flutter result model after a
  successful prediction.

`POST /extract` accepts multipart `image`, optional `extract_target` (currently
only `conjunctiva`), and `include_visuals`. With visuals enabled, the response
contains `visuals.original`, `visuals.mask`, `visuals.masked`, and
`visuals.roi` as base64-encoded JPEG/PNG data, plus `backend`, `usable`,
`bbox`, and `quality`. Empty masks return a 422 error rather than a
success-shaped crop.

## Flutter Extract ROI status

Before this work, the dropdown and image picker called `/extract`, but the
backend route was absent and the result screen displayed only metadata.
The flow now:

- uses the existing gallery/camera image selection;
- requests `include_visuals=true`;
- shows loading feedback and existing error SnackBars;
- displays the original input, actual segmentation mask, masked image, and
  extracted ROI;
- removes the unsupported palpebral-conjunctiva option.

## Validation

- Production checkpoint hashes verified.
- Python backend modules compile successfully.
- Backend model loading and one valid-image extraction were exercised after
  the changes. The valid response returned original `2448x3264`, mask/masked
  `512x512`, and ROI `224x224` images; invalid input returned 400 and an empty
  mask returned 422.
- Existing extractor regression tests: `4 passed`.
- Flutter `flutter analyze`/device UI execution could not be run in this
  environment because the Flutter SDK is unavailable. The Dart changes still
  require analyzer/device verification before release.

No model weights were changed, no model was retrained, no experimental model
was loaded, and no deployment or commit was performed.
