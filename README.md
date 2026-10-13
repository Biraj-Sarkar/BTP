# BTP Anemia Model Suite

This directory is a combined, reviewable copy of the anemia and haemoglobin
prediction work. The individual algorithm directories are kept separate so
their source files and training instructions remain easy to inspect.

## Contents

- `algorithms/Anubhav/` – conjunctival-image anemia/Hb pipeline, segmentation,
  preprocessing, regression, serving code, and tests.
- `algorithms/HBPredictor/` – ConvNeXt-Tiny multimodal Hb regressor using an
  extracted conjunctival ROI plus age and sex.
- `App/` – unchanged copy of the Flutter application from `project/App`.
- `datasets/` – supplied capture metadata and reference images.

## Dataset status

The supplied CSV contains patient metadata, Hb labels, and JSON image paths;
the supplied reference images are staged alongside it. See `datasets/README.md`
for the expected image and ROI layout for each model. No model training is
claimed by this cleanup step.

## Source mapping

The package contains the complete named `BTP-anubhav-ai-psychiatrist-research`
Anubhav pipeline, the HBPredictor package, and an unchanged copy of the app.
Generated experiments, bundled sample images, and training artifacts were
removed from `algorithms/Anubhav`; the shared reference data remains under
`datasets/`.
