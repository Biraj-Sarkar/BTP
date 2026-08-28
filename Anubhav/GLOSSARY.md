# Glossary and Abbreviations

Every abbreviation used across this project, defined once. Documents may repeat
a definition at first use, but this is the reference.

## Clinical

| Abbreviation | Expansion | Meaning here |
|---|---|---|
| **Hb** | Haemoglobin | The protein carrying oxygen in red blood cells; what makes blood red, and the quantity being estimated. Measured in grams per decilitre. |
| **g/dL** | grams per decilitre | Unit of haemoglobin concentration. Normal adult ranges are roughly 12–16. |
| **WHO** | World Health Organization | Source of the anaemia thresholds used (11.0 / 11.5 / 12.0 / 13.0 g/dL depending on age, sex and pregnancy). |
| **CBC** | Complete Blood Count | The full laboratory panel accompanying the local cohort. |
| **HCT** | Haematocrit | Percentage of blood volume occupied by red cells. |
| **RBC** | Red Blood Cell count | Cells per unit volume. |
| **MCV** | Mean Corpuscular Volume | Average red cell size; distinguishes types of anaemia. |
| **MCH** | Mean Corpuscular Haemoglobin | Average haemoglobin mass per red cell. |
| **MCHC** | Mean Corpuscular Haemoglobin Concentration | Haemoglobin concentration within red cells. |
| **RDW** | Red cell Distribution Width | Variation in red cell size; with MCV, helps identify iron-deficiency anaemia. |
| **MPV** | Mean Platelet Volume | Average platelet size. |
| **TLC** | Total Leukocyte Count | White blood cell count. |
| **BPL / APL** | Below / Above Poverty Line | Socio-economic marker recorded in the cohort metadata. |

## Colour and imaging

| Abbreviation | Expansion | Meaning here |
|---|---|---|
| **RGB** | Red, Green, Blue | How images are stored; three intensities per pixel, 0–255. |
| **CIELAB** (or **LAB**) | Commission Internationale de l'Éclairage L\*a\*b\* | Colour space separating lightness from colour. |
| **L\*** | Lightness | The brightness axis of CIELAB, 0 (black) to 100 (white). |
| **a\*** | Green–red axis | Negative is green, positive is red. The main haemoglobin correlate. |
| **b\*** | Blue–yellow axis | Negative is blue, positive is yellow. Used to suppress skin, which is red *and* yellow. |
| **HSV** | Hue, Saturation, Value | An alternative colour space, not used here (hue wraps at 360°, complicating averaging). |
| **sRGB** | standard Red Green Blue | The default colour profile; the cohort images use it. |
| **P3** | Display P3 | A wider-gamut colour profile; iPhone captures use it, requiring conversion. |
| **CLAHE** | Contrast Limited Adaptive Histogram Equalisation | Brightness normalisation applied in small tiles with a cap on contrast stretching. |
| **ISP** | Image Signal Processor | The phone's built-in image processing, which applies its own colour correction. |
| **HEIC** | High Efficiency Image Container | Apple's default photo format; converted to JPEG for processing. |
| **JPEG / PNG** | Joint Photographic Experts Group / Portable Network Graphics | Lossy and lossless image formats respectively. |
| **ROI** | Region Of Interest | The area being extracted — here, the conjunctiva. |
| **QC** | Quality Control | The gate rejecting unusable captures (blur, mask area, clipping). |

## Machine learning

| Abbreviation | Expansion | Meaning here |
|---|---|---|
| **CNN** | Convolutional Neural Network | The network family used for image tasks. |
| **ResNet** | Residual Network | The CNN architecture used for regression; "residual" refers to its skip connections. |
| **ViT** | Vision Transformer | An alternative architecture, rejected as too data-hungry for ~900 images. |
| **BN** | Batch Normalisation | A layer normalising activations; its running statistics need explicit freezing. |
| **LR** | Linear Regression *or* Learning Rate | Context-dependent. In this project, usually **Linear Regression** (the simple baseline models). Where the optimiser step size is meant, it is written "learning rate" in full. |
| **OLS** | Ordinary Least Squares | Linear regression with no penalty term. |
| **Ridge** | — | Linear regression with an L2 penalty (`α·Σβ²`); shrinks coefficients, keeps all features. |
| **Lasso** | Least Absolute Shrinkage and Selection Operator | Linear regression with an L1 penalty (`α·Σ\|β\|`); drives some coefficients to exactly zero. |
| **L1 / L2** | — | Penalty types: L1 sums absolute coefficient values, L2 sums their squares. |
| **CV** | Cross-Validation | Evaluating on held-out data by rotating which part is held out. |
| **LOO / LOOCV** | Leave-One-Out Cross-Validation | CV where each sample is the test set exactly once. |
| **SGD** | Stochastic Gradient Descent | A basic optimiser; AdamW is used instead. |
| **AdamW** | Adam with decoupled Weight decay | The optimiser used; adapts step size per parameter. |
| **GMM** | Gaussian Mixture Model | The colour model grabCut fits to foreground and background. |
| **API** | Application Programming Interface | The server interface the app would call. |

## Metrics

| Abbreviation | Expansion | Meaning here |
|---|---|---|
| **MSE** | Mean Squared Error | Average of squared errors; units (g/dL)². |
| **RMSE** | Root Mean Squared Error | Square root of MSE; back in g/dL. |
| **MAE** | Mean Absolute Error | Average absolute error; the most readable error figure. |
| **R²** | Coefficient of determination | Fraction of variance explained. 0 = no better than the mean; negative = worse. |
| **LoA** | Limits of Agreement | Bland–Altman interval containing ~95% of errors (`bias ± 1.96·SD`). |
| **TP / TN / FP / FN** | True/False Positive/Negative | Confusion matrix entries. Positive class = anaemic. |
| **PPV** | Positive Predictive Value | Same as precision: `TP/(TP+FP)`. |
| **NPV** | Negative Predictive Value | `TN/(TN+FN)`; how often "normal" is correct. |
| **F1** | F1 score | Harmonic mean of precision and recall. |
| **AUC** | Area Under the Curve | Threshold-independent classifier summary; quoted from published work for comparison. |
| **IoU** | Intersection over Union | Segmentation overlap: `\|A∩B\| / \|A∪B\|`. |
| **Dice** | Dice coefficient | Segmentation overlap: `2\|A∩B\| / (\|A\|+\|B\|)`. |
| **SD** | Standard Deviation | Spread of a distribution. |
| **ρ (rho)** | Spearman's rank correlation | Correlation of ranks rather than values; robust to non-linearity. |
| **r** | Pearson correlation | Linear correlation between two variables, −1 to +1. |

## Datasets and project terms

| Term | Meaning |
|---|---|
| **Eyes-defy-anemia** | Public dataset, 218 images, India + Italy, adults. The only source with pixel-level conjunctiva masks. |
| **CP-AnemiC** | Public dataset, 710 conjunctival images, children 6–59 months, Ghana. |
| **Local cohort** | 26 patients photographed here, both eyes, with laboratory Hb and full blood count. |
| **Phantom** | A synthetic eye image generated for testing; flat-coloured, with redness a deterministic function of Hb. |
| **Palpebral conjunctiva** | The moist inner surface of the lower eyelid — the tissue being photographed. |
| **Sclera** | The white of the eye. A confounder: pale and colourless, so brightness-based methods find it instead. |
| **Gray-world** | The white-balance assumption that a scene averages to grey. |
| **Baseline** | Predicting the training mean (regression) or the majority class (classification). Every result is reported against it. |
| **Out-of-fold** | A prediction made on data the model did not train on. |
