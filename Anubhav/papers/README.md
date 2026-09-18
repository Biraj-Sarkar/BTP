# Papers

Reading for the anemia-detection BTP. Only openly licensed PDFs are stored here;
anything behind a publisher block is listed with a link to fetch yourself.

Files in `conjunctiva-anemia/` are **numbered in the order they should be read**.
Each row says what to take from the paper *for this project*, not just what it is.

## The path — conjunctiva and anaemia

| # | Paper | What to take from it |
|---|---|---|
| 01 | **Collings 2016**, PLOS ONE — non-invasive detection of anaemia using digital photographs of the conjunctiva | The simplest version of our method: erythema index from a conjunctival photo, consumer camera vs smartphone, **with a colour calibration card in every frame**, plus clinician pallor ratings for comparison. Read it beside `experiments/18_white_balance/` — the card is the principled answer to the camera problem we measured. |
| 02 | **Suner 2021**, PLOS ONE — prediction of anemia and estimation of haemoglobin using a smartphone camera | The classic pipeline and the closest published relative of our linear model: pick a region of interest, extract colour features, stepwise regression, validate in a second phase. The fairest benchmark for our numbers. |
| 03 | **Zhao 2024**, PLOS ONE — prediction of anemia in real time using a smartphone camera | The same group's deployed app, 435 patients, RAW capture. Note the **limits of agreement of ±4.8 g/dL** and that accuracy peaks at the 7 and 9 g/dL transfusion thresholds — the published version of the screening categories in `experiments/17_screening_categories/`. |
| 04 | **Mannino 2018**, Nature Communications — smartphone app for anaemia from patient-sourced photos | Fingernail beds rather than conjunctiva, but the key result is ours: **personalised calibration** sharply improves accuracy. Published support for the per-site recalibration that `experiments/14_pooled_regression/` forces on us. |
| 05 | **Kasiviswanathan 2020**, Electronics — semantic segmentation of the conjunctiva region | The paper behind **Eyes-Defy-Anemia**, the dataset our extractor trains on and our 217-patient evaluation uses. Read the annotation protocol, given `scripts/prepare_eyes_defy.py` had to reject ~45% of its masks as bounding boxes. |
| 06 | **Dhalla 2023**, Procedia CS — semantic segmentation of palpebral conjunctiva with deep architectures | Segmentation networks compared for exactly our task. Read against `experiments/13_learned_extractor/`, and check whether they measure the **downstream** effect on haemoglobin or only Dice — that is where our comparison is stronger. |
| 07 | **Kato 2024**, Br J Haematol — machine/deep learning-assisted haemoglobin prediction | End-to-end learning in a clinical journal, mostly paediatric. Read last, and pass to the partner building the network: it is the closest precedent for that work. |
| 08 | **Healthcare Informatics Research 2025** — anaemia detection from conjunctiva images | Machine and deep learning on conjunctiva images with the data-augmentation strategy for small cohorts. |
| 09 | **2025**, arXiv — post-training quantization for CNN conjunctival pallor detection | Uses CP-AnemiC, and covers shrinking a model for deployment. Short. |

## Method background — `methods/`

Read these when the main path needs them, not front to back.

| File | When |
|---|---|
| `2015_UNet_biomedical_segmentation.pdf` | Before paper 05 or 06. The architecture our extractor actually uses; short and readable. |
| `2015_ResNet_deep_residual_learning.pdf` | Before paper 07. §3 (residual connections) is the part that matters. |
| `2021_Mask2Former.pdf` | Optional. An alternative segmentation family; the pipeline keeps a backend for it but the U-Net is what ships. |
| `2021_deep_imbalanced_regression.pdf` | When the network training starts: why rare haemoglobin values need up-weighting. |

`psychiatry-llm/` is unrelated to this project.

## Licences

| Paper | Licence |
|---|---|
| 01, 02, 03 (PLOS ONE), 04 (Nature Communications), 05 (MDPI Electronics) | CC BY — free to store and redistribute |
| 06 (Procedia CS) | CC BY-NC-ND — open access, no derivatives |
| 07 (Br J Haematol, Wiley) | **Unverified.** If it is not open access, remove the PDF from this repository and keep the DOI instead: 10.1111/bjh.19621 |

## Not stored — fetch manually if needed

| Paper | Where |
|---|---|
| Eyes-defy-anemia dataset paper (Artificial Intelligence in Medicine, 2023) | https://www.sciencedirect.com/science/article/pii/S0933365722002299 — cite when using the dataset; may need institute access |
| CP-AnemiC dataset paper | https://data.mendeley.com/datasets/m53vz6b7fx/1 |
| Eye Conjunctiva Segmentation Dataset (Mendeley, used to train the extractor) | https://data.mendeley.com/datasets/yxwjgcndg2/1 — see `DATASETS.md` |

*(MDPI's site blocks scripted downloads from `mdpi.com`, but the PDF is
retrievable from their CDN at `mdpi-res.com` — that is how 05 was fetched.)*
