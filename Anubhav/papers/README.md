# Papers

Reading for the anemia-detection BTP. Only openly licensed PDFs are stored
here; anything behind a publisher block is listed with a link to fetch yourself.

## Read first — directly about this project

| File | Why it matters |
|---|---|
| `conjunctiva-anemia/2024_PLOSONE_smartphone_conjunctiva_realtime_anemia.pdf` | Closest published work to our app: smartphone photo of the conjunctiva → anemia prediction. Read the methods and the Bland–Altman results; this is the comparison our numbers will be judged against. |
| `conjunctiva-anemia/2025_HIR_deep_learning_anemia_conjunctiva.pdf` | CNN on conjunctiva images, with the data-augmentation strategy for small cohorts — our exact constraint at 26 patients. |
| `conjunctiva-anemia/2025_CPAnemiC_quantization_arxiv.pdf` | Uses the CP-AnemiC dataset we plan to train on, and covers shrinking the model for deployment. Short (6 pages). |

## Method background

| File | Why it matters |
|---|---|
| `methods/2015_ResNet_deep_residual_learning.pdf` | The regressor's backbone. Read §3 (residual connections) — skip the ImageNet leaderboard tables. |
| `methods/2015_UNet_biomedical_segmentation.pdf` | The standard segmentation architecture in medical imaging; the mental model for what a learned segmenter does. Short and readable. |
| `methods/2021_Mask2Former.pdf` | The segmentation model our pipeline fine-tunes. Read the intro and the task formulation; the architecture internals are optional. |
| `methods/2021_deep_imbalanced_regression.pdf` | Why rare Hb values need up-weighting, and the principled version of the inverse-frequency weighting the pipeline uses. |

## Not stored — fetch manually if needed

| Paper | Where |
|---|---|
| Semantic Segmentation of Conjunctiva Region for Non-Invasive Anemia Detection (Electronics 2020) | https://www.mdpi.com/2079-9292/9/8/1309 — open access, but the publisher blocks scripted downloads. Click "Download PDF". |
| Eyes-defy-anemia dataset paper (Artificial Intelligence in Medicine, 2023) | https://www.sciencedirect.com/science/article/pii/S0933365722002299 — cite when using the dataset; may need institute access. |
| CP-AnemiC dataset paper | https://data.mendeley.com/datasets/m53vz6b7fx/1 — dataset landing page with the citation. |

## Other topics

`psychiatry-llm/` — earlier reading on large language models in psychiatry,
unrelated to the anemia pipeline.
