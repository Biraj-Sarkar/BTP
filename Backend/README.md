---
title: BTP Anaemia Backend
emoji: 🩸
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
---

# Standalone backend

This folder is independent of `algorithms/`. It downloads the U-Net and Hb
checkpoints from Hugging Face on first use, extracts a conjunctiva ROI, and
returns haemoglobin plus the sex-based screening result.

Set `HF_MODEL_REPO` when all files share one repository, or set
`HF_UNET_REPO`, `HF_ANUBHAV_REPO`, and `HF_HB_REPO` separately. The default
filenames are `unet.pt`, `best.pt`, `best_model.pt`, and `data_statistics.json`.
Private repositories use the `HF_TOKEN` environment variable.
The API serves the Anubhav model by default. Set `HB_MODEL=hbpredictor` only
if you want the larger ConvNeXt predictor instead.

```powershell
cd D:\BTP\BTP-Anemia-Model-Suite\Backend
D:\BTP\.venv\Scripts\python.exe -m uvicorn server:app --host 0.0.0.0 --port 7860
```
