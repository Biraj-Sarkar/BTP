---
title: BTP Anaemia Backend
emoji: 🩸
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
---

# Standalone backend

This folder is independent of `algorithms/`. It loads the approved production
U-Net and ResNet18 checkpoints from the repository, extracts a conjunctiva ROI,
and returns haemoglobin plus the sex-based screening result. Experimental
Attention U-Net and matched predictor checkpoints are not selected by the
backend.

The `/extract` endpoint can optionally return base64-encoded original, mask,
masked, and 224x224 ROI images for the Flutter visualization screen.

```powershell
cd D:\BTP\BTP-Anemia-Model-Suite\Backend
D:\BTP\.venv\Scripts\python.exe -m uvicorn server:app --host 0.0.0.0 --port 7860
```
