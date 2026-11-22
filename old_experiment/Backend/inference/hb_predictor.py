"""Server-side Hemoglobin predictor wrapper."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import torch
import torchvision.transforms as T

SUITE_ROOT = Path(__file__).resolve().parents[2]
if str(SUITE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUITE_ROOT))

from predictor.training.checkpoint import load_checkpoint
from predictor.models.common import BaseHbRegressor


class BackendHbPredictor:
    """Wraps trained deep regression models for continuous Hb and anemia classification."""

    def __init__(self, checkpoint_path: Path | str, device: Optional[torch.device | str] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        try:
            self.model, self.bundle = load_checkpoint(checkpoint_path, device=self.device)
            self.arch_name = self.bundle.get("architecture", "predictor")
            self.variant_name = self.bundle.get("variant", "")
        except Exception:
            # Fallback loader for legacy Anubhav checkpoint if present
            raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            from torchvision.models import resnet18
            net = resnet18(weights=None)
            net.fc = torch.nn.Sequential(torch.nn.Dropout(0.3), torch.nn.Linear(net.fc.in_features, 1))
            from predictor.models.resnet import ResNetRegressor
            model = ResNetRegressor(
                variant="resnet18",
                pretrained=False,
                target_mean=float(raw.get("target_mean", 12.6)),
                target_std=float(raw.get("target_std", 2.3)),
            )
            # Remove buffer prefixes if any
            state = {k.replace("net.", "backbone."): v for k, v in raw["state_dict"].items() if not k.startswith("norm_")}
            model.load_state_dict(state, strict=False)
            model.eval()
            self.model = model.to(self.device)
            self.arch_name = "resnet"
            self.variant_name = "resnet18"
            self.bundle = raw

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
        ])

    @torch.inference_mode()
    def predict(self, crop_rgb: np.ndarray, age_years: float, sex: str) -> Dict[str, Any]:
        """Runs model inference on 224x224 crop and applies WHO clinical anemia thresholds."""
        tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)
        hb_val = float(self.model.predict_hb(tensor).item())

        is_male = (sex or "").strip().lower().startswith("m")
        threshold = 13.0 if is_male else 12.0
        anemic = hb_val < threshold

        return {
            "hb_g_dl": round(hb_val, 2),
            "anemic": bool(anemic),
            "threshold_g_dl": threshold,
            "model": f"{self.arch_name}_{self.variant_name}".strip("_"),
        }

