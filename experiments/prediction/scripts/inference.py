from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT = (
    ROOT
    / "models"
    / "predictors"
    / "densenet121_seed_20261118_best.pt"
)
IMAGE_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def build_model() -> torch.nn.Module:
    model = models.densenet121(weights=None)
    model.classifier = torch.nn.Linear(model.classifier.in_features, 1)
    return model


def load_model(
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
) -> tuple[torch.nn.Module, dict]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"DenseNet checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Invalid checkpoint format: expected a mapping in {checkpoint_path}")
    required_keys = {"architecture", "model_state_dict", "seed", "target_mean_g_dl", "target_std_g_dl"}
    missing_keys = required_keys.difference(checkpoint)
    if missing_keys:
        raise RuntimeError(
            f"Invalid checkpoint {checkpoint_path}; missing keys: {sorted(missing_keys)}"
        )
    if checkpoint["architecture"] != "densenet121":
        raise RuntimeError(f"Unexpected checkpoint architecture: {checkpoint['architecture']}")
    if float(checkpoint["target_std_g_dl"]) <= 0:
        raise RuntimeError(f"Invalid target standard deviation in checkpoint: {checkpoint_path}")
    model = build_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to("cpu")
    model.eval()
    metadata = {
        "checkpoint": str(checkpoint_path),
        "architecture": checkpoint["architecture"],
        "seed": checkpoint["seed"],
        "target_mean_g_dl": float(checkpoint["target_mean_g_dl"]),
        "target_std_g_dl": float(checkpoint["target_std_g_dl"]),
        "preprocessing": "RGB resize 224x224, ImageNet normalization",
    }
    return model, metadata


def preprocess_image(image_path: Path) -> torch.Tensor:
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"ROI image not found: {image_path}")
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    with Image.open(image_path) as image:
        image.load()
        return transform(image.convert("RGB"))


def inverse_target(value: torch.Tensor, metadata: dict) -> torch.Tensor:
    return value * metadata["target_std_g_dl"] + metadata["target_mean_g_dl"]


def predict_image(
    image_path: Path,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
) -> dict[str, float | str]:
    model, metadata = load_model(checkpoint_path)
    image = preprocess_image(image_path)
    with torch.inference_mode():
        standardized = model(image.unsqueeze(0)).reshape(-1)[0]
        prediction = inverse_target(standardized, metadata)
    if not torch.isfinite(prediction):
        raise RuntimeError(f"Model returned a non-finite Hb prediction for {image_path}")
    return {
        "image": str(image_path),
        "checkpoint": metadata["checkpoint"],
        "predicted_hb_g_dl": float(prediction.item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run selected DenseNet Hb inference on one ROI image.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    print(json.dumps(predict_image(args.image, args.checkpoint), indent=2))


if __name__ == "__main__":
    main()
