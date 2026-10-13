from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hbpredictor.data import DataStatistics, build_transform
from hbpredictor.model import HbPredictor


def main():
    parser = argparse.ArgumentParser(description="Predict Hb from one pre-extracted conjunctival ROI.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--roi", required=True)
    parser.add_argument("--age", required=True, type=float)
    parser.add_argument("--sex", required=True, help="Category exactly as used in training, e.g. female")
    args = parser.parse_args()
    model_dir = Path(args.model_dir); stats = DataStatistics.load(model_dir / "data_statistics.json")
    sex = args.sex.strip().lower()
    if sex not in stats.sex_to_index: raise ValueError(f"Unknown sex {args.sex!r}; expected one of {list(stats.sex_to_index)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_dir / "best_model.pt", map_location=device, weights_only=True)
    model = HbPredictor(checkpoint["num_sexes"], pretrained=False).to(device); model.load_state_dict(checkpoint["model_state"]); model.eval()
    image = build_transform(False)(Image.open(args.roi).convert("RGB")).unsqueeze(0).to(device)
    age = torch.tensor([[(args.age - stats.age_mean) / stats.age_std]], dtype=torch.float32, device=device)
    sex_index = torch.tensor([stats.sex_to_index[sex]], dtype=torch.long, device=device)
    with torch.no_grad(): prediction = model(image, age, sex_index).item()
    print(f"Predicted Hb: {prediction:.2f} g/dL")


if __name__ == "__main__": main()
