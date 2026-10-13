from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hbpredictor.data import HbDataset, make_statistics, patient_split, read_manifest
from hbpredictor.metrics import bland_altman_plot, regression_metrics
from hbpredictor.model import HbPredictor


def parse_args():
    parser = argparse.ArgumentParser(description="Train a multimodal conjunctival ROI Hb regression model.")
    parser.add_argument("--manifest", required=True, help="CSV containing patient_id,image_id,roi_224x224,age,sex,hb")
    parser.add_argument("--output-dir", default="runs/hb_predictor")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--freeze-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true", help="Do not initialize ConvNeXt with ImageNet weights")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device):
    model.eval(); actual, predicted = [], []
    with torch.no_grad():
        for batch in loader:
            prediction = model(batch["image"].to(device), batch["age"].to(device), batch["sex"].to(device))
            predicted.extend(prediction.cpu().tolist()); actual.extend(batch["hb"].tolist())
    return actual, predicted, regression_metrics(actual, predicted)


def main():
    args = parse_args(); seed_everything(args.seed)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    frame = read_manifest(args.manifest)
    train_frame, val_frame, test_frame = patient_split(frame, args.seed)
    for name, split in (("train", train_frame), ("validation", val_frame), ("test", test_frame)):
        split.to_csv(output / f"{name}_split.csv", index=False)
    stats = make_statistics(train_frame); stats.save(output / "data_statistics.json")
    loaders = {
        "train": DataLoader(HbDataset(train_frame, stats, train=True), batch_size=args.batch_size, shuffle=True, num_workers=args.workers),
        "validation": DataLoader(HbDataset(val_frame, stats), batch_size=args.batch_size, num_workers=args.workers),
        "test": DataLoader(HbDataset(test_frame, stats), batch_size=args.batch_size, num_workers=args.workers),
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HbPredictor(len(stats.sex_to_index), pretrained=not args.no_pretrained).to(device)
    model.freeze_backbone(args.freeze_epochs > 0)
    optimizer = AdamW(filter(lambda parameter: parameter.requires_grad, model.parameters()), lr=args.learning_rate, weight_decay=1e-4)
    criterion = nn.HuberLoss(delta=1.0)
    best_mae, history = float("inf"), []
    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_epochs + 1:
            model.freeze_backbone(False)
            optimizer = AdamW(model.parameters(), lr=args.learning_rate / 10, weight_decay=1e-4)
        model.train(); total_loss = 0.0
        for batch in tqdm(loaders["train"], desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            optimizer.zero_grad()
            prediction = model(batch["image"].to(device), batch["age"].to(device), batch["sex"].to(device))
            loss = criterion(prediction, batch["hb"].to(device)); loss.backward(); optimizer.step()
            total_loss += loss.item() * len(prediction)
        _, _, validation = evaluate(model, loaders["validation"], device)
        record = {"epoch": epoch, "train_huber_loss": total_loss / len(train_frame), **{f"validation_{key}": value for key, value in validation.items()}}
        history.append(record); print(json.dumps(record))
        if validation["mae"] < best_mae:
            best_mae = validation["mae"]
            torch.save({"model_state": model.state_dict(), "num_sexes": len(stats.sex_to_index), "architecture": "convnext_tiny_multimodal"}, output / "best_model.pt")
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    checkpoint = torch.load(output / "best_model.pt", map_location=device, weights_only=True)
    model = HbPredictor(checkpoint["num_sexes"], pretrained=False).to(device); model.load_state_dict(checkpoint["model_state"])
    actual, predicted, metrics = evaluate(model, loaders["test"], device)
    metrics.update(bland_altman_plot(actual, predicted, output / "bland_altman_test.png"))
    (output / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    test_frame[["patient_id", "image_id", "hb"]].assign(predicted_hb=predicted).to_csv(output / "test_predictions.csv", index=False)
    print("Test metrics:", json.dumps(metrics, indent=2)); print(f"Saved run to {output.resolve()}")


if __name__ == "__main__": main()
