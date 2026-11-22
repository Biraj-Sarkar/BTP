"""Unified Trainer for Hemoglobin Predictor architectures."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..models.common import BaseHbRegressor
from .losses import get_loss_fn
from .metrics import compute_metrics
from .checkpoint import save_checkpoint, load_checkpoint


class Trainer:
    """Standardized training, validation, checkpointing, and evaluation engine."""

    def __init__(
        self,
        model: BaseHbRegressor,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: Dict[str, Any],
        output_dir: Path | str,
        device: Optional[torch.device | str] = None,
        audit_summary: Optional[Dict[str, Any]] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.audit_summary = audit_summary or {}

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Training hyperparameters
        self.epochs = int(config.get("epochs", 30))
        self.lr = float(config.get("learning_rate", 1e-4))
        self.weight_decay = float(config.get("weight_decay", 1e-4))
        self.patience = int(config.get("patience", 10))
        self.severe_hb_weight = float(config.get("severe_hb_weight", 1.0))
        self.loss_type = str(config.get("loss_type", "smooth_l1"))

        self.loss_fn = get_loss_fn(
            loss_name=self.loss_type,
            severe_hb_weight=self.severe_hb_weight,
        )

        # Optimizer: only optimize parameters that require gradients
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)

        # LR scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3
        )

        self.best_val_mae = float("inf")
        self.best_epoch = 0
        self.history: List[Dict[str, Any]] = []

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        count = 0

        for batch in self.train_loader:
            images = batch["image"].to(self.device)
            targets = batch["target"].to(self.device)
            raw_hb = batch["hb_raw"].clone().detach().float().to(self.device) if isinstance(batch["hb_raw"], torch.Tensor) else torch.tensor(batch["hb_raw"], dtype=torch.float32, device=self.device)

            self.optimizer.zero_grad()
            preds_z = self.model(images)
            loss = self.loss_fn(preds_z, targets, raw_hb=raw_hb)

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            count += batch_size

        return total_loss / max(count, 1)

    @torch.inference_mode()
    def evaluate(self, loader: DataLoader) -> Dict[str, Any]:
        self.model.eval()
        y_true: List[float] = []
        y_pred: List[float] = []
        patients: List[str] = []
        images_ids: List[str] = []

        total_loss = 0.0
        count = 0

        for batch in loader:
            images = batch["image"].to(self.device)
            targets = batch["target"].to(self.device)
            raw_hb = batch["hb_raw"].clone().detach().float().to(self.device) if isinstance(batch["hb_raw"], torch.Tensor) else torch.tensor(batch["hb_raw"], dtype=torch.float32, device=self.device)

            preds_z = self.model(images)
            loss = self.loss_fn(preds_z, targets, raw_hb=raw_hb)

            # Convert to actual g/dL
            preds_hb = self.model.denormalize_target(preds_z).cpu().numpy().reshape(-1)
            actual_hb = np.asarray(batch["hb_raw"], dtype=np.float64).reshape(-1)

            y_pred.extend(preds_hb.tolist())
            y_true.extend(actual_hb.tolist())
            patients.extend(batch["patient_id"])
            images_ids.extend(batch["image_id"])

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            count += batch_size

        metrics = compute_metrics(y_true, y_pred)
        metrics["loss"] = total_loss / max(count, 1)
        metrics["predictions"] = [
            {"patient_id": p, "image_id": img, "true_hb": round(t, 2), "pred_hb": round(pr, 2), "error": round(pr - t, 2)}
            for p, img, t, pr in zip(patients, images_ids, y_true, y_pred)
        ]
        return metrics

    def run(self) -> Dict[str, Any]:
        """Execute full training loop with early stopping, best model restoration, and test evaluation."""
        start_time = time.time()
        best_ckpt_path = self.output_dir / "best.pt"
        arch = self.config.get("architecture", "resnet")
        variant = self.config.get("variant", "resnet18")

        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_results = self.evaluate(self.val_loader)
            val_mae = val_results["mae_g_dl"]
            val_loss = val_results["loss"]

            self.scheduler.step(val_mae)
            lr_curr = self.optimizer.param_groups[0]["lr"]

            epoch_record = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "val_mae": val_mae,
                "val_rmse": val_results["rmse_g_dl"],
                "val_r2": val_results["r2"],
                "val_pearson_r": val_results["pearson_r"],
                "lr": lr_curr,
            }
            self.history.append(epoch_record)

            is_best = val_mae < self.best_val_mae
            if is_best:
                self.best_val_mae = val_mae
                self.best_epoch = epoch
                patience_counter = 0
                save_checkpoint(
                    model=self.model,
                    path=best_ckpt_path,
                    architecture=arch,
                    variant=variant,
                    epoch=epoch,
                    val_metrics=val_results,
                    config=self.config,
                    audit_summary=self.audit_summary,
                )
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print(f"Early stopping triggered at epoch {epoch}. Best epoch was {self.best_epoch}.")
                break

        training_duration = time.time() - start_time

        # Restore best checkpoint
        if best_ckpt_path.is_file():
            best_model, _ = load_checkpoint(best_ckpt_path, device=self.device)
            self.model = best_model

        # Evaluate on held-out test patients
        test_results = self.evaluate(self.test_loader)
        test_preds = test_results.pop("predictions", [])

        # Measure inference latency on test set
        t0 = time.time()
        with torch.inference_mode():
            for batch in self.test_loader:
                _ = self.model.predict_hb(batch["image"].to(self.device))
        test_latency_ms = ((time.time() - t0) / max(len(test_preds), 1)) * 1000.0

        # Save test predictions CSV
        preds_file = self.output_dir / "test_predictions.csv"
        if test_preds:
            with preds_file.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(test_preds[0].keys()))
                writer.writeheader()
                writer.writerows(test_preds)

        # Save metrics and training history
        summary = {
            "architecture": arch,
            "variant": variant,
            "best_epoch": self.best_epoch,
            "training_duration_seconds": round(training_duration, 1),
            "inference_latency_ms_per_image": round(test_latency_ms, 2),
            "parameters": self.model.count_parameters(),
            "best_val_mae_g_dl": self.best_val_mae,
            "test_metrics": test_results,
            "config": self.config,
        }

        (self.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (self.output_dir / "training_history.json").write_text(json.dumps(self.history, indent=2), encoding="utf-8")

        print(f"[{arch.upper()}] Training complete. Best Epoch: {self.best_epoch}, Val MAE: {self.best_val_mae:.3f} g/dL, Test MAE: {test_results['mae_g_dl']:.3f} g/dL, Test R2: {test_results['r2']:.4f}")
        return summary

