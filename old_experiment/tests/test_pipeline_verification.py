"""Comprehensive unit and smoke tests covering dataset, extraction, architectures, and inference."""

import os
import sys
import unittest
from pathlib import Path
import numpy as np
import torch

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extractor import ROIExtractor, load_unet
from predictor.data import (
    load_and_validate_manifest,
    create_patient_splits,
    verify_split_leakage,
    get_train_transforms,
    get_val_transforms,
    HbDataset,
    SampleRecord,
)
from predictor.models import build_model, ResNetRegressor, VGGRegressor, MobileNetRegressor
from predictor.training import (
    compute_metrics,
    get_loss_fn,
    save_checkpoint,
    load_checkpoint,
)


class TestPipelineVerification(unittest.TestCase):

    def test_01_manifest_validation(self):
        manifest_path = ROOT / "ROI" / "roi_manifest.csv"
        records, audit = load_and_validate_manifest(manifest_path, filter_usable=True)
        self.assertGreater(len(records), 200)
        self.assertGreater(audit["unique_patients"], 200)
        self.assertAlmostEqual(audit["hb_mean"], 12.6, delta=1.0)
        for r in records:
            self.assertTrue(r.image_path.is_file())
            self.assertTrue(3.0 <= r.hb <= 25.0)

    def test_02_patient_split_integrity_and_no_leakage(self):
        manifest_path = ROOT / "ROI" / "roi_manifest.csv"
        records, _ = load_and_validate_manifest(manifest_path, filter_usable=True)
        train_recs, val_recs, test_recs, report = create_patient_splits(
            records=records, val_fraction=0.15, test_fraction=0.15, seed=42
        )
        # Verify zero leakage
        verify_split_leakage(train_recs, val_recs, test_recs)

        train_patients = {r.patient_id for r in train_recs}
        val_patients = {r.patient_id for r in val_recs}
        test_patients = {r.patient_id for r in test_recs}

        self.assertEqual(len(train_patients & val_patients), 0)
        self.assertEqual(len(train_patients & test_patients), 0)
        self.assertEqual(len(val_patients & test_patients), 0)
        self.assertGreater(len(train_recs), len(val_recs))
        self.assertGreater(len(train_recs), len(test_recs))

    def test_03_transforms_and_dimensions(self):
        sample_img = np.full((300, 400, 3), 120, dtype=np.uint8)
        train_tf = get_train_transforms((224, 224))
        val_tf = get_val_transforms((224, 224))

        t_tensor = train_tf(sample_img)
        v_tensor = val_tf(sample_img)

        self.assertEqual(t_tensor.shape, (3, 224, 224))
        self.assertEqual(v_tensor.shape, (3, 224, 224))
        self.assertTrue(0.0 <= t_tensor.min() <= t_tensor.max() <= 1.0)

    def test_04_unet_loading_and_extraction(self):
        unet_path = ROOT / "Models" / "UNet_extractor" / "unet.pt"
        self.assertTrue(unet_path.is_file())
        extractor = ROIExtractor(checkpoint_path=unet_path)
        dummy_eye = np.zeros((512, 512, 3), dtype=np.uint8)
        res = extractor.extract(dummy_eye)
        self.assertEqual(res.crop.shape, (224, 224, 3))
        self.assertEqual(res.mask.shape, (512, 512))

    def test_05_all_three_architectures_forward_and_backward(self):
        architectures = [
            ("resnet", "resnet18"),
            ("vgg", "vgg16_bn"),
            ("mobilenet", "mobilenet_v3_large"),
        ]
        x = torch.randn(2, 3, 224, 224)
        target = torch.tensor([[0.5], [-0.5]], dtype=torch.float32)
        loss_fn = get_loss_fn("smooth_l1")

        for arch, variant in architectures:
            with self.subTest(arch=arch, variant=variant):
                model = build_model(
                    architecture=arch,
                    variant=variant,
                    pretrained=False,
                    target_mean=12.5,
                    target_std=2.0,
                )
                model.train()
                pred_z = model(x)
                self.assertEqual(pred_z.shape, (2, 1))

                loss = loss_fn(pred_z, target)
                loss.backward()

                # Test Hb prediction in g/dL
                model.eval()
                hb_pred = model.predict_hb(x)
                self.assertEqual(hb_pred.shape, (2, 1))

    def test_06_target_normalization_round_trip(self):
        model = build_model("resnet", pretrained=False, target_mean=12.6, target_std=2.3)
        z = torch.tensor([[-1.0], [0.0], [1.0]])
        denorm = model.denormalize_target(z)
        expected = torch.tensor([[12.6 - 2.3], [12.6], [12.6 + 2.3]])
        self.assertTrue(torch.allclose(denorm, expected, atol=1e-5))

    def test_07_checkpoint_save_and_load_round_trip(self):
        model = build_model("resnet", pretrained=False, target_mean=12.6, target_std=2.3)
        tmp_path = ROOT / "tests" / "scratch_best.pt"
        try:
            save_checkpoint(
                model=model,
                path=tmp_path,
                architecture="resnet",
                variant="resnet18",
                epoch=5,
                val_metrics={"mae_g_dl": 1.25},
                config={"epochs": 10},
            )
            self.assertTrue(tmp_path.is_file())
            loaded_model, bundle = load_checkpoint(tmp_path)
            self.assertEqual(bundle["architecture"], "resnet")
            self.assertAlmostEqual(loaded_model.target_mean.item(), 12.6, delta=1e-4)
            self.assertAlmostEqual(loaded_model.target_std.item(), 2.3, delta=1e-4)
        finally:
            if tmp_path.is_file():
                tmp_path.unlink()

    def test_08_evaluation_metrics(self):
        y_true = [10.0, 12.0, 14.0, 16.0]
        y_pred = [10.5, 12.0, 13.5, 15.5]
        m = compute_metrics(y_true, y_pred)
        # Errors: [0.5, 0.0, -0.5, -0.5], abs: [0.5, 0.0, 0.5, 0.5], mean = 0.375
        self.assertAlmostEqual(m["mae_g_dl"], 0.375, delta=1e-3)
        self.assertAlmostEqual(m["rmse_g_dl"], 0.433, delta=1e-3)
        self.assertGreater(m["r2"], 0.9)
        self.assertAlmostEqual(m["pearson_r"], 1.0, delta=0.05)


if __name__ == "__main__":
    unittest.main()
