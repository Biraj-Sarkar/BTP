"""Pipeline tests that run without the real datasets."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anemia.config import PreprocessConfig, QualityConfig  # noqa: E402
from anemia.data import Sample, anemia_threshold, is_anemic  # noqa: E402
from anemia.imaging import gray_world_white_balance, pad_to_square  # noqa: E402
from anemia.metrics import (  # noqa: E402
    dice_score,
    inverse_frequency_weights,
    regression_metrics,
    screening_metrics,
)
from anemia.preprocess import assess_quality, prepare  # noqa: E402
from anemia.segment import (  # noqa: E402
    heuristic_conjunctiva_mask,
    heuristic_segmenter,
    legacy_bright_neutral_mask,
    refined_conjunctiva_mask,
)
from anemia.splits import kfold  # noqa: E402
from synthetic import make_eye  # noqa: E402


class TestSegmentation:
    def test_fixed_heuristic_beats_legacy_on_conjunctiva(self):
        """The core segmentation claim, measured rather than asserted.

        A brightness-based mask keeps bright, low-chroma pixels, which is the
        sclera. The signal for haemoglobin is in the red conjunctiva.
        """

        fixed_scores, legacy_scores = [], []
        for hb in (8.0, 11.0, 14.0):
            image, truth = make_eye(hb=hb, seed=int(hb))
            balanced = gray_world_white_balance(image)
            fixed_scores.append(dice_score(heuristic_conjunctiva_mask(balanced), truth))
            legacy_scores.append(dice_score(legacy_bright_neutral_mask(balanced), truth))

        assert np.mean(fixed_scores) > np.mean(legacy_scores)
        assert np.mean(fixed_scores) > 0.5, f"fixed heuristic Dice too low: {fixed_scores}"

    def test_legacy_heuristic_actually_selects_sclera(self):
        """Confirms the failure mode is what we diagnosed, not something else."""

        image, conjunctiva = make_eye(hb=13.0, seed=3)
        legacy = legacy_bright_neutral_mask(gray_world_white_balance(image))
        assert dice_score(legacy, conjunctiva) < 0.2

    def test_refined_mask_works_on_phantoms_too(self):
        """The real-image extractor must not regress on the controlled case."""

        scores = []
        for hb in (8.0, 11.0, 14.0):
            image, truth = make_eye(hb=hb, seed=int(hb))
            balanced = gray_world_white_balance(image)
            scores.append(dice_score(refined_conjunctiva_mask(balanced), truth))
        assert np.mean(scores) > 0.6, f"refined extractor Dice on phantoms: {scores}"

    def test_segmenter_returns_mask_and_backend_name(self):
        image, _ = make_eye(seed=1)
        mask, backend = heuristic_segmenter(image)
        assert mask.shape == image.shape[:2]
        assert set(np.unique(mask)).issubset({0, 255})
        assert backend in {"refined", "heuristic", "grabcut"}


class TestPreprocess:
    def test_crop_is_square_and_undistorted(self):
        image, _ = make_eye(seed=2)
        prepared = prepare(image, heuristic_segmenter, PreprocessConfig(), QualityConfig())
        assert prepared.crop.shape == (224, 224, 3)

    def test_pad_to_square_preserves_aspect(self):
        wide = np.full((40, 120, 3), 200, dtype=np.uint8)
        padded = pad_to_square(wide)
        assert padded.shape[0] == padded.shape[1] == 120
        # Original content survives intact in the middle band.
        assert np.array_equal(padded[40:80, :, :], wide)

    def test_quality_gate_rejects_blur(self):
        sharp, _ = make_eye(seed=4)
        blurred, _ = make_eye(seed=4, blur=12.0)
        config = QualityConfig()
        sharp_prep = prepare(sharp, heuristic_segmenter, PreprocessConfig(), config)
        blurred_prep = prepare(blurred, heuristic_segmenter, PreprocessConfig(), config)
        assert sharp_prep.quality.focus > blurred_prep.quality.focus
        assert not blurred_prep.quality.passed


class TestQualityGate:
    """Every statistic QC computes must also be able to reject a capture.

    Blur and mask area were enforced from the start; clipping and illuminant
    colour were measured, reported, and then ignored, so a blown-out or
    strongly tinted photo produced a confident number. These tests exist so
    that cannot come back.
    """

    def test_blown_out_capture_is_rejected(self):
        image, _ = make_eye(seed=5)
        blown = np.full_like(image, 255)
        blown[:20, :20] = image[:20, :20]  # keep a sliver so segmentation runs
        mask = np.zeros(image.shape[:2], np.uint8)
        mask[200:300, 150:350] = 255

        report = assess_quality(blown, mask, QualityConfig())
        assert not report.passed
        assert any("clipped" in reason for reason in report.reasons)

    def test_strongly_tinted_capture_is_rejected(self):
        """Under a heavy cast a channel is empty, so the redness was never captured."""

        image, truth = make_eye(seed=6)
        tinted = image.copy()
        tinted[:, :, 0] = 1  # red channel effectively dead, as under blue light

        report = assess_quality(image, truth, QualityConfig(), raw_rgb=tinted)
        assert not report.passed
        assert any("coloured lighting" in reason for reason in report.reasons)
        assert report.cast_ratio > 3.0

    def test_normal_capture_passes_both_new_gates(self):
        image, truth = make_eye(seed=7)
        report = assess_quality(image, truth, QualityConfig(), raw_rgb=image)
        assert report.passed, report.reasons
        assert report.cast_ratio < 3.0

    def test_cast_check_is_skipped_when_prebalance_image_is_absent(self):
        """Skipped, not silently passed on a meaningless value."""

        image, truth = make_eye(seed=8)
        report = assess_quality(image, truth, QualityConfig())
        assert np.isnan(report.cast_ratio)
        assert report.to_dict()["cast_ratio"] is None


class TestLinearModelProvenance:
    def test_extractor_travels_with_the_model(self, tmp_path):
        """Features measured inside a different mask are not comparable, so the
        extractor name is part of the model, not a serving-time default."""

        from anemia.linear_model import LinearModel, fit_linear_model

        X = np.array([[0.1, -0.02], [0.2, 0.01], [0.15, 0.0], [0.05, -0.03], [0.25, 0.02]])
        y = np.array([9.0, 12.0, 11.0, 8.5, 13.0])
        model = fit_linear_model(X, y, "erythema", extractor="cielab")

        path = tmp_path / "m.json"
        model.save(path)
        assert LinearModel.load(path).extractor == "cielab"

    def test_incomplete_model_file_refuses_to_load(self, tmp_path):
        import json

        from anemia.linear_model import LinearModel

        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"coefficients": [1.0, 2.0]}), encoding="utf-8")
        with pytest.raises(ValueError, match="cannot"):
            LinearModel.load(path)


class TestLearnedSegmenter:
    """The trained backend is written by one file and read by another.

    `scripts/train_segmenter.py` is the only writer; `anemia.segment_learned`
    is the only reader and cannot write. These tests pin the contract between
    them, because a mismatch would surface as a served model quietly measuring
    haemoglobin inside a meaningless mask rather than as an error.
    """

    @staticmethod
    def _bundle(tmp_path, input_size=64):
        smp = pytest.importorskip("segmentation_models_pytorch")
        pytest.importorskip("torch")
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from train_segmenter import save_bundle

        from anemia.segment_learned import SegmenterConfig

        config = SegmenterConfig(encoder="resnet18", input_size=input_size)
        model = smp.Unet(encoder_name=config.encoder, encoder_weights=None,
                         in_channels=3, classes=1)
        path = tmp_path / "unet.pt"
        save_bundle(path, model, config,
                    provenance={"trained_at": "2026-01-01T00:00:00", "datasets": ["synthetic"]},
                    metrics={"val_dice": 0.5})
        return path

    def test_bundle_round_trips_and_honours_the_extractor_contract(self, tmp_path):
        from anemia.segment_learned import LearnedSegmenter

        segmenter = LearnedSegmenter(self._bundle(tmp_path))
        image, _ = make_eye(seed=11)
        mask, name = segmenter(image)

        assert mask.shape == image.shape[:2]
        assert set(np.unique(mask)).issubset({0, 255})
        assert name.startswith("unet")

    def test_training_time_conditioning_travels_with_the_weights(self, tmp_path):
        """Serving must condition images exactly as training did, so the input
        size lives in the bundle rather than defaulting in the serving code."""

        from anemia.segment_learned import LearnedSegmenter

        segmenter = LearnedSegmenter(self._bundle(tmp_path, input_size=96))
        assert segmenter.config.input_size == 96
        assert segmenter.config.encoder == "resnet18"

    def test_retrained_segmenter_invalidates_cached_features(self, tmp_path):
        """Features cached under an old mask must not be reused under a new one."""

        from anemia.segment_learned import LearnedSegmenter

        assert LearnedSegmenter(self._bundle(tmp_path)).cache_key == "unet-2026-01-01T00:00:00"

    def test_quality_gate_uses_the_extractors_own_mask_floor(self):
        """A tight, correct mask must not be rejected as 'too small' by a floor
        calibrated for a different extractor's much larger masks.

        The global 1.5% floor once rejected 30 of 52 real captures under the
        learned extractor, whose masks are ~1.9% of the frame on median.
        """

        image, truth = make_eye(seed=13)
        small = np.zeros_like(truth)
        ys, xs = np.nonzero(truth)
        cy, cx = int(ys.mean()), int(xs.mean())
        small[cy - 6:cy + 6, cx - 20:cx + 20] = 255  # well under 1.5% of frame

        def global_floor(rgb):
            return small, "fixed"

        def own_floor(rgb):
            return small, "fixed"
        own_floor.min_mask_ratio = 0.0005

        cfg = PreprocessConfig()
        rejected = prepare(image, global_floor, cfg, QualityConfig())
        accepted = prepare(image, own_floor, cfg, QualityConfig())

        assert any("too small" in r for r in rejected.quality.reasons)
        assert not any("too small" in r for r in accepted.quality.reasons)

    def test_probability_map_is_exposed_for_weighted_pooling(self, tmp_path):
        from anemia.segment_learned import LearnedSegmenter

        segmenter = LearnedSegmenter(self._bundle(tmp_path))
        image, _ = make_eye(seed=12)
        probability = segmenter.probability(image)

        assert probability.shape == image.shape[:2]
        assert 0.0 <= float(probability.min()) and float(probability.max()) <= 1.0

    @pytest.mark.parametrize(
        "bundle,expected",
        [
            ({"format": "anemia-segmenter", "version": 1, "state_dict": {}}, "missing"),
            ({"weights": [1, 2, 3]}, "missing"),
            ({"format": "something-else", "version": 1, "state_dict": {},
              "config": {}, "provenance": {}}, "format"),
        ],
    )
    def test_unservable_bundle_refuses_to_load(self, tmp_path, bundle, expected):
        """Refusing beats loading and segmenting nonsense — the same rule
        `LinearModel.load` follows for incomplete model files."""

        torch = pytest.importorskip("torch")
        from anemia.segment_learned import LearnedSegmenter

        path = tmp_path / "bad.pt"
        torch.save(bundle, path)
        with pytest.raises(ValueError, match=expected):
            LearnedSegmenter(path)


class TestClinicalLabels:
    @pytest.mark.parametrize(
        "age,sex,expected",
        [
            (2.0, "F", 11.0),    # child 6-59 months
            (8.0, "M", 11.5),
            (30.0, "M", 13.0),   # adult man
            (30.0, "F", 12.0),   # non-pregnant adult woman
            (None, None, 12.0),  # unknown demographics
        ],
    )
    def test_who_thresholds(self, age, sex, expected):
        assert anemia_threshold(age, sex) == expected

    @pytest.mark.parametrize("raw,expected", [
        (15.1, 15.1),          # an ordinary numeric cell
        ("15,1", 15.1),        # European decimal comma, stored as text
        (" 9,8 ", 9.8),
        ("", None),
        ("n/a", None),
        (float("nan"), None),
    ])
    def test_spreadsheet_numbers_accept_a_decimal_comma(self, raw, expected):
        """The Italy sheet mixes both forms in one column; 19 of 123 patients
        are text. A parser that fails on them drops those patients, and one
        that strips the comma turns 15.1 g/dL into 151."""

        from anemia.data import _to_float

        result = _to_float(raw)
        if expected is None:
            assert result is None
        else:
            assert result == pytest.approx(expected)

    def test_local_cohort_loader_reads_both_eyes_with_labels(self, tmp_path):
        """The local loader once shipped referencing an undefined variable after
        an edit meant for a different loader; nothing exercised it, so every
        test still passed. This one builds the cohort's real layout in miniature."""

        from anemia.data import load_local_cohort

        (tmp_path / "DATASAMPLE.csv").write_text(
            "Image ID,Created On,Date Of Birth,Gender,Haemoglobin (gm/dL)\n"
            "1,Tue Jun 21 2022 09:00:22 GMT+0530 (India Standard Time),09/12/2015,M,12\n"
            "2,Wed Mar 01 2023 11:15:33 GMT+0530 (India Standard Time),04/14/2011,F,8.5\n",
            encoding="utf-8",
        )
        image, _ = make_eye(seed=31)
        for eye in ("left_eye", "right_eye"):
            (tmp_path / eye).mkdir()
            for n in (1, 2):
                cv2.imwrite(str(tmp_path / eye / f"{n}.jpeg"), image)

        samples = load_local_cohort(tmp_path)

        assert len(samples) == 4
        assert {s.patient_id for s in samples} == {"local:1", "local:2"}
        by_patient = {s.patient_id: s for s in samples}
        assert by_patient["local:1"].hb == pytest.approx(12.0)
        assert by_patient["local:2"].hb == pytest.approx(8.5)
        assert by_patient["local:2"].sex == "F"
        assert 6.0 < by_patient["local:1"].age_years < 7.5

    def test_hardcoded_eleven_would_miss_anemic_men(self):
        """Why a single fixed threshold fails, made explicit."""

        hb = 12.0
        assert is_anemic(hb, age_years=30, sex="M")   # correct: below 13.0
        assert not hb < 11.0                          # fixed 11.0 rule: missed


class TestSplits:
    def test_no_patient_spans_train_and_test(self):
        samples = [
            Sample(f"p{i // 2}", "synthetic", Path(f"/tmp/{i}.jpg"), hb=8.0 + (i % 9))
            for i in range(40)
        ]
        for split in kfold(samples, folds=4, bins=3, seed=0):
            train_patients = {s.patient_id for s in split.train}
            test_patients = {s.patient_id for s in split.test}
            assert not (train_patients & test_patients)
            assert split.test, "fold produced an empty test set"

    def test_every_sample_appears_in_exactly_one_test_fold(self):
        samples = [
            Sample(f"p{i}", "synthetic", Path(f"/tmp/{i}.jpg"), hb=7.0 + (i % 10))
            for i in range(30)
        ]
        seen = []
        for split in kfold(samples, folds=5, bins=3, seed=0):
            seen.extend(s.patient_id for s in split.test)
        assert sorted(seen) == sorted(s.patient_id for s in samples)


class TestMetrics:
    def test_baseline_comparison_exposes_a_useless_model(self):
        """A model that always predicts the mean must show zero gain."""

        targets = [9.0, 11.0, 13.0, 15.0]
        mean = float(np.mean(targets))
        metrics = regression_metrics([mean] * 4, targets, baseline=mean)
        assert metrics["mae_vs_baseline"] == pytest.approx(0.0)

    def test_rare_labels_get_more_weight(self):
        targets = [12.0] * 20 + [7.0]
        weights = inverse_frequency_weights(targets, bins=5)
        assert weights[-1] > weights[0] * 5
        assert weights.mean() == pytest.approx(1.0, rel=1e-5)

    def test_screening_metrics_report_sensitivity(self):
        result = screening_metrics([True, False, True, False], [True, True, False, False])
        assert result["tp"] == 1 and result["fn"] == 1
        assert result["sensitivity"] == pytest.approx(0.5)


class TestPhantomSignal:
    def test_phantom_pallor_tracks_hb(self):
        """Sanity check on the phantoms themselves: redness must vary with Hb."""

        from anemia.imaging import redness_index

        low, mask = make_eye(hb=7.0, seed=9)
        high, _ = make_eye(hb=16.0, seed=9)
        selected = mask > 0
        assert redness_index(high)[selected].mean() > redness_index(low)[selected].mean()
