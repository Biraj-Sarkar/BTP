"""Pipeline tests that run without the real datasets."""

from __future__ import annotations

import sys
from pathlib import Path

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
from anemia.preprocess import prepare  # noqa: E402
from anemia.segment import (  # noqa: E402
    heuristic_conjunctiva_mask,
    heuristic_segmenter,
    legacy_bright_neutral_mask,
)
from anemia.splits import kfold  # noqa: E402
from synthetic import make_eye  # noqa: E402


class TestSegmentation:
    def test_fixed_heuristic_beats_legacy_on_conjunctiva(self):
        """The core claim of the rework, measured rather than asserted.

        The inherited heuristic keeps bright, low-chroma pixels, which is the
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

    def test_segmenter_returns_mask_and_backend_name(self):
        image, _ = make_eye(seed=1)
        mask, backend = heuristic_segmenter(image)
        assert mask.shape == image.shape[:2]
        assert set(np.unique(mask)).issubset({0, 255})
        assert backend in {"heuristic", "grabcut"}


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

    def test_hardcoded_eleven_would_miss_anemic_men(self):
        """The bug the inherited threshold introduced, made explicit."""

        hb = 12.0
        assert is_anemic(hb, age_years=30, sex="M")   # correct: below 13.0
        assert not hb < 11.0                          # inherited rule: missed


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
