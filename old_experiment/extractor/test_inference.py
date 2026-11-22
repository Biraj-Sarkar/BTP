"""Regression tests for explicit candidate and legacy ROI extraction paths."""
from pathlib import Path
import numpy as np

from extractor import ROIExtractor, UNetConfig, load_attention_unet, load_unet

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "extractor" / "artifacts" / "attention_unet_320" / "attention_unet_candidate.pt"
LEGACY = ROOT / "Models" / "UNet_extractor" / "unet.pt"
SAMPLE = ROOT / "segmentation_dataset" / "images" / "1731.jpg"


def test_attention_candidate_loads_and_probability_is_finite():
    segmenter = load_attention_unet(CANDIDATE, device="cpu")
    assert segmenter.architecture == "attention_unet"
    assert segmenter.config.input_size == 320
    assert segmenter.config.threshold == 0.8
    image = np.zeros((120, 180, 3), dtype=np.uint8)
    probability = segmenter.predict_probability(image)
    assert probability.shape == (120, 180)
    assert np.isfinite(probability).all()
    assert np.array_equal(probability, segmenter.predict_probability(image))


def test_legacy_checkpoint_still_loads():
    segmenter = load_unet(LEGACY, device="cpu")
    assert segmenter.architecture in {"unet", "unetplusplus", "unet++"}
    assert segmenter.config.input_size in {224, 320}


def test_candidate_end_to_end_crop_contract():
    result = ROIExtractor(checkpoint_path=CANDIDATE).extract_from_path(SAMPLE)
    assert result.crop.shape == (224, 224, 3)
    assert result.crop.dtype == np.uint8
    assert result.mask.ndim == 2
    assert result.mask.shape == (512, 512)
    assert result.bbox[2] > result.bbox[0]
    assert result.bbox[3] > result.bbox[1]


def test_empty_mask_fallback_contract():
    class EmptySegmenter:
        config = UNetConfig(input_size=320, threshold=0.8)
        architecture = "attention_unet"

        def predict_probability(self, image):
            return np.zeros(image.shape[:2], dtype=np.float32)

    result = ROIExtractor(segmenter=EmptySegmenter()).extract(np.full((64, 80, 3), 120, dtype=np.uint8))
    assert result.crop.shape == (224, 224, 3)
    assert result.crop.dtype == np.uint8
    assert not result.mask.any()
    assert result.bbox == (0, 0, 0, 0)
    assert result.backend == "unet-empty"
