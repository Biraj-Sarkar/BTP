"""Train/evaluate conjunctival segmentation models."""
from .models import build_model
from .dataset import discover_pairs, split_pairs
__all__ = ["build_model", "discover_pairs", "split_pairs"]
