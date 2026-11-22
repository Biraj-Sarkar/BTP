"""Non-invasive anemia screening from palpebral conjunctiva photographs."""

__version__ = "0.2.0"

from .config import Config, PreprocessConfig, QualityConfig
from .data import Sample, anemia_threshold, is_anemic, load_dataset, summarise

__all__ = [
    "Config",
    "PreprocessConfig",
    "QualityConfig",
    "Sample",
    "anemia_threshold",
    "is_anemic",
    "load_dataset",
    "summarise",
]
