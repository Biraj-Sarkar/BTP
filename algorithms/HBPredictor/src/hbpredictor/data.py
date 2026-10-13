from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset
from torchvision import transforms as T

REQUIRED_COLUMNS = {"patient_id", "image_id", "roi_224x224", "age", "sex", "hb"}
CAPTURED_COLUMNS = {"RegistrationNo", "Gender", "AgeInYears", "Haemoglobin", "Images"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class DataStatistics:
    age_mean: float
    age_std: float
    sex_to_index: dict[str, int]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "DataStatistics":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def read_manifest(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    frame = pd.read_csv(path)
    # Normalize the shared captured_data.csv export into this model's internal
    # manifest. The shared reference images are resized by HbDataset below.
    if CAPTURED_COLUMNS.issubset(frame.columns):
        rows = []
        for _, row in frame.iterrows():
            if pd.isna(row["Haemoglobin"]):
                continue
            patient_id = str(row["RegistrationNo"]).strip()
            image = path.parent / f"{patient_id}.jpg"
            if not image.is_file():
                for suffix in (".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
                    candidate = path.parent / f"{patient_id}{suffix}"
                    if candidate.is_file():
                        image = candidate
                        break
            if not image.is_file():
                continue
            rows.append({
                "patient_id": patient_id,
                "image_id": image.stem,
                "roi_224x224": str(image),
                "age": row["AgeInYears"],
                "sex": row["Gender"],
                "hb": row["Haemoglobin"],
            })
        frame = pd.DataFrame(rows, columns=sorted(REQUIRED_COLUMNS))
        if frame.empty:
            raise ValueError(f"No labelled images matched captured-data CSV {path}")
    if "usable" in frame.columns:
        frame = frame[frame["usable"].astype(str).str.strip().str.lower() == "true"].copy()
        if frame.empty:
            raise ValueError(f"No quality-passed ROI rows found in {path}")
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
    if frame[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Manifest contains missing values in required columns")
    frame["roi_224x224"] = frame["roi_224x224"].map(
        lambda value: str((path.parent / str(value)).resolve()) if not Path(str(value)).is_absolute() else str(value)
    )
    absent = [value for value in frame["roi_224x224"] if not Path(value).is_file()]
    if absent:
        raise FileNotFoundError(f"ROI images not found (first): {absent[0]}")
    frame["age"] = pd.to_numeric(frame["age"], errors="raise")
    frame["hb"] = pd.to_numeric(frame["hb"], errors="raise")
    frame["sex"] = frame["sex"].astype(str).str.strip().str.lower()
    return frame


def patient_split(frame: pd.DataFrame, seed: int = 42, val_fraction: float = 0.15, test_fraction: float = 0.15):
    if not 0 < val_fraction < 1 or not 0 < test_fraction < 1 or val_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must be positive and total less than one")
    if frame.patient_id.nunique() < 3:
        raise ValueError("At least three distinct patients are required for train/validation/test splitting")
    groups = frame["patient_id"]
    test_splitter = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    train_val_idx, test_idx = next(test_splitter.split(frame, groups=groups))
    train_val = frame.iloc[train_val_idx]
    val_relative = val_fraction / (1 - test_fraction)
    val_splitter = GroupShuffleSplit(n_splits=1, test_size=val_relative, random_state=seed + 1)
    train_idx_local, val_idx_local = next(val_splitter.split(train_val, groups=train_val["patient_id"]))
    return train_val.iloc[train_idx_local].copy(), train_val.iloc[val_idx_local].copy(), frame.iloc[test_idx].copy()


def make_statistics(train: pd.DataFrame) -> DataStatistics:
    sex_values = sorted(train["sex"].unique().tolist())
    if not sex_values:
        raise ValueError("Training set is empty")
    std = float(train["age"].std(ddof=0))
    return DataStatistics(float(train["age"].mean()), max(std, 1e-6), {sex: idx for idx, sex in enumerate(sex_values)})


def build_transform(train: bool):
    operations: list = [T.Resize((224, 224))]
    if train:
        operations += [T.RandomAffine(degrees=8, translate=(0.04, 0.04), scale=(0.96, 1.04)), T.RandomHorizontalFlip()]
    operations += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(operations)


class HbDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, statistics: DataStatistics, train: bool = False):
        self.frame = frame.reset_index(drop=True)
        self.statistics = statistics
        self.transform = build_transform(train)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image = Image.open(row.roi_224x224).convert("RGB")
        sex = str(row.sex)
        if sex not in self.statistics.sex_to_index:
            raise ValueError(f"Unknown sex category {sex!r}; it was not present in training data")
        age = (float(row.age) - self.statistics.age_mean) / self.statistics.age_std
        return {
            "image": self.transform(image),
            "age": torch.tensor([age], dtype=torch.float32),
            "sex": torch.tensor(self.statistics.sex_to_index[sex], dtype=torch.long),
            "hb": torch.tensor(float(row.hb), dtype=torch.float32),
            "patient_id": str(row.patient_id),
            "image_id": str(row.image_id),
        }
