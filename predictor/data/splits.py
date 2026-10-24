"""Patient-level grouped dataset partitioning with clinical stratification and leakage verification."""

from __future__ import annotations

from typing import List, Dict, Tuple, Set
import numpy as np
import pandas as pd
from .manifest import SampleRecord


def verify_split_leakage(
    train_records: List[SampleRecord],
    val_records: List[SampleRecord],
    test_records: List[SampleRecord],
) -> None:
    """Verifies that no patient ID appears in multiple partitions."""
    train_patients = {r.patient_id for r in train_records}
    val_patients = {r.patient_id for r in val_records}
    test_patients = {r.patient_id for r in test_records}

    leak_train_val = train_patients & val_patients
    leak_train_test = train_patients & test_patients
    leak_val_test = val_patients & test_patients

    if leak_train_val or leak_train_test or leak_val_test:
        raise ValueError(
            f"Patient leakage detected! "
            f"Train/Val overlap: {len(leak_train_val)}, "
            f"Train/Test overlap: {len(leak_train_test)}, "
            f"Val/Test overlap: {len(leak_val_test)}"
        )


def create_patient_splits(
    records: List[SampleRecord],
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    stratify_bins: int = 4,
    seed: int = 42,
) -> Tuple[List[SampleRecord], List[SampleRecord], List[SampleRecord], Dict[str, Any]]:
    """Partition dataset at the patient level with stratified Hb distribution.
    
    Ensures zero data leakage between training, validation, and held-out test sets.
    """
    if not records:
        raise ValueError("Cannot partition empty record list.")

    # Group records by unique patient
    patient_to_records: Dict[str, List[SampleRecord]] = {}
    for r in records:
        patient_to_records.setdefault(r.patient_id, []).append(r)

    unique_patients = sorted(list(patient_to_records.keys()))
    rng = np.random.RandomState(seed)

    # Compute patient-level mean Hb for stratification
    patient_hb = np.array([
        np.mean([r.hb for r in patient_to_records[p]])
        for p in unique_patients
    ])

    # Assign strata based on Hb quantiles
    num_bins = min(stratify_bins, len(unique_patients) // 3)
    if num_bins > 1:
        bins = pd.qcut(patient_hb, q=num_bins, labels=False, duplicates="drop")
    else:
        bins = np.zeros(len(unique_patients), dtype=int)

    train_patients: List[str] = []
    val_patients: List[str] = []
    test_patients: List[str] = []

    for bin_idx in np.unique(bins):
        idx_in_bin = np.where(bins == bin_idx)[0]
        rng.shuffle(idx_in_bin)
        n = len(idx_in_bin)

        n_test = max(1, int(round(n * test_fraction)))
        n_val = max(1, int(round(n * val_fraction)))
        n_train = n - n_test - n_val

        if n_train <= 0:
            # Fallback for very small bins
            n_train = max(1, n - 2)
            n_val = 1 if (n - n_train) >= 1 else 0
            n_test = n - n_train - n_val

        bin_test = idx_in_bin[:n_test]
        bin_val = idx_in_bin[n_test:n_test + n_val]
        bin_train = idx_in_bin[n_test + n_val:]

        test_patients.extend([unique_patients[i] for i in bin_test])
        val_patients.extend([unique_patients[i] for i in bin_val])
        train_patients.extend([unique_patients[i] for i in bin_train])

    # Expand patients to sample records
    train_records = [r for p in train_patients for r in patient_to_records[p]]
    val_records = [r for p in val_patients for r in patient_to_records[p]]
    test_records = [r for p in test_patients for r in patient_to_records[p]]

    # Ensure zero leakage
    verify_split_leakage(train_records, val_records, test_records)

    split_report = {
        "seed": seed,
        "total_patients": len(unique_patients),
        "total_samples": len(records),
        "train_patients": len(train_patients),
        "train_samples": len(train_records),
        "train_hb_mean": float(np.mean([r.hb for r in train_records])),
        "train_hb_std": float(np.std([r.hb for r in train_records])),
        "val_patients": len(val_patients),
        "val_samples": len(val_records),
        "val_hb_mean": float(np.mean([r.hb for r in val_records])),
        "val_hb_std": float(np.std([r.hb for r in val_records])),
        "test_patients": len(test_patients),
        "test_samples": len(test_records),
        "test_hb_mean": float(np.mean([r.hb for r in test_records])),
        "test_hb_std": float(np.std([r.hb for r in test_records])),
    }

    return train_records, val_records, test_records, split_report

