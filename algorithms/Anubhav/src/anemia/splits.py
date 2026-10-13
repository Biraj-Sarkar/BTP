"""Patient-grouped, Hb-stratified splits.

Two independent ways to inflate your reported accuracy are avoided here:

* **Leakage.** Eyes-defy-anemia stores several captures per patient. Splitting
  on images puts near-duplicate photos of the same eye in train and test, and
  the model scores well by recognising the patient rather than the pallor.
  Grouping is on `patient_id`.
* **Unlucky folds.** With ~900 images and few severe cases, a random split can
  leave a fold with no anemic patients at all. Groups are stratified by the
  patient's mean Hb bin before assignment.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterator, List, Sequence, Tuple

import numpy as np

from .data import Sample


@dataclass
class Split:
    train: List[Sample]
    test: List[Sample]

    def describe(self) -> str:
        return (
            f"train={len(self.train)} images / {len({s.patient_id for s in self.train})} patients, "
            f"test={len(self.test)} images / {len({s.patient_id for s in self.test})} patients"
        )


def _patient_hb(samples: Sequence[Sample]) -> Dict[str, float]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for sample in samples:
        grouped[sample.patient_id].append(sample.hb)
    return {pid: float(np.mean(values)) for pid, values in grouped.items()}


def _stratified_groups(
    samples: Sequence[Sample], bins: int, seed: int
) -> List[Tuple[str, int]]:
    """Order patients so that consecutive runs alternate across Hb strata."""

    patient_hb = _patient_hb(samples)
    patients = sorted(patient_hb)
    values = np.array([patient_hb[p] for p in patients])

    if values.size == 0:
        return []

    quantiles = np.quantile(values, np.linspace(0, 1, bins + 1)[1:-1]) if bins > 1 else np.array([])
    strata = np.digitize(values, quantiles)

    rng = np.random.default_rng(seed)
    ordered: List[Tuple[str, int]] = []
    for stratum in range(int(strata.max()) + 1):
        members = [patients[i] for i in np.where(strata == stratum)[0]]
        rng.shuffle(members)
        ordered.extend((patient, stratum) for patient in members)
    return ordered


def kfold(samples: Sequence[Sample], folds: int, bins: int, seed: int) -> Iterator[Split]:
    """Yield `folds` patient-grouped, Hb-stratified train/test splits."""

    ordered = _stratified_groups(samples, bins, seed)
    if len(ordered) < folds:
        raise ValueError(f"Need at least {folds} patients for {folds}-fold CV, got {len(ordered)}")

    assignment: Dict[str, int] = {}
    per_stratum_counter: Dict[int, int] = defaultdict(int)
    for patient, stratum in ordered:
        assignment[patient] = per_stratum_counter[stratum] % folds
        per_stratum_counter[stratum] += 1

    for fold in range(folds):
        train = [s for s in samples if assignment[s.patient_id] != fold]
        test = [s for s in samples if assignment[s.patient_id] == fold]
        yield Split(train=train, test=test)


def holdout(samples: Sequence[Sample], fraction: float, bins: int, seed: int) -> Split:
    """Single patient-grouped holdout, for quick iteration only.

    Report cross-validated numbers in the thesis; this exists so a debugging run
    finishes in one fold's time.
    """

    ordered = _stratified_groups(samples, bins, seed)
    per_stratum_counter: Dict[int, int] = defaultdict(int)
    test_patients = set()
    for patient, stratum in ordered:
        index = per_stratum_counter[stratum]
        per_stratum_counter[stratum] += 1
        if (index % max(int(round(1 / max(fraction, 1e-6))), 2)) == 0:
            test_patients.add(patient)

    return Split(
        train=[s for s in samples if s.patient_id not in test_patients],
        test=[s for s in samples if s.patient_id in test_patients],
    )
