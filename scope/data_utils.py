"""Pure data helpers shared by SCOPE training and preprocessing."""

from __future__ import annotations

import random
from collections.abc import Sequence


def sample_disjoint_index(
    target_index: int,
    dataset_size: int,
    rng: random.Random | None = None,
) -> int:
    """Sample uniformly from every dataset index except ``target_index``."""
    if dataset_size < 2:
        raise ValueError("Register calibration requires at least two training examples.")
    if not 0 <= target_index < dataset_size:
        raise IndexError(
            f"target_index must be in [0, {dataset_size}), got {target_index}"
        )

    generator = rng if rng is not None else random
    sampled = generator.randrange(dataset_size - 1)
    return sampled + 1 if sampled >= target_index else sampled


def retained_teacher_records(records: Sequence[dict]) -> list[dict]:
    """Return non-filtered, non-empty teacher records in dataset-ready form."""
    retained = []
    for record in records:
        if record.get("filtered", False):
            continue
        reasoning = record.get("teacher_reasoning", "")
        if not isinstance(reasoning, str) or not reasoning.strip():
            continue
        retained.append(
            {
                "problem": record["problem"],
                "solution": record["solution"],
                "teacher_reasoning": reasoning,
            }
        )
    return retained
