# -*- coding: utf-8 -*-
"""V5.1 constrained candidate generation for the existing smart-pick UI.

This module only consumes locally supplied probabilities and pair scores. It
never accesses remote lottery services or retrieves draw data.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from v50_predictive_engine import generate_tickets


def generate_constrained_tickets(
    probabilities: np.ndarray,
    pair_scores: np.ndarray | None = None,
    count: int = 10,
    seed: int = 2026,
    pair_strength: float = 0.5,
    max_overlap: int = 4,
) -> tuple[list[list[int]], int, bool]:
    """Generate unique tickets with a pairwise overlap ceiling.

    Returns (tickets, effective_overlap_ceiling, was_relaxed). If the requested
    ceiling is infeasible for the requested batch, it is relaxed one number at
    a time, with the actual ceiling returned explicitly.
    """
    count = int(count)
    requested = int(max_overlap)
    if not 1 <= count <= 20:
        raise ValueError("候选注数必须位于1—20。")
    if not 0 <= requested <= 5:
        raise ValueError("max_overlap必须位于0—5。")
    probs = np.asarray(probabilities, dtype=float).reshape(-1)
    if probs.shape != (45,) or not np.all(np.isfinite(probs)):
        raise ValueError("V5号码概率必须包含45个有限值。")
    if not np.isclose(float(probs.sum()), 6.0, atol=1e-5):
        raise ValueError("V5号码概率总和必须约为6。")

    for ceiling in range(requested, 6):
        accepted: list[list[int]] = []
        max_attempts = max(2500, count * 500)
        for attempt in range(max_attempts):
            # Distinct deterministic seed per draw attempt; stable across reruns.
            attempt_seed = int(
                (int(seed) + attempt * 104729 + ceiling * 1000003) % 2147483647
            )
            candidate = generate_tickets(
                probs, pair_scores, count=1, seed=attempt_seed,
                pair_strength=float(pair_strength),
            )[0]
            if candidate in accepted:
                continue
            if all(
                len(set(candidate).intersection(existing)) <= ceiling
                for existing in accepted
            ):
                accepted.append(candidate)
                if len(accepted) == count:
                    return accepted, ceiling, ceiling > requested
    raise RuntimeError("在允许的跨组重号范围内，无法生成所需数量的不同组合。")
