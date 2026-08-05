"""Support-conditioned distributionally robust selective decisions.

The functions in this module expose the decision layer used by the manuscript.
They operate on saved score-stratum bounds and do not fit a predictive model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class RobustCapacity:
    """A capacity and its exact rectangular worst-case residual risk."""

    capacity: float
    worst_case_risk: float
    feasible: bool
    support_passed: bool


def rectangular_worst_case_risk(
    deployment_weights: Sequence[float],
    stratum_upper_bounds: Sequence[float],
    supported: Sequence[bool] | None = None,
) -> float:
    """Return the exact supremum of a weighted rectangular risk set.

    Each supported stratum risk is allowed to range from zero to its supplied
    upper endpoint. A positive deployment weight in an unsupported stratum
    makes the policy inadmissible and raises ``ValueError``.
    """

    weights = np.asarray(deployment_weights, dtype=float)
    upper = np.asarray(stratum_upper_bounds, dtype=float)
    if weights.ndim != 1 or upper.ndim != 1 or weights.size != upper.size:
        raise ValueError("weights and upper bounds must be one-dimensional with equal length")
    if np.any(~np.isfinite(weights)) or np.any(~np.isfinite(upper)):
        raise ValueError("weights and upper bounds must be finite")
    if np.any(weights < 0) or np.any(upper < 0) or np.any(upper > 1):
        raise ValueError("weights and upper bounds must lie in their probability domains")
    total = float(weights.sum())
    if not np.isclose(total, 1.0, atol=1e-8):
        raise ValueError("deployment weights must sum to one")
    if supported is None:
        supported_mask = np.ones(weights.size, dtype=bool)
    else:
        supported_mask = np.asarray(supported, dtype=bool)
        if supported_mask.shape != weights.shape:
            raise ValueError("supported mask must match the stratum arrays")
    if np.any((weights > 0) & ~supported_mask):
        raise ValueError("positive deployment mass in an unsupported stratum requires abstention")
    return float(np.dot(weights, upper))


def scalar_shift_worst_case_risk(upper_bound: float, shift_budget: float) -> float:
    """Return the prior-record scalar envelope ``upper_bound + shift_budget``."""

    upper = float(upper_bound)
    shift = float(shift_budget)
    if not (0.0 <= upper <= 1.0 and 0.0 <= shift <= 1.0):
        raise ValueError("upper bound and shift budget must be in [0, 1]")
    return min(1.0, upper + shift)


def choose_minimax_capacity(
    capacities: Iterable[float],
    worst_case_risks: Iterable[float],
    risk_target: float,
    *,
    support_passed: bool = True,
) -> RobustCapacity | None:
    """Select the smallest feasible capacity on a declared grid.

    Capacities are expected in ascending order. ``None`` represents abstention
    when support fails or every grid policy exceeds the risk target.
    """

    target = float(risk_target)
    if not 0.0 <= target <= 1.0:
        raise ValueError("risk target must be in [0, 1]")
    rows = list(zip((float(c) for c in capacities), (float(r) for r in worst_case_risks)))
    if not rows:
        raise ValueError("at least one capacity is required")
    if any(not (0.0 <= c <= 1.0 and 0.0 <= r <= 1.0) for c, r in rows):
        raise ValueError("capacities and risks must be in [0, 1]")
    if not support_passed:
        return None
    for capacity, risk in rows:
        if risk <= target:
            return RobustCapacity(capacity, risk, True, True)
    return None
