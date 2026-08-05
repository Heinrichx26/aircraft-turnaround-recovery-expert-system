from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from multitask_recoverability import exact_binomial_upper


IDENTIFIERS = {"episode_id", "joint_opportunity"}


def residual(target: np.ndarray, score: np.ndarray, capacity: float) -> np.ndarray:
    selected = max(1, int(math.ceil(len(target) * capacity)))
    order = np.argsort(-score, kind="mergesort")
    return target[order[selected:]]


def block_bounds(
    target: np.ndarray,
    score: np.ndarray,
    capacity: float,
    blocks: int,
    delta: float,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for block_id, indices in enumerate(np.array_split(np.arange(len(target)), blocks)):
        values = residual(target[indices], score[indices], capacity)
        count = int(len(values))
        events = int(values.sum())
        upper = exact_binomial_upper(events, count, delta / blocks) if count else 1.0
        rows.append(
            {
                "block": block_id,
                "events": events,
                "count": count,
                "empirical_risk": float(values.mean()) if count else 0.0,
                "risk_upper_bound": upper,
            }
        )
    return rows


def certify_method(
    calibration_target: np.ndarray,
    calibration_score: np.ndarray,
    validation_target: np.ndarray,
    validation_score: np.ndarray,
    risk_target: float,
    capacities_descending: list[float],
    blocks: int,
    delta: float,
    required_drift_radius: float,
    safety_margin: float,
    minimum_residual_per_block: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    frontier: list[dict[str, object]] = []
    certified: list[dict[str, object]] = []
    sequence_open = True
    for capacity in capacities_descending:
        bounds = block_bounds(
            calibration_target, calibration_score, capacity, blocks, delta
        )
        upper = max(float(row["risk_upper_bound"]) for row in bounds)
        minimum_count = min(int(row["count"]) for row in bounds)
        drift_radius = risk_target - upper
        passes = bool(
            sequence_open
            and minimum_count >= minimum_residual_per_block
            and upper + required_drift_radius + safety_margin <= risk_target
        )
        if sequence_open and not passes:
            sequence_open = False
        if passes:
            certified.append(
                {
                    "capacity": capacity,
                    "risk_upper_bound": upper,
                    "drift_radius": drift_radius,
                }
            )
        frontier.append(
            {
                "capacity": capacity,
                "risk_upper_bound": upper,
                "drift_radius": drift_radius,
                "required_drift_radius": required_drift_radius,
                "safety_margin": safety_margin,
                "minimum_residual_per_block": minimum_count,
                "fixed_sequence_certified": passes,
                "block_results": bounds,
            }
        )
    chosen = certified[-1] if certified else None
    if chosen is None:
        validation_rate = None
        validation_count = None
    else:
        values = residual(
            validation_target, validation_score, float(chosen["capacity"])
        )
        validation_rate = float(values.mean()) if len(values) else 0.0
        validation_count = int(len(values))
    report = {
        "risk_control_feasible": chosen is not None,
        "selected_capacity": None if chosen is None else chosen["capacity"],
        "selected_risk_upper_bound": None if chosen is None else chosen["risk_upper_bound"],
        "certified_drift_radius": None if chosen is None else chosen["drift_radius"],
        "required_drift_radius": required_drift_radius,
        "safety_margin": safety_margin,
        "validation_residual_rate": validation_rate,
        "validation_residual_count": validation_count,
        "validation_risk_passed": bool(
            validation_rate is not None and validation_rate <= risk_target
        ),
    }
    return report, frontier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--risk-ratio", type=float, default=0.50)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--min-capacity", type=float, default=0.50)
    parser.add_argument("--max-capacity", type=float, default=0.98)
    parser.add_argument("--capacity-step", type=float, default=0.01)
    parser.add_argument("--required-drift-radius", type=float, default=0.001)
    parser.add_argument("--safety-margin", type=float, default=0.0)
    parser.add_argument("--minimum-residual-per-block", type=int, default=500)
    args = parser.parse_args()

    score_dir = Path(args.score_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration = pd.read_csv(score_dir / "calibration_scores.csv")
    validation = pd.read_csv(score_dir / "validation_scores.csv")
    target_column = "joint_opportunity"
    methods = [column for column in calibration.columns if column not in IDENTIFIERS]
    calibration_target = calibration[target_column].to_numpy(dtype=float)
    validation_target = validation[target_column].to_numpy(dtype=float)
    risk_target = float(args.risk_ratio * calibration_target.mean())
    count = int(round((args.max_capacity - args.min_capacity) / args.capacity_step))
    capacities = [args.max_capacity - i * args.capacity_step for i in range(count + 1)]

    reports: dict[str, object] = {}
    rows: list[dict[str, object]] = []
    for method in methods:
        report, frontier = certify_method(
            calibration_target,
            calibration[method].to_numpy(dtype=float),
            validation_target,
            validation[method].to_numpy(dtype=float),
            risk_target,
            capacities,
            args.blocks,
            args.delta,
            args.required_drift_radius,
            args.safety_margin,
            args.minimum_residual_per_block,
        )
        reports[method] = report
        for row in frontier:
            block_results = row.pop("block_results")
            for block in block_results:
                rows.append({"method": method, **row, **block})

    proposed = reports.get("structured_multitask_relational_model", {})
    payload = {
        "score_directory": score_dir.name,
        "risk_target": risk_target,
        "calibration_prevalence": float(calibration_target.mean()),
        "validation_prevalence": float(validation_target.mean()),
        "fixed_sequence_order": "maximum_to_minimum_review capacity",
        "familywise_delta": args.delta,
        "temporal_blocks": args.blocks,
        "required_drift_radius": args.required_drift_radius,
        "safety_margin": args.safety_margin,
        "minimum_residual_per_block": args.minimum_residual_per_block,
        "methods": reports,
        "proposed_certificate_passed": bool(
            proposed.get("risk_control_feasible")
            and proposed.get("validation_risk_passed")
        ),
    }
    pd.DataFrame(rows).to_csv(output_dir / "nested_capacity_frontier.csv", index=False)
    (output_dir / "report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
