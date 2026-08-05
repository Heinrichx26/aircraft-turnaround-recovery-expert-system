from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from multitask_recoverability import exact_binomial_upper


def assign_bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(edges[1:-1], values, side="right"), 0, len(edges) - 2)


def simultaneous_conditional_bounds(
    target: np.ndarray,
    score: np.ndarray,
    blocks: int,
    bins: int,
    delta: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float | int]]]:
    edges = np.quantile(score, np.linspace(0.0, 1.0, bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    bin_ids = assign_bins(score, edges)
    block_indices = np.array_split(np.arange(len(target)), blocks)
    alpha = delta / (blocks * bins)
    upper = np.zeros(bins, dtype=float)
    rows = []
    for block_id, indices in enumerate(block_indices):
        for bin_id in range(bins):
            mask = bin_ids[indices] == bin_id
            block_target = target[indices][mask]
            events = int(block_target.sum())
            count = int(len(block_target))
            bound = exact_binomial_upper(events, count, alpha) if count else 1.0
            upper[bin_id] = max(upper[bin_id], bound)
            rows.append(
                {
                    "block": block_id,
                    "score_bin": bin_id,
                    "events": events,
                    "count": count,
                    "empirical_risk": float(events / count) if count else np.nan,
                    "risk_upper_bound": bound,
                }
            )
    return edges, upper, rows


def choose_capacity(
    deployment_score: np.ndarray,
    edges: np.ndarray,
    conditional_upper: np.ndarray,
    risk_target: float,
    capacity_grid: list[float],
) -> tuple[dict[str, float | int | bool], list[dict[str, float | int]]]:
    order = np.argsort(-deployment_score)
    deployment_bins = assign_bins(deployment_score, edges)
    rows = []
    for capacity in capacity_grid:
        selected = max(1, int(math.ceil(len(deployment_score) * capacity)))
        unreviewed = order[selected:]
        counts = np.bincount(deployment_bins[unreviewed], minlength=len(conditional_upper))
        weights = counts / max(int(counts.sum()), 1)
        transported_upper = float(np.sum(weights * conditional_upper))
        rows.append(
            {
                "capacity": capacity,
                "selected": selected,
                "unreviewed": int(len(unreviewed)),
                "transported_risk_upper_bound": transported_upper,
            }
        )
    feasible = [row for row in rows if row["transported_risk_upper_bound"] <= risk_target]
    chosen = feasible[0] if feasible else rows[-1]
    return {**chosen, "risk_control_feasible": bool(feasible)}, rows


def residual_rate(target: np.ndarray, score: np.ndarray, capacity: float) -> float:
    selected = max(1, int(math.ceil(len(target) * capacity)))
    order = np.argsort(-score)
    residual = target[order[selected:]]
    return float(residual.mean()) if len(residual) else 0.0


def assess(
    directory: Path,
    risk_ratio: float,
    blocks: int,
    bins: int,
    delta: float,
    capacity_grid: list[float],
    maximum_density_ratio: float,
    minimum_cell_size: int,
    maximum_bins: int,
    minimum_observed_cell_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    calibration = pd.read_csv(directory / "calibration_scores.csv")
    deployment = pd.read_csv(directory / "validation_scores.csv")
    target_column = "joint_opportunity"
    methods = [column for column in calibration.columns if column not in {"episode_id", target_column}]
    calibration_target = calibration[target_column].to_numpy(dtype=float)
    deployment_target = deployment[target_column].to_numpy(dtype=float)
    if bins <= 0:
        bins = max(
            2,
            min(maximum_bins, int(len(calibration_target) // (blocks * minimum_cell_size))),
        )
    risk_target = float(risk_ratio * calibration_target.mean())
    capacity_rows = []
    bin_rows = []
    summaries = []
    for method in methods:
        calibration_score = calibration[method].to_numpy(dtype=float)
        deployment_score = deployment[method].to_numpy(dtype=float)
        edges, upper, rows = simultaneous_conditional_bounds(
            calibration_target,
            calibration_score,
            blocks,
            bins,
            delta,
        )
        minimum_observed_count = min(int(row["count"]) for row in rows)
        calibration_bins = assign_bins(calibration_score, edges)
        deployment_bins = assign_bins(deployment_score, edges)
        calibration_share = np.bincount(calibration_bins, minlength=bins) / len(calibration_bins)
        deployment_share = np.bincount(deployment_bins, minlength=bins) / len(deployment_bins)
        density_ratio = deployment_share / np.maximum(calibration_share, 1.0 / len(calibration_bins))
        support_passed = bool(
            np.isfinite(deployment_score).all()
            and float(density_ratio.max()) <= maximum_density_ratio
            and minimum_observed_count >= minimum_observed_cell_size
        )
        chosen, frontier = choose_capacity(
            deployment_score,
            edges,
            upper,
            risk_target,
            capacity_grid,
        )
        chosen["risk_control_feasible"] = bool(chosen["risk_control_feasible"] and support_passed)
        validation_residual = residual_rate(
            deployment_target, deployment_score, float(chosen["capacity"])
        )
        for row in rows:
            row.update({"experiment": directory.name, "method": method})
            bin_rows.append(row)
        for row in frontier:
            row.update({"experiment": directory.name, "method": method})
            capacity_rows.append(row)
        summaries.append(
            {
                "method": method,
                "capacity": float(chosen["capacity"]),
                "transported_risk_upper_bound": float(chosen["transported_risk_upper_bound"]),
                "risk_control_feasible": bool(chosen["risk_control_feasible"]),
                "support_passed": support_passed,
                "maximum_density_ratio": float(density_ratio.max()),
                "minimum_observed_cell_count": minimum_observed_count,
                "validation_residual_rate": validation_residual,
                "validation_risk_passed": bool(validation_residual <= risk_target),
            }
        )
    summary_frame = pd.DataFrame(summaries)
    proposed = summary_frame[
        summary_frame["method"].eq("structured_multitask_relational_model")
    ].iloc[0]
    feasible_baselines = summary_frame[
        ~summary_frame["method"].eq("structured_multitask_relational_model")
        & summary_frame["risk_control_feasible"]
        & summary_frame["validation_risk_passed"]
    ]
    baseline_capacity = (
        float(feasible_baselines["capacity"].min()) if len(feasible_baselines) else math.nan
    )
    saving = baseline_capacity - float(proposed["capacity"]) if len(feasible_baselines) else math.nan
    relative = saving / baseline_capacity if baseline_capacity > 0 else math.nan
    report = {
        "experiment": directory.name,
        "score_bins": bins,
        "risk_target": risk_target,
        "calibration_prevalence": float(calibration_target.mean()),
        "validation_prevalence": float(deployment_target.mean()),
        "proposed_capacity": float(proposed["capacity"]),
        "best_feasible_baseline_capacity": None if math.isnan(baseline_capacity) else baseline_capacity,
        "relative_capacity_saving": None if math.isnan(relative) else relative,
        "proposed_validation_residual_rate": float(proposed["validation_residual_rate"]),
        "proposed_validation_risk_passed": bool(proposed["validation_risk_passed"]),
        "support_passed": bool(proposed["support_passed"]),
        "gate_passed": bool(
            proposed["risk_control_feasible"]
            and proposed["validation_risk_passed"]
            and not math.isnan(relative)
            and relative >= 0.20 - 1e-12
        ),
        "methods": summaries,
    }
    return pd.DataFrame(capacity_rows), pd.DataFrame(bin_rows), report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("directories", nargs="+")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--risk-ratio", type=float, default=0.50)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--minimum-cell-size", type=int, default=2500)
    parser.add_argument("--maximum-bins", type=int, default=20)
    parser.add_argument("--minimum-observed-cell-size", type=int, default=1000)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--capacity-step", type=float, default=0.01)
    parser.add_argument("--max-capacity", type=float, default=0.50)
    parser.add_argument("--maximum-density-ratio", type=float, default=2.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    count = int(math.floor(args.max_capacity / args.capacity_step + 1e-12))
    grid = [args.capacity_step * value for value in range(1, count + 1)]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_capacity = []
    all_bins = []
    reports = []
    for value in args.directories:
        capacity, bins, report = assess(
            Path(value),
            args.risk_ratio,
            args.blocks,
            args.bins,
            args.delta,
            grid,
            args.maximum_density_ratio,
            args.minimum_cell_size,
            args.maximum_bins,
            args.minimum_observed_cell_size,
        )
        all_capacity.append(capacity)
        all_bins.append(bins)
        reports.append(report)
    pd.concat(all_capacity, ignore_index=True).to_csv(output_dir / "capacity_frontier.csv", index=False)
    pd.concat(all_bins, ignore_index=True).to_csv(output_dir / "conditional_risk_bounds.csv", index=False)
    payload = {
        "risk_ratio": args.risk_ratio,
        "temporal_blocks": args.blocks,
        "score_bins": "support_adaptive" if args.bins <= 0 else args.bins,
        "minimum_calibration_cell_size": args.minimum_cell_size,
        "maximum_score_bins": args.maximum_bins,
        "minimum_observed_cell_size": args.minimum_observed_cell_size,
        "familywise_delta": args.delta,
        "maximum_density_ratio": args.maximum_density_ratio,
        "experiments": reports,
        "all_experiments_passed": bool(all(report["gate_passed"] for report in reports)),
    }
    (output_dir / "report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
