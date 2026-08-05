from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def parse_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq("true") | series.eq(True)


def load(path: Path) -> pd.DataFrame:
    columns = [
        "episode_id",
        "sched_dep_dt",
        "month",
        "recover_h4",
        "pred_recover",
        "supported",
        "stressed",
    ]
    parts = []
    for chunk in pd.read_csv(path, usecols=columns, chunksize=200_000):
        keep = parse_bool(chunk["supported"]) & parse_bool(chunk["stressed"])
        parts.append(chunk.loc[keep].copy())
    frame = pd.concat(parts, ignore_index=True)
    frame["sched_dep_dt"] = pd.to_datetime(frame["sched_dep_dt"], errors="coerce")
    for column in ["month", "recover_h4", "pred_recover"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["sched_dep_dt", "recover_h4", "pred_recover"]).sort_values(
        "sched_dep_dt"
    ).reset_index(drop=True)


def split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(frame)
    boundaries = [int(math.floor(n * value)) for value in [0.40, 0.60, 0.75]]
    return (
        frame.iloc[: boundaries[0]].copy(),
        frame.iloc[boundaries[0] : boundaries[1]].copy(),
        frame.iloc[boundaries[1] : boundaries[2]].copy(),
        frame.iloc[boundaries[2] :].copy(),
    )


def expected_calibration_error(target: np.ndarray, probability: np.ndarray, bins: int) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_id = np.clip(np.searchsorted(edges[1:-1], probability, side="right"), 0, bins - 1)
    error = 0.0
    for value in range(bins):
        mask = bin_id == value
        if mask.any():
            error += float(mask.mean()) * abs(float(target[mask].mean()) - float(probability[mask].mean()))
    return error


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -35.0, 35.0)))


def fit_logistic_calibrator(score: np.ndarray, target: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(score)), score])
    parameters = np.asarray([0.0, 1.0], dtype=float)
    for _ in range(100):
        probability = sigmoid(design @ parameters)
        weight = np.maximum(probability * (1.0 - probability), 1e-8)
        gradient = design.T @ (probability - target)
        hessian = (design.T * weight) @ design + 1e-8 * np.eye(2)
        step = np.linalg.solve(hessian, gradient)
        parameters -= step
        if float(np.max(np.abs(step))) < 1e-10:
            break
    return parameters


def roc_auc(target: np.ndarray, probability: np.ndarray) -> float:
    order = np.argsort(probability, kind="mergesort")
    ranks = np.empty(len(probability), dtype=float)
    sorted_probability = probability[order]
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and sorted_probability[stop] == sorted_probability[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    positives = target == 1
    n_positive = int(positives.sum())
    n_negative = int(len(target) - n_positive)
    return float((ranks[positives].sum() - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative))


def average_precision(target: np.ndarray, probability: np.ndarray) -> float:
    order = np.argsort(-probability, kind="mergesort")
    sorted_target = target[order]
    cumulative = np.cumsum(sorted_target)
    precision = cumulative / np.arange(1, len(target) + 1)
    return float(np.sum(precision * sorted_target) / max(int(sorted_target.sum()), 1))


def metrics(target: np.ndarray, probability: np.ndarray, label: str, bins: int) -> dict[str, object]:
    return {
        "period": label,
        "episodes": int(len(target)),
        "recovery_prevalence": float(target.mean()),
        "auc": roc_auc(target, probability),
        "average_precision": average_precision(target, probability),
        "brier_score": float(np.mean((target - probability) ** 2)),
        "expected_calibration_error": expected_calibration_error(target, probability, bins),
        "mean_probability": float(probability.mean()),
    }


def run(input_path: Path, output_dir: Path, bins: int) -> None:
    frame = load(input_path)
    training, tuning, calibration, validation = split(frame)
    tuning_score = np.clip(tuning["pred_recover"].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    tuning_logit = np.log(tuning_score / (1.0 - tuning_score)).reshape(-1, 1)
    parameters = fit_logistic_calibrator(
        tuning_logit[:, 0], tuning["recover_h4"].to_numpy(dtype=int)
    )

    rows = []
    monthly_rows = []
    for label, period in [("risk_calibration", calibration), ("validation", validation)]:
        raw = np.clip(period["pred_recover"].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
        logit = np.log(raw / (1.0 - raw)).reshape(-1, 1)
        calibrated = sigmoid(parameters[0] + parameters[1] * logit[:, 0])
        target = period["recover_h4"].to_numpy(dtype=int)
        rows.append(metrics(target, raw, f"{label}_raw", bins))
        rows.append(metrics(target, calibrated, f"{label}_calibrated", bins))
        period = period.copy()
        period["calibrated_probability"] = calibrated
        for month, group in period.groupby("month", sort=True):
            monthly_rows.append(
                {
                    "period": label,
                    "month": int(month),
                    **metrics(
                        group["recover_h4"].to_numpy(dtype=int),
                        group["calibrated_probability"].to_numpy(dtype=float),
                        "monthly",
                        bins,
                    ),
                }
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "calibration_metrics.csv", index=False)
    pd.DataFrame(monthly_rows).to_csv(output_dir / "monthly_calibration.csv", index=False)
    report = {
        "input": input_path.parent.name,
        "split_sizes": {
            "training": len(training),
            "tuning": len(tuning),
            "risk_calibration": len(calibration),
            "validation": len(validation),
        },
        "calibration_model": "logistic calibration on tuning-period score logits",
        "calibration_bins": bins,
        "intercept": float(parameters[0]),
        "slope": float(parameters[1]),
        "metrics": rows,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output_dir")
    parser.add_argument("--bins", type=int, default=15)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(ROOT / args.input, ROOT / args.output_dir, args.bins)
