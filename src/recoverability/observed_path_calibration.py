from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def calibrate(values: pd.Series, intercept: float, slope: float) -> pd.Series:
    probability = pd.to_numeric(values, errors="coerce").clip(1e-6, 1.0 - 1e-6)
    logit = np.log(probability / (1.0 - probability))
    return 1.0 / (1.0 + np.exp(-np.clip(intercept + slope * logit, -35.0, 35.0)))


def donor_lower_bound(recovery_rate: pd.Series, donor_count: pd.Series) -> np.ndarray:
    rate = pd.to_numeric(recovery_rate, errors="coerce").clip(0.0, 1.0).to_numpy(dtype=float)
    count = np.maximum(
        pd.to_numeric(donor_count, errors="coerce").to_numpy(dtype=float), 1.0
    )
    z = 1.6448536269514722
    denominator = 1.0 + z**2 / count
    center = rate + z**2 / (2.0 * count)
    radius = z * np.sqrt(rate * (1.0 - rate) / count + z**2 / (4.0 * count**2))
    return np.clip((center - radius) / denominator, 0.0, 1.0)


def run(
    input_path: Path,
    calibration_report: Path,
    output_path: Path,
    month: int,
) -> None:
    report = json.loads(calibration_report.read_text(encoding="utf-8"))
    intercept = float(report["intercept"])
    slope = float(report["slope"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    first = True
    source_rows = 0
    output_rows = 0
    for chunk in pd.read_csv(input_path, chunksize=150_000, low_memory=False):
        if month > 0:
            chunk = chunk[pd.to_numeric(chunk["month"], errors="coerce").eq(month)].copy()
        if chunk.empty:
            continue
        source_rows += len(chunk)
        chunk["pred_recover_raw"] = chunk["pred_recover"]
        chunk["pred_recover"] = calibrate(chunk["pred_recover"], intercept, slope)
        for column in ["donor_pred_mean", "donor_pred_max"]:
            chunk[f"{column}_raw"] = chunk[column]
            chunk[column] = calibrate(chunk[column], intercept, slope)
        chunk["donor_recovery_lcb"] = donor_lower_bound(
            chunk["donor_actual_recover_mean"], chunk["donor_count"]
        )
        chunk["ctrg_gap_mean"] = chunk["donor_recovery_lcb"] - chunk["pred_recover"]
        chunk["ctrg_gap_max"] = chunk["ctrg_gap_mean"]
        output_rows += len(chunk)
        chunk.to_csv(output_path, mode="w" if first else "a", header=first, index=False)
        first = False
    if source_rows != output_rows:
        raise RuntimeError("Calibration transformation changed the episode count.")
    print(f"Transformed episodes: {output_rows:,}")
    print(f"Calibration intercept: {intercept:.8f}")
    print(f"Calibration slope: {slope:.8f}")
    print(f"Saved {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("calibration_report")
    parser.add_argument("output")
    parser.add_argument("--month", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(ROOT / args.input, ROOT / args.calibration_report, ROOT / args.output, args.month)
