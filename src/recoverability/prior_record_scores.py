from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


HISTORICAL_COLUMNS = [
    "episode_id",
    "hist_supported",
    "hist_donor_count",
    "hist_donor_pred_mean",
    "hist_donor_pred_max",
    "hist_donor_actual_recover_mean",
    "hist_donor_median_time_gap",
    "hist_donor_median_available_turn",
]


def parse_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq("true") | series.eq(True)


def run(source_path: Path, historical_path: Path, output_path: Path) -> None:
    historical_header = pd.read_csv(historical_path, nrows=0)
    missing = sorted(set(HISTORICAL_COLUMNS) - set(historical_header.columns))
    if missing:
        raise ValueError(
            "Historical donor output lacks required prior-record fields: "
            + ", ".join(missing)
        )
    historical = pd.read_csv(
        historical_path,
        usecols=HISTORICAL_COLUMNS,
        low_memory=False,
    )
    if historical["episode_id"].duplicated().any():
        raise ValueError("Historical donor output contains duplicate episode identifiers.")
    historical = historical.rename(
        columns={
            "hist_supported": "prior_supported",
            "hist_donor_count": "prior_donor_count",
            "hist_donor_pred_mean": "prior_donor_pred_mean",
            "hist_donor_pred_max": "prior_donor_pred_max",
            "hist_donor_actual_recover_mean": "prior_donor_actual_recover_mean",
            "hist_donor_median_time_gap": "prior_donor_median_time_gap",
            "hist_donor_median_available_turn": "prior_donor_median_available_turn",
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    first = True
    source_rows = 0
    merged_rows = 0
    supported_rows = 0
    stressed_rows = 0
    supported_stressed_rows = 0
    for chunk in pd.read_csv(source_path, chunksize=150_000, low_memory=False):
        source_rows += len(chunk)
        merged = chunk.merge(historical, on="episode_id", how="left", validate="one_to_one")
        merged_rows += len(merged)
        prior_supported = parse_bool(merged["prior_supported"].fillna(False))
        merged["donor_count"] = pd.to_numeric(
            merged["prior_donor_count"], errors="coerce"
        ).fillna(0).astype(int)
        for target, source in [
            ("donor_pred_mean", "prior_donor_pred_mean"),
            ("donor_pred_max", "prior_donor_pred_max"),
            ("donor_actual_recover_mean", "prior_donor_actual_recover_mean"),
            ("donor_median_time_gap", "prior_donor_median_time_gap"),
            ("donor_median_available_turn", "prior_donor_median_available_turn"),
        ]:
            merged[target] = pd.to_numeric(merged[source], errors="coerce")
        merged["ctrg_gap_mean"] = merged["donor_pred_mean"] - pd.to_numeric(
            merged["pred_recover"], errors="coerce"
        )
        merged["ctrg_gap_max"] = merged["donor_pred_max"] - pd.to_numeric(
            merged["pred_recover"], errors="coerce"
        )
        merged["supported"] = prior_supported
        supported_rows += int(prior_supported.sum())
        stressed = parse_bool(merged["stressed"])
        stressed_rows += int(stressed.sum())
        supported_stressed_rows += int((stressed & prior_supported).sum())
        merged = merged.drop(
            columns=[column for column in merged.columns if column.startswith("prior_")]
        )
        merged.to_csv(output_path, mode="w" if first else "a", header=first, index=False)
        first = False

    if source_rows != merged_rows:
        raise RuntimeError("Prior-record merge changed the number of focal episodes.")
    share = supported_stressed_rows / stressed_rows if stressed_rows else np.nan
    print(f"Source episodes: {source_rows:,}")
    print(f"Prior-supported episodes: {supported_rows:,}")
    print(f"Stressed episodes: {stressed_rows:,}")
    print(f"Prior-supported stressed episodes: {supported_stressed_rows:,}")
    print(f"Prior-record support share: {share:.6f}")
    print(f"Saved {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("historical")
    parser.add_argument("output")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(ROOT / args.source, ROOT / args.historical, ROOT / args.output)
