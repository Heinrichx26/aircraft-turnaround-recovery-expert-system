"""Cluster-aware robustness audit for the public-record continuation ranking.

The audit is kept separate from the full model pipeline. It reads saved episode
scores, evaluates a fixed continuation-evidence ranking at several queue sizes,
and bootstraps airport and carrier aggregates to expose dependence sensitivity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CAPACITIES = (0.05, 0.10, 0.20, 0.30, 0.50)
USECOLS = [
    "airport",
    "carrier",
    "stressed",
    "supported",
    "recoverable_despite_severe",
    "ctrg_gap_max",
]


def load_public_scores(source: Path, max_rows: int | None) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    remaining = max_rows
    for chunk in pd.read_csv(source, usecols=USECOLS, chunksize=150_000, low_memory=False):
        if remaining is not None:
            chunk = chunk.iloc[:remaining]
            remaining -= len(chunk)
        chunks.append(chunk)
        if remaining is not None and remaining <= 0:
            break
    data = pd.concat(chunks, ignore_index=True)
    data = data[data["stressed"].astype(bool) & data["supported"].astype(bool)].copy()
    data["opportunity"] = data["recoverable_despite_severe"].astype(int)
    data["priority"] = data["ctrg_gap_max"].fillna(0.0).clip(lower=0.0)
    return data


def bootstrap_capture(
    data: pd.DataFrame,
    cluster_column: str,
    capacity: float,
    repetitions: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, int]:
    threshold = data["priority"].quantile(1.0 - capacity)
    selected = data["priority"].ge(threshold)
    work = data.assign(selected_opportunity=data["opportunity"] * selected.astype(int))
    aggregate = (
        work.groupby(cluster_column, dropna=False)
        .agg(total=("opportunity", "sum"), captured=("selected_opportunity", "sum"))
        .reset_index(drop=True)
    )
    estimates: list[float] = []
    for _ in range(repetitions):
        indices = rng.choice(len(aggregate), len(aggregate), replace=True)
        draw = aggregate.iloc[indices]
        estimates.append(float(draw["captured"].sum() / max(draw["total"].sum(), 1)))
    values = np.asarray(estimates)
    return (
        float(values.mean()),
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
        int(len(aggregate)),
    )


def run_audit(data: pd.DataFrame, repetitions: int, seed: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    for capacity in CAPACITIES:
        for cluster_column, cluster_type in (("airport", "airport"), ("carrier", "carrier")):
            estimate, ci_low, ci_high, clusters = bootstrap_capture(
                data, cluster_column, capacity, repetitions, rng
            )
            rows.append(
                {
                    "capacity": capacity,
                    "cluster_type": cluster_type,
                    "estimate": estimate,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "clusters": clusters,
                    "bootstrap_repetitions": repetitions,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()
    if args.repetitions < 20:
        raise ValueError("repetitions must be at least 20")
    data = load_public_scores(args.source, args.max_rows)
    result = run_audit(data, args.repetitions, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_dir / "public_record_cluster_robustness.csv", index=False)
    manifest = {
        "source": str(args.source),
        "rows_after_filter": int(len(data)),
        "opportunity_count": int(data["opportunity"].sum()),
        "repetitions": args.repetitions,
        "seed": args.seed,
        "clusters": {k: int(data[k].nunique(dropna=False)) for k in ("airport", "carrier")},
        "ranking": "ctrg_gap_max",
        "target": "recoverable_despite_severe",
    }
    (args.output_dir / "public_record_cluster_robustness_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
