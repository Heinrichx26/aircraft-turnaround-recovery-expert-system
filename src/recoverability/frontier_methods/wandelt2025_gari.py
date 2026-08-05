from __future__ import annotations

import numpy as np
import pandas as pd

from .common import FrontierMethod, normalize_by_train, rank01


def airport_resilience_profile(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["delayed15"] = (pd.to_numeric(work["out_dep_delay"], errors="coerce") >= 15).astype(float)
    route = work.groupby(["airport", "dest"], dropna=False).size().rename("route_count").reset_index()
    entropy_rows = []
    for airport, g in route.groupby("airport"):
        p = g["route_count"] / g["route_count"].sum()
        entropy = float(-(p * np.log(p)).sum() / np.log(len(p))) if len(p) > 1 else 0.0
        entropy_rows.append({"airport": airport, "route_entropy": entropy})
    profile = (
        work.groupby("airport", dropna=False)
        .agg(
            departures=("episode_id", "size"),
            route_count=("dest", "nunique"),
            carrier_count=("carrier", "nunique"),
            tail_count=("tail", "nunique"),
            delayed_share=("delayed15", "mean"),
            cancel_share=("is_cancelled", "mean"),
            mean_taxi_out=("taxi_out", "mean"),
        )
        .reset_index()
        .merge(pd.DataFrame(entropy_rows), on="airport", how="left")
    )
    return profile


def score_wandelt2025_gari(train: pd.DataFrame, test: pd.DataFrame, eval_df: pd.DataFrame, horizon: int, random_state: int) -> FrontierMethod:
    train_profile = airport_resilience_profile(train)
    test_profile = airport_resilience_profile(test)
    merged = test_profile.merge(train_profile, on="airport", suffixes=("_test", "_train"), how="left")
    resilience = (
        0.22 * normalize_by_train(train_profile["route_count"], merged["route_count_train"])
        + 0.18 * normalize_by_train(train_profile["carrier_count"], merged["carrier_count_train"])
        + 0.16 * normalize_by_train(train_profile["tail_count"], merged["tail_count_train"])
        + 0.18 * merged["route_entropy_train"].fillna(0.0)
        + 0.14 * (1.0 - normalize_by_train(train_profile["delayed_share"], merged["delayed_share_train"]))
        + 0.12 * (1.0 - normalize_by_train(train_profile["cancel_share"], merged["cancel_share_train"]))
    )
    airport_score = pd.Series(1.0 - resilience.to_numpy(dtype=float), index=merged["airport"])
    local = eval_df.copy()
    pressure = local["airport"].map(airport_score).fillna(airport_score.median())
    score = 0.50 * pressure + 0.25 * rank01(local["out_dep_delay"], ascending=True) + 0.15 * rank01(-local["available_turn"], ascending=True) + 0.10 * (1.0 - rank01(local["donor_count"], ascending=True))
    return FrontierMethod(
        method="Wandelt et al. 2025 GARI adaptation",
        source="Wandelt et al. (2025)",
        journal="Transportation Research Part D",
        year=2025,
        adaptation="Global Airport Resilience Index concept adapted into airport-level route, carrier, tail, delay, and cancellation resilience pressure.",
        score=score,
    )
