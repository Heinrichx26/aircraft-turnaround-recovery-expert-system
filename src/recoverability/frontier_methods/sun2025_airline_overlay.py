from __future__ import annotations

import numpy as np
import pandas as pd

from .common import FrontierMethod, normalize_by_train, rank01


def overlay_profile(df: pd.DataFrame) -> pd.DataFrame:
    route = df.groupby(["carrier", "airport", "dest"], dropna=False).size().rename("count").reset_index()
    rows = []
    for (carrier, airport), g in route.groupby(["carrier", "airport"], dropna=False):
        p = g["count"] / g["count"].sum()
        entropy = float(-(p * np.log(p)).sum() / np.log(len(p))) if len(p) > 1 else 0.0
        rows.append(
            {
                "carrier": carrier,
                "airport": airport,
                "carrier_airport_routes": int(g["dest"].nunique()),
                "carrier_airport_departures": int(g["count"].sum()),
                "carrier_route_entropy": entropy,
                "top_route_share": float(g["count"].max() / g["count"].sum()),
            }
        )
    tails = (
        df.groupby(["carrier", "airport"], dropna=False)
        .agg(carrier_airport_tails=("tail", "nunique"))
        .reset_index()
    )
    return pd.DataFrame(rows).merge(tails, on=["carrier", "airport"], how="left")


def score_sun2025_airline_overlay(train: pd.DataFrame, test: pd.DataFrame, eval_df: pd.DataFrame, horizon: int, random_state: int) -> FrontierMethod:
    train_p = overlay_profile(train)
    test_p = overlay_profile(test)
    merged = test_p.merge(train_p, on=["carrier", "airport"], suffixes=("_test", "_train"), how="left")
    fragility = (
        0.25 * normalize_by_train(train_p["top_route_share"], merged["top_route_share_train"])
        + 0.22 * (1.0 - normalize_by_train(train_p["carrier_airport_routes"], merged["carrier_airport_routes_train"]))
        + 0.20 * (1.0 - normalize_by_train(train_p["carrier_airport_tails"], merged["carrier_airport_tails_train"]))
        + 0.18 * (1.0 - merged["carrier_route_entropy_train"].fillna(0.0))
        + 0.15 * (1.0 - normalize_by_train(train_p["carrier_airport_departures"], merged["carrier_airport_departures_train"]))
    )
    key = merged["carrier"].astype(str) + "|" + merged["airport"].astype(str)
    overlay_score = pd.Series(fragility.to_numpy(dtype=float), index=key)
    local = eval_df.copy()
    local_key = local["carrier"].astype(str) + "|" + local["airport"].astype(str)
    score = local_key.map(overlay_score).fillna(overlay_score.median())
    score = 0.48 * score + 0.22 * rank01(local["out_dep_delay"], ascending=True) + 0.18 * rank01(1.0 - local["pred_recover"], ascending=True) + 0.12 * (1.0 - rank01(local["donor_count"], ascending=True))
    return FrontierMethod(
        method="Sun et al. 2025 airline-overlay resilience",
        source="Sun et al. (2025)",
        journal="Transportation Research Part E",
        year=2025,
        adaptation="Airline-overlay resilience adapted into carrier-airport route concentration, tail diversity, and local fragility pressure.",
        score=score,
    )
