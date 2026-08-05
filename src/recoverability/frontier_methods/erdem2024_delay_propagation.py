from __future__ import annotations

import pandas as pd

from .common import BASE_CATEGORICAL, BASE_NUMERIC, FrontierMethod, fit_hgb_classifier, predict_hgb


def add_lagged_pressure(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour_bin"] = out["sched_dep_dt"].dt.floor("h")
    out["delayed15"] = (pd.to_numeric(out["out_dep_delay"], errors="coerce") >= 15).astype(float)
    specs = {
        "airport": ["airport"],
        "carrier": ["carrier"],
        "airport_carrier": ["airport", "carrier"],
        "route": ["airport", "dest"],
    }
    for prefix, keys in specs.items():
        stats = (
            out.groupby(keys + ["hour_bin"], dropna=False)
            .agg(
                dep_count=("episode_id", "size"),
                mean_delay=("out_dep_delay", "mean"),
                delayed_share=("delayed15", "mean"),
                mean_taxi_out=("taxi_out", "mean"),
            )
            .reset_index()
            .sort_values(keys + ["hour_bin"])
        )
        for col in ["dep_count", "mean_delay", "delayed_share", "mean_taxi_out"]:
            stats[f"{prefix}_lag1_{col}"] = stats.groupby(keys, dropna=False)[col].shift(1)
            stats[f"{prefix}_lag2_{col}"] = stats.groupby(keys, dropna=False)[col].shift(2)
        keep = keys + ["hour_bin"] + [c for c in stats.columns if c.startswith(f"{prefix}_lag")]
        out = out.merge(stats[keep], on=keys + ["hour_bin"], how="left")
    return out


def score_erdem2024_delay_propagation(train: pd.DataFrame, test: pd.DataFrame, eval_df: pd.DataFrame, horizon: int, random_state: int) -> FrontierMethod:
    all_df = pd.concat([train.assign(split="train"), test.assign(split="test")], ignore_index=True)
    with_pressure = add_lagged_pressure(all_df)
    train_p = with_pressure[with_pressure["split"].eq("train")].copy()
    test_p = with_pressure[with_pressure["split"].eq("test")].copy()
    pressure_features = [c for c in with_pressure.columns if "_lag" in c]
    model = fit_hgb_classifier(
        train_p,
        f"fail_h{horizon}",
        BASE_NUMERIC + pressure_features,
        BASE_CATEGORICAL,
        random_state + 1201,
        max_iter=220,
        learning_rate=0.04,
        min_samples_leaf=35,
        l2_regularization=0.04,
    )
    score_all = pd.Series(predict_hgb(model, test_p), index=test_p["episode_id"])
    score = eval_df["episode_id"].map(score_all)
    return FrontierMethod(
        method="Erdem and Bilgic 2024 propagation learner",
        source="Erdem and Bilgic (2024)",
        journal="Journal of Air Transport Management",
        year=2024,
        adaptation="Daily delay-propagation learning adapted with airport, carrier, airport-carrier, and route lagged pressure features.",
        score=score,
    )
