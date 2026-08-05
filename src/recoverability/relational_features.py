from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "results" / "ctrg" / "full" / "episode_scores.csv"
DEFAULT_OUTPUT = ROOT / "results" / "ctrg" / "relational_features"


def parse_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq("true") | series.eq(True)


def rank01(values: np.ndarray | pd.Series) -> np.ndarray:
    return pd.Series(np.asarray(values, dtype=float)).rank(pct=True).fillna(0.5).to_numpy()


def load_month(path: Path, month: int, max_rows: int, seed: int) -> pd.DataFrame:
    usecols = [
        "episode_id",
        "tail",
        "carrier",
        "airport",
        "dest",
        "sched_dep_dt",
        "month",
        "hour",
        "out_dep_delay",
        "available_turn",
        "distance_group",
        "fail_h4",
        "pred_recover",
        "donor_count",
        "donor_pred_mean",
        "donor_pred_max",
        "donor_median_time_gap",
        "donor_median_available_turn",
        "ctrg_gap_mean",
        "ctrg_gap_max",
        "supported",
        "stressed",
        "recoverable_despite_severe",
        "carrier_delay",
        "weather_delay",
        "nas_delay",
        "late_aircraft_delay",
    ]
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=150_000):
        keep = pd.to_numeric(chunk["month"], errors="coerce").eq(month)
        keep &= parse_bool(chunk["supported"])
        keep &= parse_bool(chunk["stressed"])
        if keep.any():
            parts.append(chunk.loc[keep].copy())
    if not parts:
        raise ValueError(f"No supported stressed episodes found for month {month}.")
    data = pd.concat(parts, ignore_index=True)
    data["sched_dep_dt"] = pd.to_datetime(data["sched_dep_dt"], errors="coerce")
    numeric = [
        "hour",
        "out_dep_delay",
        "available_turn",
        "distance_group",
        "fail_h4",
        "pred_recover",
        "donor_count",
        "donor_pred_mean",
        "donor_pred_max",
        "donor_median_time_gap",
        "donor_median_available_turn",
        "ctrg_gap_mean",
        "ctrg_gap_max",
        "carrier_delay",
        "weather_delay",
        "nas_delay",
        "late_aircraft_delay",
    ]
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["recoverable_despite_severe"] = parse_bool(data["recoverable_despite_severe"])
    data = data.dropna(
        subset=[
            "sched_dep_dt",
            "fail_h4",
            "pred_recover",
            "donor_count",
            "donor_pred_mean",
            "donor_pred_max",
            "donor_median_time_gap",
            "ctrg_gap_mean",
            "ctrg_gap_max",
            "out_dep_delay",
        ]
    ).sort_values("sched_dep_dt")
    if len(data) > max_rows:
        rng = np.random.default_rng(seed)
        positions = np.sort(rng.choice(len(data), size=max_rows, replace=False))
        data = data.iloc[positions].sort_values("sched_dep_dt")
    return data.reset_index(drop=True)


def split_time(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    first = int(math.floor(len(df) * 0.45))
    second = int(math.floor(len(df) * 0.70))
    training = df.iloc[:first].copy().reset_index(drop=True)
    tuning = df.iloc[first:second].copy().reset_index(drop=True)
    validation = df.iloc[second:].copy().reset_index(drop=True)
    return training, tuning, validation


def review_reward(
    df: pd.DataFrame,
    capacity: float = 0.10,
    failure_multiplier: float = 2.0,
) -> np.ndarray:
    prevalence = max(float(df["fail_h4"].mean()), 1e-6)
    failure_weight = 0.35 + 0.25 * capacity / prevalence
    gap = np.clip(df["ctrg_gap_max"].to_numpy(dtype=float) / 0.40, 0.0, 1.0)
    severe = df["recoverable_despite_severe"].astype(float).to_numpy()
    failure = df["fail_h4"].to_numpy(dtype=float)
    return failure_multiplier * failure_weight * failure + 0.25 * severe + 0.15 * gap


RELATIONS = [
    ["airport"],
    ["carrier"],
    ["airport", "carrier"],
    ["airport", "hour"],
    ["carrier", "hour"],
    ["airport", "dest"],
]


def relation_key(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    return df[columns].astype(str).agg("|".join, axis=1)


def smoothed_relation_messages(
    training: pd.DataFrame,
    other: pd.DataFrame,
    target: np.ndarray,
    smoothing: float,
    leave_one_out: bool,
) -> pd.DataFrame:
    prior = float(np.mean(target))
    output = pd.DataFrame(index=other.index)
    target_series = pd.Series(target, index=training.index)
    for relation_id, columns in enumerate(RELATIONS):
        train_key = relation_key(training, columns)
        stats = pd.DataFrame({"key": train_key, "target": target_series}).groupby("key")[
            "target"
        ].agg(["sum", "count"])
        other_key = relation_key(other, columns)
        sums = other_key.map(stats["sum"]).fillna(0.0).to_numpy(dtype=float)
        counts = other_key.map(stats["count"]).fillna(0.0).to_numpy(dtype=float)
        if leave_one_out:
            sums = sums - target
            counts = np.maximum(counts - 1.0, 0.0)
        message = (sums + smoothing * prior) / (counts + smoothing)
        output[f"relation_{relation_id}_message"] = message
        output[f"relation_{relation_id}_support"] = np.log1p(counts)
    return output


def base_features(df: pd.DataFrame) -> pd.DataFrame:
    def numeric(name: str, default: float = 0.0) -> pd.Series:
        values = pd.to_numeric(df[name], errors="coerce")
        if values.notna().any():
            return values.fillna(float(values.median()))
        return pd.Series(default, index=df.index, dtype=float)

    risk = 1.0 - df["pred_recover"]
    donor_dispersion = (df["donor_pred_max"] - df["donor_pred_mean"]).clip(lower=0.0)
    delay = np.log1p(df["out_dep_delay"].clip(lower=0.0))
    hour_angle = 2.0 * np.pi * df["hour"].fillna(0.0) / 24.0
    urgency = (
        1.0 / (1.0 + np.exp(-np.clip((df["out_dep_delay"] - 60.0) / 12.0, -30.0, 30.0)))
    ) * (
        1.0
        / (
            1.0
            + np.exp(-np.clip((df["donor_pred_max"] - 0.70) / 0.04, -30.0, 30.0))
        )
    )
    frame = pd.DataFrame(
        {
            "pred_recover": df["pred_recover"],
            "observed_risk": risk,
            "gap_max": df["ctrg_gap_max"],
            "gap_mean": df["ctrg_gap_mean"],
            "donor_max": df["donor_pred_max"],
            "donor_mean": df["donor_pred_mean"],
            "donor_dispersion": donor_dispersion,
            "log_donor_count": np.log1p(df["donor_count"].clip(lower=0.0)),
            "donor_time_gap": df["donor_median_time_gap"] / 120.0,
            "donor_turn": df["donor_median_available_turn"] / 240.0,
            "log_delay": delay,
            "turn_pressure": -df["available_turn"] / 240.0,
            "distance_group": df["distance_group"].fillna(df["distance_group"].median()),
            "hour_sin": np.sin(hour_angle),
            "hour_cos": np.cos(hour_angle),
            "carrier_delay": np.log1p(df["carrier_delay"].clip(lower=0.0)),
            "weather_delay": np.log1p(df["weather_delay"].clip(lower=0.0)),
            "nas_delay": np.log1p(df["nas_delay"].clip(lower=0.0)),
            "late_aircraft_delay": np.log1p(df["late_aircraft_delay"].clip(lower=0.0)),
            "recovery_urgency": urgency,
            "risk_x_gap": risk * df["ctrg_gap_max"],
            "gap_x_support": df["ctrg_gap_max"] * np.log1p(df["donor_count"].clip(lower=0.0)),
            "risk_x_urgency": risk * urgency,
        }
    )
    if "airport_elevation_ft" in df.columns:
        frame["airport_elevation_kft"] = numeric("airport_elevation_ft") / 1000.0
        frame["airport_latitude_scaled"] = numeric("airport_latitude") / 50.0
        frame["airport_longitude_scaled"] = numeric("airport_longitude") / 150.0
    if "year_mfr" in df.columns:
        flight_year = pd.to_datetime(df["sched_dep_dt"], errors="coerce").dt.year
        aircraft_age = (flight_year - numeric("year_mfr")).clip(lower=0.0, upper=100.0)
        frame["aircraft_age_decades"] = aircraft_age / 10.0
        frame["log_aircraft_seats"] = np.log1p(numeric("aircraft_seat_count").clip(lower=0.0))
        frame["aircraft_engine_count"] = numeric("aircraft_engine_count")
        frame["aircraft_weight_class"] = numeric("aircraft_weight_class_number")
        frame["aircraft_cruise_speed_scaled"] = numeric("aircraft_cruise_speed") / 500.0
        frame["aircraft_evidence_available"] = numeric("aircraft_evidence_available")
    if "temperature_c" in df.columns:
        temperature = numeric("temperature_c")
        dewpoint = numeric("dewpoint_c")
        direction = np.deg2rad(numeric("wind_direction_deg"))
        frame["temperature_scaled"] = temperature / 40.0
        frame["dewpoint_depression_scaled"] = (temperature - dewpoint).clip(-20.0, 50.0) / 20.0
        frame["pressure_anomaly_scaled"] = (numeric("sea_level_pressure_hpa") - 1013.25) / 30.0
        frame["log_visibility_km"] = np.log1p(numeric("visibility_m").clip(lower=0.0) / 1000.0)
        frame["wind_speed_scaled"] = numeric("wind_speed_ms") / 20.0
        frame["wind_direction_sin"] = np.sin(direction)
        frame["wind_direction_cos"] = np.cos(direction)
        frame["log_ceiling_km"] = np.log1p(numeric("ceiling_height_m").clip(lower=0.0) / 1000.0)
        frame["log_precipitation"] = np.log1p(numeric("precipitation_mm").clip(lower=0.0))
        frame["weather_evidence_available"] = numeric("weather_evidence_available")
    return frame.replace([np.inf, -np.inf], np.nan)


def build_features(
    training: pd.DataFrame,
    tuning: pd.DataFrame,
    validation: pd.DataFrame,
    target: np.ndarray,
    smoothing: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_base = base_features(training)
    tune_base = base_features(tuning)
    valid_base = base_features(validation)
    train_messages = smoothed_relation_messages(
        training, training, target, smoothing, leave_one_out=True
    )
    tune_messages = smoothed_relation_messages(
        training, tuning, target, smoothing, leave_one_out=False
    )
    valid_messages = smoothed_relation_messages(
        training, validation, target, smoothing, leave_one_out=False
    )
    return (
        pd.concat([train_base, train_messages], axis=1),
        pd.concat([tune_base, tune_messages], axis=1),
        pd.concat([valid_base, valid_messages], axis=1),
    )


@dataclass
class TreeNode:
    feature: int | None = None
    threshold: int | None = None
    value: float | None = None
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None


def best_split(
    bins: np.ndarray,
    residual: np.ndarray,
    indices: np.ndarray,
    min_leaf: int,
) -> tuple[int | None, int | None, float]:
    parent = residual[indices]
    parent_sse = float(np.sum((parent - parent.mean()) ** 2))
    best_feature = None
    best_threshold = None
    best_gain = 0.0
    for feature in range(bins.shape[1]):
        values = bins[indices, feature]
        max_bin = int(values.max())
        if max_bin <= 0:
            continue
        counts = np.bincount(values, minlength=max_bin + 1).astype(float)
        sums = np.bincount(values, weights=parent, minlength=max_bin + 1)
        squares = np.bincount(values, weights=parent**2, minlength=max_bin + 1)
        left_count = np.cumsum(counts)[:-1]
        left_sum = np.cumsum(sums)[:-1]
        left_square = np.cumsum(squares)[:-1]
        total_count = counts.sum()
        total_sum = sums.sum()
        total_square = squares.sum()
        right_count = total_count - left_count
        valid = (left_count >= min_leaf) & (right_count >= min_leaf)
        if not valid.any():
            continue
        left_sse = left_square - left_sum**2 / np.maximum(left_count, 1.0)
        right_sum = total_sum - left_sum
        right_square = total_square - left_square
        right_sse = right_square - right_sum**2 / np.maximum(right_count, 1.0)
        gain = parent_sse - left_sse - right_sse
        gain[~valid] = -np.inf
        threshold = int(np.argmax(gain))
        if float(gain[threshold]) > best_gain:
            best_gain = float(gain[threshold])
            best_feature = feature
            best_threshold = threshold
    return best_feature, best_threshold, best_gain


def fit_tree(
    bins: np.ndarray,
    residual: np.ndarray,
    indices: np.ndarray,
    depth: int,
    max_depth: int,
    min_leaf: int,
) -> TreeNode:
    if depth >= max_depth or len(indices) < 2 * min_leaf:
        return TreeNode(value=float(residual[indices].mean()))
    feature, threshold, gain = best_split(bins, residual, indices, min_leaf)
    if feature is None or threshold is None or gain <= 1e-10:
        return TreeNode(value=float(residual[indices].mean()))
    left_indices = indices[bins[indices, feature] <= threshold]
    right_indices = indices[bins[indices, feature] > threshold]
    return TreeNode(
        feature=feature,
        threshold=threshold,
        left=fit_tree(bins, residual, left_indices, depth + 1, max_depth, min_leaf),
        right=fit_tree(bins, residual, right_indices, depth + 1, max_depth, min_leaf),
    )


def predict_tree(node: TreeNode, bins: np.ndarray) -> np.ndarray:
    output = np.empty(len(bins), dtype=float)

    def assign(current: TreeNode, indices: np.ndarray) -> None:
        if current.value is not None:
            output[indices] = current.value
            return
        mask = bins[indices, int(current.feature)] <= int(current.threshold)
        assign(current.left, indices[mask])
        assign(current.right, indices[~mask])

    assign(node, np.arange(len(bins)))
    return output


def bin_features(
    training: pd.DataFrame,
    others: list[pd.DataFrame],
    max_bins: int,
) -> tuple[np.ndarray, list[np.ndarray], pd.Series]:
    median = training.median(numeric_only=True)
    train = training.fillna(median)
    other_filled = [frame.fillna(median) for frame in others]
    thresholds: list[np.ndarray] = []
    for column in train.columns:
        values = train[column].to_numpy(dtype=float)
        quantiles = np.quantile(values, np.linspace(0.0, 1.0, max_bins + 1)[1:-1])
        thresholds.append(np.unique(quantiles))

    def transform(frame: pd.DataFrame) -> np.ndarray:
        output = np.zeros((len(frame), len(frame.columns)), dtype=np.int16)
        for index, column in enumerate(frame.columns):
            output[:, index] = np.searchsorted(
                thresholds[index], frame[column].to_numpy(dtype=float), side="right"
            )
        return output

    return transform(train), [transform(frame) for frame in other_filled], median


@dataclass
class BoostingModel:
    initial: float
    learning_rate: float
    trees: list[TreeNode]


def fit_boosting(
    train_bins: np.ndarray,
    target: np.ndarray,
    trees: int,
    learning_rate: float,
    depth: int,
    min_leaf: int,
) -> BoostingModel:
    initial = float(np.mean(target))
    prediction = np.full(len(target), initial, dtype=float)
    fitted_trees: list[TreeNode] = []
    indices = np.arange(len(target))
    for _ in range(trees):
        residual = target - prediction
        tree = fit_tree(train_bins, residual, indices, 0, depth, min_leaf)
        prediction += learning_rate * predict_tree(tree, train_bins)
        fitted_trees.append(tree)
    return BoostingModel(initial, learning_rate, fitted_trees)


def predict_boosting(model: BoostingModel, bins: np.ndarray) -> np.ndarray:
    prediction = np.full(len(bins), model.initial, dtype=float)
    for tree in model.trees:
        prediction += model.learning_rate * predict_tree(tree, bins)
    return prediction


def evaluate(df: pd.DataFrame, score: np.ndarray, method: str) -> dict[str, float | int | str]:
    values = np.asarray(score, dtype=float)
    n = max(1, int(math.ceil(len(df) * 0.10)))
    selected = np.argpartition(-values, n - 1)[:n]
    picked = df.iloc[selected]
    failure_rate = float(picked["fail_h4"].mean())
    failure_capture = float(picked["fail_h4"].sum() / df["fail_h4"].sum())
    mean_gap = float(picked["ctrg_gap_max"].mean())
    severe_share = float(picked["recoverable_despite_severe"].mean())
    utility = (
        0.35 * failure_rate
        + 0.25 * failure_capture
        + 0.25 * severe_share
        + 0.15 * min(mean_gap / 0.40, 1.0)
    )
    return {
        "method": method,
        "episodes": int(len(df)),
        "top10_n": int(n),
        "failure_rate": failure_rate,
        "failure_capture": failure_capture,
        "mean_gap": mean_gap,
        "severe_high_continuation_share": severe_share,
        "review_utility": float(utility),
    }


PARAMETERS = [
    {"trees": 50, "learning_rate": 0.05, "depth": 2, "min_leaf": 80},
    {"trees": 70, "learning_rate": 0.04, "depth": 3, "min_leaf": 80},
    {"trees": 90, "learning_rate": 0.03, "depth": 3, "min_leaf": 120},
]


def run(
    input_path: Path,
    output_dir: Path,
    month: int,
    max_rows: int,
    seed: int,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_month(input_path, month, max_rows, seed)
    training, tuning, validation = split_time(data)
    target = review_reward(training)
    train_features, tune_features, valid_features = build_features(
        training, tuning, validation, target, smoothing=25.0
    )
    train_bins, [tune_bins, valid_bins], _ = bin_features(
        train_features, [tune_features, valid_features], max_bins=24
    )

    models = []
    tuning_predictions = []
    validation_predictions = []
    model_rows = []
    for model_id, parameters in enumerate(PARAMETERS):
        model = fit_boosting(train_bins, target, **parameters)
        tune_prediction = predict_boosting(model, tune_bins)
        valid_prediction = predict_boosting(model, valid_bins)
        metrics = evaluate(tuning, tune_prediction, f"Relational model {model_id}")
        model_rows.append({"model_id": model_id, **parameters, **metrics})
        models.append(model)
        tuning_predictions.append(tune_prediction)
        validation_predictions.append(valid_prediction)

    tuning_matrix = np.column_stack(tuning_predictions)
    validation_matrix = np.column_stack(validation_predictions)
    tuning_mean = tuning_matrix.mean(axis=1)
    tuning_std = tuning_matrix.std(axis=1)
    validation_mean = validation_matrix.mean(axis=1)
    validation_std = validation_matrix.std(axis=1)
    tuning_gap = rank01(tuning["ctrg_gap_max"])
    validation_gap = rank01(validation["ctrg_gap_max"])
    selection_rows = []
    for blend in np.linspace(0.0, 1.0, 11):
        for uncertainty_penalty in [0.0, 0.25, 0.50, 1.0]:
            learned = rank01(tuning_mean - uncertainty_penalty * tuning_std)
            score = blend * learned + (1.0 - blend) * tuning_gap
            metrics = evaluate(tuning, score, "Relational temporal utility ranker")
            selection_rows.append(
                {
                    "blend": float(blend),
                    "uncertainty_penalty": float(uncertainty_penalty),
                    **metrics,
                }
            )
    selection_table = pd.DataFrame(selection_rows).sort_values(
        ["review_utility", "mean_gap", "failure_rate"],
        ascending=[False, False, False],
    )
    best = selection_table.iloc[0]
    learned_validation = rank01(
        validation_mean - float(best["uncertainty_penalty"]) * validation_std
    )
    proposed_score = float(best["blend"]) * learned_validation + (
        1.0 - float(best["blend"])
    ) * validation_gap
    comparison_rows = [
        evaluate(validation, proposed_score, "Relational temporal utility ranker"),
        evaluate(
            validation,
            validation["ctrg_gap_max"].to_numpy(dtype=float),
            "KCCRES max-gap certificate",
        ),
        evaluate(
            validation,
            (rank01(validation["ctrg_gap_max"]) ** 2.0)
            * ((rank01(1.0 - validation["pred_recover"]) + 0.02) ** 0.75),
            "KCCRES dual-channel audit certificate",
        ),
        evaluate(
            validation,
            1.0 - validation["pred_recover"].to_numpy(dtype=float),
            "Observed-path risk",
        ),
        evaluate(
            validation,
            validation["ctrg_gap_mean"].to_numpy(dtype=float),
            "Mean-donor gap",
        ),
    ]
    comparison = pd.DataFrame(comparison_rows).sort_values(
        "review_utility", ascending=False
    )
    proposed = comparison[
        comparison["method"].eq("Relational temporal utility ranker")
    ].iloc[0]
    best_existing = comparison[
        ~comparison["method"].eq("Relational temporal utility ranker")
    ].iloc[0]

    pd.DataFrame(model_rows).to_csv(output_dir / "model_tuning.csv", index=False)
    selection_table.to_csv(output_dir / "selection_tuning.csv", index=False)
    comparison.to_csv(output_dir / "validation_comparison.csv", index=False)
    report = {
        "month": int(month),
        "sampled_episodes": int(len(data)),
        "training_episodes": int(len(training)),
        "tuning_episodes": int(len(tuning)),
        "validation_episodes": int(len(validation)),
        "feature_count": int(train_features.shape[1]),
        "relation_count": int(len(RELATIONS)),
        "selected_blend": float(best["blend"]),
        "selected_uncertainty_penalty": float(best["uncertainty_penalty"]),
        "validation_comparison": comparison.to_dict(orient="records"),
        "utility_gain_over_best_existing": float(
            proposed["review_utility"] - best_existing["review_utility"]
        ),
        "utility_gain_ge_0_02": bool(
            proposed["review_utility"] - best_existing["review_utility"] >= 0.02
        ),
        "failure_rate_not_below_best_by_more_than_0_02": bool(
            proposed["failure_rate"] >= best_existing["failure_rate"] - 0.02
        ),
        "mean_gap_not_below_best_by_more_than_0_02": bool(
            proposed["mean_gap"] >= best_existing["mean_gap"] - 0.02
        ),
    }
    (output_dir / "evaluation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return report


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--month", type=int, default=7)
    parser.add_argument("--max-rows", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    run(Path(args.input), Path(args.output_dir), args.month, args.max_rows, args.seed)


if __name__ == "__main__":
    main(sys.argv[1:])
