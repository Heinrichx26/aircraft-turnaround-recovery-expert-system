from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from relational_features import RELATIONS, base_features, rank01, relation_key
from temporal_relation_model import (
    Standardizer,
    adam_update,
    attention_summary,
    forward,
    initialize_parameters,
    loss_and_gradients,
    parameter_items,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "results" / "ctrg" / "full" / "episode_scores.csv"
DEFAULT_OUTPUT = ROOT / "results" / "ctrg" / "risk_controlled_opportunity"


USECOLS = [
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
    "donor_actual_recover_mean",
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
    "airport_elevation_ft",
    "airport_latitude",
    "airport_longitude",
    "year_mfr",
    "aircraft_engine_count",
    "aircraft_seat_count",
    "aircraft_weight_class_number",
    "aircraft_cruise_speed",
    "temperature_c",
    "dewpoint_c",
    "sea_level_pressure_hpa",
    "visibility_m",
    "wind_direction_deg",
    "wind_speed_ms",
    "ceiling_height_m",
    "precipitation_mm",
    "present_weather_code",
    "weather_evidence_available",
    "aircraft_evidence_available",
]

REQUIRED_USECOLS = USECOLS[:28]


def parse_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq("true") | series.eq(True)


def donor_recovery_lower_bound(
    recovery_rate: np.ndarray | pd.Series,
    donor_count: np.ndarray | pd.Series,
    confidence: float = 0.90,
) -> np.ndarray:
    rate = np.clip(np.asarray(recovery_rate, dtype=float), 0.0, 1.0)
    count = np.maximum(np.asarray(donor_count, dtype=float), 1.0)
    if abs(confidence - 0.90) > 1e-12:
        raise ValueError("The evaluation implementation currently supports a 90% Wilson lower bound.")
    z = 1.6448536269514722
    denominator = 1.0 + z**2 / count
    center = rate + z**2 / (2.0 * count)
    radius = z * np.sqrt(rate * (1.0 - rate) / count + z**2 / (4.0 * count**2))
    return np.clip((center - radius) / denominator, 0.0, 1.0)


def load_month(path: Path, month: int, max_rows: int, seed: int) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    header = pd.read_csv(path, nrows=0)
    missing = sorted(set(REQUIRED_USECOLS) - set(header.columns))
    if missing:
        raise ValueError(f"Episode score file lacks required fields: {', '.join(missing)}")
    available_usecols = [column for column in USECOLS if column in header.columns]
    for chunk in pd.read_csv(path, usecols=available_usecols, chunksize=150_000):
        mask = parse_bool(chunk["supported"])
        if month > 0:
            mask &= pd.to_numeric(chunk["month"], errors="coerce").eq(month)
        mask &= parse_bool(chunk["stressed"])
        if mask.any():
            parts.append(chunk.loc[mask].copy())
    if not parts:
        raise ValueError(f"No supported stressed episodes found for month {month}.")
    data = pd.concat(parts, ignore_index=True)
    data["sched_dep_dt"] = pd.to_datetime(data["sched_dep_dt"], errors="coerce")
    numeric = [column for column in available_usecols if column not in {
        "episode_id", "tail", "carrier", "airport", "dest", "sched_dep_dt",
        "supported", "stressed", "recoverable_despite_severe",
        "weather_evidence_available", "aircraft_evidence_available"
    }]
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["recoverable_despite_severe"] = parse_bool(data["recoverable_despite_severe"])
    for column in ["weather_evidence_available", "aircraft_evidence_available"]:
        if column in data.columns:
            data[column] = parse_bool(data[column]).astype(float)
    required = [
        "sched_dep_dt", "fail_h4", "pred_recover", "donor_count",
        "donor_pred_mean", "donor_pred_max", "donor_actual_recover_mean",
        "donor_median_time_gap", "ctrg_gap_mean", "ctrg_gap_max", "out_dep_delay",
    ]
    data = data.dropna(subset=required).sort_values("sched_dep_dt")
    if max_rows > 0 and len(data) > max_rows:
        rng = np.random.default_rng(seed)
        positions = np.sort(rng.choice(len(data), size=max_rows, replace=False))
        data = data.iloc[positions].sort_values("sched_dep_dt")
    data = data.reset_index(drop=True)
    data["donor_recovery_lcb"] = donor_recovery_lower_bound(
        data["donor_actual_recover_mean"], data["donor_count"]
    )
    return data


def split_time(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(data)
    first = int(math.floor(n * 0.40))
    second = int(math.floor(n * 0.60))
    third = int(math.floor(n * 0.75))
    return (
        data.iloc[:first].copy().reset_index(drop=True),
        data.iloc[first:second].copy().reset_index(drop=True),
        data.iloc[second:third].copy().reset_index(drop=True),
        data.iloc[third:].copy().reset_index(drop=True),
    )


def opportunity_label(
    data: pd.DataFrame,
    reliability_threshold: float,
    advantage_margin: float,
) -> np.ndarray:
    return (
        (data["fail_h4"].to_numpy(dtype=float) >= 0.5)
        & (data["donor_recovery_lcb"].to_numpy(dtype=float) >= reliability_threshold)
        & (
            data["donor_recovery_lcb"].to_numpy(dtype=float)
            - data["pred_recover"].to_numpy(dtype=float)
            >= advantage_margin
        )
    ).astype(float)


def relation_tokens(
    training: pd.DataFrame,
    other: pd.DataFrame,
    target: np.ndarray,
    smoothing: float,
    leave_one_out: bool,
) -> tuple[np.ndarray, list[str]]:
    signals = np.column_stack(
        [
            target,
            training["fail_h4"].to_numpy(dtype=float),
            training["donor_recovery_lcb"].to_numpy(dtype=float),
            np.clip(training["ctrg_gap_max"].to_numpy(dtype=float) / 0.40, 0.0, 1.0),
            1.0 - training["pred_recover"].to_numpy(dtype=float),
            training["donor_pred_mean"].to_numpy(dtype=float),
        ]
    )
    names = [
        "robust_opportunity",
        "failed_exit",
        "donor_recovery_lower_bound",
        "recoverability_gap",
        "observed_path_risk",
        "predicted_donor_recovery",
        "log_support",
    ]
    output = np.zeros((len(other), len(RELATIONS), len(names)), dtype=np.float64)
    for relation_id, columns in enumerate(RELATIONS):
        train_key = relation_key(training, columns)
        other_key = relation_key(other, columns)
        counts = train_key.value_counts()
        mapped_counts = other_key.map(counts).fillna(0.0).to_numpy(dtype=float)
        adjusted_counts = np.maximum(mapped_counts - 1.0, 0.0) if leave_one_out else mapped_counts
        output[:, relation_id, -1] = np.log1p(adjusted_counts)
        for signal_id in range(signals.shape[1]):
            sums = pd.DataFrame({"key": train_key, "value": signals[:, signal_id]}).groupby("key")[
                "value"
            ].sum()
            mapped_sums = other_key.map(sums).fillna(0.0).to_numpy(dtype=float, copy=True)
            if leave_one_out:
                mapped_sums -= signals[:, signal_id]
            prior = float(signals[:, signal_id].mean())
            output[:, relation_id, signal_id] = (
                mapped_sums + smoothing * prior
            ) / (adjusted_counts + smoothing)
    return output, names


def prepare_features(
    training: pd.DataFrame,
    tuning: pd.DataFrame,
    calibration: pd.DataFrame,
    validation: pd.DataFrame,
    target: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray], list[str], list[str]]:
    frames = [base_features(frame) for frame in [training, tuning, calibration, validation]]
    base_scaler = Standardizer.fit(frames[0].to_numpy(dtype=float))
    base_arrays = [base_scaler.transform(frame.to_numpy(dtype=float)) for frame in frames]
    tokens_train, names = relation_tokens(
        training, training, target, smoothing=25.0, leave_one_out=True
    )
    token_arrays = [tokens_train]
    for frame in [tuning, calibration, validation]:
        token_arrays.append(
            relation_tokens(training, frame, target, smoothing=25.0, leave_one_out=False)[0]
        )
    scaler = Standardizer.fit(tokens_train.reshape(-1, tokens_train.shape[-1]))
    token_arrays = [
        scaler.transform(tokens.reshape(-1, tokens.shape[-1])).reshape(tokens.shape)
        for tokens in token_arrays
    ]
    return base_arrays, token_arrays, list(frames[0].columns), names


def opportunity_metrics(
    data: pd.DataFrame,
    target: np.ndarray,
    score: np.ndarray,
    capacity: float,
    method: str,
) -> dict[str, float | int | str]:
    values = np.asarray(score, dtype=float)
    selected_count = max(1, int(math.ceil(len(data) * capacity)))
    order = np.argsort(-values)
    selected = order[:selected_count]
    unreviewed = order[selected_count:]
    total_opportunities = max(float(target.sum()), 1.0)
    precision = float(target[selected].mean())
    capture = float(target[selected].sum() / total_opportunities)
    residual_rate = float(target[unreviewed].mean()) if len(unreviewed) else 0.0
    return {
        "method": method,
        "episodes": int(len(data)),
        "capacity": float(capacity),
        "selected": int(selected_count),
        "opportunity_precision": precision,
        "opportunity_capture": capture,
        "unreviewed_opportunity_rate": residual_rate,
        "mean_gap": float(data.iloc[selected]["ctrg_gap_max"].mean()),
        "failed_exit_rate": float(data.iloc[selected]["fail_h4"].mean()),
        "mean_donor_recovery_lcb": float(data.iloc[selected]["donor_recovery_lcb"].mean()),
    }


def fit_model(
    base_train: np.ndarray,
    token_train: np.ndarray,
    target: np.ndarray,
    base_tune: np.ndarray,
    token_tune: np.ndarray,
    tuning: pd.DataFrame,
    tuning_target: np.ndarray,
    setting: dict[str, float | int],
    seed: int,
) -> tuple[object, dict[str, float | int]]:
    target_mean = float(target.mean())
    target_scale = max(float(target.std()), 1e-6)
    normalized_target = (target - target_mean) / target_scale
    sample_weight = 1.0 + float(setting["positive_weight"]) * target
    parameters = initialize_parameters(
        base_train.shape[1],
        token_train.shape[2],
        token_train.shape[1],
        int(setting["hidden_dim"]),
        seed,
    )
    first_moment = {name: np.zeros_like(value) for name, value in parameter_items(parameters)}
    second_moment = {name: np.zeros_like(value) for name, value in parameter_items(parameters)}
    best_parameters = copy.deepcopy(parameters)
    best_value = -np.inf
    best_epoch = 0
    patience = 0
    final_loss = math.nan
    for epoch in range(1, int(setting["epochs"]) + 1):
        final_loss, gradients, _ = loss_and_gradients(
            base_train,
            token_train,
            normalized_target,
            sample_weight,
            parameters,
            float(setting["l2"]),
            float(setting["rank_weight"]),
            0.10,
            float(setting["temperature"]),
        )
        adam_update(
            parameters,
            gradients,
            first_moment,
            second_moment,
            epoch,
            float(setting["learning_rate"]),
        )
        if epoch % 20 == 0 or epoch == int(setting["epochs"]):
            prediction, _ = forward(base_tune, token_tune, parameters)
            prediction = prediction * target_scale + target_mean
            metrics = opportunity_metrics(
                tuning, tuning_target, prediction, 0.10, "temporal relation model"
            )
            value = 0.65 * float(metrics["opportunity_precision"]) + 0.35 * float(
                metrics["opportunity_capture"]
            )
            if value > best_value + 1e-6:
                best_value = value
                best_epoch = epoch
                best_parameters = copy.deepcopy(parameters)
                patience = 0
            else:
                patience += 1
            if patience >= 8:
                break
    return best_parameters, {
        "best_epoch": int(best_epoch),
        "best_tuning_objective": float(best_value),
        "final_training_loss": float(final_loss),
        "target_mean": target_mean,
        "target_scale": target_scale,
    }


MODEL_SETTINGS = [
    {
        "hidden_dim": 8,
        "positive_weight": 3.0,
        "learning_rate": 0.008,
        "epochs": 480,
        "l2": 1e-4,
        "rank_weight": 0.8,
        "temperature": 0.55,
    },
    {
        "hidden_dim": 12,
        "positive_weight": 5.0,
        "learning_rate": 0.006,
        "epochs": 560,
        "l2": 2e-4,
        "rank_weight": 1.2,
        "temperature": 0.45,
    },
    {
        "hidden_dim": 16,
        "positive_weight": 7.0,
        "learning_rate": 0.005,
        "epochs": 640,
        "l2": 3e-4,
        "rank_weight": 1.6,
        "temperature": 0.35,
    },
]


def baseline_scores(data: pd.DataFrame) -> dict[str, np.ndarray]:
    risk = rank01(1.0 - data["pred_recover"])
    gap = rank01(data["ctrg_gap_max"])
    support = rank01(np.log1p(data["donor_count"]))
    return {
        "Observed-path risk": risk,
        "KCCRES max-gap certificate": gap,
        "KCCRES dual-channel certificate": gap**2 * (risk + 0.02) ** 0.75,
        "Risk-gap-support certificate": risk * gap * (0.5 + 0.5 * support),
    }


def hoeffding_upper_bound(
    target: np.ndarray,
    score: np.ndarray,
    capacity: float,
    delta: float,
    comparisons: int,
) -> tuple[float, float]:
    selected_count = max(1, int(math.ceil(len(target) * capacity)))
    order = np.argsort(-np.asarray(score, dtype=float))
    residual = target[order[selected_count:]]
    if len(residual) == 0:
        return 0.0, 0.0
    empirical = float(residual.mean())
    radius = math.sqrt(math.log(comparisons / delta) / (2.0 * len(residual)))
    return empirical, min(1.0, empirical + radius)


CAPACITY_GRID = [value / 100.0 for value in range(5, 61, 5)]


def calibrate_capacity(
    target: np.ndarray,
    score: np.ndarray,
    risk_target: float,
    delta: float,
) -> dict[str, float | bool]:
    rows = []
    for capacity in CAPACITY_GRID:
        empirical, upper = hoeffding_upper_bound(
            target, score, capacity, delta, len(CAPACITY_GRID)
        )
        rows.append((capacity, empirical, upper))
    feasible = [row for row in rows if row[2] <= risk_target]
    chosen = feasible[0] if feasible else (1.0, 0.0, 0.0)
    return {
        "capacity": float(chosen[0]),
        "calibration_residual_rate": float(chosen[1]),
        "calibration_risk_upper_bound": float(chosen[2]),
        "risk_control_feasible": bool(feasible),
    }


def run(
    input_path: Path,
    output_dir: Path,
    month: int,
    max_rows: int,
    seed: int,
    reliability_threshold: float,
    advantage_margin: float,
    residual_risk_target: float,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_month(input_path, month, max_rows, seed)
    training, tuning, calibration, validation = split_time(data)
    targets = [
        opportunity_label(frame, reliability_threshold, advantage_margin)
        for frame in [training, tuning, calibration, validation]
    ]
    base_arrays, token_arrays, base_names, token_names = prepare_features(
        training, tuning, calibration, validation, targets[0]
    )

    tuning_predictions = []
    calibration_predictions = []
    validation_predictions = []
    training_rows = []
    attention_rows = []
    for model_id, setting in enumerate(MODEL_SETTINGS):
        model, summary = fit_model(
            base_arrays[0],
            token_arrays[0],
            targets[0],
            base_arrays[1],
            token_arrays[1],
            tuning,
            targets[1],
            setting,
            seed + model_id * 101,
        )
        predictions = []
        for base, tokens in zip(base_arrays[1:], token_arrays[1:]):
            prediction, _ = forward(base, tokens, model)
            predictions.append(prediction * float(summary["target_scale"]) + float(summary["target_mean"]))
        tuning_predictions.append(predictions[0])
        calibration_predictions.append(predictions[1])
        validation_predictions.append(predictions[2])
        training_rows.append({"model_id": model_id, **setting, **summary})
        attention_rows.append(
            {"model_id": model_id, **attention_summary(base_arrays[1], token_arrays[1], model)}
        )

    tune_matrix = np.column_stack(tuning_predictions)
    calibration_matrix = np.column_stack(calibration_predictions)
    validation_matrix = np.column_stack(validation_predictions)
    tune_baselines = baseline_scores(tuning)
    calibration_baselines = baseline_scores(calibration)
    validation_baselines = baseline_scores(validation)

    selection_rows = []
    candidate_arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for model_id in range(tune_matrix.shape[1]):
        candidate_arrays[f"attention_model_{model_id}"] = (
            rank01(tune_matrix[:, model_id]),
            rank01(calibration_matrix[:, model_id]),
            rank01(validation_matrix[:, model_id]),
        )
    candidate_arrays["attention_ensemble"] = (
        rank01(tune_matrix.mean(axis=1)),
        rank01(calibration_matrix.mean(axis=1)),
        rank01(validation_matrix.mean(axis=1)),
    )
    for candidate_name, arrays in candidate_arrays.items():
        for learned_weight in [0.50, 0.70, 0.85, 1.00]:
            for risk_weight in [0.00, 0.10, 0.20, 0.30]:
                for gap_weight in [0.00, 0.10, 0.20, 0.30]:
                    if learned_weight + risk_weight + gap_weight > 1.0:
                        continue
                    support_weight = 1.0 - learned_weight - risk_weight - gap_weight
                    tune_score = (
                        learned_weight * arrays[0]
                        + risk_weight * tune_baselines["Observed-path risk"]
                        + gap_weight * tune_baselines["KCCRES max-gap certificate"]
                        + support_weight
                        * rank01(np.log1p(tuning["donor_count"]))
                    )
                    metrics = opportunity_metrics(
                        tuning,
                        targets[1],
                        tune_score,
                        0.10,
                        "Risk-controlled opportunity graph",
                    )
                    objective = 0.65 * float(metrics["opportunity_precision"]) + 0.35 * float(
                        metrics["opportunity_capture"]
                    )
                    selection_rows.append(
                        {
                            "candidate": candidate_name,
                            "learned_weight": learned_weight,
                            "risk_weight": risk_weight,
                            "gap_weight": gap_weight,
                            "support_weight": support_weight,
                            "selection_objective": objective,
                            **metrics,
                        }
                    )
    selection = pd.DataFrame(selection_rows).sort_values(
        ["selection_objective", "opportunity_precision", "opportunity_capture"],
        ascending=False,
    )
    best = selection.iloc[0]
    arrays = candidate_arrays[str(best["candidate"])]

    def compose(
        frame: pd.DataFrame,
        learned: np.ndarray,
        baselines: dict[str, np.ndarray],
    ) -> np.ndarray:
        return (
            float(best["learned_weight"]) * learned
            + float(best["risk_weight"]) * baselines["Observed-path risk"]
            + float(best["gap_weight"]) * baselines["KCCRES max-gap certificate"]
            + float(best["support_weight"]) * rank01(np.log1p(frame["donor_count"]))
        )

    proposed_calibration_score = compose(calibration, arrays[1], calibration_baselines)
    proposed_validation_score = compose(validation, arrays[2], validation_baselines)
    proposed_top10 = opportunity_metrics(
        validation,
        targets[3],
        proposed_validation_score,
        0.10,
        "Risk-controlled opportunity graph",
    )

    method_scores_calibration = {
        "Risk-controlled opportunity graph": proposed_calibration_score,
        **calibration_baselines,
    }
    method_scores_validation = {
        "Risk-controlled opportunity graph": proposed_validation_score,
        **validation_baselines,
    }
    top10_rows = [proposed_top10]
    for name, score in validation_baselines.items():
        top10_rows.append(opportunity_metrics(validation, targets[3], score, 0.10, name))
    top10_table = pd.DataFrame(top10_rows).sort_values(
        ["opportunity_precision", "opportunity_capture"], ascending=False
    )

    capacity_rows = []
    calibration_results = {}
    for name, score in method_scores_calibration.items():
        calibrated = calibrate_capacity(
            targets[2], score, residual_risk_target, delta=0.05
        )
        calibration_results[name] = calibrated
        validation_metrics = opportunity_metrics(
            validation,
            targets[3],
            method_scores_validation[name],
            float(calibrated["capacity"]),
            name,
        )
        capacity_rows.append({**calibrated, **validation_metrics})
    capacity_table = pd.DataFrame(capacity_rows).sort_values("capacity")

    proposed_top = top10_table[
        top10_table["method"].eq("Risk-controlled opportunity graph")
    ].iloc[0]
    best_baseline_top = top10_table[
        ~top10_table["method"].eq("Risk-controlled opportunity graph")
    ].iloc[0]
    proposed_capacity = calibration_results["Risk-controlled opportunity graph"]
    baseline_capacities = [
        float(value["capacity"])
        for name, value in calibration_results.items()
        if name != "Risk-controlled opportunity graph"
    ]
    best_baseline_capacity = min(baseline_capacities)
    precision_gain = float(
        proposed_top["opportunity_precision"] - best_baseline_top["opportunity_precision"]
    )
    capacity_saving = float(best_baseline_capacity - float(proposed_capacity["capacity"]))
    mean_relation_weights = np.mean(
        [row["mean_relation_weights"] for row in attention_rows], axis=0
    )
    maximum_relation_weight = float(mean_relation_weights.max())
    proposed_capacity_validation = capacity_table[
        capacity_table["method"].eq("Risk-controlled opportunity graph")
    ].iloc[0]

    pd.DataFrame(training_rows).to_csv(output_dir / "model_training.csv", index=False)
    pd.DataFrame(attention_rows).to_json(
        output_dir / "relation_attention.json", orient="records", indent=2
    )
    selection.to_csv(output_dir / "selection_tuning.csv", index=False)
    top10_table.to_csv(output_dir / "validation_top10.csv", index=False)
    capacity_table.to_csv(output_dir / "validation_capacity_control.csv", index=False)
    report = {
        "problem": (
            "Identify focal chains that fail while their compatible donor set has a 90% Wilson "
            "lower recovery bound above the declared threshold and a declared improvement margin "
            "over the focal recovery estimate, then control the opportunity rate remaining outside "
            "the review pool."
        ),
        "month": int(month),
        "sampled_episodes": int(len(data)),
        "split_sizes": {
            "training": int(len(training)),
            "tuning": int(len(tuning)),
            "risk_calibration": int(len(calibration)),
            "validation": int(len(validation)),
        },
        "reliability_threshold": float(reliability_threshold),
        "advantage_margin": float(advantage_margin),
        "residual_risk_target": float(residual_risk_target),
        "opportunity_prevalence": {
            "training": float(targets[0].mean()),
            "tuning": float(targets[1].mean()),
            "risk_calibration": float(targets[2].mean()),
            "validation": float(targets[3].mean()),
        },
        "base_feature_count": int(base_arrays[0].shape[1]),
        "relation_count": int(token_arrays[0].shape[1]),
        "relation_signal_count": int(token_arrays[0].shape[2]),
        "base_features": base_names,
        "relation_signals": token_names,
        "selected_score": {
            "candidate": str(best["candidate"]),
            "learned_weight": float(best["learned_weight"]),
            "risk_weight": float(best["risk_weight"]),
            "gap_weight": float(best["gap_weight"]),
            "support_weight": float(best["support_weight"]),
        },
        "mean_relation_weights": [float(value) for value in mean_relation_weights],
        "maximum_relation_weight": maximum_relation_weight,
        "top10_validation": top10_table.to_dict(orient="records"),
        "capacity_control_validation": capacity_table.to_dict(orient="records"),
        "top10_precision_gain_over_best_baseline": precision_gain,
        "review_capacity_saving_over_best_baseline": capacity_saving,
        "validation_residual_risk_at_calibrated_capacity": float(
            proposed_capacity_validation["unreviewed_opportunity_rate"]
        ),
        "top10_precision_gain_ge_0_05": bool(precision_gain >= 0.05),
        "capacity_saving_ge_0_10": bool(capacity_saving >= 0.10),
        "residual_risk_control_passed": bool(
            float(proposed_capacity_validation["unreviewed_opportunity_rate"])
            <= residual_risk_target
        ),
        "no_relation_collapse": bool(maximum_relation_weight < 0.90),
        "evaluation_gate_passed": bool(
            (precision_gain >= 0.05 or capacity_saving >= 0.10)
            and float(proposed_capacity_validation["unreviewed_opportunity_rate"])
            <= residual_risk_target
            and maximum_relation_weight < 0.90
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
    parser.add_argument("--reliability-threshold", type=float, default=0.50)
    parser.add_argument("--advantage-margin", type=float, default=0.0)
    parser.add_argument("--residual-risk-target", type=float, default=0.10)
    args = parser.parse_args(argv)
    run(
        Path(args.input),
        Path(args.output_dir),
        args.month,
        args.max_rows,
        args.seed,
        args.reliability_threshold,
        args.advantage_margin,
        args.residual_risk_target,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
