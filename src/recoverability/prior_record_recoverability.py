from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from compatible_history import build_historical_library_scores


ROOT = Path(__file__).resolve().parents[2]

PRIOR_NUMERIC = [
    "in_arr_delay",
    "out_dep_delay",
    "available_turn",
    "planned_turn",
    "turn_slack",
    "taxi_in",
    "distance_group",
    "same_tail_turn_index",
    "airport_hour_dep_rate",
    "hour",
    "day_of_week",
]

PRIOR_CATEGORICAL = ["airport", "carrier", "dep_time_blk", "arr_time_blk"]


def load_turnarounds(path: Path, horizon: int) -> pd.DataFrame:
    required = sorted(
        set(
            [
                "episode_id", "tail", "carrier", "airport", "dest", "flight_date",
                "sched_dep_dt", "month", "hour", "day_of_week", "out_dep_delay",
                "available_turn", "distance_group", "dep_time_blk", "arr_time_blk",
                "is_cancelled", "is_diverted", "carrier_delay", "weather_delay",
                "nas_delay", "late_aircraft_delay",
                f"recover_h{horizon}", f"fail_h{horizon}", f"endpoint_obs_h{horizon}",
                *PRIOR_NUMERIC,
                *PRIOR_CATEGORICAL,
            ]
        )
    )
    header = pd.read_csv(path, nrows=0)
    usecols = [column for column in required if column in header.columns]
    frame = pd.read_csv(path, usecols=usecols, low_memory=False)
    frame["sched_dep_dt"] = pd.to_datetime(frame["sched_dep_dt"], errors="coerce")
    for column in [
        *PRIOR_NUMERIC,
        f"recover_h{horizon}", f"fail_h{horizon}", f"endpoint_obs_h{horizon}",
        "is_cancelled", "is_diverted", "carrier_delay", "weather_delay",
        "nas_delay", "late_aircraft_delay",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in [*PRIOR_CATEGORICAL, "episode_id", "tail", "dest"]:
        if column in frame.columns:
            frame[column] = frame[column].fillna("UNK").astype(str)
    return frame


class TemporalEncodedLogit:
    def __init__(self, seed: int, smoothing: float = 50.0) -> None:
        self.seed = seed
        self.smoothing = smoothing
        self.medians: pd.Series | None = None
        self.means: np.ndarray | None = None
        self.scales: np.ndarray | None = None
        self.prior = 0.5
        self.encodings: dict[str, pd.Series] = {}
        self.weights: np.ndarray | None = None

    @staticmethod
    def _keys(frame: pd.DataFrame) -> dict[str, pd.Series]:
        airport = frame["airport"].fillna("UNK").astype(str)
        carrier = frame["carrier"].fillna("UNK").astype(str)
        hour = pd.to_numeric(frame["hour"], errors="coerce").fillna(-1).astype(int).astype(str)
        return {
            "airport": airport,
            "carrier": carrier,
            "departure_block": frame["dep_time_blk"].fillna("UNK").astype(str),
            "arrival_block": frame["arr_time_blk"].fillna("UNK").astype(str),
            "airport_carrier": airport + "|" + carrier,
            "airport_hour": airport + "|" + hour,
            "carrier_hour": carrier + "|" + hour,
        }

    def _matrix(self, frame: pd.DataFrame) -> np.ndarray:
        if self.medians is None or self.means is None or self.scales is None:
            raise RuntimeError("TemporalEncodedLogit has not been fitted.")
        numeric = (
            frame[PRIOR_NUMERIC]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(self.medians)
            .to_numpy(dtype=np.float32)
        )
        numeric = np.clip((numeric - self.means) / self.scales, -8.0, 8.0)
        encoded = []
        for name, key in self._keys(frame).items():
            encoded.append(
                key.map(self.encodings[name]).fillna(self.prior).to_numpy(dtype=np.float32)
            )
        hour = pd.to_numeric(frame["hour"], errors="coerce").fillna(0).to_numpy(dtype=float)
        delay = pd.to_numeric(frame["out_dep_delay"], errors="coerce").fillna(0).to_numpy(dtype=float)
        extras = np.column_stack(
            [
                np.sin(2.0 * np.pi * hour / 24.0),
                np.cos(2.0 * np.pi * hour / 24.0),
                np.log1p(np.clip(delay, 0.0, None)) / 5.0,
                (delay >= 60.0).astype(float),
            ]
        ).astype(np.float32)
        return np.column_stack(
            [np.ones(len(frame), dtype=np.float32), numeric, np.column_stack(encoded), extras]
        ).astype(np.float32)

    def fit(self, frame: pd.DataFrame, target: pd.Series) -> "TemporalEncodedLogit":
        y = target.to_numpy(dtype=np.float32)
        self.prior = float(y.mean())
        numeric = (
            frame[PRIOR_NUMERIC]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
        )
        self.medians = numeric.median().fillna(0.0)
        filled = numeric.fillna(self.medians).to_numpy(dtype=np.float32)
        self.means = filled.mean(axis=0)
        self.scales = np.maximum(filled.std(axis=0), 1e-3)
        keys = self._keys(frame)
        for name, key in keys.items():
            grouped = pd.DataFrame({"key": key, "target": y}).groupby("key")["target"].agg(["sum", "count"])
            self.encodings[name] = (
                grouped["sum"] + self.smoothing * self.prior
            ) / (grouped["count"] + self.smoothing)
        matrix = self._matrix(frame)
        weights = np.zeros(matrix.shape[1], dtype=np.float64)
        weights[0] = np.log(self.prior / max(1.0 - self.prior, 1e-6))
        first = np.zeros_like(weights)
        second = np.zeros_like(weights)
        rng = np.random.default_rng(self.seed)
        batch_size = min(65_536, len(matrix))
        step = 0
        for _ in range(18):
            for start in rng.permutation(np.arange(0, len(matrix), batch_size)):
                stop = min(int(start) + batch_size, len(matrix))
                x_batch = matrix[int(start):stop].astype(np.float64, copy=False)
                y_batch = y[int(start):stop].astype(np.float64, copy=False)
                probability = 1.0 / (1.0 + np.exp(-np.clip(x_batch @ weights, -30.0, 30.0)))
                gradient = x_batch.T @ (probability - y_batch) / len(y_batch)
                gradient[1:] += 2e-4 * weights[1:]
                step += 1
                first = 0.9 * first + 0.1 * gradient
                second = 0.999 * second + 0.001 * gradient**2
                corrected_first = first / (1.0 - 0.9**step)
                corrected_second = second / (1.0 - 0.999**step)
                weights -= 0.025 * corrected_first / (np.sqrt(corrected_second) + 1e-8)
        self.weights = weights
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("TemporalEncodedLogit has not been fitted.")
        logits = self._matrix(frame).astype(np.float64, copy=False) @ self.weights
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))


def binary_auc(target: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    sorted_ranks = np.empty(len(score), dtype=float)
    start = 0
    while start < len(score):
        stop = start + 1
        while stop < len(score) and sorted_score[stop] == sorted_score[start]:
            stop += 1
        sorted_ranks[start:stop] = 0.5 * (start + 1 + stop)
        start = stop
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = sorted_ranks
    positives = target == 1
    n_pos = int(positives.sum())
    n_neg = int(len(target) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def binary_average_precision(target: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(-score, kind="mergesort")
    ordered_target = target[order]
    ordered_score = score[order]
    n_pos = int(ordered_target.sum())
    if n_pos == 0:
        return 0.0
    threshold_ends = np.flatnonzero(
        np.r_[ordered_score[1:] != ordered_score[:-1], True]
    )
    cumulative_true = np.cumsum(ordered_target)[threshold_ends]
    cumulative_count = threshold_ends + 1
    precision = cumulative_true / cumulative_count
    recall = cumulative_true / n_pos
    recall_increment = np.diff(np.r_[0.0, recall])
    return float(np.sum(precision * recall_increment))


def split_time(frame: pd.DataFrame, month: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = frame.sort_values("sched_dep_dt").reset_index(drop=True)
    if month > 0:
        ordered = ordered[ordered["sched_dep_dt"].dt.month.eq(month)].reset_index(drop=True)
    months = sorted(ordered["month"].dropna().unique())
    if len(months) >= 4:
        split_month = months[len(months) // 2]
        training = ordered[ordered["month"] < split_month].copy()
        evaluation = ordered[ordered["month"] >= split_month].copy()
    else:
        boundary = int(math.floor(0.60 * len(ordered)))
        training = ordered.iloc[:boundary].copy()
        evaluation = ordered.iloc[boundary:].copy()
    return training.reset_index(drop=True), evaluation.reset_index(drop=True)


def train_models(training: pd.DataFrame, horizon: int, seed: int):
    target = training[f"recover_h{horizon}"].astype(int)
    return TemporalEncodedLogit(seed).fit(training, target)


def predict(frame: pd.DataFrame, model: TemporalEncodedLogit) -> np.ndarray:
    return model.predict_proba(frame)


def prepare_output(scored: pd.DataFrame, horizon: int) -> pd.DataFrame:
    evaluation = scored[scored["is_evaluation"]].copy()
    evaluation = evaluation.rename(
        columns={
            "hist_donor_count": "donor_count",
            "hist_donor_pred_mean": "donor_pred_mean",
            "hist_donor_pred_max": "donor_pred_max",
            "hist_donor_actual_recover_mean": "donor_actual_recover_mean",
            "hist_donor_median_time_gap": "donor_median_time_gap",
            "hist_donor_median_available_turn": "donor_median_available_turn",
        }
    )
    evaluation["donor_actual_recover_any"] = (
        evaluation["donor_actual_recover_mean"] > 0
    ).astype(float)
    evaluation["ctrg_gap_mean"] = evaluation["donor_pred_mean"] - evaluation["pred_recover"]
    evaluation["ctrg_gap_max"] = evaluation["donor_pred_max"] - evaluation["pred_recover"]
    evaluation["supported"] = evaluation["donor_count"] > 0
    evaluation["stressed"] = evaluation["out_dep_delay"] >= 15
    evaluation["severe_start_delay"] = evaluation["out_dep_delay"] >= 60
    evaluation["structural_brittle"] = (
        evaluation["stressed"]
        & evaluation["supported"]
        & evaluation["ctrg_gap_max"].le(0)
    )
    evaluation["recoverable_despite_severe"] = (
        evaluation["severe_start_delay"]
        & evaluation["supported"]
        & evaluation["donor_pred_max"].ge(0.70)
    )
    for column in ["carrier_delay", "weather_delay", "nas_delay", "late_aircraft_delay"]:
        evaluation[column] = 0.0
    columns = [
        "episode_id",
        "tail",
        "carrier",
        "airport",
        "dest",
        "flight_date",
        "sched_dep_dt",
        "month",
        "hour",
        "out_dep_delay",
        "available_turn",
        "distance_group",
        f"recover_h{horizon}",
        f"fail_h{horizon}",
        "pred_recover",
        "donor_count",
        "donor_pred_mean",
        "donor_pred_max",
        "donor_actual_recover_mean",
        "donor_actual_recover_any",
        "donor_median_time_gap",
        "donor_median_available_turn",
        "ctrg_gap_mean",
        "ctrg_gap_max",
        "supported",
        "stressed",
        "severe_start_delay",
        "structural_brittle",
        "recoverable_despite_severe",
        "carrier_delay",
        "weather_delay",
        "nas_delay",
        "late_aircraft_delay",
    ]
    return evaluation[columns].rename(
        columns={f"recover_h{horizon}": "recover_h4", f"fail_h{horizon}": "fail_h4"}
    )


def run(args: argparse.Namespace) -> None:
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    turn = load_turnarounds(ROOT / args.turnarounds, args.horizon)
    turn = turn.sort_values(["tail", "sched_dep_dt"]).reset_index(drop=True)
    turn["endpoint_close_dt_h4"] = turn.groupby("tail", sort=False)["sched_dep_dt"].shift(
        -(args.horizon - 1)
    )
    eligible = turn[
        turn[f"endpoint_obs_h{args.horizon}"].eq(1) & turn["out_dep_delay"].notna()
    ].copy()
    training, evaluation = split_time(eligible, args.month)
    print(f"Prior-record split: train={len(training):,}, evaluation={len(evaluation):,}", flush=True)
    model = train_models(training, args.horizon, args.seed)
    training["pred_recover"] = predict(training, model)
    evaluation["pred_recover"] = predict(evaluation, model)
    target = evaluation[f"recover_h{args.horizon}"].astype(int)
    target_array = target.to_numpy(dtype=int)
    score_array = evaluation["pred_recover"].to_numpy(dtype=float)
    auc = binary_auc(target_array, score_array)
    ap = binary_average_precision(target_array, score_array)

    training["is_evaluation"] = False
    evaluation["is_evaluation"] = True
    pool = pd.concat([training, evaluation], ignore_index=True).sort_values(
        ["airport", "carrier", "sched_dep_dt"]
    ).reset_index(drop=True)
    pool.loc[~pool["is_evaluation"], "out_dep_delay"] = 0.0
    scored, edge_count = build_historical_library_scores(
        pool,
        donor_window_minutes=args.donor_window_minutes,
        max_donors_per_episode=args.max_donors_per_episode,
    )
    output = prepare_output(scored, args.horizon)
    output.to_csv(output_dir / "episode_scores.csv", index=False)
    stressed = output[output["stressed"]]
    supported = stressed[stressed["supported"]]
    summary = {
        "year": args.year,
        "month": args.month,
        "training_episodes": int(len(training)),
        "evaluation_episodes": int(len(evaluation)),
        "stressed_episodes": int(len(stressed)),
        "supported_stressed_episodes": int(len(supported)),
        "support_share": float(len(supported) / len(stressed)) if len(stressed) else np.nan,
        "donor_edges": int(edge_count),
        "observed_path_auc": float(auc),
        "observed_path_average_precision": float(ap),
        "median_donor_count": float(supported["donor_count"].median()) if len(supported) else np.nan,
        "median_donor_time_gap": float(supported["donor_median_time_gap"].median()) if len(supported) else np.nan,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turnarounds", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--donor-window-minutes", type=int, default=120)
    parser.add_argument("--max-donors-per-episode", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
