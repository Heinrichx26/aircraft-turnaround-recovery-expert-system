from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


BTS_URL = (
    "https://transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)

USECOLS = [
    "Year",
    "Quarter",
    "Month",
    "DayofMonth",
    "DayOfWeek",
    "FlightDate",
    "Reporting_Airline",
    "Tail_Number",
    "Flight_Number_Reporting_Airline",
    "Origin",
    "Dest",
    "CRSDepTime",
    "DepTime",
    "DepDelay",
    "DepDelayMinutes",
    "DepDel15",
    "DepartureDelayGroups",
    "DepTimeBlk",
    "TaxiOut",
    "WheelsOff",
    "WheelsOn",
    "TaxiIn",
    "CRSArrTime",
    "ArrTime",
    "ArrDelay",
    "ArrDelayMinutes",
    "ArrDel15",
    "ArrivalDelayGroups",
    "ArrTimeBlk",
    "Cancelled",
    "Diverted",
    "CRSElapsedTime",
    "ActualElapsedTime",
    "AirTime",
    "Distance",
    "DistanceGroup",
    "CarrierDelay",
    "WeatherDelay",
    "NASDelay",
    "SecurityDelay",
    "LateAircraftDelay",
]

NUMERIC_COLS = [
    "CRSDepTime",
    "DepTime",
    "DepDelay",
    "DepDelayMinutes",
    "DepDel15",
    "DepartureDelayGroups",
    "TaxiOut",
    "WheelsOff",
    "WheelsOn",
    "TaxiIn",
    "CRSArrTime",
    "ArrTime",
    "ArrDelay",
    "ArrDelayMinutes",
    "ArrDel15",
    "ArrivalDelayGroups",
    "Cancelled",
    "Diverted",
    "CRSElapsedTime",
    "ActualElapsedTime",
    "AirTime",
    "Distance",
    "DistanceGroup",
    "CarrierDelay",
    "WeatherDelay",
    "NASDelay",
    "SecurityDelay",
    "LateAircraftDelay",
]

MODEL_NUMERIC = [
    "in_arr_delay",
    "out_dep_delay",
    "available_turn",
    "planned_turn",
    "turn_slack",
    "taxi_out",
    "taxi_in",
    "distance_group",
    "same_tail_turn_index",
    "airport_hour_dep_rate",
    "airport_hour_mean_dep_delay",
    "airport_day_weather_delay_share",
    "airport_day_late_aircraft_share",
    "airport_day_cancel_share",
    "hour",
    "day_of_week",
]

MODEL_CATEGORICAL = [
    "airport",
    "carrier",
    "dep_time_blk",
    "arr_time_blk",
]


@dataclass
class ExperimentConfig:
    mode: str
    years: list[int]
    months: list[int]
    airports: list[str]
    horizon: int
    min_turn: int
    max_turn: int
    donor_window_minutes: int
    max_donors_per_episode: int
    random_state: int


def log(message: str) -> None:
    print(message, flush=True)


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def download_month(year: int, month: int, raw_dir: Path) -> Path:
    ensure_dirs(raw_dir)
    out = raw_dir / f"bts_otp_{year}_{month:02d}.zip"
    if out.exists() and out.stat().st_size > 1_000_000:
        return out
    url = BTS_URL.format(year=year, month=month)
    log(f"Downloading BTS {year}-{month:02d} ...")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 public aviation research"},
    )
    part = out.with_suffix(out.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            if part.exists():
                part.unlink()
            with urllib.request.urlopen(request, timeout=180) as response, part.open("wb") as fh:
                while True:
                    block = response.read(1 << 20)
                    if not block:
                        break
                    fh.write(block)
            if part.stat().st_size < 1_000_000:
                raise RuntimeError(f"Downloaded file is unexpectedly small: {part}")
            with zipfile.ZipFile(part) as archive:
                failed_member = archive.testzip()
                if failed_member is not None:
                    raise RuntimeError(f"ZIP integrity check failed for {failed_member}")
            part.replace(out)
            return out
        except Exception as exc:
            last_error = exc
            if part.exists():
                part.unlink()
            if attempt < 3:
                log(f"Retrying BTS {year}-{month:02d} after download error ({attempt}/3).")
                time.sleep(2**attempt)
    raise RuntimeError(f"Failed to download BTS {year}-{month:02d}") from last_error


def read_bts_zip(zip_path: Path, airports: set[str]) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise RuntimeError(f"No CSV found in {zip_path}")
        with zf.open(csv_names[0]) as fh:
            header = pd.read_csv(fh, nrows=0)
        usecols = [c for c in USECOLS if c in set(header.columns)]
        with zf.open(csv_names[0]) as fh:
            df = pd.read_csv(fh, usecols=usecols, low_memory=False)
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in USECOLS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[df["Tail_Number"].notna()].copy()
    df = df[(df["Origin"].isin(airports)) | (df["Dest"].isin(airports))].copy()
    return df


def hhmm_to_minutes(value: float | int | str) -> float:
    if pd.isna(value):
        return np.nan
    try:
        raw = int(float(value))
    except (TypeError, ValueError):
        return np.nan
    if raw == 2400:
        raw = 0
    hour = raw // 100
    minute = raw % 100
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        return np.nan
    return float(hour * 60 + minute)


def add_basic_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["flight_date_dt"] = pd.to_datetime(df["FlightDate"], errors="coerce")
    df["crs_dep_min"] = df["CRSDepTime"].map(hhmm_to_minutes)
    df["crs_arr_min"] = df["CRSArrTime"].map(hhmm_to_minutes)
    df["sched_dep_dt"] = df["flight_date_dt"] + pd.to_timedelta(df["crs_dep_min"], unit="m")
    df["dep_delay_signed"] = pd.to_numeric(df["DepDelay"], errors="coerce")
    df["arr_delay_signed"] = pd.to_numeric(df["ArrDelay"], errors="coerce")
    df["dep_delay_minutes"] = pd.to_numeric(df["DepDelayMinutes"], errors="coerce").fillna(
        df["dep_delay_signed"].clip(lower=0)
    )
    df["arr_delay_minutes"] = pd.to_numeric(df["ArrDelayMinutes"], errors="coerce").fillna(
        df["arr_delay_signed"].clip(lower=0)
    )
    df["is_cancelled"] = pd.to_numeric(df["Cancelled"], errors="coerce").fillna(0).astype(float)
    df["is_diverted"] = pd.to_numeric(df["Diverted"], errors="coerce").fillna(0).astype(float)
    return df


def choose_arrival_datetime(prev: pd.Series, next_sched_dep: pd.Timestamp) -> tuple[pd.Timestamp | pd.NaT, float]:
    if pd.isna(prev["flight_date_dt"]) or pd.isna(prev["crs_arr_min"]) or pd.isna(next_sched_dep):
        return pd.NaT, np.nan
    bases = [
        prev["flight_date_dt"] + pd.Timedelta(days=offset) + pd.to_timedelta(prev["crs_arr_min"], unit="m")
        for offset in (-1, 0, 1, 2)
    ]
    best = None
    best_planned = None
    for cand in bases:
        planned = (next_sched_dep - cand).total_seconds() / 60.0
        if -120 <= planned <= 1440:
            if best is None or abs(planned) < abs(best_planned):
                best = cand
                best_planned = planned
    if best is None:
        return pd.NaT, np.nan
    actual = best + pd.to_timedelta(prev["arr_delay_signed"] if pd.notna(prev["arr_delay_signed"]) else 0.0, unit="m")
    return actual, float(best_planned)


def reconstruct_turnarounds(df: pd.DataFrame, airports: set[str], min_turn: int, max_turn: int) -> pd.DataFrame:
    df = add_basic_datetimes(df)
    df = df.sort_values(["Tail_Number", "sched_dep_dt", "Flight_Number_Reporting_Airline"]).reset_index(drop=True)
    prev = df.copy()
    nxt = df.groupby("Tail_Number", sort=False).shift(-1)
    pair = pd.DataFrame(
        {
            "tail": prev["Tail_Number"],
            "prev_dest": prev["Dest"],
            "prev_flight_date_dt": prev["flight_date_dt"],
            "prev_crs_arr_min": prev["crs_arr_min"],
            "prev_arr_delay_signed": prev["arr_delay_signed"],
            "prev_arr_delay_minutes": prev["arr_delay_minutes"],
            "prev_taxi_in": prev["TaxiIn"],
            "prev_arr_time_blk": prev["ArrTimeBlk"],
            "airport": nxt["Origin"],
            "dest": nxt["Dest"],
            "carrier": nxt["Reporting_Airline"],
            "flight_date": nxt["FlightDate"],
            "flight_number": nxt["Flight_Number_Reporting_Airline"],
            "sched_dep_dt": nxt["sched_dep_dt"],
            "month": nxt["Month"],
            "day_of_week": nxt["DayOfWeek"],
            "out_dep_delay": nxt["dep_delay_minutes"],
            "out_dep_delay_signed": nxt["dep_delay_signed"],
            "out_arr_delay": nxt["arr_delay_minutes"],
            "taxi_out": nxt["TaxiOut"],
            "distance_group": nxt["DistanceGroup"],
            "dep_time_blk": nxt["DepTimeBlk"],
            "is_cancelled": nxt["is_cancelled"],
            "is_diverted": nxt["is_diverted"],
            "carrier_delay": nxt["CarrierDelay"],
            "weather_delay": nxt["WeatherDelay"],
            "nas_delay": nxt["NASDelay"],
            "security_delay": nxt["SecurityDelay"],
            "late_aircraft_delay": nxt["LateAircraftDelay"],
        }
    )
    pair = pair[
        (pair["prev_dest"] == pair["airport"])
        & (pair["airport"].isin(airports))
        & pair["sched_dep_dt"].notna()
        & pair["prev_flight_date_dt"].notna()
        & pair["prev_crs_arr_min"].notna()
    ].copy()
    if pair.empty:
        return pair

    sched_dep = pair["sched_dep_dt"]
    best_planned = pd.Series(np.nan, index=pair.index, dtype=float)
    best_arrival = pd.Series(pd.NaT, index=pair.index, dtype="datetime64[ns]")
    for offset in (-1, 0, 1, 2):
        cand_arr = pair["prev_flight_date_dt"] + pd.to_timedelta(offset, unit="D") + pd.to_timedelta(
            pair["prev_crs_arr_min"], unit="m"
        )
        planned = (sched_dep - cand_arr).dt.total_seconds() / 60.0
        valid = planned.between(-120, 1440)
        better = valid & (best_planned.isna() | (planned.abs() < best_planned.abs()))
        best_planned.loc[better] = planned.loc[better]
        best_arrival.loc[better] = cand_arr.loc[better]
    actual_arr = best_arrival + pd.to_timedelta(pair["prev_arr_delay_signed"].fillna(0.0), unit="m")
    available_turn = (sched_dep - actual_arr).dt.total_seconds() / 60.0
    actual_dep = sched_dep + pd.to_timedelta(pair["out_dep_delay_signed"].fillna(0.0), unit="m")
    actual_turn = (actual_dep - actual_arr).dt.total_seconds() / 60.0

    pair["planned_turn"] = best_planned
    pair["available_turn"] = available_turn
    pair["actual_turn"] = actual_turn
    pair = pair[
        pair["planned_turn"].between(min_turn, max_turn)
        & pair["available_turn"].between(-120, max_turn)
    ].copy()
    if pair.empty:
        return pair
    pair["episode_id"] = (
        pair["tail"].astype(str)
        + "_"
        + pair["flight_date"].astype(str)
        + "_"
        + pair["flight_number"].astype(str)
        + "_"
        + pair.groupby("tail").cumcount().astype(str)
    )
    pair["hour"] = pair["sched_dep_dt"].dt.hour.astype(int)
    pair["turn_slack"] = pair["available_turn"] - min_turn
    pair["same_tail_turn_index"] = pair.groupby("tail").cumcount()
    pair["taxi_in"] = pair["prev_taxi_in"]
    pair["in_arr_delay"] = pair["prev_arr_delay_minutes"]
    pair["in_arr_delay_signed"] = pair["prev_arr_delay_signed"]
    pair["arr_time_blk"] = pair["prev_arr_time_blk"].fillna("UNK").astype(str)
    pair["dep_time_blk"] = pair["dep_time_blk"].fillna("UNK").astype(str)
    for col in [
        "carrier_delay",
        "weather_delay",
        "nas_delay",
        "security_delay",
        "late_aircraft_delay",
        "is_cancelled",
        "is_diverted",
    ]:
        pair[col] = pd.to_numeric(pair[col], errors="coerce").fillna(0.0)
    keep_cols = [
        "episode_id",
        "tail",
        "carrier",
        "airport",
        "dest",
        "flight_date",
        "sched_dep_dt",
        "hour",
        "month",
        "day_of_week",
        "in_arr_delay",
        "in_arr_delay_signed",
        "out_dep_delay",
        "out_dep_delay_signed",
        "out_arr_delay",
        "available_turn",
        "planned_turn",
        "actual_turn",
        "turn_slack",
        "taxi_out",
        "taxi_in",
        "distance_group",
        "dep_time_blk",
        "arr_time_blk",
        "is_cancelled",
        "is_diverted",
        "carrier_delay",
        "weather_delay",
        "nas_delay",
        "late_aircraft_delay",
        "same_tail_turn_index",
    ]
    turn = pair[keep_cols].copy()
    if turn.empty:
        return turn
    turn = turn.sort_values(["tail", "sched_dep_dt"]).reset_index(drop=True)
    return add_context_features(add_recovery_labels(turn))


def add_recovery_labels(turn: pd.DataFrame, horizon_values: tuple[int, ...] = (2, 4, 6)) -> pd.DataFrame:
    turn = turn.copy()
    for h in horizon_values:
        turn[f"recover_h{h}"] = np.nan
        turn[f"fail_h{h}"] = np.nan
        turn[f"endpoint_obs_h{h}"] = 0
    group = turn.groupby("tail", sort=False)
    max_h = max(horizon_values)
    future_delay = {}
    future_bad = {}
    future_obs = {}
    base_obs = pd.Series(1, index=turn.index, dtype=float)
    bad = ((turn["is_cancelled"] > 0) | (turn["is_diverted"] > 0)).astype(float)
    for k in range(max_h):
        future_delay[k] = group["out_dep_delay"].shift(-k)
        future_bad[k] = bad.groupby(turn["tail"], sort=False).shift(-k)
        future_obs[k] = base_obs.groupby(turn["tail"], sort=False).shift(-k)
    for h in horizon_values:
        obs = pd.concat([future_obs[k] for k in range(h)], axis=1).notna().all(axis=1)
        delay_block = pd.concat([future_delay[k] for k in range(h)], axis=1)
        bad_block = pd.concat([future_bad[k] for k in range(h)], axis=1)
        recovered = delay_block.le(15).any(axis=1) & obs
        bad_terminal = bad_block.gt(0).any(axis=1) & obs
        recovered = recovered & ~bad_terminal
        failed = (~recovered | bad_terminal) & obs
        turn.loc[obs, f"recover_h{h}"] = recovered.loc[obs].astype(float)
        turn.loc[obs, f"fail_h{h}"] = failed.loc[obs].astype(float)
        turn.loc[obs, f"endpoint_obs_h{h}"] = 1
    return turn


def add_context_features(turn: pd.DataFrame) -> pd.DataFrame:
    turn = turn.copy()
    turn["airport_date"] = turn["airport"].astype(str) + "_" + turn["flight_date"].astype(str)
    turn["airport_hour"] = (
        turn["airport"].astype(str)
        + "_"
        + turn["flight_date"].astype(str)
        + "_"
        + turn["hour"].astype(str)
    )
    hour_stats = turn.groupby("airport_hour").agg(
        airport_hour_dep_rate=("episode_id", "size"),
        airport_hour_mean_dep_delay=("out_dep_delay", "mean"),
    )
    day_stats = turn.groupby("airport_date").agg(
        airport_day_weather_delay_share=("weather_delay", lambda s: float((s > 0).mean())),
        airport_day_late_aircraft_share=("late_aircraft_delay", lambda s: float((s > 0).mean())),
        airport_day_cancel_share=("is_cancelled", "mean"),
    )
    turn = turn.join(hour_stats, on="airport_hour")
    turn = turn.join(day_stats, on="airport_date")
    return turn


def train_model(train: pd.DataFrame, horizon: int, random_state: int):
    y = train[f"recover_h{horizon}"].astype(int)
    X = train[MODEL_NUMERIC + MODEL_CATEGORICAL].copy()
    numeric_pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    categorical_pipe = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore", min_frequency=10),
    )
    preprocess = ColumnTransformer(
        [
            ("num", numeric_pipe, MODEL_NUMERIC),
            ("cat", categorical_pipe, MODEL_CATEGORICAL),
        ]
    )
    model = LogisticRegression(max_iter=1000, C=0.5, class_weight="balanced", random_state=random_state)
    clf = make_pipeline(preprocess, model)
    clf.fit(X, y)
    return clf


def train_tree_model(train: pd.DataFrame, horizon: int, random_state: int):
    encoded = pd.get_dummies(train[MODEL_CATEGORICAL], dummy_na=True)
    X = pd.concat([train[MODEL_NUMERIC].reset_index(drop=True), encoded.reset_index(drop=True)], axis=1)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    y = train[f"recover_h{horizon}"].astype(int)
    clf = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=180,
        min_samples_leaf=40,
        l2_regularization=0.02,
        random_state=random_state,
    )
    clf.fit(X, y)
    return clf, X.columns.tolist(), X.median(numeric_only=True)


def predict_tree(model_tuple, df: pd.DataFrame) -> np.ndarray:
    clf, columns, med = model_tuple
    encoded = pd.get_dummies(df[MODEL_CATEGORICAL], dummy_na=True)
    X = pd.concat([df[MODEL_NUMERIC].reset_index(drop=True), encoded.reset_index(drop=True)], axis=1)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.reindex(columns=columns, fill_value=0)
    X = X.fillna(med)
    return clf.predict_proba(X)[:, 1]


def split_train_eval(turn: pd.DataFrame, mode: str, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible = turn[(turn[f"endpoint_obs_h{horizon}"] == 1) & (turn["out_dep_delay"].notna())].copy()
    eligible = eligible.sort_values("sched_dep_dt").reset_index(drop=True)
    if mode == "evaluation":
        cutoff = eligible["sched_dep_dt"].quantile(0.55)
        train = eligible[eligible["sched_dep_dt"] <= cutoff].copy()
        test = eligible[eligible["sched_dep_dt"] > cutoff].copy()
    else:
        months = sorted(eligible["month"].dropna().unique())
        if len(months) >= 4:
            split_month = months[len(months) // 2]
            train = eligible[eligible["month"] < split_month].copy()
            test = eligible[eligible["month"] >= split_month].copy()
        else:
            cutoff = eligible["sched_dep_dt"].quantile(0.6)
            train = eligible[eligible["sched_dep_dt"] <= cutoff].copy()
            test = eligible[eligible["sched_dep_dt"] > cutoff].copy()
    return train.reset_index(drop=True), test.reset_index(drop=True)


def build_donor_scores(
    test: pd.DataFrame,
    donor_window_minutes: int,
    max_donors_per_episode: int,
    horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    test = test.copy().reset_index(drop=True)
    stale_cols = [
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
    ]
    test = test.drop(columns=[col for col in stale_cols if col in test.columns])
    donor_rows: list[dict] = []
    summary_rows: list[dict] = []
    grouped = {
        key: g.sort_values("sched_dep_dt").reset_index(drop=True)
        for key, g in test.groupby(["airport", "carrier"], sort=False)
    }
    group_times = {
        key: g["sched_dep_dt"].reset_index(drop=True).to_numpy(dtype="datetime64[ns]")
        for key, g in grouped.items()
    }
    for idx, row in test.iterrows():
        if row["out_dep_delay"] < 15:
            summary_rows.append({"episode_id": row["episode_id"], "donor_count": 0})
            continue
        key = (row["airport"], row["carrier"])
        pool = grouped.get(key)
        if pool is None or pool.empty:
            summary_rows.append({"episode_id": row["episode_id"], "donor_count": 0})
            continue
        times = group_times[key]
        lower = row["sched_dep_dt"] - pd.Timedelta(minutes=donor_window_minutes)
        upper = row["sched_dep_dt"] + pd.Timedelta(minutes=donor_window_minutes)
        lo = int(np.searchsorted(times, np.datetime64(lower.to_datetime64(), "ns"), side="left"))
        hi = int(np.searchsorted(times, np.datetime64(upper.to_datetime64(), "ns"), side="right"))
        candidates = pool.iloc[lo:hi].copy()
        candidates = candidates[
            (candidates["tail"] != row["tail"])
            & (candidates["available_turn"] >= 35)
            & (candidates["is_cancelled"] == 0)
            & (candidates["is_diverted"] == 0)
        ]
        if pd.notna(row.get("distance_group", np.nan)):
            candidates = candidates[
                (candidates["distance_group"].isna())
                | ((candidates["distance_group"] - row["distance_group"]).abs() <= 2)
            ]
        if candidates.empty:
            summary_rows.append({"episode_id": row["episode_id"], "donor_count": 0})
            continue
        candidates["time_gap"] = (candidates["sched_dep_dt"] - row["sched_dep_dt"]).abs().dt.total_seconds() / 60.0
        candidates["slack_gap"] = (candidates["available_turn"] - row["available_turn"]).abs()
        candidates = candidates.sort_values(["time_gap", "slack_gap"], ascending=True).head(max_donors_per_episode)
        donor_recover = candidates[f"recover_h{horizon}"].mean()
        donor_pred_mean = candidates["pred_recover"].mean()
        donor_pred_max = candidates["pred_recover"].max()
        donor_actual_max = candidates[f"recover_h{horizon}"].max()
        summary_rows.append(
            {
                "episode_id": row["episode_id"],
                "donor_count": int(len(candidates)),
                "donor_pred_mean": float(donor_pred_mean),
                "donor_pred_max": float(donor_pred_max),
                "donor_actual_recover_mean": float(donor_recover),
                "donor_actual_recover_any": float(donor_actual_max),
                "donor_median_time_gap": float(candidates["time_gap"].median()),
                "donor_median_available_turn": float(candidates["available_turn"].median()),
            }
        )
        for _, cand in candidates.iterrows():
            donor_rows.append(
                {
                    "episode_id": row["episode_id"],
                    "donor_episode_id": cand["episode_id"],
                    "time_gap": float(cand["time_gap"]),
                    "donor_pred_recover": float(cand["pred_recover"]),
                    "donor_actual_recover": float(cand[f"recover_h{horizon}"]),
                }
            )
    donor_summary = pd.DataFrame(summary_rows)
    if donor_summary.empty:
        donor_summary = pd.DataFrame({"episode_id": test["episode_id"], "donor_count": 0})
    donor_edges = pd.DataFrame(donor_rows)
    merged = test.merge(donor_summary, on="episode_id", how="left")
    merged["donor_count"] = merged["donor_count"].fillna(0).astype(int)
    for col in [
        "donor_pred_mean",
        "donor_pred_max",
        "donor_actual_recover_mean",
        "donor_actual_recover_any",
        "donor_median_time_gap",
        "donor_median_available_turn",
    ]:
        if col not in merged:
            merged[col] = np.nan
    merged["ctrg_gap_mean"] = merged["donor_pred_mean"] - merged["pred_recover"]
    merged["ctrg_gap_max"] = merged["donor_pred_max"] - merged["pred_recover"]
    merged["supported"] = merged["donor_count"] > 0
    merged["stressed"] = merged["out_dep_delay"] >= 15
    merged["severe_start_delay"] = merged["out_dep_delay"] >= 60
    merged["structural_brittle"] = merged["stressed"] & merged["supported"] & (merged["ctrg_gap_max"] <= 0)
    merged["recoverable_despite_severe"] = (
        merged["severe_start_delay"] & merged["supported"] & (merged["donor_pred_max"] >= 0.70)
    )
    return merged, donor_edges


def top_slice_table(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    supported = df[(df["supported"]) & (df["stressed"])].copy()
    rows = []
    if supported.empty:
        return pd.DataFrame(rows)
    labels = {
        "ctrg_gap_max": "CTRG max-gap",
        "ctrg_gap_mean": "CTRG mean-gap",
        "pred_recover": "Observed-path recoverability",
        "out_dep_delay": "Delay-only",
        "available_turn": "Slack-only",
    }
    failure_label = f"fail_h{horizon}"
    base_fail = supported[failure_label].mean()
    base_recover = supported[f"recover_h{horizon}"].mean()
    for score, label in labels.items():
        ascending = score in {"pred_recover", "available_turn"}
        ranked = supported.sort_values(score, ascending=ascending)
        for frac in (0.05, 0.10, 0.20, 0.30):
            n = max(1, int(math.ceil(len(ranked) * frac)))
            top = ranked.head(n)
            rows.append(
                {
                    "ranking": label,
                    "slice": f"top_{int(frac * 100)}pct",
                    "n": int(n),
                    "coverage_of_supported_stressed": float(n / len(supported)),
                    "failure_rate": float(top[failure_label].mean()),
                    "failure_lift_vs_supported": float(top[failure_label].mean() / base_fail) if base_fail > 0 else np.nan,
                    "recovery_rate": float(top[f"recover_h{horizon}"].mean()),
                    "donor_actual_recover_mean": float(top["donor_actual_recover_mean"].mean()),
                    "mean_start_delay": float(top["out_dep_delay"].mean()),
                    "mean_pred_recover": float(top["pred_recover"].mean()),
                    "mean_donor_pred_max": float(top["donor_pred_max"].mean()),
                    "mean_gap_max": float(top["ctrg_gap_max"].mean()),
                    "base_failure_rate": float(base_fail),
                    "base_recovery_rate": float(base_recover),
                }
            )
    return pd.DataFrame(rows)


def monthly_table(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    rows = []
    for month, g in df.groupby("month"):
        stressed = g[g["stressed"]]
        supported = stressed[stressed["supported"]]
        if len(stressed) == 0:
            continue
        rows.append(
            {
                "month": int(month),
                "test_episodes": int(len(g)),
                "stressed_episodes": int(len(stressed)),
                "supported_stressed": int(len(supported)),
                "supported_share_stressed": float(len(supported) / len(stressed)),
                "failure_rate_supported_stressed": float(supported[f"fail_h{horizon}"].mean()) if len(supported) else np.nan,
                "mean_gap_max_supported": float(supported["ctrg_gap_max"].mean()) if len(supported) else np.nan,
                "severe_high_continuation_share": float(supported["recoverable_despite_severe"].mean()) if len(supported) else np.nan,
                "structural_brittle_share": float(supported["structural_brittle"].mean()) if len(supported) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def delay_band_table(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    supported = df[(df["supported"]) & (df["stressed"])].copy()
    if supported.empty:
        return pd.DataFrame()
    bins = [15, 30, 60, 120, 240, np.inf]
    labels = ["15-29", "30-59", "60-119", "120-239", "240+"]
    supported["delay_band"] = pd.cut(supported["out_dep_delay"], bins=bins, labels=labels, right=False)
    rows = []
    for band, g in supported.groupby("delay_band", observed=True):
        rows.append(
            {
                "delay_band_min": str(band),
                "episodes": int(len(g)),
                "failure_rate": float(g[f"fail_h{horizon}"].mean()),
                "mean_pred_recover": float(g["pred_recover"].mean()),
                "mean_donor_pred_max": float(g["donor_pred_max"].mean()),
                "mean_gap_max": float(g["ctrg_gap_max"].mean()),
                "high_continuation_share": float((g["donor_pred_max"] >= 0.70).mean()),
                "structural_brittle_share": float(g["structural_brittle"].mean()),
            }
        )
    return pd.DataFrame(rows)


def cause_table(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    supported = df[(df["supported"]) & (df["stressed"])].copy()
    if supported.empty:
        return pd.DataFrame()
    causes = {
        "carrier_delay": "Carrier",
        "weather_delay": "Weather",
        "nas_delay": "National Air System",
        "late_aircraft_delay": "Late aircraft",
    }
    cause_cols = list(causes.keys())
    supported["dominant_cause_col"] = supported[cause_cols].idxmax(axis=1)
    supported.loc[supported[cause_cols].max(axis=1) <= 0, "dominant_cause_col"] = "unreported_or_small"
    label_map = {**causes, "unreported_or_small": "Unreported or below reporting threshold"}
    rows = []
    for cause, g in supported.groupby("dominant_cause_col"):
        rows.append(
            {
                "dominant_delay_cause": label_map.get(cause, str(cause)),
                "episodes": int(len(g)),
                "failure_rate": float(g[f"fail_h{horizon}"].mean()),
                "mean_pred_recover": float(g["pred_recover"].mean()),
                "mean_donor_pred_max": float(g["donor_pred_max"].mean()),
                "mean_gap_max": float(g["ctrg_gap_max"].mean()),
                "severe_high_continuation_share": float(g["recoverable_despite_severe"].mean()),
                "structural_brittle_share": float(g["structural_brittle"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("episodes", ascending=False)


def bootstrap_top_slice(df: pd.DataFrame, horizon: int, random_state: int, n_boot: int = 300) -> pd.DataFrame:
    supported = df[(df["supported"]) & (df["stressed"])].copy()
    if supported.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(random_state)
    rows = []
    airports = supported["airport"].dropna().unique()
    for score, label, ascending in [
        ("ctrg_gap_max", "CTRG max-gap", False),
        ("out_dep_delay", "Delay-only", False),
        ("available_turn", "Slack-only", True),
        ("pred_recover", "Observed-path risk", True),
    ]:
        for frac in (0.05, 0.10, 0.20, 0.30):
            vals = []
            for _ in range(n_boot):
                sampled_airports = rng.choice(airports, size=len(airports), replace=True)
                sample = pd.concat(
                    [supported[supported["airport"] == airport] for airport in sampled_airports],
                    ignore_index=True,
                )
                ranked = sample.sort_values(score, ascending=ascending)
                n = max(1, int(math.ceil(len(ranked) * frac)))
                vals.append(float(ranked.head(n)[f"fail_h{horizon}"].mean()))
            rows.append(
                {
                    "ranking": label,
                    "slice": f"top_{int(frac * 100)}pct",
                    "failure_rate_boot_mean": float(np.mean(vals)),
                    "failure_rate_ci_low": float(np.quantile(vals, 0.025)),
                    "failure_rate_ci_high": float(np.quantile(vals, 0.975)),
                }
            )
    return pd.DataFrame(rows)


def airport_table(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    rows = []
    for airport, g in df.groupby("airport"):
        stressed = g[g["stressed"]]
        supported = stressed[stressed["supported"]]
        if len(stressed) == 0:
            continue
        rows.append(
            {
                "airport": airport,
                "test_episodes": int(len(g)),
                "stressed_episodes": int(len(stressed)),
                "supported_stressed": int(len(supported)),
                "supported_share_stressed": float(len(supported) / len(stressed)),
                "failure_rate_stressed": float(stressed[f"fail_h{horizon}"].mean()),
                "failure_rate_supported_stressed": float(supported[f"fail_h{horizon}"].mean()) if len(supported) else np.nan,
                "mean_gap_max_supported": float(supported["ctrg_gap_max"].mean()) if len(supported) else np.nan,
                "severe_high_continuation_share": float(supported["recoverable_despite_severe"].mean()) if len(supported) else np.nan,
                "structural_brittle_share": float(supported["structural_brittle"].mean()) if len(supported) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize_experiment(
    config: ExperimentConfig,
    turn: pd.DataFrame,
    train: pd.DataFrame,
    test_graph: pd.DataFrame,
    donor_edges: pd.DataFrame,
    auc: float,
    ap: float,
) -> dict:
    horizon = config.horizon
    stressed = test_graph[test_graph["stressed"]]
    supported = stressed[stressed["supported"]]
    severe = supported[supported["severe_start_delay"]]
    moderate = supported[(supported["out_dep_delay"] >= 15) & (supported["out_dep_delay"] < 60)]
    summary = {
        "config": asdict(config),
        "counts": {
            "raw_turnarounds": int(len(turn)),
            "train_endpoint_episodes": int(len(train)),
            "test_endpoint_episodes": int(len(test_graph)),
            "test_stressed_episodes": int(len(stressed)),
            "supported_stressed_episodes": int(len(supported)),
            "donor_edges": int(len(donor_edges)),
        },
        "model": {
            "recover_auc": float(auc),
            "recover_average_precision": float(ap),
        },
        "support": {
            "supported_share_among_stressed": float(len(supported) / len(stressed)) if len(stressed) else np.nan,
            "median_donor_count_supported": float(supported["donor_count"].median()) if len(supported) else np.nan,
            "median_donor_time_gap_minutes": float(supported["donor_median_time_gap"].median()) if len(supported) else np.nan,
        },
        "continuation_patterns": {
            "supported_stressed_failure_rate": float(supported[f"fail_h{horizon}"].mean()) if len(supported) else np.nan,
            "supported_stressed_recovery_rate": float(supported[f"recover_h{horizon}"].mean()) if len(supported) else np.nan,
            "mean_gap_max_supported": float(supported["ctrg_gap_max"].mean()) if len(supported) else np.nan,
            "severe_high_continuation_share": float(supported["recoverable_despite_severe"].mean()) if len(supported) else np.nan,
            "structural_brittle_share": float(supported["structural_brittle"].mean()) if len(supported) else np.nan,
            "severe_mean_donor_pred_max": float(severe["donor_pred_max"].mean()) if len(severe) else np.nan,
            "moderate_mean_donor_pred_max": float(moderate["donor_pred_max"].mean()) if len(moderate) else np.nan,
            "severe_count": int(len(severe)),
            "moderate_count": int(len(moderate)),
        },
    }
    return summary


def run(config: ExperimentConfig, root: Path) -> None:
    data_dir = root / "data" / "ctrg"
    raw_dir = data_dir / "raw_bts"
    processed_dir = data_dir / "processed"
    results_dir = root / "results" / "ctrg" / config.mode
    ensure_dirs(raw_dir, processed_dir, results_dir)

    airports = set(config.airports)
    frames = []
    for year in config.years:
        for month in config.months:
            zip_path = download_month(year, month, raw_dir)
            frames.append(read_bts_zip(zip_path, airports))
    flights = pd.concat(frames, ignore_index=True)
    log(f"Loaded flight rows after airport/tail filter: {len(flights):,}")

    turn = reconstruct_turnarounds(flights, airports, config.min_turn, config.max_turn)
    if turn.empty:
        raise RuntimeError("No turnarounds reconstructed.")
    turn.to_csv(processed_dir / f"{config.mode}_turnarounds.csv", index=False)
    log(f"Reconstructed turnarounds: {len(turn):,}")

    train, test = split_train_eval(turn, config.mode, config.horizon)
    if train.empty or test.empty:
        raise RuntimeError("Empty train or test split.")

    linear_model = train_model(train, config.horizon, config.random_state)
    tree_model = train_tree_model(train, config.horizon, config.random_state)
    test = test.copy()
    test["pred_recover_linear"] = linear_model.predict_proba(test[MODEL_NUMERIC + MODEL_CATEGORICAL])[:, 1]
    test["pred_recover_tree"] = predict_tree(tree_model, test)
    test["pred_recover"] = 0.5 * test["pred_recover_linear"] + 0.5 * test["pred_recover_tree"]
    y = test[f"recover_h{config.horizon}"].astype(int)
    auc = roc_auc_score(y, test["pred_recover"]) if y.nunique() > 1 else np.nan
    ap = average_precision_score(y, test["pred_recover"]) if y.nunique() > 1 else np.nan
    log(f"Observed-path recovery model: AUC={auc:.3f}, AP={ap:.3f}")

    test_graph, donor_edges = build_donor_scores(
        test,
        donor_window_minutes=config.donor_window_minutes,
        max_donors_per_episode=config.max_donors_per_episode,
        horizon=config.horizon,
    )
    top_table = top_slice_table(test_graph, config.horizon)
    monthly = monthly_table(test_graph, config.horizon)
    airport = airport_table(test_graph, config.horizon)
    delay_band = delay_band_table(test_graph, config.horizon)
    cause = cause_table(test_graph, config.horizon)
    boot = bootstrap_top_slice(test_graph, config.horizon, config.random_state, n_boot=150 if config.mode == "full" else 300)
    summary = summarize_experiment(config, turn, train, test_graph, donor_edges, auc, ap)

    test_graph.to_csv(results_dir / "episode_scores.csv", index=False)
    donor_edges.to_csv(results_dir / "donor_edges.csv", index=False)
    top_table.to_csv(results_dir / "top_slice_table.csv", index=False)
    airport.to_csv(results_dir / "airport_table.csv", index=False)
    monthly.to_csv(results_dir / "monthly_table.csv", index=False)
    delay_band.to_csv(results_dir / "delay_band_table.csv", index=False)
    cause.to_csv(results_dir / "cause_table.csv", index=False)
    boot.to_csv(results_dir / "top_slice_bootstrap.csv", index=False)
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(json.dumps(summary, indent=2))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["evaluation", "full"], default="evaluation")
    parser.add_argument("--years", nargs="+", type=int, default=[2024])
    parser.add_argument("--months", nargs="+", type=int, default=[1])
    parser.add_argument("--airports", nargs="+", default=["ATL", "DFW", "DEN", "ORD", "LAX"])
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--min-turn", type=int, default=35)
    parser.add_argument("--max-turn", type=int, default=720)
    parser.add_argument("--donor-window-minutes", type=int, default=120)
    parser.add_argument("--max-donors-per-episode", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=2026)
    return parser.parse_args(argv)


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    config = ExperimentConfig(
        mode=args.mode,
        years=args.years,
        months=args.months,
        airports=args.airports,
        horizon=args.horizon,
        min_turn=args.min_turn,
        max_turn=args.max_turn,
        donor_window_minutes=args.donor_window_minutes,
        max_donors_per_episode=args.max_donors_per_episode,
        random_state=args.random_state,
    )
    run(config, root)


if __name__ == "__main__":
    main(sys.argv[1:])
