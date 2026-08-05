from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


USECOLS = [
    "episode_id",
    "tail",
    "carrier",
    "airport",
    "sched_dep_dt",
    "out_dep_delay",
    "available_turn",
    "distance_group",
    "recover_h4",
    "fail_h4",
    "pred_recover",
    "is_cancelled",
    "is_diverted",
]


def log(message: str) -> None:
    print(message, flush=True)


def load_episode_scores(
    path: Path,
    *,
    month: int = 0,
    max_rows: int | None = None,
) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    usecols = [col for col in USECOLS if col in set(header.columns)]
    df = pd.read_csv(path, usecols=usecols, nrows=max_rows, low_memory=False)
    df["sched_dep_dt"] = pd.to_datetime(df["sched_dep_dt"], errors="coerce")
    if month > 0:
        df = df[df["sched_dep_dt"].dt.month.eq(month)].copy()
    for col in [
        "out_dep_delay",
        "available_turn",
        "distance_group",
        "recover_h4",
        "fail_h4",
        "pred_recover",
        "is_cancelled",
        "is_diverted",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "is_cancelled" not in df.columns:
        df["is_cancelled"] = 0.0
    if "is_diverted" not in df.columns:
        df["is_diverted"] = 0.0
    for col in ["episode_id", "tail", "carrier", "airport"]:
        df[col] = df[col].fillna("UNK").astype(str)
    df = df[df["sched_dep_dt"].notna() & df["pred_recover"].notna()].copy()
    return df.reset_index(drop=True)


def add_endpoint_close_times(test: pd.DataFrame, turnarounds_path: Path) -> pd.DataFrame:
    cols = ["episode_id", "tail", "sched_dep_dt"]
    turn = pd.read_csv(turnarounds_path, usecols=cols, low_memory=False)
    turn["sched_dep_dt"] = pd.to_datetime(turn["sched_dep_dt"], errors="coerce")
    turn = turn[turn["sched_dep_dt"].notna()].sort_values(["tail", "sched_dep_dt"]).reset_index(drop=True)
    turn["endpoint_close_dt_h4"] = turn.groupby("tail", sort=False)["sched_dep_dt"].shift(-3)
    close = turn[["episode_id", "endpoint_close_dt_h4"]].copy()
    merged = test.merge(close, on="episode_id", how="left")
    missing = merged["endpoint_close_dt_h4"].isna().sum()
    if missing:
        log(f"Endpoint-close time missing for {missing:,} rows; these rows remain out of the historical library.")
    return merged


def build_historical_library_scores(
    test: pd.DataFrame,
    *,
    donor_window_minutes: int,
    max_donors_per_episode: int,
) -> tuple[pd.DataFrame, int]:
    test = test.sort_values(["airport", "carrier", "sched_dep_dt"]).reset_index(drop=True)
    n = len(test)
    donor_count = np.zeros(n, dtype=np.int16)
    donor_pred_max = np.full(n, np.nan, dtype=float)
    donor_pred_mean = np.full(n, np.nan, dtype=float)
    donor_actual_mean = np.full(n, np.nan, dtype=float)
    donor_median_time_gap = np.full(n, np.nan, dtype=float)
    donor_median_available_turn = np.full(n, np.nan, dtype=float)

    window_ns = np.int64(donor_window_minutes) * np.int64(60_000_000_000)
    edge_count = 0
    groups = list(test.groupby(["airport", "carrier"], sort=False).indices.items())
    for group_number, (_, idx_values) in enumerate(groups, start=1):
        idx = np.asarray(idx_values, dtype=np.int64)
        g = test.iloc[idx].sort_values("sched_dep_dt")
        orig_idx = g.index.to_numpy(dtype=np.int64)
        sched = g["sched_dep_dt"].to_numpy(dtype="datetime64[ns]").astype("int64")
        minute_of_day = (
            g["sched_dep_dt"].dt.hour.to_numpy(dtype=np.int64) * 60
            + g["sched_dep_dt"].dt.minute.to_numpy(dtype=np.int64)
        )
        close = g["endpoint_close_dt_h4"].to_numpy(dtype="datetime64[ns]").astype("int64")
        tail = g["tail"].to_numpy(dtype=str)
        avail = g["available_turn"].to_numpy(dtype=float)
        dist = g["distance_group"].to_numpy(dtype=float)
        pred = g["pred_recover"].to_numpy(dtype=float)
        actual = g["recover_h4"].to_numpy(dtype=float)
        delay = g["out_dep_delay"].to_numpy(dtype=float)
        cancel = g["is_cancelled"].fillna(0).to_numpy(dtype=float)
        divert = g["is_diverted"].fillna(0).to_numpy(dtype=float)

        valid_base = (
            np.isfinite(close)
            & np.isfinite(pred)
            & (avail >= 35)
            & (cancel == 0)
            & (divert == 0)
        )
        stressed_positions = np.flatnonzero(delay >= 15)
        for pos in stressed_positions:
            cand = np.arange(0, pos, dtype=np.int64)
            if cand.size == 0:
                continue
            clock_gap = np.abs(minute_of_day[cand] - minute_of_day[pos])
            clock_gap = np.minimum(clock_gap, 1440 - clock_gap)
            mask = (
                valid_base[cand]
                & (close[cand] < sched[pos])
                & (clock_gap <= donor_window_minutes)
                & (tail[cand] != tail[pos])
            )
            if np.isfinite(dist[pos]):
                mask &= (~np.isfinite(dist[cand])) | (np.abs(dist[cand] - dist[pos]) <= 2)
            cand = cand[mask]
            if cand.size == 0:
                continue
            time_gap = np.minimum(
                np.abs(minute_of_day[cand] - minute_of_day[pos]),
                1440 - np.abs(minute_of_day[cand] - minute_of_day[pos]),
            ).astype(float)
            slack_gap = np.abs(avail[cand] - avail[pos])
            order = np.lexsort((slack_gap, time_gap))
            cand = cand[order[:max_donors_per_episode]]
            time_gap = time_gap[order[:max_donors_per_episode]]
            row_id = orig_idx[pos]
            donor_count[row_id] = int(cand.size)
            donor_pred_max[row_id] = float(np.nanmax(pred[cand]))
            donor_pred_mean[row_id] = float(np.nanmean(pred[cand]))
            donor_actual_mean[row_id] = float(np.nanmean(actual[cand]))
            donor_median_time_gap[row_id] = float(np.nanmedian(time_gap))
            donor_median_available_turn[row_id] = float(np.nanmedian(avail[cand]))
            edge_count += int(cand.size)
        if group_number % 50 == 0 or group_number == len(groups):
            log(f"Historical-library groups {group_number}/{len(groups)}; edges={edge_count:,}")

    out = test.copy()
    out["hist_donor_count"] = donor_count
    out["hist_donor_pred_max"] = donor_pred_max
    out["hist_donor_pred_mean"] = donor_pred_mean
    out["hist_donor_actual_recover_mean"] = donor_actual_mean
    out["hist_donor_median_time_gap"] = donor_median_time_gap
    out["hist_donor_median_available_turn"] = donor_median_available_turn
    out["hist_supported"] = out["hist_donor_count"] > 0
    out["hist_gap_max"] = out["hist_donor_pred_max"] - out["pred_recover"]
    return out, edge_count


def summarize(df: pd.DataFrame, edge_count: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    stressed = df[df["out_dep_delay"] >= 15].copy()
    supported = stressed[stressed["hist_supported"]].copy()
    if supported.empty:
        raise RuntimeError("Compatible-history evaluation found no supported stressed episodes.")
    base_failure = float(supported["fail_h4"].mean())
    rows = [
        {
            "mode": "Compatible-history evaluation",
            "stressed_episodes": int(len(stressed)),
            "supported_stressed": int(len(supported)),
            "support_share": float(len(supported) / len(stressed)),
            "donor_edges": int(edge_count),
            "median_donor_count": float(supported["hist_donor_count"].median()),
            "median_time_gap": float(supported["hist_donor_median_time_gap"].median()),
            "reference_failure": base_failure,
            "mean_gap": float(supported["hist_gap_max"].mean()),
            "mean_best_continuation": float(supported["hist_donor_pred_max"].mean()),
        }
    ]
    ranking_specs = [
        ("Historical CTRG max-gap", "hist_gap_max", False),
        ("Observed-path risk", "pred_recover", True),
        ("Delay-only", "out_dep_delay", False),
        ("Slack-only", "available_turn", True),
    ]
    top_rows = []
    for label, score, ascending in ranking_specs:
        ranked = supported.sort_values(score, ascending=ascending)
        for frac in (0.05, 0.10, 0.20, 0.30):
            n = max(1, int(np.ceil(len(ranked) * frac)))
            top = ranked.head(n)
            top_rows.append(
                {
                    "ranking": label,
                    "slice": f"top_{int(frac * 100)}pct",
                    "n": int(n),
                    "failure_rate": float(top["fail_h4"].mean()),
                    "lift": float(top["fail_h4"].mean() / base_failure) if base_failure > 0 else np.nan,
                    "mean_gap": float(top["hist_gap_max"].mean()),
                    "mean_best_continuation": float(top["hist_donor_pred_max"].mean()),
                    "mean_start_delay": float(top["out_dep_delay"].mean()),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(top_rows)


def run(args: argparse.Namespace) -> None:
    out_dir = ROOT / "results" / "ctrg" / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    test = load_episode_scores(
        ROOT / args.episode_scores,
        month=args.month,
        max_rows=args.max_rows,
    )
    log(f"Loaded evaluation rows: {len(test):,}")
    test = add_endpoint_close_times(test, ROOT / args.turnarounds)
    scored, edge_count = build_historical_library_scores(
        test,
        donor_window_minutes=args.donor_window_minutes,
        max_donors_per_episode=args.max_donors_per_episode,
    )
    summary, top = summarize(scored, edge_count)
    keep_cols = [
        "episode_id",
        "sched_dep_dt",
        "carrier",
        "airport",
        "out_dep_delay",
        "pred_recover",
        "fail_h4",
        "hist_supported",
        "hist_donor_count",
        "hist_donor_pred_max",
        "hist_donor_pred_mean",
        "hist_donor_actual_recover_mean",
        "hist_gap_max",
        "hist_donor_median_time_gap",
        "hist_donor_median_available_turn",
    ]
    scored[[c for c in keep_cols if c in scored.columns]].to_csv(out_dir / "historical_library_episode_scores.csv", index=False)
    summary.to_csv(out_dir / "historical_library_summary.csv", index=False)
    top.to_csv(out_dir / "historical_library_top_slices.csv", index=False)
    log(summary.to_string(index=False))
    log(top[top["slice"] == "top_10pct"].to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-scores", default="results/ctrg/full/episode_scores.csv")
    parser.add_argument("--turnarounds", default="data/ctrg/processed/full_turnarounds.csv")
    parser.add_argument("--out-dir", default="compatible_history")
    parser.add_argument("--donor-window-minutes", type=int, default=120)
    parser.add_argument("--max-donors-per-episode", type=int, default=20)
    parser.add_argument("--month", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
