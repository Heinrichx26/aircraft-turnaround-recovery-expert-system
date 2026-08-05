from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from frontier_methods import (
    FrontierMethod,
    evaluate_frontier_scores,
    score_erdem2024_delay_propagation,
    score_guo2026_uncertainty_z,
    score_rashedi2025_solution_space,
    score_sun2025_airline_overlay,
    score_tang2025_cascaded_gbm,
    score_wandelt2025_gari,
)
from frontier_methods.common import rank01
from episode_scoring import load_turnarounds
from operational_records import split_train_eval


ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = ROOT / "elsarticle" / "tables"


def load_episode_scores(path: Path, horizon: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    bool_cols = ["supported", "stressed", "severe_start_delay", "structural_brittle", "recoverable_despite_severe"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().eq("true") | df[col].eq(True)
    numeric_cols = [
        f"fail_h{horizon}",
        f"recover_h{horizon}",
        "pred_recover",
        "donor_count",
        "donor_pred_mean",
        "donor_pred_max",
        "donor_median_time_gap",
        "donor_median_available_turn",
        "ctrg_gap_mean",
        "ctrg_gap_max",
        "out_dep_delay",
        "available_turn",
        "carrier_delay",
        "weather_delay",
        "nas_delay",
        "late_aircraft_delay",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def kccres_evidence_certificate(eval_df: pd.DataFrame) -> pd.Series:
    gap = rank01(eval_df["ctrg_gap_max"], ascending=True)
    risk = rank01(1.0 - eval_df["pred_recover"], ascending=True)
    depth = rank01(eval_df["donor_count"], ascending=True)
    locality = rank01(-eval_df["donor_median_time_gap"], ascending=True)
    consensus = rank01(eval_df["donor_pred_mean"], ascending=True)
    turn_relief = rank01(eval_df["donor_median_available_turn"], ascending=True)
    return 0.72 * gap + 0.10 * risk + 0.06 * depth + 0.04 * locality + 0.04 * consensus + 0.04 * turn_relief


def kccres_dual_channel_certificate(eval_df: pd.DataFrame) -> pd.Series:
    gap = rank01(eval_df["ctrg_gap_max"], ascending=True)
    realized_risk = rank01(1.0 - eval_df["pred_recover"], ascending=True)
    return (gap ** 2.0) * ((realized_risk + 0.02) ** 0.75)


def build_methods(train: pd.DataFrame, test: pd.DataFrame, eval_df: pd.DataFrame, horizon: int, random_state: int) -> list[FrontierMethod]:
    methods: list[FrontierMethod] = [
        FrontierMethod(
            method="KCCRES dual-channel audit certificate",
            source="This study",
            journal="This study",
            year=2026,
            adaptation="Knowledge-certified nonlinear certificate combining recoverability-space contrast and realized-path failure pressure inside the CTRG support domain.",
            score=kccres_dual_channel_certificate(eval_df),
        ),
        FrontierMethod(
            method="KCCRES max-gap certificate",
            source="This study",
            journal="This study",
            year=2026,
            adaptation="Knowledge-certified counterfactual recoverability certificate using observed-path and best compatible continuation contrast.",
            score=eval_df["ctrg_gap_max"],
        ),
    ]
    scorers = [
        ("Tang et al. (Journal of Air Transport Management, 2025)", score_tang2025_cascaded_gbm),
        ("Erdem and Bilgic (Journal of Air Transport Management, 2024)", score_erdem2024_delay_propagation),
        ("Rashedi et al. (European Journal of Operational Research, 2025)", score_rashedi2025_solution_space),
        ("Wandelt et al. (Transportation Research Part D, 2025)", score_wandelt2025_gari),
        ("Sun et al. (Transportation Research Part E, 2025)", score_sun2025_airline_overlay),
        ("Guo et al. (Expert Systems with Applications, 2026)", score_guo2026_uncertainty_z),
    ]
    for label, scorer in scorers:
        print(f"Scoring adapted reproduction: {label}", flush=True)
        methods.append(scorer(train, test, eval_df, horizon, random_state))
    return methods


def pct(value: float) -> str:
    return "--" if pd.isna(value) else f"{100.0 * value:.2f}"


def num(value: float, digits: int = 3) -> str:
    return "--" if pd.isna(value) else f"{value:.{digits}f}"


def bold(text: str) -> str:
    return rf"\textbf{{{text}}}"


def write_frontier_table(top: pd.DataFrame) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    top10 = top[top["slice"].eq("top_10pct")].copy()
    max_fail = top10["failure_rate"].max()
    max_gap = top10["mean_gap"].max()
    max_severe = top10["severe_high_continuation_share"].max()
    lines = [
        r"\begin{table}[!tbp]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{1.4pt}",
        r"\renewcommand{\arraystretch}{1.04}",
        r"\caption{Full-scale comparison with recent research-family adaptations}",
        r"\label{tab:frontier-reproduction}",
        r"\begin{tabularx}{\linewidth}{@{}L{0.25\linewidth}L{0.19\linewidth}r r r r@{}}",
        r"\toprule",
        r"Method & Source journal and year & Failed-exit (\%) & Mean gap & Best cont. & Severe high-cont. (\%) \\",
        r"\midrule",
    ]
    for _, row in top10.iterrows():
        fail = pct(row.failure_rate)
        gap = num(row.mean_gap)
        severe = pct(row.severe_high_continuation_share)
        if row.failure_rate == max_fail:
            fail = bold(fail)
        if row.mean_gap == max_gap:
            gap = bold(gap)
        if row.severe_high_continuation_share == max_severe:
            severe = bold(severe)
        source = row["source"] if row["source"] == "This study" else f"{row['source']}; {row['journal']}"
        lines.append(f"{row.method} & {source} & {fail} & {gap} & {num(row.mean_best_continuation)} & {severe} \\\\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            r"\tablenote{All adapted methods are trained or constructed from the same public evidence available to the study. Failed-exit is the share of selected episodes whose four-turn endpoint remains delayed or reaches cancellation or diversion. Mean gap is best-continuation recoverability minus observed-path recoverability. Higher failed-exit rate indicates stronger realized-risk enrichment. Higher mean gap and severe high-continuation share indicate stronger recovery-space discovery. Bold marks the largest value in each reported outcome column.}",
            r"\end{table}",
        ]
    )
    (TABLE_DIR / "tab_frontier_reproduction.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_parameter_table(summary: pd.DataFrame) -> None:
    lines = [
        r"\begin{table}[!tbp]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{1.8pt}",
        r"\renewcommand{\arraystretch}{1.04}",
        r"\caption{Adapted frontier-method parameter settings}",
        r"\label{tab:supp-frontier-parameters}",
        r"\begin{tabularx}{\linewidth}{@{}L{0.22\linewidth}L{0.18\linewidth}Y@{}}",
        r"\toprule",
        r"Method & Source & Adapted parameter setting \\",
        r"\midrule",
    ]
    params = {
        "KCCRES dual-channel audit certificate": "Support domain defined by CTRG. Score is percentile-rank maximum recoverability gap raised to 2.00 multiplied by percentile-rank realized-path failure pressure plus 0.02 raised to 0.75; four-turn endpoint.",
        "KCCRES max-gap certificate": "Donor window 120 minutes; donor cap 20; same airport and carrier; different tail; minimum donor turn 35 minutes; distance-group tolerance two groups; four-turn endpoint.",
        "KCCRES multi-evidence certificate": "Weights: recoverability gap 0.72, observed-path risk 0.10, support depth 0.06, evidence locality 0.04, donor consensus 0.04, donor turn-time relief 0.04.",
        "Tang et al. 2025 cascaded gradient boosting": "Two histogram gradient-boosting stages; stage 1 uses 220 boosting iterations, learning rate 0.04, and minimum leaf size 35; stage 2 uses 240 boosting iterations, learning rate 0.035, and minimum leaf size 30; the stage 1 score enters stage 2.",
        "Erdem and Bilgic 2024 propagation learner": "Lagged airport, carrier, airport-carrier, and route pressure features at one-hour and two-hour lags; histogram gradient boosting uses 220 boosting iterations, learning rate 0.04, and minimum leaf size 35.",
        "Rashedi et al. 2025 machine-learning reduction": "Histogram gradient-boosting failed-exit prescreener with support penalty; weights: failed-exit probability 0.62, support weakness 0.14, locality weakness 0.10, slack pressure 0.08, delay pressure 0.06.",
        "Wandelt et al. 2025 GARI adaptation": "Airport resilience pressure from route count, carrier count, tail count, route entropy, delay share, and cancellation share; combined with delay, slack, and donor support weakness.",
        "Sun et al. 2025 airline-overlay resilience": "Carrier-airport overlay fragility from top-route share, route breadth, tail diversity, route entropy, and carrier-airport departures; combined with delay, realized-path risk, and support weakness.",
        "Guo et al. 2026 uncertainty-aware multi-criteria decision support": "Six bootstrap histogram gradient-boosting models; score combines ensemble mean failed-exit risk 0.58, epistemic spread 0.18, support-confidence weakness 0.14, and reliability penalty 0.10.",
    }
    for _, row in summary.iterrows():
        source = row.source if row.source == "This study" else f"{row.source}; {row.journal}, {int(row.year)}"
        lines.append(f"{row.method} & {source} & {params.get(row.method, row.adaptation)} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}"])
    (TABLE_DIR / "supp_tab_frontier_parameters.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    turn = load_turnarounds(Path(args.turnarounds), args.horizon)
    train, test = split_train_eval(turn, args.split_mode, args.horizon)
    scores = load_episode_scores(Path(args.episode_scores), args.horizon)
    eval_df = scores[scores["stressed"] & scores["supported"]].copy().reset_index(drop=True)
    print(
        f"Loaded turnarounds={len(turn):,}; train={len(train):,}; test={len(test):,}; "
        f"supported stressed episodes={len(eval_df):,}",
        flush=True,
    )
    methods = build_methods(train, test, eval_df, args.horizon, args.random_state)
    summary, top = evaluate_frontier_scores(eval_df, methods, args.horizon)
    summary.to_csv(out_dir / "frontier_reproduction_summary.csv", index=False)
    top.to_csv(out_dir / "frontier_reproduction_top_slices.csv", index=False)
    metadata = {
        "turnarounds": str(args.turnarounds),
        "episode_scores": str(args.episode_scores),
        "split_mode": args.split_mode,
        "horizon": args.horizon,
        "random_state": args.random_state,
        "supported_stressed_episodes": int(len(eval_df)),
    }
    (out_dir / "frontier_reproduction_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if args.write_tables:
        table_top = top.merge(summary[["method", "journal", "year"]], on="method", how="left")
        write_frontier_table(table_top)
        write_parameter_table(summary)
    print(summary.round(4).to_string(index=False))
    print(top[top["slice"].eq("top_10pct")].round(4).to_string(index=False))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turnarounds", required=True)
    parser.add_argument("--episode-scores", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-mode", default="full")
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=2026)
    parser.add_argument("--write-tables", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args(sys.argv[1:]))
