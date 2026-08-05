from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass(frozen=True)
class FrontierMethod:
    method: str
    source: str
    journal: str
    year: int
    adaptation: str
    score: pd.Series


BASE_NUMERIC = [
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

BASE_CATEGORICAL = ["airport", "carrier", "dep_time_blk", "arr_time_blk"]
DEFAULT_MAX_TRAIN_ROWS = None


def rank01(series: pd.Series, ascending: bool = True) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.rank(pct=True, ascending=ascending).fillna(0.5)


def encode_frame(df: pd.DataFrame, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    x_num = df[numeric].apply(pd.to_numeric, errors="coerce")
    x_cat = pd.get_dummies(df[categorical].fillna("UNK").astype(str), dummy_na=True)
    x = pd.concat([x_num.reset_index(drop=True), x_cat.reset_index(drop=True)], axis=1)
    return x.replace([np.inf, -np.inf], np.nan)


def stratified_training_frame(
    train: pd.DataFrame,
    target: str,
    random_state: int,
    max_train_rows: int | None = DEFAULT_MAX_TRAIN_ROWS,
) -> pd.DataFrame:
    work = train[train[target].notna()].copy()
    if max_train_rows is None or len(work) <= max_train_rows:
        return work
    rng = np.random.default_rng(random_state)
    pieces = []
    for _, group in work.groupby(target, dropna=False):
        take = max(1, int(round(max_train_rows * len(group) / len(work))))
        take = min(take, len(group))
        seed = int(rng.integers(0, np.iinfo(np.int32).max))
        pieces.append(group.sample(n=take, random_state=seed))
    sampled = pd.concat(pieces, ignore_index=True)
    if len(sampled) > max_train_rows:
        sampled = sampled.sample(n=max_train_rows, random_state=random_state)
    return sampled.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def fit_hgb_classifier(
    train: pd.DataFrame,
    target: str,
    numeric: list[str],
    categorical: list[str],
    random_state: int,
    *,
    max_iter: int = 180,
    learning_rate: float = 0.045,
    min_samples_leaf: int = 40,
    l2_regularization: float = 0.03,
    max_train_rows: int | None = DEFAULT_MAX_TRAIN_ROWS,
) -> tuple[HistGradientBoostingClassifier, list[str], pd.Series, list[str], list[str]]:
    work = stratified_training_frame(train, target, random_state, max_train_rows=max_train_rows)
    y = work[target].astype(int)
    x = encode_frame(work, numeric, categorical)
    med = x.median(numeric_only=True)
    x = x.fillna(med)
    clf = HistGradientBoostingClassifier(
        learning_rate=learning_rate,
        max_iter=max_iter,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        random_state=random_state,
    )
    clf.fit(x, y)
    return clf, x.columns.tolist(), med, numeric, categorical


def predict_hgb(model_tuple, df: pd.DataFrame) -> np.ndarray:
    clf, columns, med, numeric, categorical = model_tuple
    x = encode_frame(df, numeric, categorical).reindex(columns=columns, fill_value=0)
    x = x.fillna(med)
    return clf.predict_proba(x)[:, 1]


def normalize_by_train(train: pd.Series, test: pd.Series) -> pd.Series:
    tr = pd.to_numeric(train, errors="coerce")
    te = pd.to_numeric(test, errors="coerce")
    lo = float(np.nanpercentile(tr, 5)) if tr.notna().any() else 0.0
    hi = float(np.nanpercentile(tr, 95)) if tr.notna().any() else 1.0
    if hi <= lo:
        hi = lo + 1.0
    return ((te - lo) / (hi - lo)).clip(0, 1).fillna(0.5)


def evaluate_frontier_scores(eval_df: pd.DataFrame, methods: list[FrontierMethod], horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    fail_col = f"fail_h{horizon}"
    y = eval_df[fail_col].astype(int)
    base_failure = float(y.mean())
    fail_total = float(y.sum())
    severe_total = float(eval_df["recoverable_despite_severe"].sum())
    summary_rows = []
    top_rows = []
    for method in methods:
        score = pd.to_numeric(method.score, errors="coerce")
        valid = score.notna()
        scoped = eval_df.loc[valid].copy()
        scoped_score = score.loc[valid]
        yy = scoped[fail_col].astype(int)
        auc = roc_auc_score(yy, scoped_score) if yy.nunique() > 1 else np.nan
        ap = average_precision_score(yy, scoped_score) if yy.nunique() > 1 else np.nan
        summary_rows.append(
            {
                "method": method.method,
                "source": method.source,
                "journal": method.journal,
                "year": method.year,
                "adaptation": method.adaptation,
                "evaluated_episodes": int(len(scoped)),
                "missing_share": float(1.0 - len(scoped) / len(eval_df)),
                "failure_auc": float(auc),
                "failure_average_precision": float(ap),
                "base_failure_rate": base_failure,
            }
        )
        order = scoped_score.sort_values(ascending=False).index
        for frac in (0.05, 0.10, 0.20, 0.30):
            n = max(1, int(np.ceil(len(order) * frac)))
            picked = scoped.loc[order[:n]]
            failure_rate = float(picked[fail_col].mean())
            top_rows.append(
                {
                    "method": method.method,
                    "source": method.source,
                    "slice": f"top_{int(frac * 100)}pct",
                    "n": int(n),
                    "failure_rate": failure_rate,
                    "failure_lift": float(failure_rate / base_failure) if base_failure > 0 else np.nan,
                    "failure_capture": float(picked[fail_col].sum() / fail_total) if fail_total > 0 else np.nan,
                    "mean_gap": float(picked["ctrg_gap_max"].mean()),
                    "mean_best_continuation": float(picked["donor_pred_max"].mean()),
                    "severe_high_continuation_share": float(picked["recoverable_despite_severe"].mean()),
                    "severe_high_continuation_capture": float(picked["recoverable_despite_severe"].sum() / severe_total) if severe_total > 0 else np.nan,
                    "mean_start_delay": float(picked["out_dep_delay"].mean()),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(top_rows)
