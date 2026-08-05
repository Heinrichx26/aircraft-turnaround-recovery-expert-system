from __future__ import annotations

import pandas as pd

from .common import BASE_CATEGORICAL, BASE_NUMERIC, FrontierMethod, fit_hgb_classifier, predict_hgb, rank01


def score_rashedi2025_solution_space(train: pd.DataFrame, test: pd.DataFrame, eval_df: pd.DataFrame, horizon: int, random_state: int) -> FrontierMethod:
    model = fit_hgb_classifier(
        train,
        f"fail_h{horizon}",
        BASE_NUMERIC,
        BASE_CATEGORICAL,
        random_state + 1301,
        max_iter=240,
        learning_rate=0.035,
        min_samples_leaf=30,
        l2_regularization=0.05,
    )
    fail_all = pd.Series(predict_hgb(model, test), index=test["episode_id"])
    local = eval_df.copy()
    local["fail_score"] = local["episode_id"].map(fail_all)
    support_penalty = 1.0 - rank01(local["donor_count"], ascending=True)
    locality_penalty = 1.0 - rank01(-local["donor_median_time_gap"], ascending=True)
    slack_pressure = rank01(-local["available_turn"], ascending=True)
    delay_pressure = rank01(local["out_dep_delay"], ascending=True)
    score = 0.62 * local["fail_score"] + 0.14 * support_penalty + 0.10 * locality_penalty + 0.08 * slack_pressure + 0.06 * delay_pressure
    return FrontierMethod(
        method="Rashedi et al. 2025 machine-learning reduction",
        source="Rashedi et al. (2025)",
        journal="European Journal of Operational Research",
        year=2025,
        adaptation="Machine-learning solution-space reduction adapted as a failed-exit prescreening score penalized by weak donor support and time-locality.",
        score=score,
    )
