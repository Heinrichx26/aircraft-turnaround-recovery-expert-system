from __future__ import annotations

import pandas as pd

from .common import BASE_CATEGORICAL, BASE_NUMERIC, FrontierMethod, fit_hgb_classifier, predict_hgb


def score_tang2025_cascaded_gbm(train: pd.DataFrame, test: pd.DataFrame, eval_df: pd.DataFrame, horizon: int, random_state: int) -> FrontierMethod:
    numeric_stage1 = BASE_NUMERIC
    stage1 = fit_hgb_classifier(
        train,
        f"fail_h{horizon}",
        numeric_stage1,
        BASE_CATEGORICAL,
        random_state + 1101,
        max_iter=220,
        learning_rate=0.04,
        min_samples_leaf=35,
        l2_regularization=0.04,
    )
    train_stage = train.copy()
    test_stage = test.copy()
    train_stage["stage1_fail_pressure"] = predict_hgb(stage1, train)
    test_stage["stage1_fail_pressure"] = predict_hgb(stage1, test)
    numeric_stage2 = BASE_NUMERIC + ["stage1_fail_pressure"]
    stage2 = fit_hgb_classifier(
        train_stage,
        f"fail_h{horizon}",
        numeric_stage2,
        BASE_CATEGORICAL,
        random_state + 1102,
        max_iter=240,
        learning_rate=0.035,
        min_samples_leaf=30,
        l2_regularization=0.05,
    )
    score_all = pd.Series(predict_hgb(stage2, test_stage), index=test_stage["episode_id"])
    score = eval_df["episode_id"].map(score_all)
    return FrontierMethod(
        method="Tang et al. 2025 cascaded gradient boosting",
        source="Tang et al. (2025)",
        journal="Journal of Air Transport Management",
        year=2025,
        adaptation="Two-stage cascaded gradient boosting adapted from turnaround milestone prediction to four-turn failed-exit ranking.",
        score=score,
    )
