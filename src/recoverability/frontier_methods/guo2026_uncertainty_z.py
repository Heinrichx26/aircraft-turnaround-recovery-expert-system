from __future__ import annotations

import numpy as np
import pandas as pd

from .common import BASE_CATEGORICAL, BASE_NUMERIC, FrontierMethod, fit_hgb_classifier, predict_hgb, rank01


def score_guo2026_uncertainty_z(train: pd.DataFrame, test: pd.DataFrame, eval_df: pd.DataFrame, horizon: int, random_state: int) -> FrontierMethod:
    preds = []
    train_work = train[train[f"fail_h{horizon}"].notna()].reset_index(drop=True)
    rng = np.random.default_rng(random_state + 1601)
    for idx in range(6):
        sample_idx = rng.choice(len(train_work), size=max(100, int(0.78 * len(train_work))), replace=True)
        sample = train_work.iloc[sample_idx].copy()
        model = fit_hgb_classifier(
            sample,
            f"fail_h{horizon}",
            BASE_NUMERIC,
            BASE_CATEGORICAL,
            random_state + 1601 + idx,
            max_iter=170,
            learning_rate=0.045,
            min_samples_leaf=35,
            l2_regularization=0.04,
        )
        preds.append(predict_hgb(model, test))
    pred = np.vstack(preds)
    mean_fail = pd.Series(pred.mean(axis=0), index=test["episode_id"])
    epistemic = pd.Series(pred.std(axis=0), index=test["episode_id"])
    local = eval_df.copy()
    mean_component = local["episode_id"].map(mean_fail)
    uncertainty_component = local["episode_id"].map(epistemic)
    support_confidence = rank01(local["donor_count"], ascending=True) * rank01(-local["donor_median_time_gap"], ascending=True)
    z_reliability = 1.0 - uncertainty_component.fillna(uncertainty_component.median())
    score = 0.58 * mean_component + 0.18 * uncertainty_component + 0.14 * (1.0 - support_confidence) + 0.10 * (1.0 - z_reliability)
    return FrontierMethod(
        method="Guo et al. 2026 uncertainty-aware multi-criteria decision support",
        source="Guo et al. (2026)",
        journal="Expert Systems with Applications",
        year=2026,
        adaptation="Fuzzy extended Z-number decision logic adapted as ensemble mean risk, epistemic spread, and support-confidence reliability aggregation.",
        score=score,
    )
