from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from relational_features import (
    RELATIONS,
    base_features,
    evaluate,
    load_month,
    rank01,
    relation_key,
    split_time,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "results" / "ctrg" / "full" / "episode_scores.csv"
DEFAULT_OUTPUT = ROOT / "results" / "ctrg" / "temporal_relation_model"


def capacity_reward(df: pd.DataFrame, capacity: float = 0.10) -> np.ndarray:
    failure = df["fail_h4"].to_numpy(dtype=float)
    severe = df["recoverable_despite_severe"].astype(float).to_numpy()
    gap = np.clip(df["ctrg_gap_max"].to_numpy(dtype=float) / 0.40, 0.0, 1.0)
    prevalence = max(float(failure.mean()), 1e-6)
    failure_weight = 0.35 + 0.25 * capacity / prevalence
    return failure_weight * failure + 0.25 * severe + 0.15 * gap


def relation_statistics(
    training: pd.DataFrame,
    other: pd.DataFrame,
    reward: np.ndarray,
    smoothing: float,
    leave_one_out: bool,
) -> tuple[np.ndarray, list[str]]:
    signal_names = [
        "capacity_reward",
        "failed_exit",
        "severe_high_continuation",
        "recoverability_gap",
        "observed_path_risk",
        "best_continuation",
    ]
    signals = np.column_stack(
        [
            reward,
            training["fail_h4"].to_numpy(dtype=float),
            training["recoverable_despite_severe"].astype(float).to_numpy(),
            np.clip(training["ctrg_gap_max"].to_numpy(dtype=float) / 0.40, 0.0, 1.0),
            1.0 - training["pred_recover"].to_numpy(dtype=float),
            training["donor_pred_max"].to_numpy(dtype=float),
        ]
    )
    output = np.zeros((len(other), len(RELATIONS), len(signal_names) + 1), dtype=np.float64)
    for relation_id, columns in enumerate(RELATIONS):
        train_key = relation_key(training, columns)
        other_key = relation_key(other, columns)
        counts = train_key.value_counts()
        mapped_counts = other_key.map(counts).fillna(0.0).to_numpy(dtype=float)
        if leave_one_out:
            mapped_counts = np.maximum(mapped_counts - 1.0, 0.0)
        output[:, relation_id, -1] = np.log1p(mapped_counts)
        for signal_id in range(signals.shape[1]):
            frame = pd.DataFrame({"key": train_key, "value": signals[:, signal_id]})
            sums = frame.groupby("key")["value"].sum()
            mapped_sums = other_key.map(sums).fillna(0.0).to_numpy(dtype=float)
            if leave_one_out:
                mapped_sums = mapped_sums - signals[:, signal_id]
            prior = float(signals[:, signal_id].mean())
            output[:, relation_id, signal_id] = (
                mapped_sums + smoothing * prior
            ) / (mapped_counts + smoothing)
    return output, signal_names + ["log_support"]


@dataclass
class Standardizer:
    median: np.ndarray
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "Standardizer":
        median = np.nanmedian(values, axis=0)
        filled = np.where(np.isnan(values), median, values)
        mean = filled.mean(axis=0)
        scale = filled.std(axis=0)
        scale = np.where(scale < 1e-6, 1.0, scale)
        return cls(median, mean, scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        filled = np.where(np.isnan(values), self.median, values)
        return (filled - self.mean) / self.scale


def prepare_features(
    training: pd.DataFrame,
    tuning: pd.DataFrame,
    validation: pd.DataFrame,
    smoothing: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    reward = capacity_reward(training)
    base_train_frame = base_features(training)
    base_tune_frame = base_features(tuning)
    base_valid_frame = base_features(validation)
    base_scaler = Standardizer.fit(base_train_frame.to_numpy(dtype=float))
    base_train = base_scaler.transform(base_train_frame.to_numpy(dtype=float))
    base_tune = base_scaler.transform(base_tune_frame.to_numpy(dtype=float))
    base_valid = base_scaler.transform(base_valid_frame.to_numpy(dtype=float))

    token_train, token_names = relation_statistics(
        training, training, reward, smoothing=smoothing, leave_one_out=True
    )
    token_tune, _ = relation_statistics(
        training, tuning, reward, smoothing=smoothing, leave_one_out=False
    )
    token_valid, _ = relation_statistics(
        training, validation, reward, smoothing=smoothing, leave_one_out=False
    )
    flat_scaler = Standardizer.fit(token_train.reshape(-1, token_train.shape[-1]))
    token_train = flat_scaler.transform(token_train.reshape(-1, token_train.shape[-1])).reshape(
        token_train.shape
    )
    token_tune = flat_scaler.transform(token_tune.reshape(-1, token_tune.shape[-1])).reshape(
        token_tune.shape
    )
    token_valid = flat_scaler.transform(token_valid.reshape(-1, token_valid.shape[-1])).reshape(
        token_valid.shape
    )
    return (
        base_train,
        base_tune,
        base_valid,
        token_train,
        token_tune,
        token_valid,
        list(base_train_frame.columns),
        token_names,
    )


@dataclass
class AttentionParameters:
    token_projection: np.ndarray
    context_projection: np.ndarray
    relation_embedding: np.ndarray
    attention_vector: np.ndarray
    relation_bias: np.ndarray
    output_hidden: np.ndarray
    output_base: np.ndarray
    output_bias: np.ndarray


def initialize_parameters(
    base_dim: int,
    token_dim: int,
    relation_count: int,
    hidden_dim: int,
    seed: int,
) -> AttentionParameters:
    rng = np.random.default_rng(seed)
    return AttentionParameters(
        token_projection=rng.normal(0.0, 1.0 / math.sqrt(token_dim), (token_dim, hidden_dim)),
        context_projection=rng.normal(0.0, 0.25 / math.sqrt(base_dim), (base_dim, hidden_dim)),
        relation_embedding=rng.normal(0.0, 0.10, (relation_count, hidden_dim)),
        attention_vector=rng.normal(0.0, 1.0 / math.sqrt(hidden_dim), hidden_dim),
        relation_bias=np.zeros(relation_count, dtype=float),
        output_hidden=rng.normal(0.0, 1.0 / math.sqrt(hidden_dim), hidden_dim),
        output_base=rng.normal(0.0, 0.10 / math.sqrt(base_dim), base_dim),
        output_bias=np.zeros(1, dtype=float),
    )


def parameter_items(parameters: AttentionParameters) -> list[tuple[str, np.ndarray]]:
    return [(field, getattr(parameters, field)) for field in parameters.__dataclass_fields__]


def forward(
    base: np.ndarray,
    tokens: np.ndarray,
    parameters: AttentionParameters,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    projected_tokens = np.einsum("nrq,qd->nrd", tokens, parameters.token_projection)
    projected_context = base @ parameters.context_projection
    pre_activation = (
        projected_tokens
        + projected_context[:, None, :]
        + parameters.relation_embedding[None, :, :]
    )
    relation_state = np.tanh(pre_activation)
    logits = (
        np.einsum("nrd,d->nr", relation_state, parameters.attention_vector)
        / math.sqrt(relation_state.shape[-1])
        + parameters.relation_bias[None, :]
    )
    logits = logits - logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(axis=1, keepdims=True)
    hidden = np.einsum("nr,nrd->nd", weights, relation_state)
    prediction = (
        hidden @ parameters.output_hidden
        + base @ parameters.output_base
        + float(parameters.output_bias[0])
    )
    cache = {
        "base": base,
        "tokens": tokens,
        "pre_activation": pre_activation,
        "relation_state": relation_state,
        "weights": weights,
        "hidden": hidden,
    }
    return prediction, cache


def loss_and_gradients(
    base: np.ndarray,
    tokens: np.ndarray,
    target: np.ndarray,
    sample_weight: np.ndarray,
    parameters: AttentionParameters,
    l2: float,
    rank_weight: float,
    capacity: float,
    temperature: float,
) -> tuple[float, AttentionParameters, np.ndarray]:
    prediction, cache = forward(base, tokens, parameters)
    weight_sum = max(float(sample_weight.sum()), 1.0)
    residual = prediction - target
    loss = float(np.sum(sample_weight * residual**2) / weight_sum)
    derivative = 2.0 * sample_weight * residual / weight_sum
    if rank_weight > 0.0:
        selected_mass = max(capacity * len(prediction), 1.0)
        lower = float(prediction.min() - 20.0 * temperature)
        upper = float(prediction.max() + 20.0 * temperature)
        for _ in range(60):
            threshold = 0.5 * (lower + upper)
            soft_selection = 1.0 / (
                1.0
                + np.exp(
                    -np.clip((prediction - threshold) / temperature, -40.0, 40.0)
                )
            )
            if float(soft_selection.sum()) > selected_mass:
                lower = threshold
            else:
                upper = threshold
        soft_selection = 1.0 / (
            1.0
            + np.exp(
                -np.clip((prediction - 0.5 * (lower + upper)) / temperature, -40.0, 40.0)
            )
        )
        selection_slope = soft_selection * (1.0 - soft_selection)
        slope_sum = max(float(selection_slope.sum()), 1e-12)
        slope_weighted_target = float(np.sum(selection_slope * target) / slope_sum)
        derivative += (
            -rank_weight
            * selection_slope
            * (target - slope_weighted_target)
            / (temperature * selected_mass)
        )
        loss -= rank_weight * float(np.sum(soft_selection * target) / selected_mass)

    hidden = cache["hidden"]
    relation_state = cache["relation_state"]
    attention_weight = cache["weights"]
    gradients = AttentionParameters(
        token_projection=np.zeros_like(parameters.token_projection),
        context_projection=np.zeros_like(parameters.context_projection),
        relation_embedding=np.zeros_like(parameters.relation_embedding),
        attention_vector=np.zeros_like(parameters.attention_vector),
        relation_bias=np.zeros_like(parameters.relation_bias),
        output_hidden=hidden.T @ derivative + l2 * parameters.output_hidden,
        output_base=base.T @ derivative + l2 * parameters.output_base,
        output_bias=np.array([derivative.sum()]),
    )
    hidden_gradient = derivative[:, None] * parameters.output_hidden[None, :]
    state_gradient = attention_weight[:, :, None] * hidden_gradient[:, None, :]
    weight_gradient = np.einsum("nd,nrd->nr", hidden_gradient, relation_state)
    logit_gradient = attention_weight * (
        weight_gradient - np.sum(weight_gradient * attention_weight, axis=1, keepdims=True)
    )
    scale = math.sqrt(relation_state.shape[-1])
    gradients.attention_vector = (
        np.einsum("nr,nrd->d", logit_gradient, relation_state) / scale
        + l2 * parameters.attention_vector
    )
    gradients.relation_bias = logit_gradient.sum(axis=0)
    state_gradient += (
        logit_gradient[:, :, None]
        * parameters.attention_vector[None, None, :]
        / scale
    )
    pre_gradient = state_gradient * (1.0 - relation_state**2)
    gradients.token_projection = (
        np.einsum("nrq,nrd->qd", tokens, pre_gradient)
        + l2 * parameters.token_projection
    )
    gradients.context_projection = (
        base.T @ pre_gradient.sum(axis=1) + l2 * parameters.context_projection
    )
    gradients.relation_embedding = pre_gradient.sum(axis=0) + l2 * parameters.relation_embedding
    regularization = 0.5 * l2 * sum(
        float(np.sum(value**2))
        for name, value in parameter_items(parameters)
        if name != "output_bias"
    )
    return loss + regularization, gradients, prediction


def adam_update(
    parameters: AttentionParameters,
    gradients: AttentionParameters,
    first_moment: dict[str, np.ndarray],
    second_moment: dict[str, np.ndarray],
    step: int,
    learning_rate: float,
) -> None:
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    for name, value in parameter_items(parameters):
        gradient = getattr(gradients, name)
        first_moment[name] = beta1 * first_moment[name] + (1.0 - beta1) * gradient
        second_moment[name] = beta2 * second_moment[name] + (1.0 - beta2) * gradient**2
        corrected_first = first_moment[name] / (1.0 - beta1**step)
        corrected_second = second_moment[name] / (1.0 - beta2**step)
        value -= learning_rate * corrected_first / (np.sqrt(corrected_second) + epsilon)


def attention_summary(
    base: np.ndarray,
    tokens: np.ndarray,
    parameters: AttentionParameters,
) -> dict[str, float | list[float]]:
    _, cache = forward(base, tokens, parameters)
    mean_weights = cache["weights"].mean(axis=0)
    entropy = -np.sum(mean_weights * np.log(np.maximum(mean_weights, 1e-12)))
    normalized_entropy = float(entropy / math.log(len(mean_weights)))
    return {
        "mean_relation_weights": [float(value) for value in mean_weights],
        "maximum_relation_weight": float(mean_weights.max()),
        "normalized_relation_entropy": normalized_entropy,
    }


def fit_attention_model(
    base_train: np.ndarray,
    token_train: np.ndarray,
    target: np.ndarray,
    base_tune: np.ndarray,
    token_tune: np.ndarray,
    tuning: pd.DataFrame,
    hidden_dim: int,
    top_weight: float,
    learning_rate: float,
    epochs: int,
    l2: float,
    rank_weight: float,
    temperature: float,
    seed: int,
    early_metric: str = "review_utility",
) -> tuple[AttentionParameters, dict[str, float | int]]:
    target_mean = float(target.mean())
    target_scale = max(float(target.std()), 1e-6)
    normalized_target = (target - target_mean) / target_scale
    target_rank = rank01(target)
    sample_weight = 1.0 + top_weight * target_rank**3
    parameters = initialize_parameters(
        base_train.shape[1], token_train.shape[2], token_train.shape[1], hidden_dim, seed
    )
    first_moment = {name: np.zeros_like(value) for name, value in parameter_items(parameters)}
    second_moment = {name: np.zeros_like(value) for name, value in parameter_items(parameters)}
    best_parameters = copy.deepcopy(parameters)
    best_utility = -np.inf
    best_epoch = 0
    final_loss = math.nan
    patience = 0
    for epoch in range(1, epochs + 1):
        final_loss, gradients, _ = loss_and_gradients(
            base_train,
            token_train,
            normalized_target,
            sample_weight,
            parameters,
            l2,
            rank_weight,
            0.10,
            temperature,
        )
        adam_update(
            parameters,
            gradients,
            first_moment,
            second_moment,
            epoch,
            learning_rate,
        )
        if epoch % 20 == 0 or epoch == epochs:
            tune_prediction, _ = forward(base_tune, token_tune, parameters)
            metrics = evaluate(tuning, tune_prediction, "Temporal relational attention")
            monitored_value = float(metrics[early_metric])
            if monitored_value > best_utility + 1e-6:
                best_utility = monitored_value
                best_epoch = epoch
                best_parameters = copy.deepcopy(parameters)
                patience = 0
            else:
                patience += 1
            if patience >= 8:
                break
    return best_parameters, {
        "best_epoch": int(best_epoch),
        "best_tuning_utility": float(best_utility),
        "early_metric": early_metric,
        "final_training_loss": float(final_loss),
        "target_mean": target_mean,
        "target_scale": target_scale,
    }


MODEL_SETTINGS = [
    {"hidden_dim": 8, "top_weight": 2.0, "learning_rate": 0.008, "epochs": 480, "l2": 1e-4, "rank_weight": 0.5, "temperature": 0.60},
    {"hidden_dim": 12, "top_weight": 4.0, "learning_rate": 0.006, "epochs": 560, "l2": 2e-4, "rank_weight": 1.0, "temperature": 0.45},
    {"hidden_dim": 16, "top_weight": 6.0, "learning_rate": 0.005, "epochs": 640, "l2": 3e-4, "rank_weight": 1.5, "temperature": 0.35},
]


FAILURE_MODEL_SETTINGS = [
    {"hidden_dim": 12, "top_weight": 5.0, "learning_rate": 0.006, "epochs": 520, "l2": 2e-4, "rank_weight": 0.8, "temperature": 0.50},
    {"hidden_dim": 16, "top_weight": 8.0, "learning_rate": 0.005, "epochs": 600, "l2": 3e-4, "rank_weight": 1.2, "temperature": 0.40},
]


def robust_gap_score(df: pd.DataFrame, penalty: float) -> np.ndarray:
    dispersion = np.maximum(
        df["donor_pred_max"].to_numpy(dtype=float)
        - df["donor_pred_mean"].to_numpy(dtype=float),
        0.0,
    )
    support_scale = np.sqrt(np.maximum(np.log1p(df["donor_count"].to_numpy(dtype=float)), 1.0))
    robust_continuation = (
        df["donor_pred_max"].to_numpy(dtype=float)
        - penalty * dispersion / support_scale
        - 0.02 * np.minimum(df["donor_median_time_gap"].to_numpy(dtype=float) / 120.0, 1.0)
    )
    return robust_continuation - df["pred_recover"].to_numpy(dtype=float)


def conformal_lower_score(
    tuning_target: np.ndarray,
    tuning_mean: np.ndarray,
    tuning_std: np.ndarray,
    validation_mean: np.ndarray,
    validation_std: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, float, float]:
    scale = tuning_std + 0.05
    nonconformity = np.maximum(tuning_mean - tuning_target, 0.0) / scale
    rank = min(
        len(nonconformity) - 1,
        int(math.ceil((len(nonconformity) + 1) * (1.0 - alpha))) - 1,
    )
    quantile = float(np.sort(nonconformity)[max(rank, 0)])
    lower = validation_mean - quantile * (validation_std + 0.05)
    tuning_lower = tuning_mean - quantile * scale
    coverage = float(np.mean(tuning_target >= tuning_lower))
    return lower, quantile, coverage


def constant_scale_conformal_lower(
    tuning_target: np.ndarray,
    tuning_prediction: np.ndarray,
    validation_prediction: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    nonconformity = np.maximum(tuning_prediction - tuning_target, 0.0)
    rank = min(
        len(nonconformity) - 1,
        int(math.ceil((len(nonconformity) + 1) * (1.0 - alpha))) - 1,
    )
    quantile = float(np.sort(nonconformity)[max(rank, 0)])
    tuning_lower = tuning_prediction - quantile
    validation_lower = validation_prediction - quantile
    tuning_coverage = float(np.mean(tuning_target >= tuning_lower))
    return tuning_lower, validation_lower, quantile, tuning_coverage, 0.0


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
    (
        base_train,
        base_tune,
        base_valid,
        token_train,
        token_tune,
        token_valid,
        base_names,
        token_names,
    ) = prepare_features(training, tuning, validation, smoothing=25.0)
    train_target = capacity_reward(training)
    tune_target = capacity_reward(tuning)

    models: list[AttentionParameters] = []
    training_rows: list[dict[str, float | int]] = []
    tuning_predictions: list[np.ndarray] = []
    validation_predictions: list[np.ndarray] = []
    relation_rows: list[dict[str, object]] = []
    for model_id, setting in enumerate(MODEL_SETTINGS):
        model, training_summary = fit_attention_model(
            base_train,
            token_train,
            train_target,
            base_tune,
            token_tune,
            tuning,
            seed=seed + 101 * model_id,
            **setting,
        )
        tune_prediction, _ = forward(base_tune, token_tune, model)
        valid_prediction, _ = forward(base_valid, token_valid, model)
        target_mean = float(training_summary["target_mean"])
        target_scale = float(training_summary["target_scale"])
        tune_prediction = tune_prediction * target_scale + target_mean
        valid_prediction = valid_prediction * target_scale + target_mean
        models.append(model)
        tuning_predictions.append(tune_prediction)
        validation_predictions.append(valid_prediction)
        training_rows.append(
            {"model_type": "capacity_reward", "model_id": model_id, **setting, **training_summary}
        )
        relation_rows.append(
            {
                "model_type": "capacity_reward",
                "model_id": model_id,
                **attention_summary(base_tune, token_tune, model),
            }
        )

    failure_tuning_predictions: list[np.ndarray] = []
    failure_validation_predictions: list[np.ndarray] = []
    failure_target = training["fail_h4"].to_numpy(dtype=float)
    for failure_id, setting in enumerate(FAILURE_MODEL_SETTINGS):
        model, training_summary = fit_attention_model(
            base_train,
            token_train,
            failure_target,
            base_tune,
            token_tune,
            tuning,
            seed=seed + 701 + 101 * failure_id,
            early_metric="failure_rate",
            **setting,
        )
        tune_prediction, _ = forward(base_tune, token_tune, model)
        valid_prediction, _ = forward(base_valid, token_valid, model)
        target_mean = float(training_summary["target_mean"])
        target_scale = float(training_summary["target_scale"])
        tune_prediction = tune_prediction * target_scale + target_mean
        valid_prediction = valid_prediction * target_scale + target_mean
        failure_tuning_predictions.append(tune_prediction)
        failure_validation_predictions.append(valid_prediction)
        training_rows.append(
            {
                "model_type": "failed_exit",
                "model_id": failure_id,
                **setting,
                **training_summary,
            }
        )
        relation_rows.append(
            {
                "model_type": "failed_exit",
                "model_id": failure_id,
                **attention_summary(base_tune, token_tune, model),
            }
        )

    tuning_matrix = np.column_stack(tuning_predictions)
    validation_matrix = np.column_stack(validation_predictions)
    tuning_mean = tuning_matrix.mean(axis=1)
    tuning_std = tuning_matrix.std(axis=1)
    validation_mean = validation_matrix.mean(axis=1)
    validation_std = validation_matrix.std(axis=1)
    failure_tuning_score = rank01(
        np.column_stack(failure_tuning_predictions).mean(axis=1)
    )
    failure_validation_score = rank01(
        np.column_stack(failure_validation_predictions).mean(axis=1)
    )
    validation_lcb, conformal_quantile, tuning_coverage = conformal_lower_score(
        tune_target,
        tuning_mean,
        tuning_std,
        validation_mean,
        validation_std,
        alpha=0.08,
    )
    tuning_lcb = tuning_mean - conformal_quantile * (tuning_std + 0.05)

    learned_candidates: dict[str, tuple[np.ndarray, np.ndarray, float, float]] = {
        "ensemble_lcb": (
            tuning_lcb,
            validation_lcb,
            conformal_quantile,
            tuning_coverage,
        ),
    }
    for model_id in range(tuning_matrix.shape[1]):
        tune_lower, valid_lower, quantile, coverage, _ = constant_scale_conformal_lower(
            tune_target,
            tuning_matrix[:, model_id],
            validation_matrix[:, model_id],
            alpha=0.08,
        )
        learned_candidates[f"model_{model_id}_lcb"] = (
            tune_lower,
            valid_lower,
            quantile,
            coverage,
        )

    tuning_gap = rank01(tuning["ctrg_gap_max"])
    validation_gap = rank01(validation["ctrg_gap_max"])
    tuning_dual = (tuning_gap**2.0) * (
        (rank01(1.0 - tuning["pred_recover"]) + 0.02) ** 0.75
    )
    validation_dual = (validation_gap**2.0) * (
        (rank01(1.0 - validation["pred_recover"]) + 0.02) ** 0.75
    )
    selection_rows: list[dict[str, float | int | str]] = []
    for candidate_name, (candidate_tune, _, candidate_quantile, candidate_coverage) in learned_candidates.items():
        for robust_penalty in [0.0, 0.25, 0.50, 1.0]:
            tune_robust = rank01(robust_gap_score(tuning, robust_penalty))
            for learned_weight in [0.25, 0.50, 0.75, 1.0]:
                for gap_weight in [0.0, 0.25, 0.50, 0.75]:
                    if learned_weight + gap_weight > 1.0:
                        continue
                    robust_weight = 1.0 - learned_weight - gap_weight
                    learned = rank01(candidate_tune)
                    score = (
                        learned_weight * learned
                        + gap_weight * tuning_gap
                        + robust_weight * tune_robust
                    )
                    for failure_weight in [0.0, 0.10, 0.20, 0.30, 0.40]:
                        for dual_weight in [0.0, 0.10, 0.20]:
                            if failure_weight + dual_weight > 0.50:
                                continue
                            residual_weight = 1.0 - failure_weight - dual_weight
                            final_score = (
                                residual_weight * score
                                + failure_weight * failure_tuning_score
                                + dual_weight * rank01(tuning_dual)
                            )
                            metrics = evaluate(tuning, final_score, "Temporal relational attention certificate")
                            selection_rows.append(
                                {
                                    "learned_candidate": candidate_name,
                                    "candidate_conformal_quantile": candidate_quantile,
                                    "candidate_tuning_coverage": candidate_coverage,
                                    "robust_penalty": robust_penalty,
                                    "learned_weight": learned_weight,
                                    "gap_weight": gap_weight,
                                    "robust_weight": robust_weight,
                                    "failure_weight": failure_weight,
                                    "dual_weight": dual_weight,
                                    **metrics,
                                }
                            )
    selection = pd.DataFrame(selection_rows).sort_values(
        ["review_utility", "mean_gap", "failure_rate"],
        ascending=[False, False, False],
    )
    best = selection.iloc[0]
    selected_candidate = learned_candidates[str(best["learned_candidate"])]
    selected_validation_learned = selected_candidate[1]
    valid_robust = rank01(robust_gap_score(validation, float(best["robust_penalty"])))
    proposed_score = (
        float(best["learned_weight"]) * rank01(selected_validation_learned)
        + float(best["gap_weight"]) * validation_gap
        + float(best["robust_weight"]) * valid_robust
    )
    residual_weight = 1.0 - float(best["failure_weight"]) - float(best["dual_weight"])
    proposed_score = (
        residual_weight * proposed_score
        + float(best["failure_weight"]) * failure_validation_score
        + float(best["dual_weight"]) * rank01(validation_dual)
    )

    comparison_rows = [
        evaluate(validation, proposed_score, "Temporal relational attention certificate"),
        evaluate(validation, validation["ctrg_gap_max"], "KCCRES max-gap certificate"),
        evaluate(validation, validation_dual, "KCCRES dual-channel audit certificate"),
        evaluate(validation, 1.0 - validation["pred_recover"], "Observed-path risk"),
        evaluate(validation, validation["ctrg_gap_mean"], "Mean-donor gap"),
    ]
    comparison = pd.DataFrame(comparison_rows).sort_values("review_utility", ascending=False)
    proposed = comparison[
        comparison["method"].eq("Temporal relational attention certificate")
    ].iloc[0]
    best_existing = comparison[
        ~comparison["method"].eq("Temporal relational attention certificate")
    ].iloc[0]
    mean_relation_weights = np.mean(
        [row["mean_relation_weights"] for row in relation_rows], axis=0
    )
    maximum_relation_weight = float(np.max(mean_relation_weights))
    normalized_relation_entropy = float(
        -np.sum(mean_relation_weights * np.log(np.maximum(mean_relation_weights, 1e-12)))
        / math.log(len(mean_relation_weights))
    )
    validation_coverage = float(
        np.mean(capacity_reward(validation) >= selected_validation_learned)
    )
    utility_gain = float(proposed["review_utility"] - best_existing["review_utility"])

    pd.DataFrame(training_rows).to_csv(output_dir / "model_training.csv", index=False)
    pd.DataFrame(relation_rows).to_json(
        output_dir / "relation_attention.json", orient="records", indent=2
    )
    selection.to_csv(output_dir / "selection_tuning.csv", index=False)
    comparison.to_csv(output_dir / "validation_comparison.csv", index=False)
    report = {
        "month": int(month),
        "sampled_episodes": int(len(data)),
        "training_episodes": int(len(training)),
        "tuning_episodes": int(len(tuning)),
        "validation_episodes": int(len(validation)),
        "base_feature_count": int(base_train.shape[1]),
        "relation_count": int(token_train.shape[1]),
        "relation_signal_count": int(token_train.shape[2]),
        "base_features": base_names,
        "relation_signals": token_names,
        "selected_weights": {
            "learned_candidate": str(best["learned_candidate"]),
            "robust_penalty": float(best["robust_penalty"]),
            "learned_weight": float(best["learned_weight"]),
            "gap_weight": float(best["gap_weight"]),
            "robust_weight": float(best["robust_weight"]),
            "failure_weight": float(best["failure_weight"]),
            "dual_weight": float(best["dual_weight"]),
        },
        "conformal_alpha": 0.08,
        "conformal_quantile": float(selected_candidate[2]),
        "tuning_marginal_coverage": float(selected_candidate[3]),
        "validation_marginal_coverage": validation_coverage,
        "mean_relation_weights": [float(value) for value in mean_relation_weights],
        "maximum_relation_weight": maximum_relation_weight,
        "normalized_relation_entropy": normalized_relation_entropy,
        "validation_comparison": comparison.to_dict(orient="records"),
        "utility_gain_over_best_existing": utility_gain,
        "utility_gain_ge_0_02": bool(utility_gain >= 0.02),
        "no_relation_collapse": bool(maximum_relation_weight < 0.90),
        "conformal_coverage_ge_0_90": bool(validation_coverage >= 0.90),
        "evaluation_gate_passed": bool(
            utility_gain >= 0.02
            and maximum_relation_weight < 0.90
            and validation_coverage >= 0.90
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
