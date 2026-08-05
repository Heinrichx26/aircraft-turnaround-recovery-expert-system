from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import risk_controlled_opportunity as single
from temporal_relation_model import adam_update, Standardizer


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "results" / "ctrg" / "full" / "episode_scores.csv"
DEFAULT_OUTPUT = ROOT / "results" / "ctrg" / "multitask_recoverability"


RELATIONS = [
    ["airport"],
    ["carrier"],
    ["airport", "carrier"],
    ["airport", "hour"],
    ["carrier", "hour"],
    ["airport", "dest"],
    ["carrier", "dest"],
    ["airport", "carrier", "hour"],
]


@dataclass
class MultiTaskParameters:
    token_projection: np.ndarray
    context_projection: np.ndarray
    relation_embedding: np.ndarray
    attention_vector: np.ndarray
    relation_bias: np.ndarray
    head_hidden: np.ndarray
    head_base: np.ndarray
    head_bias: np.ndarray


def parameter_items(parameters: MultiTaskParameters) -> list[tuple[str, np.ndarray]]:
    return [(name, getattr(parameters, name)) for name in parameters.__dataclass_fields__]


def initialize_parameters(
    base_dim: int,
    token_dim: int,
    relation_count: int,
    hidden_dim: int,
    seed: int,
) -> MultiTaskParameters:
    rng = np.random.default_rng(seed)
    return MultiTaskParameters(
        token_projection=rng.normal(
            0.0, 1.0 / math.sqrt(token_dim), (token_dim, hidden_dim)
        ),
        context_projection=rng.normal(
            0.0, 0.25 / math.sqrt(base_dim), (base_dim, hidden_dim)
        ),
        relation_embedding=rng.normal(0.0, 0.10, (relation_count, hidden_dim)),
        attention_vector=rng.normal(0.0, 1.0 / math.sqrt(hidden_dim), hidden_dim),
        relation_bias=np.zeros(relation_count, dtype=float),
        head_hidden=rng.normal(0.0, 1.0 / math.sqrt(hidden_dim), (hidden_dim, 3)),
        head_base=rng.normal(0.0, 0.10 / math.sqrt(base_dim), (base_dim, 3)),
        head_bias=np.zeros(3, dtype=float),
    )


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -35.0, 35.0)))


def forward(
    base: np.ndarray,
    tokens: np.ndarray,
    parameters: MultiTaskParameters,
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
    logits -= logits.max(axis=1, keepdims=True)
    attention = np.exp(logits)
    attention /= attention.sum(axis=1, keepdims=True)
    hidden = np.einsum("nr,nrd->nd", attention, relation_state)
    task_logits = (
        hidden @ parameters.head_hidden
        + base @ parameters.head_base
        + parameters.head_bias[None, :]
    )
    probabilities = sigmoid(task_logits)
    return probabilities, {
        "base": base,
        "tokens": tokens,
        "relation_state": relation_state,
        "attention": attention,
        "hidden": hidden,
    }


def loss_and_gradients(
    base: np.ndarray,
    tokens: np.ndarray,
    targets: np.ndarray,
    parameters: MultiTaskParameters,
    setting: dict[str, float | int],
) -> tuple[float, MultiTaskParameters]:
    probabilities, cache = forward(base, tokens, parameters)
    task_weights = np.asarray(setting["task_weights"], dtype=float)
    positive_weights = np.asarray(setting["positive_weights"], dtype=float)
    observation_weights = 1.0 + targets * positive_weights[None, :]
    clipped = np.clip(probabilities, 1e-8, 1.0 - 1e-8)
    derivative_logits = np.zeros_like(probabilities)
    loss = 0.0
    for task_id in range(3):
        denominator = max(float(observation_weights[:, task_id].sum()), 1.0)
        bce = -(
            targets[:, task_id] * np.log(clipped[:, task_id])
            + (1.0 - targets[:, task_id]) * np.log(1.0 - clipped[:, task_id])
        )
        loss += float(
            task_weights[task_id]
            * np.sum(observation_weights[:, task_id] * bce)
            / denominator
        )
        derivative_logits[:, task_id] += (
            task_weights[task_id]
            * observation_weights[:, task_id]
            * (probabilities[:, task_id] - targets[:, task_id])
            / denominator
        )

    opportunity_score = probabilities[:, 2]
    opportunity_target = targets[:, 2]
    capacity = float(setting["capacity"])
    temperature = float(setting["temperature"])
    selected_mass = max(capacity * len(opportunity_score), 1.0)
    lower = float(opportunity_score.min() - 20.0 * temperature)
    upper = float(opportunity_score.max() + 20.0 * temperature)
    for _ in range(60):
        threshold = 0.5 * (lower + upper)
        soft_selection = sigmoid((opportunity_score - threshold) / temperature)
        if float(soft_selection.sum()) > selected_mass:
            lower = threshold
        else:
            upper = threshold
    soft_selection = sigmoid(
        (opportunity_score - 0.5 * (lower + upper)) / temperature
    )
    selection_slope = soft_selection * (1.0 - soft_selection)
    slope_sum = max(float(selection_slope.sum()), 1e-12)
    centered_target = opportunity_target - float(
        np.sum(selection_slope * opportunity_target) / slope_sum
    )
    rank_weight = float(setting["rank_weight"])
    derivative_probability = np.zeros_like(probabilities)
    derivative_probability[:, 2] += (
        -rank_weight
        * selection_slope
        * centered_target
        / (temperature * selected_mass)
    )
    loss -= rank_weight * float(
        np.sum(soft_selection * opportunity_target) / selected_mass
    )

    clear_weight = float(setting.get("clear_weight", 0.0))
    if clear_weight > 0.0:
        clear_capacity = float(setting.get("clear_capacity", 0.95))
        clear_temperature = float(setting.get("clear_temperature", temperature))
        unreviewed_mass = max((1.0 - clear_capacity) * len(opportunity_score), 1.0)
        lower = float(opportunity_score.min() - 20.0 * clear_temperature)
        upper = float(opportunity_score.max() + 20.0 * clear_temperature)
        for _ in range(60):
            threshold = 0.5 * (lower + upper)
            soft_unreviewed = sigmoid((threshold - opportunity_score) / clear_temperature)
            if float(soft_unreviewed.sum()) > unreviewed_mass:
                upper = threshold
            else:
                lower = threshold
        soft_unreviewed = sigmoid(
            (0.5 * (lower + upper) - opportunity_score) / clear_temperature
        )
        clear_slope = soft_unreviewed * (1.0 - soft_unreviewed)
        clear_slope_sum = max(float(clear_slope.sum()), 1e-12)
        clear_centered_target = opportunity_target - float(
            np.sum(clear_slope * opportunity_target) / clear_slope_sum
        )
        derivative_probability[:, 2] += (
            -clear_weight
            * clear_slope
            * clear_centered_target
            / (clear_temperature * unreviewed_mass)
        )
        loss += clear_weight * float(
            np.sum(soft_unreviewed * opportunity_target) / unreviewed_mass
        )

    failure = probabilities[:, 0]
    advantage = probabilities[:, 1]
    opportunity = probabilities[:, 2]
    logical_weight = float(setting["logical_weight"])
    product_weight = float(setting["product_weight"])
    n = max(len(targets), 1)
    violation_failure = np.maximum(opportunity - failure, 0.0)
    violation_advantage = np.maximum(opportunity - advantage, 0.0)
    loss += logical_weight * float(
        np.mean(violation_failure**2 + violation_advantage**2)
    )
    derivative_probability[:, 2] += (
        2.0 * logical_weight * (violation_failure + violation_advantage) / n
    )
    derivative_probability[:, 0] -= 2.0 * logical_weight * violation_failure / n
    derivative_probability[:, 1] -= 2.0 * logical_weight * violation_advantage / n
    product_residual = opportunity - failure * advantage
    loss += product_weight * float(np.mean(product_residual**2))
    product_gradient = 2.0 * product_weight * product_residual / n
    derivative_probability[:, 2] += product_gradient
    derivative_probability[:, 0] -= product_gradient * advantage
    derivative_probability[:, 1] -= product_gradient * failure
    derivative_logits += derivative_probability * probabilities * (1.0 - probabilities)

    l2 = float(setting["l2"])
    hidden = cache["hidden"]
    relation_state = cache["relation_state"]
    attention = cache["attention"]
    gradients = MultiTaskParameters(
        token_projection=np.zeros_like(parameters.token_projection),
        context_projection=np.zeros_like(parameters.context_projection),
        relation_embedding=np.zeros_like(parameters.relation_embedding),
        attention_vector=np.zeros_like(parameters.attention_vector),
        relation_bias=np.zeros_like(parameters.relation_bias),
        head_hidden=hidden.T @ derivative_logits + l2 * parameters.head_hidden,
        head_base=base.T @ derivative_logits + l2 * parameters.head_base,
        head_bias=derivative_logits.sum(axis=0),
    )
    hidden_gradient = derivative_logits @ parameters.head_hidden.T
    state_gradient = attention[:, :, None] * hidden_gradient[:, None, :]
    attention_gradient = np.einsum("nd,nrd->nr", hidden_gradient, relation_state)
    logit_gradient = attention * (
        attention_gradient
        - np.sum(attention_gradient * attention, axis=1, keepdims=True)
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
    gradients.relation_embedding = (
        pre_gradient.sum(axis=0) + l2 * parameters.relation_embedding
    )
    regularization = 0.5 * l2 * sum(
        float(np.sum(value**2))
        for name, value in parameter_items(parameters)
        if name != "head_bias"
    )
    return loss + regularization, gradients


MODEL_SETTINGS = [
    {
        "hidden_dim": 12,
        "learning_rate": 0.006,
        "epochs": 600,
        "l2": 2e-4,
        "rank_weight": 0.8,
        "temperature": 0.05,
        "logical_weight": 0.5,
        "product_weight": 0.5,
        "task_weights": [0.8, 0.8, 1.5],
        "positive_weights": [2.0, 3.0, 8.0],
        "capacity": 0.10,
    },
    {
        "hidden_dim": 16,
        "learning_rate": 0.005,
        "epochs": 700,
        "l2": 3e-4,
        "rank_weight": 1.2,
        "temperature": 0.04,
        "logical_weight": 1.0,
        "product_weight": 0.8,
        "task_weights": [1.0, 1.0, 1.8],
        "positive_weights": [3.0, 5.0, 10.0],
        "capacity": 0.10,
    },
    {
        "hidden_dim": 20,
        "learning_rate": 0.004,
        "epochs": 800,
        "l2": 4e-4,
        "rank_weight": 1.6,
        "temperature": 0.035,
        "logical_weight": 1.5,
        "product_weight": 1.2,
        "task_weights": [1.0, 1.2, 2.0],
        "positive_weights": [4.0, 7.0, 12.0],
        "capacity": 0.10,
    },
]


def apply_dual_tail_settings(
    settings: list[dict[str, float | int]],
    clear_weight: float,
    clear_capacity: float,
) -> None:
    for setting in settings:
        setting["clear_weight"] = clear_weight
        setting["clear_capacity"] = clear_capacity
        setting["clear_temperature"] = min(float(setting["temperature"]), 0.025)


def tail_risk(target: np.ndarray, score: np.ndarray, review_capacity: float) -> float:
    selected = max(1, int(math.ceil(len(target) * review_capacity)))
    order = np.argsort(-np.asarray(score, dtype=float), kind="mergesort")
    residual = np.asarray(target, dtype=float)[order[selected:]]
    return float(residual.mean()) if len(residual) else 0.0


def build_targets(
    data: pd.DataFrame,
    reliability_threshold: float,
    advantage_margin: float,
    opportunity_definition: str,
) -> np.ndarray:
    failure = (data["fail_h4"].to_numpy(dtype=float) >= 0.5).astype(float)
    feasible_alternative = (
        data["donor_recovery_lcb"].to_numpy(dtype=float) >= reliability_threshold
    )
    if opportunity_definition == "certified_advantage":
        feasible_alternative &= (
            data["donor_recovery_lcb"].to_numpy(dtype=float)
            - data["pred_recover"].to_numpy(dtype=float)
            >= advantage_margin
        )
    elif opportunity_definition != "feasible_alternative":
        raise ValueError(f"Unknown opportunity definition: {opportunity_definition}")
    feasible_alternative = feasible_alternative.astype(float)
    opportunity = failure * feasible_alternative
    return np.column_stack([failure, feasible_alternative, opportunity])


def relation_tokens(
    training: pd.DataFrame,
    other: pd.DataFrame,
    targets: np.ndarray,
    smoothing: float,
    leave_one_out: bool,
) -> tuple[np.ndarray, list[str]]:
    signals = np.column_stack(
        [
            targets[:, 2],
            targets[:, 0],
            targets[:, 1],
            training["donor_recovery_lcb"].to_numpy(dtype=float),
            np.clip(training["ctrg_gap_max"].to_numpy(dtype=float) / 0.40, 0.0, 1.0),
            1.0 - training["pred_recover"].to_numpy(dtype=float),
            training["donor_pred_mean"].to_numpy(dtype=float),
        ]
    )
    names = [
        "joint_opportunity_rate",
        "failed_exit_rate",
        "robust_advantage_rate",
        "donor_recovery_lower_bound",
        "recoverability_gap",
        "observed_path_risk",
        "predicted_donor_recovery",
        "log_support",
    ]
    output = np.zeros((len(other), len(RELATIONS), len(names)), dtype=np.float64)
    for relation_id, columns in enumerate(RELATIONS):
        train_key = single.relation_key(training, columns)
        other_key = single.relation_key(other, columns)
        counts = train_key.value_counts()
        mapped_counts = other_key.map(counts).fillna(0.0).to_numpy(dtype=float)
        adjusted_counts = np.maximum(mapped_counts - 1.0, 0.0) if leave_one_out else mapped_counts
        output[:, relation_id, -1] = np.log1p(adjusted_counts)
        for signal_id in range(signals.shape[1]):
            sums = pd.DataFrame(
                {"key": train_key, "value": signals[:, signal_id]}
            ).groupby("key")["value"].sum()
            mapped_sums = other_key.map(sums).fillna(0.0).to_numpy(
                dtype=float, copy=True
            )
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
    targets: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray], list[str], list[str]]:
    frames = [single.base_features(frame) for frame in [training, tuning, calibration, validation]]
    base_scaler = Standardizer.fit(frames[0].to_numpy(dtype=float))
    base_arrays = [base_scaler.transform(frame.to_numpy(dtype=float)) for frame in frames]
    tokens_train, names = relation_tokens(
        training, training, targets, smoothing=25.0, leave_one_out=True
    )
    token_arrays = [tokens_train]
    for frame in [tuning, calibration, validation]:
        token_arrays.append(
            relation_tokens(
                training, frame, targets, smoothing=25.0, leave_one_out=False
            )[0]
        )
    token_scaler = Standardizer.fit(
        tokens_train.reshape(-1, tokens_train.shape[-1])
    )
    token_arrays = [
        token_scaler.transform(tokens.reshape(-1, tokens.shape[-1])).reshape(tokens.shape)
        for tokens in token_arrays
    ]
    return base_arrays, token_arrays, list(frames[0].columns), names


def candidate_scores(probabilities: np.ndarray) -> dict[str, np.ndarray]:
    failure = probabilities[:, 0]
    advantage = probabilities[:, 1]
    joint = probabilities[:, 2]
    failure_rank = single.rank01(failure)
    advantage_rank = single.rank01(advantage)
    rank_product = failure_rank * advantage_rank
    rank_harmonic = (
        2.0
        * failure_rank
        * advantage_rank
        / np.maximum(failure_rank + advantage_rank, 1e-8)
    )
    return {
        "joint_head": single.rank01(joint),
        "factorized_product": single.rank01(failure * advantage),
        "conservative_conjunction": single.rank01(np.minimum(failure, advantage)),
        "rank_minimum_conjunction": single.rank01(
            np.minimum(failure_rank, advantage_rank)
        ),
        "rank_geometric_conjunction": single.rank01(np.sqrt(rank_product)),
        "rank_harmonic_conjunction": single.rank01(rank_harmonic),
        "failure_weighted_conjunction": single.rank01(
            failure_rank**0.60 * advantage_rank**0.40
        ),
        "advantage_weighted_conjunction": single.rank01(
            failure_rank**0.40 * advantage_rank**0.60
        ),
        "structured_geometric": single.rank01(
            np.sqrt(np.maximum(joint * failure * advantage, 0.0))
        ),
    }


def fit_model(
    base_train: np.ndarray,
    token_train: np.ndarray,
    targets_train: np.ndarray,
    base_tune: np.ndarray,
    token_tune: np.ndarray,
    tuning: pd.DataFrame,
    targets_tune: np.ndarray,
    setting: dict[str, float | int],
    seed: int,
) -> tuple[MultiTaskParameters, dict[str, float | int]]:
    parameters = initialize_parameters(
        base_train.shape[1],
        token_train.shape[2],
        token_train.shape[1],
        int(setting["hidden_dim"]),
        seed,
    )
    first = {name: np.zeros_like(value) for name, value in parameter_items(parameters)}
    second = {name: np.zeros_like(value) for name, value in parameter_items(parameters)}
    best_parameters = copy.deepcopy(parameters)
    best_precision = -np.inf
    best_epoch = 0
    patience = 0
    final_loss = math.nan
    for epoch in range(1, int(setting["epochs"]) + 1):
        final_loss, gradients = loss_and_gradients(
            base_train, token_train, targets_train, parameters, setting
        )
        adam_update(
            parameters,
            gradients,
            first,
            second,
            epoch,
            float(setting["learning_rate"]),
        )
        if epoch % 20 == 0 or epoch == int(setting["epochs"]):
            tune_probability, _ = forward(base_tune, token_tune, parameters)
            score = candidate_scores(tune_probability)["structured_geometric"]
            metric = single.opportunity_metrics(
                tuning, targets_tune[:, 2], score, 0.10, "multitask"
            )
            precision = float(metric["opportunity_precision"])
            clear_weight = float(setting.get("clear_weight", 0.0))
            clear_capacity = float(setting.get("clear_capacity", 0.95))
            low_tail = tail_risk(targets_tune[:, 2], score, clear_capacity)
            selection_value = precision - clear_weight * low_tail
            if selection_value > best_precision + 1e-6:
                best_precision = selection_value
                best_epoch = epoch
                best_parameters = copy.deepcopy(parameters)
                patience = 0
            else:
                patience += 1
            if patience >= 10:
                break
    return best_parameters, {
        "best_epoch": int(best_epoch),
        "best_tuning_dual_tail_objective": float(best_precision),
        "final_training_loss": float(final_loss),
    }


def binomial_cdf(k: int, n: int, probability: float) -> float:
    if k >= n:
        return 1.0
    if probability <= 0.0:
        return 1.0
    if probability >= 1.0:
        return 0.0
    log_p = math.log(probability)
    log_q = math.log1p(-probability)
    terms = np.asarray(
        [
            math.lgamma(n + 1)
            - math.lgamma(j + 1)
            - math.lgamma(n - j + 1)
            + j * log_p
            + (n - j) * log_q
            for j in range(k + 1)
        ],
        dtype=float,
    )
    maximum = float(terms.max())
    return float(math.exp(maximum) * np.exp(terms - maximum).sum())


def exact_binomial_upper(k: int, n: int, alpha: float) -> float:
    if n <= 0 or k >= n:
        return 1.0
    lower = k / n
    upper = 1.0
    for _ in range(70):
        midpoint = 0.5 * (lower + upper)
        if binomial_cdf(k, n, midpoint) > alpha:
            lower = midpoint
        else:
            upper = midpoint
    return float(upper)


CAPACITY_GRID = [value / 100.0 for value in range(5, 81, 5)]


def calibrate_capacity_exact(
    target: np.ndarray,
    score: np.ndarray,
    risk_target: float,
    delta: float,
) -> dict[str, float | bool]:
    rows = []
    order = np.argsort(-np.asarray(score, dtype=float))
    alpha = delta / len(CAPACITY_GRID)
    for capacity in CAPACITY_GRID:
        selected_count = max(1, int(math.ceil(len(target) * capacity)))
        residual = target[order[selected_count:]]
        events = int(residual.sum())
        upper = exact_binomial_upper(events, len(residual), alpha)
        rows.append((capacity, float(residual.mean()), upper, events, len(residual)))
    feasible = [row for row in rows if row[2] <= risk_target]
    chosen = feasible[0] if feasible else rows[-1]
    return {
        "capacity": float(chosen[0]),
        "calibration_residual_rate": float(chosen[1]),
        "calibration_risk_upper_bound": float(chosen[2]),
        "calibration_events": int(chosen[3]),
        "calibration_unreviewed": int(chosen[4]),
        "risk_control_feasible": bool(feasible),
    }


def attention_summary(
    base: np.ndarray,
    tokens: np.ndarray,
    parameters: MultiTaskParameters,
) -> dict[str, float | list[float]]:
    _, cache = forward(base, tokens, parameters)
    mean_weights = cache["attention"].mean(axis=0)
    entropy = -np.sum(mean_weights * np.log(np.maximum(mean_weights, 1e-12)))
    normalized_entropy = (
        float(entropy / math.log(len(mean_weights))) if len(mean_weights) > 1 else 0.0
    )
    return {
        "mean_relation_weights": [float(value) for value in mean_weights],
        "maximum_relation_weight": float(mean_weights.max()),
        "normalized_relation_entropy": normalized_entropy,
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
    residual_risk_ratio: float | None,
    ablation: str,
    parallel_models: int,
    opportunity_definition: str,
    clear_weight: float = 0.0,
    clear_capacity: float = 0.95,
    selection_tail_weight: float = 0.0,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = single.load_month(input_path, month, max_rows, seed)
    training, tuning, calibration, validation = single.split_time(data)
    frames = [training, tuning, calibration, validation]
    targets = [
        build_targets(
            frame,
            reliability_threshold,
            advantage_margin,
            opportunity_definition,
        )
        for frame in frames
    ]
    if residual_risk_ratio is not None:
        residual_risk_target = float(
            residual_risk_ratio * targets[2][:, 2].mean()
        )
    base_arrays, token_arrays, base_names, token_names = prepare_features(
        training, tuning, calibration, validation, targets[0]
    )
    settings_list = copy.deepcopy(MODEL_SETTINGS)
    apply_dual_tail_settings(settings_list, clear_weight, clear_capacity)
    if ablation == "no_relations":
        token_arrays = [
            np.zeros((len(tokens), 1, 1), dtype=float) for tokens in token_arrays
        ]
        token_names = ["removed"]
    elif ablation == "no_multitask":
        for setting in settings_list:
            setting["task_weights"] = [0.0, 0.0, 1.0]
            setting["logical_weight"] = 0.0
            setting["product_weight"] = 0.0
    elif ablation == "no_logic":
        for setting in settings_list:
            setting["logical_weight"] = 0.0
            setting["product_weight"] = 0.0
    elif ablation == "no_capacity_loss":
        for setting in settings_list:
            setting["rank_weight"] = 0.0
    elif ablation == "core":
        for setting in settings_list:
            setting["rank_weight"] = 0.0
            setting["logical_weight"] = 0.0
            setting["product_weight"] = 0.0
    elif ablation != "full":
        raise ValueError(f"Unknown ablation: {ablation}")

    def train_one(
        model_id: int, setting: dict[str, float | int]
    ) -> tuple[int, list[np.ndarray], dict[str, object], dict[str, object]]:
        started = time.perf_counter()
        print(f"Training model {model_id + 1}/{len(settings_list)}", flush=True)
        model, training_summary = fit_model(
            base_arrays[0],
            token_arrays[0],
            targets[0],
            base_arrays[1],
            token_arrays[1],
            tuning,
            targets[1],
            setting,
            seed + model_id * 211,
        )
        probabilities = [forward(base, tokens, model)[0] for base, tokens in zip(base_arrays, token_arrays)]
        training_row = {
            "model_id": model_id,
            **{key: value for key, value in setting.items() if not isinstance(value, list)},
            "task_weights": json.dumps(setting["task_weights"]),
            "positive_weights": json.dumps(setting["positive_weights"]),
            **training_summary,
            "elapsed_seconds": float(time.perf_counter() - started),
        }
        attention_row = {
            "model_id": model_id,
            **attention_summary(base_arrays[1], token_arrays[1], model),
        }
        print(
            f"Completed model {model_id + 1}/{len(settings_list)} "
            f"in {training_row['elapsed_seconds']:.1f}s",
            flush=True,
        )
        return model_id, probabilities, training_row, attention_row

    jobs = list(enumerate(settings_list))
    if parallel_models > 1:
        with ThreadPoolExecutor(max_workers=min(parallel_models, len(jobs))) as executor:
            fitted = list(executor.map(lambda item: train_one(*item), jobs))
    else:
        fitted = [train_one(*item) for item in jobs]
    fitted.sort(key=lambda item: item[0])
    probability_sets = [item[1] for item in fitted]
    training_rows = [item[2] for item in fitted]
    attention_rows = [item[3] for item in fitted]

    baseline_sets = [single.baseline_scores(frame) for frame in frames]
    candidates: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for model_id, probabilities in enumerate(probability_sets):
        score_sets = [candidate_scores(probability) for probability in probabilities]
        for score_name in score_sets[0]:
            candidates[f"model_{model_id}_{score_name}"] = (
                score_sets[1][score_name],
                score_sets[2][score_name],
                score_sets[3][score_name],
            )
    for score_name in candidate_scores(probability_sets[0][0]):
        candidates[f"ensemble_{score_name}"] = tuple(
            single.rank01(
                np.mean(
                    [candidate_scores(model[split_id])[score_name] for model in probability_sets],
                    axis=0,
                )
            )
            for split_id in [1, 2, 3]
        )

    selection_rows = []
    for candidate_name, candidate in candidates.items():
        for learned_weight in [0.70, 0.85, 1.00]:
            for risk_weight in [0.00, 0.10, 0.20]:
                for gap_weight in [0.00, 0.10, 0.20]:
                    if learned_weight + risk_weight + gap_weight > 1.0 + 1e-12:
                        continue
                    support_weight = max(0.0, 1.0 - learned_weight - risk_weight - gap_weight)
                    score = (
                        learned_weight * candidate[0]
                        + risk_weight * baseline_sets[1]["Observed-path risk"]
                        + gap_weight * baseline_sets[1]["KCCRES max-gap certificate"]
                        + support_weight * single.rank01(np.log1p(tuning["donor_count"]))
                    )
                    metrics = single.opportunity_metrics(
                        tuning,
                        targets[1][:, 2],
                        score,
                        0.10,
                        "Structured multitask relational model",
                    )
                    low_tail = tail_risk(
                        targets[1][:, 2], score, clear_capacity
                    )
                    selection_rows.append(
                        {
                            "candidate": candidate_name,
                            "learned_weight": learned_weight,
                            "risk_weight": risk_weight,
                            "gap_weight": gap_weight,
                            "support_weight": support_weight,
                            "low_tail_risk": low_tail,
                            "dual_tail_objective": float(
                                metrics["opportunity_precision"]
                                - selection_tail_weight * low_tail
                            ),
                            **metrics,
                        }
                    )
    selection = pd.DataFrame(selection_rows).sort_values(
        ["dual_tail_objective", "opportunity_precision", "opportunity_capture"],
        ascending=False,
    )
    best = selection.iloc[0]
    learned_scores = candidates[str(best["candidate"])]

    def compose(
        frame: pd.DataFrame,
        learned: np.ndarray,
        baselines: dict[str, np.ndarray],
    ) -> np.ndarray:
        return (
            float(best["learned_weight"]) * learned
            + float(best["risk_weight"]) * baselines["Observed-path risk"]
            + float(best["gap_weight"]) * baselines["KCCRES max-gap certificate"]
            + float(best["support_weight"]) * single.rank01(np.log1p(frame["donor_count"]))
        )

    proposed_calibration = compose(calibration, learned_scores[1], baseline_sets[2])
    proposed_validation = compose(validation, learned_scores[2], baseline_sets[3])
    calibration_scores = {
        "Structured multitask relational model": proposed_calibration,
        **baseline_sets[2],
    }
    validation_scores = {
        "Structured multitask relational model": proposed_validation,
        **baseline_sets[3],
    }
    top10_table = pd.DataFrame(
        [
            single.opportunity_metrics(
                validation, targets[3][:, 2], score, 0.10, name
            )
            for name, score in validation_scores.items()
        ]
    ).sort_values(["opportunity_precision", "opportunity_capture"], ascending=False)

    capacity_rows = []
    calibration_results = {}
    for name, score in calibration_scores.items():
        calibrated = calibrate_capacity_exact(
            targets[2][:, 2], score, residual_risk_target, delta=0.05
        )
        calibration_results[name] = calibrated
        validation_metrics = single.opportunity_metrics(
            validation,
            targets[3][:, 2],
            validation_scores[name],
            float(calibrated["capacity"]),
            name,
        )
        capacity_rows.append({**calibrated, **validation_metrics})
    capacity_table = pd.DataFrame(capacity_rows).sort_values(
        ["risk_control_feasible", "capacity"], ascending=[False, True]
    )

    calibration_export = pd.DataFrame(
        {
            "episode_id": calibration["episode_id"].astype(str),
            "joint_opportunity": targets[2][:, 2],
            **{
                name.lower().replace(" ", "_").replace("-", "_"): score
                for name, score in calibration_scores.items()
            },
        }
    )
    validation_export = pd.DataFrame(
        {
            "episode_id": validation["episode_id"].astype(str),
            "joint_opportunity": targets[3][:, 2],
            **{
                name.lower().replace(" ", "_").replace("-", "_"): score
                for name, score in validation_scores.items()
            },
        }
    )

    proposed_top = top10_table[
        top10_table["method"].eq("Structured multitask relational model")
    ].iloc[0]
    best_baseline = top10_table[
        ~top10_table["method"].eq("Structured multitask relational model")
    ].iloc[0]
    precision_gain = float(
        proposed_top["opportunity_precision"] - best_baseline["opportunity_precision"]
    )
    proposed_calibration_result = calibration_results[
        "Structured multitask relational model"
    ]
    feasible_baseline_capacities = [
        float(result["capacity"])
        for name, result in calibration_results.items()
        if name != "Structured multitask relational model"
        and bool(result["risk_control_feasible"])
    ]
    capacity_saving = (
        min(feasible_baseline_capacities) - float(proposed_calibration_result["capacity"])
        if feasible_baseline_capacities
        and bool(proposed_calibration_result["risk_control_feasible"])
        else math.nan
    )
    best_feasible_baseline_capacity = (
        min(feasible_baseline_capacities) if feasible_baseline_capacities else math.nan
    )
    relative_capacity_saving = (
        capacity_saving / best_feasible_baseline_capacity
        if not math.isnan(capacity_saving) and best_feasible_baseline_capacity > 0.0
        else math.nan
    )
    proposed_capacity_validation = capacity_table[
        capacity_table["method"].eq("Structured multitask relational model")
    ].iloc[0]
    mean_relation_weights = np.mean(
        [row["mean_relation_weights"] for row in attention_rows], axis=0
    )
    maximum_relation_weight = float(mean_relation_weights.max())
    exclusive_certificate = bool(
        proposed_calibration_result["risk_control_feasible"]
        and not feasible_baseline_capacities
        and float(proposed_calibration_result["capacity"]) <= 0.50
    )
    residual_pass = bool(
        float(proposed_capacity_validation["unreviewed_opportunity_rate"])
        <= residual_risk_target
    )
    evaluation_pass = bool(
        (
            precision_gain >= 0.05
            or (not math.isnan(capacity_saving) and capacity_saving >= 0.10 - 1e-12)
            or (
                not math.isnan(relative_capacity_saving)
                and relative_capacity_saving >= 0.20 - 1e-12
            )
            or exclusive_certificate
        )
        and bool(proposed_calibration_result["risk_control_feasible"])
        and residual_pass
        and precision_gain >= -0.01
        and maximum_relation_weight < 0.90
    )

    pd.DataFrame(training_rows).to_csv(output_dir / "model_training.csv", index=False)
    pd.DataFrame(attention_rows).to_json(
        output_dir / "relation_attention.json", orient="records", indent=2
    )
    selection.to_csv(output_dir / "selection_tuning.csv", index=False)
    top10_table.to_csv(output_dir / "validation_top10.csv", index=False)
    capacity_table.to_csv(output_dir / "validation_capacity_control.csv", index=False)
    calibration_export.to_csv(output_dir / "calibration_scores.csv", index=False)
    validation_export.to_csv(output_dir / "validation_scores.csv", index=False)
    report = {
        "problem": (
            "Identify focal chains that fail while compatible continuations retain a statistically "
            "supported recovery advantage, then select the smallest review set whose residual "
            "opportunity risk satisfies a finite-sample upper bound."
        ),
        "ablation": ablation,
        "parallel_models": int(parallel_models),
        "sampled_episodes": int(len(data)),
        "split_sizes": {
            "training": int(len(training)),
            "tuning": int(len(tuning)),
            "risk_calibration": int(len(calibration)),
            "validation": int(len(validation)),
        },
        "reliability_threshold": float(reliability_threshold),
        "opportunity_definition": opportunity_definition,
        "advantage_margin": (
            float(advantage_margin)
            if opportunity_definition == "certified_advantage"
            else None
        ),
        "residual_risk_target": float(residual_risk_target),
        "residual_risk_ratio": (
            None if residual_risk_ratio is None else float(residual_risk_ratio)
        ),
        "dual_tail_learning": {
            "clear_weight": float(clear_weight),
            "clear_capacity": float(clear_capacity),
            "selection_tail_weight": float(selection_tail_weight),
            "selected_tuning_low_tail_risk": float(best["low_tail_risk"]),
        },
        "task_prevalence": {
            label: [float(target[:, task_id].mean()) for target in targets]
            for task_id, label in enumerate(["failure", "robust_advantage", "joint_opportunity"])
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
        "review_capacity_saving_over_feasible_baseline": (
            None if math.isnan(capacity_saving) else float(capacity_saving)
        ),
        "relative_review_capacity_saving": (
            None
            if math.isnan(relative_capacity_saving)
            else float(relative_capacity_saving)
        ),
        "exclusive_finite_sample_certificate": exclusive_certificate,
        "proposed_calibration_feasible": bool(
            proposed_calibration_result["risk_control_feasible"]
        ),
        "validation_residual_risk_at_calibrated_capacity": float(
            proposed_capacity_validation["unreviewed_opportunity_rate"]
        ),
        "no_relation_collapse": bool(maximum_relation_weight < 0.90),
        "evaluation_gate_passed": evaluation_pass,
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
    parser.add_argument("--advantage-margin", type=float, default=0.20)
    parser.add_argument(
        "--opportunity-definition",
        choices=["certified_advantage", "feasible_alternative"],
        default="certified_advantage",
    )
    parser.add_argument("--residual-risk-target", type=float, default=0.02)
    parser.add_argument("--residual-risk-ratio", type=float, default=None)
    parser.add_argument(
        "--ablation",
        choices=["full", "core", "no_relations", "no_multitask", "no_logic", "no_capacity_loss"],
        default="full",
    )
    parser.add_argument("--parallel-models", type=int, default=1)
    parser.add_argument("--clear-weight", type=float, default=0.0)
    parser.add_argument("--clear-capacity", type=float, default=0.95)
    parser.add_argument("--selection-tail-weight", type=float, default=0.0)
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
        args.residual_risk_ratio,
        args.ablation,
        max(1, args.parallel_models),
        args.opportunity_definition,
        args.clear_weight,
        args.clear_capacity,
        args.selection_tail_weight,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
