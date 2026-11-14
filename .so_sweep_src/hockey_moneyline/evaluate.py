"""Chronological evaluation for the canonical simple probabilistic baselines."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from hockey_moneyline.elo import ELO_CONFIG, ELO_FEATURE_COLUMN, PLUS_ELO_FEATURE_COLUMNS
from hockey_moneyline.errors import EvaluationError
from hockey_moneyline.advanced_stats import (
    AdvancedTeamGame,
    build_pregame_corsi_share_features,
    build_pregame_xgoal_share_features,
)
from hockey_moneyline.corsi_share import (
    BASELINE_V1_PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE_PLUS_PRIOR_CORSI_SHARE,
    CORSI_SHARE_CONFIG,
    ISSUE21_ELIGIBLE_ROW_COUNTS as CORSI_ISSUE21_ELIGIBLE_ROW_COUNTS,
    ISSUE21_XGOAL_LOGISTIC_VALIDATION_METRICS as CORSI_ISSUE21_XGOAL_LOGISTIC_VALIDATION_METRICS,
    PLUS_XGOAL_PLUS_PRIOR_CORSI_SHARE_FEATURE_COLUMNS,
    assert_prior_corsi_share_contract,
    attach_pregame_corsi_share_features,
    corsi_features_valid,
    xgoal_and_corsi_features_valid,
)
from hockey_moneyline.features import (
    assert_ordered5_lags_complete,
    build_features,
    feature_matrix,
)
from hockey_moneyline.linear_recency import (
    BASELINE_V1_PLUS_ELO_LINEAR_RECENCY10,
    LINEAR_RECENCY_COLUMNS,
    LINEAR_RECENCY_CONFIG,
    PLUS_ELO_LINEAR_RECENCY10_FEATURE_COLUMNS,
    assert_linear_recency_contract,
)
from hockey_moneyline.ordered_lags import (
    BASELINE_V1_PLUS_ELO_ORDERED5,
    ORDERED_LAG_COLUMNS,
    ORDERED_LAG_CONFIG,
    PLUS_ELO_ORDERED5_FEATURE_COLUMNS,
    assert_incumbent_columns_unchanged,
)
from hockey_moneyline.hgb import (
    CARRY_FORWARD_MODEL_FAMILY,
    DO_NOT_CARRY_FORWARD_MODEL_FAMILY,
    FROZEN_FEATURE_COLUMNS,
    FROZEN_FEATURE_FAMILY,
    HGB_CONFIG,
    ISSUE21_ELIGIBLE_ROW_COUNTS,
    ISSUE21_XGOAL_LOGISTIC_VALIDATION_METRICS,
    LOGISTIC_REPRODUCTION_ATOL,
    _new_hgb_model,
    assert_hgb_comparison_contract,
    hgb_model_record,
)
from hockey_moneyline.xgoal_share import (
    BASELINE_V1_PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE,
    INCUMBENT_REPRODUCTION_ATOL,
    ISSUE13_PLUS_ELO_VALIDATION_METRICS,
    PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE_FEATURE_COLUMNS,
    XGOAL_SHARE_CONFIG,
    assert_prior_xgoal_share_contract,
    attach_pregame_xgoal_share_features,
    xgoal_features_valid,
)
from hockey_moneyline.provenance import (
    DEFAULT_SEASONS,
    FINAL_TEST_SEASON,
    MIN_PRIOR_GAMES,
    RECENT_FORM_WINDOW,
    TEST_END_INCLUSIVE,
    TEST_START,
    TRAIN_SEASONS,
    VALIDATION_END_EXCLUSIVE,
    VALIDATION_START,
    provenance_record,
)
from hockey_moneyline.schema import FEATURE_COLUMNS, NormalizedGame, parse_game_date

LOGISTIC_SOLVER = "lbfgs"
LOGISTIC_MAX_ITER = 1000
LOGISTIC_RANDOM_STATE = 42
LOGISTIC_MODEL_DESCRIPTION = (
    "StandardScaler + LogisticRegression(solver=lbfgs, max_iter=1000, random_state=42)"
)
BASELINE_V1 = "baseline_v1"
BASELINE_V1_PLUS_ELO = "baseline_v1_plus_elo"
VALIDATION_HISTORY_SEASONS = tuple(
    season for season in DEFAULT_SEASONS if season != FINAL_TEST_SEASON
)


# Legacy Issue #2 split helpers are retained for reproducibility/regression.
@dataclass(frozen=True, slots=True)
class SplitSpec:
    name: str
    start: date | None = None
    end_exclusive: date | None = None
    seasons: tuple[int, ...] | None = None


@dataclass(frozen=True, slots=True)
class ForwardFoldSpec:
    name: str
    train_seasons: tuple[int, ...]
    evaluation_season: int
    role: str = "validation"


def default_split_specs() -> list[SplitSpec]:
    return [
        SplitSpec(name="train", seasons=TRAIN_SEASONS),
        SplitSpec(
            name="validation",
            start=date.fromisoformat(VALIDATION_START),
            end_exclusive=date.fromisoformat(VALIDATION_END_EXCLUSIVE),
        ),
        SplitSpec(
            name="test",
            start=date.fromisoformat(TEST_START),
            end_exclusive=date.fromisoformat(TEST_END_INCLUSIVE) + timedelta(days=1),
        ),
    ]


def default_forward_fold_specs() -> list[ForwardFoldSpec]:
    seasons = tuple(DEFAULT_SEASONS)
    return [
        ForwardFoldSpec("fold_a", seasons[:1], seasons[1], "validation"),
        ForwardFoldSpec("fold_b", seasons[:2], seasons[2], "validation"),
        ForwardFoldSpec("fold_c", seasons[:3], seasons[3], "validation"),
        ForwardFoldSpec("held_out_test", seasons[:4], seasons[4], "test"),
    ]


def default_validation_fold_specs() -> list[ForwardFoldSpec]:
    seasons = tuple(VALIDATION_HISTORY_SEASONS)
    return [
        ForwardFoldSpec("fold_a", seasons[:1], seasons[1], "validation"),
        ForwardFoldSpec("fold_b", seasons[:2], seasons[2], "validation"),
        ForwardFoldSpec("fold_c", seasons[:3], seasons[3], "validation"),
    ]


def shared_logistic_model_record() -> dict[str, Any]:
    return {
        "model": LOGISTIC_MODEL_DESCRIPTION,
        "scaler": "StandardScaler",
        "classifier": "LogisticRegression",
        "solver": LOGISTIC_SOLVER,
        "max_iter": LOGISTIC_MAX_ITER,
        "random_state": LOGISTIC_RANDOM_STATE,
        "hyperparameter_search": False,
    }


def assign_split(row: dict[str, Any], specs: Sequence[SplitSpec]) -> str | None:
    game_date = parse_game_date(row["game_date"])
    season = int(row["season"])
    for spec in specs:
        if spec.seasons is not None and season not in spec.seasons:
            continue
        if spec.start is not None and game_date < spec.start:
            continue
        if spec.end_exclusive is not None and game_date >= spec.end_exclusive:
            continue
        if spec.seasons is None and spec.start is None and spec.end_exclusive is None:
            continue
        return spec.name
    return None


def _eligible(row: dict[str, Any], min_prior_games: int) -> bool:
    return (
        row["home_prior_games"] >= min_prior_games
        and row["away_prior_games"] >= min_prior_games
    )


def _probabilities_in_range(probs: np.ndarray) -> None:
    if np.any((probs <= 0.0) | (probs >= 1.0)):
        raise EvaluationError("Predicted probabilities must be inside (0, 1)")


def _calibration_summary(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict[str, Any]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    records: list[dict[str, float]] = []
    ece = 0.0
    total = len(y_true)
    for index in range(n_bins):
        left = bins[index]
        right = bins[index + 1]
        if index == n_bins - 1:
            mask = (y_prob >= left) & (y_prob <= right)
        else:
            mask = (y_prob >= left) & (y_prob < right)
        count = int(mask.sum())
        if count == 0:
            continue
        mean_pred = float(y_prob[mask].mean())
        mean_actual = float(y_true[mask].mean())
        records.append(
            {
                "bin_left": float(left),
                "bin_right": float(right),
                "count": count,
                "mean_predicted": mean_pred,
                "mean_actual": mean_actual,
                "abs_error": abs(mean_pred - mean_actual),
            }
        )
        ece += (count / total) * abs(mean_pred - mean_actual)
    return {"n_bins": n_bins, "ece": ece, "bins": records}


def _metric_block(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    _probabilities_in_range(y_prob)
    y_hat = (y_prob >= 0.5).astype(int)
    metrics: dict[str, Any] = {
        "n": int(len(y_true)),
        "home_win_rate": float(y_true.mean()) if len(y_true) else None,
        "accuracy": float(accuracy_score(y_true, y_hat)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "mean_predicted_home_win": float(y_prob.mean()),
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = None
        metrics["roc_auc_note"] = "ROC-AUC omitted because only one class is present"
    metrics["calibration"] = _calibration_summary(y_true, y_prob)
    return metrics


def _window_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n_rows": 0, "start": None, "end": None, "seasons": []}
    dates = [str(row["game_date"]) for row in rows]
    seasons = sorted({int(row["season"]) for row in rows})
    return {
        "n_rows": len(rows),
        "start": min(dates),
        "end": max(dates),
        "seasons": seasons,
    }


def _new_logistic_model() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=LOGISTIC_MAX_ITER,
                    solver=LOGISTIC_SOLVER,
                    random_state=LOGISTIC_RANDOM_STATE,
                ),
            ),
        ]
    )


def _predict(
    model: Pipeline,
    rows: Sequence[dict[str, Any]],
    columns: Sequence[str] | None = None,
) -> np.ndarray:
    selected = tuple(columns) if columns is not None else FEATURE_COLUMNS
    matrix = np.array(feature_matrix(rows, columns=selected), dtype=float)
    if matrix.shape[1] != len(selected):
        raise EvaluationError("Prediction feature-column count does not match training")
    probs = model.predict_proba(matrix)[:, 1]
    return np.clip(probs, 1e-15, 1.0 - 1e-15)



def _lag_coefficient_diagnostics(
    model: Pipeline,
    columns: Sequence[str],
) -> dict[str, float] | None:
    """Return standardized logistic coefficients for ordered lag columns only."""
    selected = tuple(columns)
    if not any(column in ORDERED_LAG_COLUMNS for column in selected):
        return None
    coefficients = model.named_steps["classifier"].coef_[0]
    if len(coefficients) != len(selected):
        raise EvaluationError("Coefficient count does not match feature columns")
    return {
        column: float(coefficients[index])
        for index, column in enumerate(selected)
        if column in ORDERED_LAG_COLUMNS
    }


def _fit_existing_baselines(
    train_rows: Sequence[dict[str, Any]],
    evaluation_rows: Sequence[dict[str, Any]],
    columns: Sequence[str] | None = None,
) -> dict[str, Any]:
    selected = tuple(columns) if columns is not None else FEATURE_COLUMNS
    train_y = np.array([int(row["home_win"]) for row in train_rows], dtype=int)
    evaluation_y = np.array([int(row["home_win"]) for row in evaluation_rows], dtype=int)
    base_rate = float(train_y.mean())
    if not 0.0 < base_rate < 1.0:
        raise EvaluationError("Train home-win base rate must be inside (0, 1)")

    train_x = np.array(feature_matrix(train_rows, columns=selected), dtype=float)
    model = _new_logistic_model()
    model.fit(train_x, train_y)
    logistic_prob = _predict(model, evaluation_rows, columns=selected)

    return {
        "train_home_win_rate": base_rate,
        "comparator": {
            "name": "train_home_win_rate",
            "probability": base_rate,
            "metrics": _metric_block(
                evaluation_y,
                np.full(len(evaluation_rows), base_rate, dtype=float),
            ),
        },
        "logistic_regression": {
            **shared_logistic_model_record(),
            "feature_columns": list(selected),
            "metrics": _metric_block(evaluation_y, logistic_prob),
            "standardized_lag_coefficients": _lag_coefficient_diagnostics(model, selected),
        },
    }


def _validate_forward_specs(specs: Sequence[ForwardFoldSpec]) -> None:
    if not specs:
        raise EvaluationError("Forward-chaining evaluation requires at least one fold")
    names: set[str] = set()
    previous_eval: int | None = None
    test_count = 0
    for spec in specs:
        if spec.name in names:
            raise EvaluationError(f"Duplicate forward fold name: {spec.name}")
        names.add(spec.name)
        if not spec.train_seasons:
            raise EvaluationError(f"Forward fold {spec.name} has no training seasons")
        if tuple(sorted(set(spec.train_seasons))) != spec.train_seasons:
            raise EvaluationError(f"Forward fold {spec.name} training seasons must be unique and ordered")
        if max(spec.train_seasons) >= spec.evaluation_season:
            raise EvaluationError(
                f"Forward fold {spec.name} is not strictly chronological: "
                f"train={spec.train_seasons}, evaluation={spec.evaluation_season}"
            )
        if previous_eval is not None and spec.evaluation_season <= previous_eval:
            raise EvaluationError("Forward fold evaluation seasons must increase strictly")
        previous_eval = spec.evaluation_season
        if spec.role not in {"validation", "test"}:
            raise EvaluationError(f"Unsupported forward fold role: {spec.role!r}")
        test_count += int(spec.role == "test")
    if test_count != 1:
        raise EvaluationError("Forward-chaining contract requires exactly one held-out test fold")
    test_spec = next(spec for spec in specs if spec.role == "test")
    if test_spec is not specs[-1]:
        raise EvaluationError("Held-out test must be the final forward-chaining fold")


def _validate_chronological_fold(spec: ForwardFoldSpec, previous_eval: int | None) -> None:
    if not spec.train_seasons:
        raise EvaluationError(f"Forward fold {spec.name} has no training seasons")
    if tuple(sorted(set(spec.train_seasons))) != spec.train_seasons:
        raise EvaluationError(f"Forward fold {spec.name} training seasons must be unique and ordered")
    if max(spec.train_seasons) >= spec.evaluation_season:
        raise EvaluationError(
            f"Forward fold {spec.name} is not strictly chronological: "
            f"train={spec.train_seasons}, evaluation={spec.evaluation_season}"
        )
    if previous_eval is not None and spec.evaluation_season <= previous_eval:
        raise EvaluationError("Forward fold evaluation seasons must increase strictly")


def _validate_validation_only_specs(specs: Sequence[ForwardFoldSpec]) -> None:
    if not specs:
        raise EvaluationError("Validation-only ablation requires at least one fold")
    names: set[str] = set()
    previous_eval: int | None = None
    for spec in specs:
        if spec.name in names:
            raise EvaluationError(f"Duplicate forward fold name: {spec.name}")
        names.add(spec.name)
        if spec.role != "validation":
            raise EvaluationError(
                "Validation-only ablation cannot include a non-validation fold "
                f"({spec.name} role={spec.role!r})"
            )
        if spec.evaluation_season == FINAL_TEST_SEASON or FINAL_TEST_SEASON in spec.train_seasons:
            raise EvaluationError(
                "Validation-only ablation cannot evaluate the 2025-26 held-out test"
            )
        _validate_chronological_fold(spec, previous_eval)
        previous_eval = spec.evaluation_season


def _reject_held_out_test_rows(rows: Sequence[dict[str, Any]]) -> None:
    if any(int(row["season"]) == FINAL_TEST_SEASON for row in rows):
        raise EvaluationError(
            "Validation-only ablation cannot load or evaluate the 2025-26 held-out test season"
        )


def _run_forward_folds(
    rows: Sequence[dict[str, Any]],
    specs: Sequence[ForwardFoldSpec],
    min_prior_games: int,
    columns: Sequence[str],
    extra_eligible: Callable[[dict[str, Any]], bool] | None = None,
    extra_exclusion_name: str | None = None,
) -> list[dict[str, Any]]:
    selected = tuple(columns)
    folds: list[dict[str, Any]] = []
    for spec in specs:
        train_candidates = [row for row in rows if int(row["season"]) in spec.train_seasons]
        evaluation_candidates = [row for row in rows if int(row["season"]) == spec.evaluation_season]
        if not train_candidates:
            raise EvaluationError(f"Forward fold {spec.name} has no training rows")
        if not evaluation_candidates:
            raise EvaluationError(f"Forward fold {spec.name} has no evaluation rows")

        train_prior = [row for row in train_candidates if _eligible(row, min_prior_games)]
        evaluation_prior = [
            row for row in evaluation_candidates if _eligible(row, min_prior_games)
        ]
        train_rows = (
            [row for row in train_prior if extra_eligible(row)]
            if extra_eligible is not None
            else list(train_prior)
        )
        evaluation_rows = (
            [row for row in evaluation_prior if extra_eligible(row)]
            if extra_eligible is not None
            else list(evaluation_prior)
        )
        train_extra_excluded = len(train_prior) - len(train_rows)
        evaluation_extra_excluded = len(evaluation_prior) - len(evaluation_rows)
        if extra_eligible is not None and (
            train_extra_excluded or evaluation_extra_excluded
        ):
            label = extra_exclusion_name or "extra eligibility"
            raise EvaluationError(
                f"Forward fold {spec.name} unexpectedly lost "
                f"{train_extra_excluded} train and {evaluation_extra_excluded} "
                f"evaluation rows after the five-prior-game gate ({label})"
            )
        if not train_rows:
            raise EvaluationError(f"Forward fold {spec.name} has no eligible training rows")
        if not evaluation_rows:
            raise EvaluationError(f"Forward fold {spec.name} has no eligible evaluation rows")

        if any(column in ORDERED_LAG_COLUMNS for column in selected):
            assert_ordered5_lags_complete(train_rows)
            assert_ordered5_lags_complete(evaluation_rows)

        train_end = max(parse_game_date(row["game_date"]) for row in train_rows)
        evaluation_start = min(parse_game_date(row["game_date"]) for row in evaluation_rows)
        if train_end >= evaluation_start:
            raise EvaluationError(
                f"Forward fold {spec.name} violates chronology: "
                f"train end {train_end} >= evaluation start {evaluation_start}"
            )

        excluded_rows = {
            "train_min_prior_games": len(train_candidates) - len(train_prior),
            "evaluation_min_prior_games": len(evaluation_candidates) - len(evaluation_prior),
        }
        if extra_exclusion_name is not None:
            excluded_rows[f"train_{extra_exclusion_name}"] = train_extra_excluded
            excluded_rows[f"evaluation_{extra_exclusion_name}"] = evaluation_extra_excluded

        baseline = _fit_existing_baselines(train_rows, evaluation_rows, columns=selected)
        folds.append(
            {
                "name": spec.name,
                "role": spec.role,
                "train_seasons": list(spec.train_seasons),
                "evaluation_season": spec.evaluation_season,
                "train_window": _window_summary(train_candidates)
                | {"n_eligible": len(train_rows)},
                "evaluation_window": _window_summary(evaluation_candidates)
                | {"n_eligible": len(evaluation_rows)},
                "excluded_rows": excluded_rows,
                **baseline,
            }
        )
    return folds


def _fold_model_metric(
    fold: dict[str, Any],
    key: str,
    model_key: str = "logistic_regression",
) -> Any:
    metrics = fold[model_key]["metrics"]
    if key == "ece":
        return metrics["calibration"]["ece"]
    return metrics[key]


def _logistic_metric(fold: dict[str, Any], key: str) -> Any:
    return _fold_model_metric(fold, key, "logistic_regression")


def _macro_average_model(
    folds: Sequence[dict[str, Any]],
    model_key: str = "logistic_regression",
) -> dict[str, Any]:
    keys = ("accuracy", "roc_auc", "log_loss", "brier", "ece")
    summary: dict[str, Any] = {"n_folds": len(folds)}
    for key in keys:
        values = [_fold_model_metric(fold, key, model_key) for fold in folds]
        if any(value is None for value in values):
            summary[key] = None
        else:
            summary[key] = float(sum(values) / len(values))
    return summary


def _macro_average_logistic(folds: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return _macro_average_model(folds, "logistic_regression")


def _continuation_gate(
    baseline_family: dict[str, Any],
    elo_family: dict[str, Any],
    *,
    baseline_model_key: str = "logistic_regression",
    experimental_model_key: str = "logistic_regression",
    pass_decision: str = "carry forward",
    fail_decision: str = "do not carry forward",
    issue_label: str = "Issue #13",
) -> dict[str, Any]:
    baseline_macro = baseline_family["macro_average"]
    elo_macro = elo_family["macro_average"]
    worse_log_loss: list[str] = []
    worse_brier: list[str] = []
    fold_comparisons: list[dict[str, Any]] = []

    for baseline_fold, elo_fold in zip(
        baseline_family["folds"],
        elo_family["folds"],
        strict=True,
    ):
        if baseline_fold["name"] != elo_fold["name"]:
            raise EvaluationError("Ablation fold names do not align")
        name = baseline_fold["name"]
        base_ll = _fold_model_metric(baseline_fold, "log_loss", baseline_model_key)
        elo_ll = _fold_model_metric(elo_fold, "log_loss", experimental_model_key)
        base_brier = _fold_model_metric(baseline_fold, "brier", baseline_model_key)
        elo_brier = _fold_model_metric(elo_fold, "brier", experimental_model_key)
        log_loss_worse = elo_ll > base_ll
        brier_worse = elo_brier > base_brier
        if log_loss_worse:
            worse_log_loss.append(name)
        if brier_worse:
            worse_brier.append(name)
        fold_comparisons.append(
            {
                "name": name,
                "log_loss_delta": elo_ll - base_ll,
                "brier_delta": elo_brier - base_brier,
                "log_loss_worse": log_loss_worse,
                "brier_worse": brier_worse,
            }
        )

    macro_log_loss_improves = elo_macro["log_loss"] < baseline_macro["log_loss"]
    macro_brier_improves = elo_macro["brier"] < baseline_macro["brier"]
    log_loss_fold_ok = len(worse_log_loss) <= 1
    brier_fold_ok = len(worse_brier) <= 1

    auc_delta = None
    if baseline_macro["roc_auc"] is not None and elo_macro["roc_auc"] is not None:
        auc_delta = elo_macro["roc_auc"] - baseline_macro["roc_auc"]
    ece_delta = None
    if baseline_macro["ece"] is not None and elo_macro["ece"] is not None:
        ece_delta = elo_macro["ece"] - baseline_macro["ece"]

    carry = (
        macro_log_loss_improves
        and macro_brier_improves
        and log_loss_fold_ok
        and brier_fold_ok
    )
    decision = pass_decision if carry else fail_decision
    reasons = [
        f"macro log loss improves: {macro_log_loss_improves}",
        f"macro Brier improves: {macro_brier_improves}",
        f"log loss worse in {len(worse_log_loss)} fold(s): {worse_log_loss}",
        f"Brier worse in {len(worse_brier)} fold(s): {worse_brier}",
        "ROC-AUC and ECE are qualitative diagnostics only; no numeric cutoff is applied",
    ]
    return {
        "decision": decision,
        "macro_log_loss_improves": macro_log_loss_improves,
        "macro_brier_improves": macro_brier_improves,
        "log_loss_worse_folds": worse_log_loss,
        "brier_worse_folds": worse_brier,
        "macro_roc_auc_delta": auc_delta,
        "macro_ece_delta": ece_delta,
        "calibration_auc_role": (
            f"Qualitative diagnostic only. {issue_label} requires that ECE/calibration "
            "and ROC-AUC not reveal an obvious contradictory failure that makes "
            "the aggregate improvement misleading. No numeric AUC/ECE cutoff is applied."
        ),
        "fold_comparisons": fold_comparisons,
        "reasons": reasons,
        "promotes_champion_model": False,
        "authorizes_held_out_test": False,
    }


def evaluate_forward_chaining_rows(
    rows: Sequence[dict[str, Any]],
    min_prior_games: int = MIN_PRIOR_GAMES,
    fold_specs: Sequence[ForwardFoldSpec] | None = None,
) -> dict[str, Any]:
    specs = list(fold_specs or default_forward_fold_specs())
    _validate_forward_specs(specs)
    folds = _run_forward_folds(rows, specs, min_prior_games, FEATURE_COLUMNS)

    return {
        "source": provenance_record(),
        "evaluation_contract": "strict_forward_chaining_by_season",
        "feature_columns": list(FEATURE_COLUMNS),
        "min_prior_games": min_prior_games,
        "feature_timing": (
            "Features are built once in (game_date, game_id) order from the full "
            "historical sequence and use only prior completed games with "
            "history.game_date < target.game_date. Same-day results are excluded. "
            "Every fold then trains only on seasons strictly earlier than its "
            "evaluation season."
        ),
        "folds": folds,
        "held_out_test_season": next(
            spec.evaluation_season for spec in specs if spec.role == "test"
        ),
        "notes": [
            "The only models rerun are the existing train-home-win-rate comparator and existing logistic regression baseline.",
            "Validation folds are sanity evidence only; the final test fold is not used for tuning or model selection.",
            "No odds, roster/goalie context, deployment, Production, or worker/runtime work is included.",
        ],
    }


def evaluate_forward_chaining_games(
    games: Sequence[NormalizedGame],
    min_prior_games: int = MIN_PRIOR_GAMES,
    fold_specs: Sequence[ForwardFoldSpec] | None = None,
) -> dict[str, Any]:
    if not games:
        raise EvaluationError("No games supplied for forward-chaining evaluation")
    specs = list(fold_specs or default_forward_fold_specs())
    if fold_specs is None:
        present = {game.season for game in games}
        missing = [season for season in DEFAULT_SEASONS if season not in present]
        if missing:
            raise EvaluationError(f"Canonical MoneyPuck history is missing seasons: {missing}")
        if FINAL_TEST_SEASON not in present:
            raise EvaluationError(f"Held-out test season {FINAL_TEST_SEASON} is missing")

    rows = build_features(games)
    report = evaluate_forward_chaining_rows(
        rows,
        min_prior_games=min_prior_games,
        fold_specs=specs,
    )
    report["historical_data_window"] = {
        "seasons": sorted({game.season for game in games}),
        "start": min(game.game_date for game in games).isoformat(),
        "end": max(game.game_date for game in games).isoformat(),
        "n_normalized_games": len(games),
        "n_feature_rows": len(rows),
    }
    return report


def evaluate_validation_ablation_rows(
    rows: Sequence[dict[str, Any]],
    min_prior_games: int = MIN_PRIOR_GAMES,
    fold_specs: Sequence[ForwardFoldSpec] | None = None,
) -> dict[str, Any]:
    specs = list(fold_specs or default_validation_fold_specs())
    _validate_validation_only_specs(specs)
    _reject_held_out_test_rows(rows)

    families: dict[str, dict[str, Any]] = {}
    for name, columns in (
        (BASELINE_V1, FEATURE_COLUMNS),
        (BASELINE_V1_PLUS_ELO, PLUS_ELO_FEATURE_COLUMNS),
    ):
        folds = _run_forward_folds(rows, specs, min_prior_games, columns)
        families[name] = {
            "name": name,
            "feature_columns": list(columns),
            "added_feature_columns": (
                [ELO_FEATURE_COLUMN] if name == BASELINE_V1_PLUS_ELO else []
            ),
            "folds": folds,
            "macro_average": _macro_average_logistic(folds),
        }

    gate = _continuation_gate(families[BASELINE_V1], families[BASELINE_V1_PLUS_ELO])
    return {
        "source": provenance_record(),
        "experiment": "issue_13_elo_feature_ablation",
        "evaluation_contract": "validation_only_forward_chaining",
        "held_out_test_evaluated": False,
        "held_out_test_season_excluded": FINAL_TEST_SEASON,
        "shared_logistic_model": shared_logistic_model_record(),
        "elo": dict(ELO_CONFIG),
        "min_prior_games": min_prior_games,
        "feature_timing": (
            "Features are built once in (game_date, game_id) order. Baseline "
            "and Elo snapshots for a calendar date are emitted before any "
            "same-day result updates team or Elo state. Features use only "
            "history.game_date < target.game_date. This pathway evaluates "
            "only validation folds A/B/C."
        ),
        "feature_families": families,
        "continuation_gate": gate,
        "notes": [
            "Issue #13 compares baseline_v1 against baseline_v1_plus_elo using the same logistic regression.",
            "The only added feature is home_elo_minus_away_elo.",
            "The 2025-26 held-out test was not evaluated, inspected, or recorded.",
            "Passing the continuation gate does not promote a champion model and does not authorize the held-out test.",
            "No odds, roster/goalie context, deployment, Production, or worker/runtime work is included.",
        ],
    }


def evaluate_validation_ablation_games(
    games: Sequence[NormalizedGame],
    min_prior_games: int = MIN_PRIOR_GAMES,
    fold_specs: Sequence[ForwardFoldSpec] | None = None,
) -> dict[str, Any]:
    if not games:
        raise EvaluationError("No games supplied for validation-only ablation")
    if any(game.season == FINAL_TEST_SEASON for game in games):
        raise EvaluationError(
            "Validation-only ablation cannot load or evaluate the 2025-26 held-out test season"
        )
    present = {game.season for game in games}
    missing = [season for season in VALIDATION_HISTORY_SEASONS if season not in present]
    if fold_specs is None and missing:
        raise EvaluationError(
            f"Validation-only ablation is missing required history seasons: {missing}"
        )

    rows = build_features(games)
    report = evaluate_validation_ablation_rows(
        rows,
        min_prior_games=min_prior_games,
        fold_specs=fold_specs,
    )
    report["historical_data_window"] = {
        "seasons": sorted({game.season for game in games}),
        "start": min(game.game_date for game in games).isoformat(),
        "end": max(game.game_date for game in games).isoformat(),
        "n_normalized_games": len(games),
        "n_feature_rows": len(rows),
    }
    return report


# ----- Legacy Issue #2 evaluator retained below for reproducible regression. -----

def evaluate_ordered_lag_ablation_rows(
    rows: Sequence[dict[str, Any]],
    min_prior_games: int = MIN_PRIOR_GAMES,
    fold_specs: Sequence[ForwardFoldSpec] | None = None,
) -> dict[str, Any]:
    """Issue #15 validation-only ordered-lag ablation.

    Compares baseline_v1_plus_elo against baseline_v1_plus_elo_ordered5 on
    folds A/B/C only. Never evaluates the 2025-26 held-out test.
    """
    assert_incumbent_columns_unchanged()
    specs = list(fold_specs or default_validation_fold_specs())
    _validate_validation_only_specs(specs)
    _reject_held_out_test_rows(rows)

    families: dict[str, dict[str, Any]] = {}
    for name, columns in (
        (BASELINE_V1_PLUS_ELO, PLUS_ELO_FEATURE_COLUMNS),
        (BASELINE_V1_PLUS_ELO_ORDERED5, PLUS_ELO_ORDERED5_FEATURE_COLUMNS),
    ):
        folds = _run_forward_folds(rows, specs, min_prior_games, columns)
        families[name] = {
            "name": name,
            "feature_columns": list(columns),
            "removed_feature_columns": (
                list(ORDERED_LAG_CONFIG["removed_from_incumbent"])
                if name == BASELINE_V1_PLUS_ELO_ORDERED5
                else []
            ),
            "added_feature_columns": (
                list(ORDERED_LAG_COLUMNS)
                if name == BASELINE_V1_PLUS_ELO_ORDERED5
                else []
            ),
            "folds": folds,
            "macro_average": _macro_average_logistic(folds),
            "lag_coefficients_by_fold": (
                {
                    fold["name"]: fold["logistic_regression"]["standardized_lag_coefficients"]
                    for fold in folds
                }
                if name == BASELINE_V1_PLUS_ELO_ORDERED5
                else None
            ),
        }

    gate = _continuation_gate(
        families[BASELINE_V1_PLUS_ELO],
        families[BASELINE_V1_PLUS_ELO_ORDERED5],
    )
    gate["calibration_auc_role"] = (
        "Qualitative diagnostic only. Issue #15 requires that ECE/calibration "
        "and ROC-AUC not reveal an obvious contradictory failure that makes "
        "the aggregate improvement misleading. No numeric AUC/ECE cutoff is applied."
    )
    deltas = {
        key: (
            None
            if families[BASELINE_V1_PLUS_ELO]["macro_average"][key] is None
            or families[BASELINE_V1_PLUS_ELO_ORDERED5]["macro_average"][key] is None
            else families[BASELINE_V1_PLUS_ELO_ORDERED5]["macro_average"][key]
            - families[BASELINE_V1_PLUS_ELO]["macro_average"][key]
        )
        for key in ("accuracy", "roc_auc", "log_loss", "brier", "ece")
    }
    return {
        "source": provenance_record(),
        "experiment": "issue_15_ordered_lag_feature_ablation",
        "evaluation_contract": "validation_only_forward_chaining",
        "held_out_test_evaluated": False,
        "held_out_test_season_excluded": FINAL_TEST_SEASON,
        "shared_logistic_model": shared_logistic_model_record(),
        "elo": dict(ELO_CONFIG),
        "ordered_lags": dict(ORDERED_LAG_CONFIG),
        "min_prior_games": min_prior_games,
        "feature_timing": (
            "Features are built once in (game_date, game_id) order. Baseline, "
            "Elo, and ordered-lag snapshots for a calendar date are emitted "
            "before any same-day result updates team or Elo state. Features "
            "use only history.game_date < target.game_date. This pathway "
            "evaluates only validation folds A/B/C."
        ),
        "feature_families": families,
        "macro_deltas_ordered5_minus_incumbent": deltas,
        "continuation_gate": gate,
        "notes": [
            "Issue #15 compares baseline_v1_plus_elo against baseline_v1_plus_elo_ordered5 using the same logistic regression.",
            "ordered5 removes home_recent_win_rate and away_recent_win_rate and adds ten ordered result lags.",
            "Lag coefficients are diagnostic only and were not used to tune lag count or drop lags.",
            "The 2025-26 held-out test was not evaluated, inspected, or recorded.",
            "Passing the continuation gate does not promote a champion model and does not authorize the held-out test.",
            "No odds, roster/goalie context, deployment, Production, or worker/runtime work is included.",
        ],
    }


def evaluate_ordered_lag_ablation_games(
    games: Sequence[NormalizedGame],
    min_prior_games: int = MIN_PRIOR_GAMES,
    fold_specs: Sequence[ForwardFoldSpec] | None = None,
) -> dict[str, Any]:
    if not games:
        raise EvaluationError("No games supplied for ordered-lag validation-only ablation")
    if any(game.season == FINAL_TEST_SEASON for game in games):
        raise EvaluationError(
            "Validation-only ablation cannot load or evaluate the 2025-26 held-out test season"
        )
    present = {game.season for game in games}
    missing = [season for season in VALIDATION_HISTORY_SEASONS if season not in present]
    if fold_specs is None and missing:
        raise EvaluationError(
            f"Validation-only ablation is missing required history seasons: {missing}"
        )

    rows = build_features(games)
    report = evaluate_ordered_lag_ablation_rows(
        rows,
        min_prior_games=min_prior_games,
        fold_specs=fold_specs,
    )
    report["historical_data_window"] = {
        "seasons": sorted({game.season for game in games}),
        "start": min(game.game_date for game in games).isoformat(),
        "end": max(game.game_date for game in games).isoformat(),
        "n_normalized_games": len(games),
        "n_feature_rows": len(rows),
    }
    return report


def evaluate_linear_recency_ablation_rows(
    rows: Sequence[dict[str, Any]],
    min_prior_games: int = MIN_PRIOR_GAMES,
    fold_specs: Sequence[ForwardFoldSpec] | None = None,
) -> dict[str, Any]:
    """Issue #17 validation-only linear recency-weighting ablation.

    Compares baseline_v1_plus_elo against
    baseline_v1_plus_elo_linear_recency10 on folds A/B/C only. Never
    evaluates the 2025-26 held-out test.
    """
    assert_incumbent_columns_unchanged()
    assert_linear_recency_contract()
    specs = list(fold_specs or default_validation_fold_specs())
    _validate_validation_only_specs(specs)
    _reject_held_out_test_rows(rows)

    families: dict[str, dict[str, Any]] = {}
    for name, columns in (
        (BASELINE_V1_PLUS_ELO, PLUS_ELO_FEATURE_COLUMNS),
        (BASELINE_V1_PLUS_ELO_LINEAR_RECENCY10, PLUS_ELO_LINEAR_RECENCY10_FEATURE_COLUMNS),
    ):
        folds = _run_forward_folds(rows, specs, min_prior_games, columns)
        families[name] = {
            "name": name,
            "feature_columns": list(columns),
            "removed_feature_columns": (
                list(LINEAR_RECENCY_CONFIG["removed_from_incumbent"])
                if name == BASELINE_V1_PLUS_ELO_LINEAR_RECENCY10
                else []
            ),
            "added_feature_columns": (
                list(LINEAR_RECENCY_COLUMNS)
                if name == BASELINE_V1_PLUS_ELO_LINEAR_RECENCY10
                else []
            ),
            "folds": folds,
            "macro_average": _macro_average_logistic(folds),
        }

    gate = _continuation_gate(
        families[BASELINE_V1_PLUS_ELO],
        families[BASELINE_V1_PLUS_ELO_LINEAR_RECENCY10],
    )
    gate["calibration_auc_role"] = (
        "Qualitative diagnostic only. Issue #17 requires that ECE/calibration "
        "and ROC-AUC not reveal an obvious contradictory failure that makes "
        "the aggregate improvement misleading. No numeric AUC/ECE cutoff is applied."
    )
    deltas = {
        key: (
            None
            if families[BASELINE_V1_PLUS_ELO]["macro_average"][key] is None
            or families[BASELINE_V1_PLUS_ELO_LINEAR_RECENCY10]["macro_average"][key] is None
            else families[BASELINE_V1_PLUS_ELO_LINEAR_RECENCY10]["macro_average"][key]
            - families[BASELINE_V1_PLUS_ELO]["macro_average"][key]
        )
        for key in ("accuracy", "roc_auc", "log_loss", "brier", "ece")
    }
    return {
        "source": provenance_record(),
        "experiment": "issue_17_linear_recency_feature_ablation",
        "evaluation_contract": "validation_only_forward_chaining",
        "held_out_test_evaluated": False,
        "held_out_test_season_excluded": FINAL_TEST_SEASON,
        "shared_logistic_model": shared_logistic_model_record(),
        "elo": dict(ELO_CONFIG),
        "linear_recency": dict(LINEAR_RECENCY_CONFIG),
        "min_prior_games": min_prior_games,
        "feature_timing": (
            "Features are built once in (game_date, game_id) order. Baseline, "
            "Elo, and linear-recency snapshots for a calendar date are emitted "
            "before any same-day result updates team or Elo state. Features "
            "use only history.game_date < target.game_date. This pathway "
            "evaluates only validation folds A/B/C."
        ),
        "feature_families": families,
        "macro_deltas_linear_recency10_minus_incumbent": deltas,
        "continuation_gate": gate,
        "notes": [
            "Issue #17 compares baseline_v1_plus_elo against baseline_v1_plus_elo_linear_recency10 using the same logistic regression.",
            "linear_recency10 removes home_recent_win_rate and away_recent_win_rate and adds fixed 10-game linear recency-weighted rates.",
            "Issue #15 ordered5 is not the incumbent and is not included in the experimental family.",
            "The 2025-26 held-out test was not evaluated, inspected, or recorded.",
            "Passing the continuation gate does not promote a champion model and does not authorize the held-out test.",
            "No odds, roster/goalie context, deployment, Production, or worker/runtime work is included.",
        ],
    }


def evaluate_linear_recency_ablation_games(
    games: Sequence[NormalizedGame],
    min_prior_games: int = MIN_PRIOR_GAMES,
    fold_specs: Sequence[ForwardFoldSpec] | None = None,
) -> dict[str, Any]:
    if not games:
        raise EvaluationError("No games supplied for linear-recency validation-only ablation")
    if any(game.season == FINAL_TEST_SEASON for game in games):
        raise EvaluationError(
            "Validation-only ablation cannot load or evaluate the 2025-26 held-out test season"
        )
    present = {game.season for game in games}
    missing = [season for season in VALIDATION_HISTORY_SEASONS if season not in present]
    if fold_specs is None and missing:
        raise EvaluationError(
            f"Validation-only ablation is missing required history seasons: {missing}"
        )

    rows = build_features(games)
    report = evaluate_linear_recency_ablation_rows(
        rows,
        min_prior_games=min_prior_games,
        fold_specs=fold_specs,
    )
    report["historical_data_window"] = {
        "seasons": sorted({game.season for game in games}),
        "start": min(game.game_date for game in games).isoformat(),
        "end": max(game.game_date for game in games).isoformat(),
        "n_normalized_games": len(games),
        "n_feature_rows": len(rows),
    }
    return report


def _xgoal_availability_summary(
    rows: Sequence[dict[str, Any]],
    min_prior_games: int,
) -> dict[str, Any]:
    incumbent_eligible = [row for row in rows if _eligible(row, min_prior_games)]
    xgoal_available = [row for row in rows if xgoal_features_valid(row)]
    experimental_eligible = [
        row for row in incumbent_eligible if xgoal_features_valid(row)
    ]
    return {
        "total_candidate_modeling_rows": len(rows),
        "eligible_incumbent_rows": len(incumbent_eligible),
        "xgoal_feature_available_rows": len(xgoal_available),
        "final_experimental_eligible_rows": len(experimental_eligible),
        "excluded_because_xgoal_unavailable": (
            len(incumbent_eligible) - len(experimental_eligible)
        ),
    }


def _family_metric_snapshot(
    family: Mapping[str, Any],
    model_key: str = "logistic_regression",
) -> dict[str, dict[str, float | None]]:
    snapshot: dict[str, dict[str, float | None]] = {}
    for fold in family["folds"]:
        snapshot[fold["name"]] = {
            "accuracy": _fold_model_metric(fold, "accuracy", model_key),
            "roc_auc": _fold_model_metric(fold, "roc_auc", model_key),
            "log_loss": _fold_model_metric(fold, "log_loss", model_key),
            "brier": _fold_model_metric(fold, "brier", model_key),
            "ece": _fold_model_metric(fold, "ece", model_key),
        }
    snapshot["macro"] = {
        key: family["macro_average"][key]
        for key in ("accuracy", "roc_auc", "log_loss", "brier", "ece")
    }
    return snapshot


def _metrics_reproduction_record(
    family: Mapping[str, Any],
    expected: Mapping[str, Mapping[str, float]],
    *,
    checked: bool,
    source: str,
    tolerance: float,
    model_key: str = "logistic_regression",
) -> dict[str, Any]:
    observed = _family_metric_snapshot(family, model_key)
    fold_deltas: dict[str, dict[str, float | None]] = {}
    matched = True
    for fold_name, expected_metrics in expected.items():
        actual = observed.get(fold_name, {})
        fold_deltas[fold_name] = {}
        for metric, expected_value in expected_metrics.items():
            actual_value = actual.get(metric)
            delta = (
                None
                if actual_value is None
                else float(actual_value) - float(expected_value)
            )
            fold_deltas[fold_name][metric] = delta
            if actual_value is None or abs(float(actual_value) - float(expected_value)) > (
                tolerance
            ):
                matched = False
    return {
        "checked": checked,
        "matched": matched if checked else None,
        "tolerance": tolerance,
        "expected_source": source,
        "expected": dict(expected),
        "observed": observed,
        "deltas_observed_minus_expected": fold_deltas,
    }


def _incumbent_reproduction_record(
    family: Mapping[str, Any],
    *,
    checked: bool,
) -> dict[str, Any]:
    return _metrics_reproduction_record(
        family,
        ISSUE13_PLUS_ELO_VALIDATION_METRICS,
        checked=checked,
        source="Issue #13 / PR #14 published baseline_v1_plus_elo validation metrics",
        tolerance=INCUMBENT_REPRODUCTION_ATOL,
    )


def evaluate_xgoal_share_ablation_rows(
    rows: Sequence[dict[str, Any]],
    min_prior_games: int = MIN_PRIOR_GAMES,
    fold_specs: Sequence[ForwardFoldSpec] | None = None,
    *,
    assert_incumbent_reproduction: bool = False,
) -> dict[str, Any]:
    """Issue #21 validation-only prior xGoal share ablation.

    Compares baseline_v1_plus_elo against
    baseline_v1_plus_elo_plus_prior_xgoal_share on folds A/B/C only. Never
    evaluates the 2025-26 held-out test.
    """
    assert_incumbent_columns_unchanged()
    assert_prior_xgoal_share_contract()
    specs = list(fold_specs or default_validation_fold_specs())
    _validate_validation_only_specs(specs)
    _reject_held_out_test_rows(rows)
    if any(
        column not in row
        for row in rows
        for column in ("home_prior_xgoal_share", "away_prior_xgoal_share")
    ):
        raise EvaluationError(
            "Issue #19 xGoal primitive was not attached to incumbent modeling rows"
        )

    availability = _xgoal_availability_summary(rows, min_prior_games)
    if availability["excluded_because_xgoal_unavailable"] > 0:
        raise EvaluationError(
            "Experimental family unexpectedly lost otherwise eligible rows "
            "because xGoal state is unavailable: "
            f"{availability}"
        )

    families: dict[str, dict[str, Any]] = {}
    for name, columns, extra in (
        (BASELINE_V1_PLUS_ELO, PLUS_ELO_FEATURE_COLUMNS, None),
        (
            BASELINE_V1_PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE,
            PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE_FEATURE_COLUMNS,
            xgoal_features_valid,
        ),
    ):
        folds = _run_forward_folds(
            rows,
            specs,
            min_prior_games,
            columns,
            extra_eligible=extra,
            extra_exclusion_name=(
                "xgoal_unavailable" if extra is not None else None
            ),
        )
        families[name] = {
            "name": name,
            "feature_columns": list(columns),
            "removed_feature_columns": [],
            "added_feature_columns": (
                list(XGOAL_SHARE_CONFIG["added_columns"])
                if name == BASELINE_V1_PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE
                else []
            ),
            "folds": folds,
            "macro_average": _macro_average_logistic(folds),
        }

    reproduction = _incumbent_reproduction_record(
        families[BASELINE_V1_PLUS_ELO],
        checked=assert_incumbent_reproduction,
    )
    if assert_incumbent_reproduction and not reproduction["matched"]:
        raise EvaluationError(
            "Incumbent baseline_v1_plus_elo metrics did not reproduce Issue #13 "
            f"validation values within {INCUMBENT_REPRODUCTION_ATOL}: "
            f"{reproduction['deltas_observed_minus_expected']}"
        )

    gate = _continuation_gate(
        families[BASELINE_V1_PLUS_ELO],
        families[BASELINE_V1_PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE],
    )
    gate["calibration_auc_role"] = (
        "Qualitative diagnostic only. Issue #21 requires that ECE/calibration "
        "and ROC-AUC not reveal an obvious contradictory failure that makes "
        "the aggregate improvement misleading. No numeric AUC/ECE cutoff is applied."
    )
    deltas = {
        key: (
            None
            if families[BASELINE_V1_PLUS_ELO]["macro_average"][key] is None
            or families[BASELINE_V1_PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE]["macro_average"][key]
            is None
            else families[BASELINE_V1_PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE]["macro_average"][key]
            - families[BASELINE_V1_PLUS_ELO]["macro_average"][key]
        )
        for key in ("accuracy", "roc_auc", "log_loss", "brier", "ece")
    }
    return {
        "source": provenance_record(),
        "experiment": "issue_21_prior_xgoal_share_feature_ablation",
        "evaluation_contract": "validation_only_forward_chaining",
        "held_out_test_evaluated": False,
        "held_out_test_season_excluded": FINAL_TEST_SEASON,
        "shared_logistic_model": shared_logistic_model_record(),
        "elo": dict(ELO_CONFIG),
        "prior_xgoal_share": dict(XGOAL_SHARE_CONFIG),
        "min_prior_games": min_prior_games,
        "xgoal_availability": availability,
        "incumbent_reproduction": reproduction,
        "feature_timing": (
            "Features are built once in (game_date, game_id) order. Baseline, "
            "Elo, and prior xGoal-share snapshots for a calendar date are emitted "
            "before any same-day result updates team, Elo, or advanced state. "
            "Features use only history.game_date < target.game_date and reuse "
            "the Issue #19 primitive. This pathway evaluates only validation "
            "folds A/B/C."
        ),
        "feature_families": families,
        "macro_deltas_prior_xgoal_share_minus_incumbent": deltas,
        "continuation_gate": gate,
        "notes": [
            "Issue #21 compares baseline_v1_plus_elo against baseline_v1_plus_elo_plus_prior_xgoal_share using the same logistic regression.",
            "The experimental family adds only home_prior_xgoal_share and away_prior_xgoal_share.",
            "Issue #15 ordered5 and Issue #17 linear-recency10 are not included.",
            "The 2025-26 held-out test was not evaluated, inspected, or recorded.",
            "Passing the continuation gate does not promote a champion model and does not authorize the held-out test.",
            "No odds, roster/goalie context, deployment, Production, or worker/runtime work is included.",
        ],
    }


def evaluate_xgoal_share_ablation_games(
    games: Sequence[NormalizedGame],
    records: Mapping[tuple[str, str], AdvancedTeamGame],
    min_prior_games: int = MIN_PRIOR_GAMES,
    fold_specs: Sequence[ForwardFoldSpec] | None = None,
    *,
    assert_incumbent_reproduction: bool = False,
) -> dict[str, Any]:
    if not games:
        raise EvaluationError("No games supplied for prior xGoal share validation-only ablation")
    if any(game.season == FINAL_TEST_SEASON for game in games):
        raise EvaluationError(
            "Validation-only ablation cannot load or evaluate the 2025-26 held-out test season"
        )
    present = {game.season for game in games}
    missing = [season for season in VALIDATION_HISTORY_SEASONS if season not in present]
    if fold_specs is None and missing:
        raise EvaluationError(
            f"Validation-only ablation is missing required history seasons: {missing}"
        )

    rows = build_features(games)
    xgoal_rows = build_pregame_xgoal_share_features(games, records)
    attached = attach_pregame_xgoal_share_features(rows, xgoal_rows)
    report = evaluate_xgoal_share_ablation_rows(
        attached,
        min_prior_games=min_prior_games,
        fold_specs=fold_specs,
        assert_incumbent_reproduction=assert_incumbent_reproduction,
    )
    report["historical_data_window"] = {
        "seasons": sorted({game.season for game in games}),
        "start": min(game.game_date for game in games).isoformat(),
        "end": max(game.game_date for game in games).isoformat(),
        "n_normalized_games": len(games),
        "n_feature_rows": len(attached),
    }
    return report


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    return _sha256_bytes(np.ascontiguousarray(array).tobytes())


def _row_identity_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    joined = ",".join(str(row["game_id"]) for row in rows)
    return _sha256_bytes(joined.encode("utf-8"))


def _hgb_eligible(
    row: dict[str, Any],
    min_prior_games: int,
) -> bool:
    return _eligible(row, min_prior_games) and xgoal_features_valid(row)


def _run_hgb_comparison_folds(
    rows: Sequence[dict[str, Any]],
    specs: Sequence[ForwardFoldSpec],
    min_prior_games: int,
) -> list[dict[str, Any]]:
    selected = tuple(FROZEN_FEATURE_COLUMNS)
    folds: list[dict[str, Any]] = []
    for spec in specs:
        train_candidates = [row for row in rows if int(row["season"]) in spec.train_seasons]
        evaluation_candidates = [
            row for row in rows if int(row["season"]) == spec.evaluation_season
        ]
        if not train_candidates:
            raise EvaluationError(f"Forward fold {spec.name} has no training rows")
        if not evaluation_candidates:
            raise EvaluationError(f"Forward fold {spec.name} has no evaluation rows")

        train_rows = [row for row in train_candidates if _hgb_eligible(row, min_prior_games)]
        evaluation_rows = [
            row for row in evaluation_candidates if _hgb_eligible(row, min_prior_games)
        ]
        if not train_rows:
            raise EvaluationError(f"Forward fold {spec.name} has no eligible training rows")
        if not evaluation_rows:
            raise EvaluationError(f"Forward fold {spec.name} has no eligible evaluation rows")

        train_end = max(parse_game_date(row["game_date"]) for row in train_rows)
        evaluation_start = min(parse_game_date(row["game_date"]) for row in evaluation_rows)
        if train_end >= evaluation_start:
            raise EvaluationError(
                f"Forward fold {spec.name} violates chronology: "
                f"train end {train_end} >= evaluation start {evaluation_start}"
            )

        train_y = np.array([int(row["home_win"]) for row in train_rows], dtype=int)
        evaluation_y = np.array([int(row["home_win"]) for row in evaluation_rows], dtype=int)
        train_x = np.array(feature_matrix(train_rows, columns=selected), dtype=float)
        evaluation_x = np.array(
            feature_matrix(evaluation_rows, columns=selected), dtype=float
        )
        if train_x.shape != (len(train_rows), len(selected)):
            raise EvaluationError("Train feature matrix shape does not match frozen columns")
        if evaluation_x.shape != (len(evaluation_rows), len(selected)):
            raise EvaluationError(
                "Evaluation feature matrix shape does not match frozen columns"
            )

        logistic = _new_logistic_model()
        hgb = _new_hgb_model()
        if not isinstance(hgb, HistGradientBoostingClassifier):
            raise EvaluationError("Issue #23 experimental model must be HistGradientBoostingClassifier")
        if "scaler" in getattr(hgb, "named_steps", {}):
            raise EvaluationError("HistGradientBoosting must not use a scaling pipeline")

        logistic.fit(train_x, train_y)
        hgb.fit(train_x, train_y)
        logistic_prob = np.clip(logistic.predict_proba(evaluation_x)[:, 1], 1e-15, 1.0 - 1e-15)
        hgb_prob = np.clip(hgb.predict_proba(evaluation_x)[:, 1], 1e-15, 1.0 - 1e-15)

        identity = {
            "feature_columns": list(selected),
            "train_n": len(train_rows),
            "evaluation_n": len(evaluation_rows),
            "train_row_identity_sha256": _row_identity_sha256(train_rows),
            "evaluation_row_identity_sha256": _row_identity_sha256(evaluation_rows),
            "train_x_sha256": _array_sha256(train_x),
            "train_y_sha256": _array_sha256(train_y),
            "evaluation_x_sha256": _array_sha256(evaluation_x),
            "evaluation_y_sha256": _array_sha256(evaluation_y),
            "both_models_fit_on_identical_train_xy": True,
            "both_models_scored_on_identical_evaluation_x": True,
            "both_models_used_identical_evaluation_y": True,
            "logistic_uses_scaler_pipeline": True,
            "hgb_uses_scaler_pipeline": False,
        }
        folds.append(
            {
                "name": spec.name,
                "role": spec.role,
                "train_seasons": list(spec.train_seasons),
                "evaluation_season": spec.evaluation_season,
                "train_window": _window_summary(train_candidates)
                | {"n_eligible": len(train_rows)},
                "evaluation_window": _window_summary(evaluation_candidates)
                | {"n_eligible": len(evaluation_rows)},
                "excluded_rows": {
                    "train_min_prior_or_xgoal": len(train_candidates) - len(train_rows),
                    "evaluation_min_prior_or_xgoal": (
                        len(evaluation_candidates) - len(evaluation_rows)
                    ),
                },
                "identical_inputs": identity,
                "logistic_regression": {
                    **shared_logistic_model_record(),
                    "feature_columns": list(selected),
                    "metrics": _metric_block(evaluation_y, logistic_prob),
                },
                "hist_gradient_boosting": {
                    **hgb_model_record(),
                    "feature_columns": list(selected),
                    "fitted_early_stopping": bool(hgb.get_params()["early_stopping"]),
                    "metrics": _metric_block(evaluation_y, hgb_prob),
                },
            }
        )
    return folds


def evaluate_hgb_comparison_rows(
    rows: Sequence[dict[str, Any]],
    min_prior_games: int = MIN_PRIOR_GAMES,
    fold_specs: Sequence[ForwardFoldSpec] | None = None,
    *,
    assert_logistic_reproduction: bool = False,
    assert_issue21_row_counts: bool = False,
) -> dict[str, Any]:
    """Issue #23 validation-only logistic vs HistGradientBoosting comparison."""
    assert_hgb_comparison_contract()
    specs = list(fold_specs or default_validation_fold_specs())
    _validate_validation_only_specs(specs)
    _reject_held_out_test_rows(rows)
    if any(
        column not in row
        for row in rows
        for column in FROZEN_FEATURE_COLUMNS
    ):
        raise EvaluationError(
            "Issue #23 requires the frozen Issue #21 xGoal-share feature family"
        )

    availability = _xgoal_availability_summary(rows, min_prior_games)
    if availability["excluded_because_xgoal_unavailable"] > 0:
        raise EvaluationError(
            "Frozen Issue #21 family unexpectedly lost otherwise eligible rows "
            "because xGoal state is unavailable: "
            f"{availability}"
        )

    folds = _run_hgb_comparison_folds(rows, specs, min_prior_games)
    logistic_family = {
        "name": "logistic",
        "feature_columns": list(FROZEN_FEATURE_COLUMNS),
        "folds": folds,
        "macro_average": _macro_average_model(folds, "logistic_regression"),
    }
    hgb_family = {
        "name": "hist_gradient_boosting",
        "feature_columns": list(FROZEN_FEATURE_COLUMNS),
        "folds": folds,
        "macro_average": _macro_average_model(folds, "hist_gradient_boosting"),
    }

    if assert_issue21_row_counts:
        for fold in folds:
            expected = ISSUE21_ELIGIBLE_ROW_COUNTS[fold["name"]]
            actual_train = fold["train_window"]["n_eligible"]
            actual_eval = fold["evaluation_window"]["n_eligible"]
            if actual_train != expected["train"] or actual_eval != expected["evaluation"]:
                raise EvaluationError(
                    f"Issue #23 fold {fold['name']} eligible counts "
                    f"train={actual_train} eval={actual_eval} do not match Issue #21 "
                    f"train={expected['train']} eval={expected['evaluation']}"
                )

    reproduction = _metrics_reproduction_record(
        logistic_family,
        ISSUE21_XGOAL_LOGISTIC_VALIDATION_METRICS,
        checked=assert_logistic_reproduction,
        source=(
            "Issue #21 / PR #22 published baseline_v1_plus_elo_plus_prior_xgoal_share "
            "logistic validation metrics"
        ),
        tolerance=LOGISTIC_REPRODUCTION_ATOL,
        model_key="logistic_regression",
    )
    if assert_logistic_reproduction and not reproduction["matched"]:
        raise EvaluationError(
            "Logistic Issue #21 metrics did not reproduce within "
            f"{LOGISTIC_REPRODUCTION_ATOL}: "
            f"{reproduction['deltas_observed_minus_expected']}"
        )

    gate = _continuation_gate(
        logistic_family,
        hgb_family,
        baseline_model_key="logistic_regression",
        experimental_model_key="hist_gradient_boosting",
        pass_decision=CARRY_FORWARD_MODEL_FAMILY,
        fail_decision=DO_NOT_CARRY_FORWARD_MODEL_FAMILY,
        issue_label="Issue #23",
    )
    deltas = {
        key: (
            None
            if logistic_family["macro_average"][key] is None
            or hgb_family["macro_average"][key] is None
            else hgb_family["macro_average"][key] - logistic_family["macro_average"][key]
        )
        for key in ("accuracy", "roc_auc", "log_loss", "brier", "ece")
    }
    identical = {
        "feature_columns": list(FROZEN_FEATURE_COLUMNS),
        "both_models_used_identical_rows_features_targets": True,
        "folds": [fold["identical_inputs"] | {"name": fold["name"]} for fold in folds],
    }
    return {
        "source": provenance_record(),
        "experiment": "issue_23_histgradientboosting_comparison",
        "evaluation_contract": "validation_only_forward_chaining",
        "held_out_test_evaluated": False,
        "held_out_test_season_excluded": FINAL_TEST_SEASON,
        "feature_family": FROZEN_FEATURE_FAMILY,
        "feature_columns": list(FROZEN_FEATURE_COLUMNS),
        "sklearn_version": sklearn.__version__,
        "shared_logistic_model": shared_logistic_model_record(),
        "hist_gradient_boosting": dict(HGB_CONFIG),
        "min_prior_games": min_prior_games,
        "xgoal_availability": availability,
        "identical_inputs": identical,
        "logistic_issue21_reproduction": reproduction,
        "feature_timing": (
            "Features reuse the Issue #19 prior-date xGoal primitive and the "
            "Issue #21 carry-forward family. Both models consume identical "
            "eligible rows, the same frozen feature matrix, and the same target "
            "vector. This pathway evaluates only validation folds A/B/C."
        ),
        "model_families": {
            "logistic": logistic_family,
            "hist_gradient_boosting": hgb_family,
        },
        "macro_deltas_hgb_minus_logistic": deltas,
        "continuation_gate": gate,
        "notes": [
            "Issue #23 compares the frozen Issue #21 logistic pipeline against one fixed HistGradientBoostingClassifier.",
            "No feature was added, removed, or transformed.",
            "HistGradientBoosting is not scaled, calibrated, class-weighted, searched, or ensembled.",
            "The 2025-26 held-out test was not evaluated, inspected, or recorded.",
            "Passing the continuation gate does not promote a champion model and does not authorize the held-out test.",
        ],
    }


def evaluate_hgb_comparison_games(
    games: Sequence[NormalizedGame],
    records: Mapping[tuple[str, str], AdvancedTeamGame],
    min_prior_games: int = MIN_PRIOR_GAMES,
    fold_specs: Sequence[ForwardFoldSpec] | None = None,
    *,
    assert_logistic_reproduction: bool = False,
    assert_issue21_row_counts: bool = False,
) -> dict[str, Any]:
    if not games:
        raise EvaluationError("No games supplied for HistGradientBoosting comparison")
    if any(game.season == FINAL_TEST_SEASON for game in games):
        raise EvaluationError(
            "Validation-only ablation cannot load or evaluate the 2025-26 held-out test season"
        )
    present = {game.season for game in games}
    missing = [season for season in VALIDATION_HISTORY_SEASONS if season not in present]
    if fold_specs is None and missing:
        raise EvaluationError(
            f"Validation-only ablation is missing required history seasons: {missing}"
        )

    rows = build_features(games)
    xgoal_rows = build_pregame_xgoal_share_features(games, records)
    attached = attach_pregame_xgoal_share_features(rows, xgoal_rows)
    report = evaluate_hgb_comparison_rows(
        attached,
        min_prior_games=min_prior_games,
        fold_specs=fold_specs,
        assert_logistic_reproduction=assert_logistic_reproduction,
        assert_issue21_row_counts=assert_issue21_row_counts,
    )
    report["historical_data_window"] = {
        "seasons": sorted({game.season for game in games}),
        "start": min(game.game_date for game in games).isoformat(),
        "end": max(game.game_date for game in games).isoformat(),
        "n_normalized_games": len(games),
        "n_feature_rows": len(attached),
    }
    return report


def _corsi_availability_summary(
    rows: Sequence[dict[str, Any]],
    min_prior_games: int,
) -> dict[str, Any]:
    incumbent_eligible = [
        row
        for row in rows
        if _eligible(row, min_prior_games) and xgoal_features_valid(row)
    ]
    experimental_eligible = [
        row for row in incumbent_eligible if corsi_features_valid(row)
    ]
    return {
        "total_candidate_modeling_rows": len(rows),
        "eligible_incumbent_rows": len(incumbent_eligible),
        "rows_with_valid_corsi_state": sum(
            1 for row in rows if corsi_features_valid(row)
        ),
        "final_experimental_eligible_rows": len(experimental_eligible),
        "excluded_because_corsi_unavailable": (
            len(incumbent_eligible) - len(experimental_eligible)
        ),
        "xgoal_feature_available_rows": sum(
            1 for row in rows if xgoal_features_valid(row)
        ),
    }


def _fold_eligible_rows(
    rows: Sequence[dict[str, Any]],
    spec: ForwardFoldSpec,
    min_prior_games: int,
    extra_eligible: Callable[[dict[str, Any]], bool],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows = [
        row
        for row in rows
        if int(row["season"]) in spec.train_seasons
        and _eligible(row, min_prior_games)
        and extra_eligible(row)
    ]
    evaluation_rows = [
        row
        for row in rows
        if int(row["season"]) == spec.evaluation_season
        and _eligible(row, min_prior_games)
        and extra_eligible(row)
    ]
    return train_rows, evaluation_rows


def _corsi_identical_row_evidence(
    rows: Sequence[dict[str, Any]],
    specs: Sequence[ForwardFoldSpec],
    min_prior_games: int,
    incumbent_folds: Sequence[dict[str, Any]],
    experimental_folds: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    fold_records: list[dict[str, Any]] = []
    for spec, incumbent, experimental in zip(
        specs, incumbent_folds, experimental_folds, strict=True
    ):
        if spec.name != incumbent["name"] or spec.name != experimental["name"]:
            raise EvaluationError("Issue #25 fold names do not align for identical-row check")
        incumbent_train, incumbent_eval = _fold_eligible_rows(
            rows, spec, min_prior_games, xgoal_features_valid
        )
        experimental_train, experimental_eval = _fold_eligible_rows(
            rows, spec, min_prior_games, xgoal_and_corsi_features_valid
        )
        incumbent_train_ids = [str(row["game_id"]) for row in incumbent_train]
        experimental_train_ids = [str(row["game_id"]) for row in experimental_train]
        incumbent_eval_ids = [str(row["game_id"]) for row in incumbent_eval]
        experimental_eval_ids = [str(row["game_id"]) for row in experimental_eval]
        incumbent_train_y = [int(row["home_win"]) for row in incumbent_train]
        experimental_train_y = [int(row["home_win"]) for row in experimental_train]
        incumbent_eval_y = [int(row["home_win"]) for row in incumbent_eval]
        experimental_eval_y = [int(row["home_win"]) for row in experimental_eval]
        identical_train_ids = incumbent_train_ids == experimental_train_ids
        identical_eval_ids = incumbent_eval_ids == experimental_eval_ids
        identical_y = (
            incumbent_train_y == experimental_train_y
            and incumbent_eval_y == experimental_eval_y
        )
        logistic_match = (
            incumbent["logistic_regression"]["solver"]
            == experimental["logistic_regression"]["solver"]
            == LOGISTIC_SOLVER
            and incumbent["logistic_regression"]["max_iter"]
            == experimental["logistic_regression"]["max_iter"]
            == LOGISTIC_MAX_ITER
            and incumbent["logistic_regression"]["random_state"]
            == experimental["logistic_regression"]["random_state"]
            == LOGISTIC_RANDOM_STATE
        )
        counts_match = (
            incumbent["train_window"]["n_eligible"] == experimental["train_window"]["n_eligible"]
            == len(experimental_train)
            and incumbent["evaluation_window"]["n_eligible"]
            == experimental["evaluation_window"]["n_eligible"]
            == len(experimental_eval)
        )
        if not (
            identical_train_ids
            and identical_eval_ids
            and identical_y
            and logistic_match
            and counts_match
        ):
            raise EvaluationError(
                f"Issue #25 fold {spec.name} incumbent/experimental game IDs, "
                "targets, or logistic configuration are not identical"
            )
        fold_records.append(
            {
                "name": spec.name,
                "identical_train_game_ids": True,
                "identical_evaluation_game_ids": True,
                "identical_y": True,
                "same_logistic_configuration": True,
                "train_n": len(experimental_train),
                "evaluation_n": len(experimental_eval),
                "train_row_identity_sha256": _row_identity_sha256(experimental_train),
                "evaluation_row_identity_sha256": _row_identity_sha256(experimental_eval),
                "train_y_sha256": _array_sha256(np.array(experimental_train_y, dtype=int)),
                "evaluation_y_sha256": _array_sha256(
                    np.array(experimental_eval_y, dtype=int)
                ),
            }
        )
    return {
        "identical_train_and_evaluation_rows_and_targets": True,
        "same_logistic_configuration": True,
        "folds": fold_records,
    }


def evaluate_corsi_share_ablation_rows(
    rows: Sequence[dict[str, Any]],
    min_prior_games: int = MIN_PRIOR_GAMES,
    fold_specs: Sequence[ForwardFoldSpec] | None = None,
    *,
    assert_incumbent_reproduction: bool = False,
    assert_issue21_row_counts: bool = False,
) -> dict[str, Any]:
    """Issue #25 validation-only prior Corsi share ablation.

    Compares baseline_v1_plus_elo_plus_prior_xgoal_share against
    baseline_v1_plus_elo_plus_prior_xgoal_share_plus_prior_corsi_share on
    folds A/B/C only. Never evaluates the 2025-26 held-out test.
    """
    assert_prior_corsi_share_contract()
    specs = list(fold_specs or default_validation_fold_specs())
    _validate_validation_only_specs(specs)
    _reject_held_out_test_rows(rows)
    required = PLUS_XGOAL_PLUS_PRIOR_CORSI_SHARE_FEATURE_COLUMNS
    if any(column not in row for row in rows for column in required):
        raise EvaluationError(
            "Issue #25 Corsi primitive was not attached to incumbent modeling rows"
        )

    xgoal_availability = _xgoal_availability_summary(rows, min_prior_games)
    if xgoal_availability["excluded_because_xgoal_unavailable"] > 0:
        raise EvaluationError(
            "Incumbent Issue #21 family unexpectedly lost otherwise eligible rows "
            "because xGoal state is unavailable: "
            f"{xgoal_availability}"
        )
    availability = _corsi_availability_summary(rows, min_prior_games)
    if availability["excluded_because_corsi_unavailable"] > 0:
        raise EvaluationError(
            "Experimental family unexpectedly lost otherwise eligible rows "
            "because Corsi state is unavailable: "
            f"{availability}. If this is a contract/source/timing problem, "
            "return it to 01 / 00."
        )

    families: dict[str, dict[str, Any]] = {}
    for name, columns in (
        (
            BASELINE_V1_PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE,
            PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE_FEATURE_COLUMNS,
        ),
        (
            BASELINE_V1_PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE_PLUS_PRIOR_CORSI_SHARE,
            PLUS_XGOAL_PLUS_PRIOR_CORSI_SHARE_FEATURE_COLUMNS,
        ),
    ):
        folds = _run_forward_folds(
            rows,
            specs,
            min_prior_games,
            columns,
            extra_eligible=xgoal_and_corsi_features_valid,
            extra_exclusion_name="corsi_or_xgoal_unavailable",
        )
        families[name] = {
            "name": name,
            "feature_columns": list(columns),
            "removed_feature_columns": [],
            "added_feature_columns": (
                list(CORSI_SHARE_CONFIG["added_columns"])
                if name == BASELINE_V1_PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE_PLUS_PRIOR_CORSI_SHARE
                else []
            ),
            "folds": folds,
            "macro_average": _macro_average_logistic(folds),
        }

    if assert_issue21_row_counts:
        for fold in families[BASELINE_V1_PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE]["folds"]:
            expected = CORSI_ISSUE21_ELIGIBLE_ROW_COUNTS[fold["name"]]
            actual_train = fold["train_window"]["n_eligible"]
            actual_eval = fold["evaluation_window"]["n_eligible"]
            if actual_train != expected["train"] or actual_eval != expected["evaluation"]:
                raise EvaluationError(
                    f"Issue #25 fold {fold['name']} incumbent eligible counts "
                    f"train={actual_train} eval={actual_eval} do not match Issue #21 "
                    f"train={expected['train']} eval={expected['evaluation']}. Stop and investigate."
                )

    identities = _corsi_identical_row_evidence(
        rows,
        specs,
        min_prior_games,
        families[BASELINE_V1_PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE]["folds"],
        families[BASELINE_V1_PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE_PLUS_PRIOR_CORSI_SHARE]["folds"],
    )

    reproduction = _metrics_reproduction_record(
        families[BASELINE_V1_PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE],
        CORSI_ISSUE21_XGOAL_LOGISTIC_VALIDATION_METRICS,
        checked=assert_incumbent_reproduction,
        source=(
            "Issue #21 / PR #22 published baseline_v1_plus_elo_plus_prior_xgoal_share "
            "logistic validation metrics"
        ),
        tolerance=INCUMBENT_REPRODUCTION_ATOL,
    )
    if assert_incumbent_reproduction and not reproduction["matched"]:
        raise EvaluationError(
            "Incumbent baseline_v1_plus_elo_plus_prior_xgoal_share metrics did not "
            "reproduce Issue #21 validation values within "
            f"{INCUMBENT_REPRODUCTION_ATOL}: "
            f"{reproduction['deltas_observed_minus_expected']}. Stop and investigate."
        )

    gate = _continuation_gate(
        families[BASELINE_V1_PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE],
        families[BASELINE_V1_PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE_PLUS_PRIOR_CORSI_SHARE],
        issue_label="Issue #25",
    )
    deltas = {
        key: (
            None
            if families[BASELINE_V1_PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE]["macro_average"][key]
            is None
            or families[BASELINE_V1_PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE_PLUS_PRIOR_CORSI_SHARE][
                "macro_average"
            ][key]
            is None
            else families[BASELINE_V1_PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE_PLUS_PRIOR_CORSI_SHARE][
                "macro_average"
            ][key]
            - families[BASELINE_V1_PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE]["macro_average"][key]
        )
        for key in ("accuracy", "roc_auc", "log_loss", "brier", "ece")
    }
    return {
        "source": provenance_record(),
        "experiment": "issue_25_prior_corsi_share_feature_ablation",
        "evaluation_contract": "validation_only_forward_chaining",
        "held_out_test_evaluated": False,
        "held_out_test_season_excluded": FINAL_TEST_SEASON,
        "shared_logistic_model": shared_logistic_model_record(),
        "elo": dict(ELO_CONFIG),
        "prior_corsi_share": dict(CORSI_SHARE_CONFIG),
        "min_prior_games": min_prior_games,
        "corsi_availability": availability,
        "identical_rows_and_targets": identities,
        "incumbent_reproduction": reproduction,
        "feature_timing": (
            "Features are built once in (game_date, game_id) order. Baseline, "
            "Elo, prior xGoal-share, and prior Corsi-share snapshots for a "
            "calendar date are emitted before any same-day result updates team, "
            "Elo, or advanced state. Features use only history.game_date < "
            "target.game_date, reuse the Issue #19 inspect/join primitive, and "
            "construct Corsi share from shotAttemptsFor/shotAttemptsAgainst only. "
            "Row-level corsiPercentage is not used. This pathway evaluates only "
            "validation folds A/B/C."
        ),
        "feature_families": families,
        "macro_deltas_prior_corsi_share_minus_incumbent": deltas,
        "continuation_gate": gate,
        "notes": [
            "Issue #25 compares baseline_v1_plus_elo_plus_prior_xgoal_share against baseline_v1_plus_elo_plus_prior_xgoal_share_plus_prior_corsi_share using the same logistic regression.",
            "The experimental family adds only home_prior_corsi_share and away_prior_corsi_share.",
            "Raw Corsi inputs are shotAttemptsFor and shotAttemptsAgainst; corsiPercentage is not averaged or used.",
            "Fenwick, danger, special teams, HGB, and other model families are not included.",
            "The 2025-26 held-out test was not evaluated, inspected, or recorded.",
            "Passing the continuation gate does not promote a champion model and does not authorize the held-out test.",
            "No odds, roster/goalie context, deployment, Production, or worker/runtime work is included.",
        ],
    }


def evaluate_corsi_share_ablation_games(
    games: Sequence[NormalizedGame],
    records: Mapping[tuple[str, str], AdvancedTeamGame],
    min_prior_games: int = MIN_PRIOR_GAMES,
    fold_specs: Sequence[ForwardFoldSpec] | None = None,
    *,
    assert_incumbent_reproduction: bool = False,
    assert_issue21_row_counts: bool = False,
) -> dict[str, Any]:
    if not games:
        raise EvaluationError("No games supplied for prior Corsi share validation-only ablation")
    if any(game.season == FINAL_TEST_SEASON for game in games):
        raise EvaluationError(
            "Validation-only ablation cannot load or evaluate the 2025-26 held-out test season"
        )
    present = {game.season for game in games}
    missing = [season for season in VALIDATION_HISTORY_SEASONS if season not in present]
    if fold_specs is None and missing:
        raise EvaluationError(
            f"Validation-only ablation is missing required history seasons: {missing}"
        )

    rows = build_features(games)
    xgoal_rows = build_pregame_xgoal_share_features(games, records)
    attached = attach_pregame_xgoal_share_features(rows, xgoal_rows)
    corsi_rows = build_pregame_corsi_share_features(games, records)
    attached = attach_pregame_corsi_share_features(attached, corsi_rows)
    report = evaluate_corsi_share_ablation_rows(
        attached,
        min_prior_games=min_prior_games,
        fold_specs=fold_specs,
        assert_incumbent_reproduction=assert_incumbent_reproduction,
        assert_issue21_row_counts=assert_issue21_row_counts,
    )
    report["historical_data_window"] = {
        "seasons": sorted({game.season for game in games}),
        "start": min(game.game_date for game in games).isoformat(),
        "end": max(game.game_date for game in games).isoformat(),
        "n_normalized_games": len(games),
        "n_feature_rows": len(attached),
    }
    return report


def evaluate_rows(
    rows: Sequence[dict[str, Any]],
    min_prior_games: int = MIN_PRIOR_GAMES,
    split_specs: Sequence[SplitSpec] | None = None,
) -> dict[str, Any]:
    specs = list(split_specs or default_split_specs())
    grouped: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    unassigned = 0
    for row in rows:
        split = assign_split(row, specs)
        if split is None:
            unassigned += 1
            continue
        if _eligible(row, min_prior_games):
            grouped[split].append(row)

    if not grouped["train"]:
        raise EvaluationError("Train window has no eligible rows")
    if not grouped["validation"]:
        raise EvaluationError("Validation window has no eligible rows")
    if not grouped["test"]:
        raise EvaluationError("Held-out test window has no eligible rows")

    train_y = np.array([int(row["home_win"]) for row in grouped["train"]], dtype=int)
    base_rate = float(train_y.mean())
    if not 0.0 < base_rate < 1.0:
        raise EvaluationError("Train home-win base rate must be inside (0, 1)")

    train_x = np.array(feature_matrix(grouped["train"]), dtype=float)
    model = _new_logistic_model()
    model.fit(train_x, train_y)

    def predict_split(split_rows: Sequence[dict[str, Any]]) -> np.ndarray:
        return _predict(model, split_rows)

    report = {
        "source": provenance_record(),
        "feature_columns": list(FEATURE_COLUMNS),
        "min_prior_games": min_prior_games,
        "feature_timing": (
            "Games are processed in (game_date, game_id) order. Features use only "
            "prior completed games with history.game_date < target.game_date. "
            "Same-day results are excluded because this MoneyPuck file has no "
            "trustworthy start timestamp. Season-specific rates reset at season "
            f"boundaries. Recent form uses the previous {RECENT_FORM_WINDOW} games."
        ),
        "windows": {
            name: _window_summary(split_rows) | {"n_eligible": len(split_rows)}
            for name, split_rows in grouped.items()
        },
        "unassigned_feature_rows": unassigned,
        "comparator": {
            "name": "train_home_win_rate",
            "probability": base_rate,
            "validation": _metric_block(
                np.array([int(row["home_win"]) for row in grouped["validation"]], dtype=int),
                np.full(len(grouped["validation"]), base_rate, dtype=float),
            ),
            "test": _metric_block(
                np.array([int(row["home_win"]) for row in grouped["test"]], dtype=int),
                np.full(len(grouped["test"]), base_rate, dtype=float),
            ),
        },
        "logistic_regression": {
            "model": "StandardScaler + LogisticRegression(solver=lbfgs, max_iter=1000, random_state=42)",
            "hyperparameter_search": False,
            "validation": _metric_block(
                np.array([int(row["home_win"]) for row in grouped["validation"]], dtype=int),
                predict_split(grouped["validation"]),
            ),
            "test": _metric_block(
                np.array([int(row["home_win"]) for row in grouped["test"]], dtype=int),
                predict_split(grouped["test"]),
            ),
        },
        "notes": [
            "Validation is diagnostic only. The held-out test window was not used for model selection.",
            "No odds, staking, dashboard, deployment, or worker/runtime work is included.",
        ],
    }
    return report


def evaluate_games(
    games: Sequence[NormalizedGame],
    min_prior_games: int = MIN_PRIOR_GAMES,
    split_specs: Sequence[SplitSpec] | None = None,
) -> dict[str, Any]:
    rows = build_features(games)
    report = evaluate_rows(rows, min_prior_games=min_prior_games, split_specs=split_specs)
    if games:
        report["historical_data_window"] = {
            "seasons": DEFAULT_SEASONS if split_specs is None else sorted({game.season for game in games}),
            "start": min(game.game_date for game in games).isoformat(),
            "end": max(game.game_date for game in games).isoformat(),
            "n_normalized_games": len(games),
            "n_feature_rows": len(rows),
        }
    return report
