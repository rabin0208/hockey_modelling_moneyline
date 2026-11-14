"""Issue #23 frozen-feature HistGradientBoosting comparison contract.

Both models use the Issue #21 carry-forward feature family unchanged.
Logistic remains the exact existing pipeline. HistGradientBoosting uses one
fixed, untuned configuration with early stopping disabled.
"""

from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

from hockey_moneyline.xgoal_share import (
    BASELINE_V1_PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE,
    PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE_FEATURE_COLUMNS,
    assert_prior_xgoal_share_contract,
)

FROZEN_FEATURE_FAMILY = BASELINE_V1_PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE
FROZEN_FEATURE_COLUMNS = PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE_FEATURE_COLUMNS

HGB_LOSS = "log_loss"
HGB_LEARNING_RATE = 0.1
HGB_MAX_ITER = 100
HGB_MAX_LEAF_NODES = 31
HGB_MIN_SAMPLES_LEAF = 20
HGB_L2_REGULARIZATION = 0.0
HGB_EARLY_STOPPING = False
HGB_RANDOM_STATE = 42

HGB_PARAMS = {
    "loss": HGB_LOSS,
    "learning_rate": HGB_LEARNING_RATE,
    "max_iter": HGB_MAX_ITER,
    "max_leaf_nodes": HGB_MAX_LEAF_NODES,
    "min_samples_leaf": HGB_MIN_SAMPLES_LEAF,
    "l2_regularization": HGB_L2_REGULARIZATION,
    "early_stopping": HGB_EARLY_STOPPING,
    "random_state": HGB_RANDOM_STATE,
}

HGB_MODEL_DESCRIPTION = (
    "HistGradientBoostingClassifier(loss=log_loss, learning_rate=0.1, "
    "max_iter=100, max_leaf_nodes=31, min_samples_leaf=20, "
    "l2_regularization=0.0, early_stopping=False, random_state=42)"
)

CARRY_FORWARD_MODEL_FAMILY = "carry forward model-family candidate"
DO_NOT_CARRY_FORWARD_MODEL_FAMILY = "do not carry forward model-family candidate"

ISSUE21_XGOAL_LOGISTIC_VALIDATION_METRICS = {
    "fold_a": {
        "accuracy": 0.619556,
        "roc_auc": 0.654698,
        "log_loss": 0.660324,
        "brier": 0.233328,
        "ece": 0.041597,
    },
    "fold_b": {
        "accuracy": 0.591057,
        "roc_auc": 0.628234,
        "log_loss": 0.669898,
        "brier": 0.238316,
        "ece": 0.054279,
    },
    "fold_c": {
        "accuracy": 0.591870,
        "roc_auc": 0.605962,
        "log_loss": 0.674706,
        "brier": 0.240804,
        "ece": 0.044352,
    },
    "macro": {
        "accuracy": 0.600828,
        "roc_auc": 0.629631,
        "log_loss": 0.668309,
        "brier": 0.237483,
        "ece": 0.046743,
    },
}

ISSUE21_ELIGIBLE_ROW_COUNTS = {
    "fold_a": {"train": 1121, "evaluation": 1217},
    "fold_b": {"train": 2338, "evaluation": 1230},
    "fold_c": {"train": 3568, "evaluation": 1230},
}

LOGISTIC_REPRODUCTION_ATOL = 1e-6

HGB_CONFIG = {
    "model": HGB_MODEL_DESCRIPTION,
    "estimator": "HistGradientBoostingClassifier",
    "scaler": None,
    "pipeline": False,
    "hyperparameter_search": False,
    "calibration": False,
    "class_weight": None,
    "ensemble": False,
    **HGB_PARAMS,
}


def _new_hgb_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(**HGB_PARAMS)


def hgb_model_record() -> dict[str, object]:
    return dict(HGB_CONFIG)


def assert_hgb_comparison_contract() -> None:
    """Guard frozen Issue #21 columns and the exact Issue #23 HGB configuration."""
    assert_prior_xgoal_share_contract()
    if FROZEN_FEATURE_COLUMNS != PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE_FEATURE_COLUMNS:
        raise RuntimeError("Issue #23 feature columns must remain the Issue #21 carry-forward family")
    if len(FROZEN_FEATURE_COLUMNS) != 21:
        raise RuntimeError("frozen Issue #21 family must remain 21 columns")
    if HGB_PARAMS != {
        "loss": "log_loss",
        "learning_rate": 0.1,
        "max_iter": 100,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 0.0,
        "early_stopping": False,
        "random_state": 42,
    }:
        raise RuntimeError("HistGradientBoosting configuration was altered")
    model = _new_hgb_model()
    if isinstance(model, Pipeline):
        raise RuntimeError("HistGradientBoosting must not be wrapped in a scaling pipeline")
    params = model.get_params()
    for key, expected in HGB_PARAMS.items():
        if params.get(key) != expected:
            raise RuntimeError(
                f"HistGradientBoosting parameter {key}={params.get(key)!r} != {expected!r}"
            )
    if params.get("early_stopping") is not False:
        raise RuntimeError("HistGradientBoosting early_stopping must be False")
