"""Issue #32 dedicated frozen-candidate 2025-26 heldout runner.

Stage A adds this guarded pathway only. It does not change frozen
feature/model/data semantics from commit
``6742ae8552ea7728eb571f531338c24f24abead9``.

This runner is separate from Issue #21 / #23 / #25 validation-only ablation
paths, which continue to reject 2025-26. It never refreshes provider data and
fails closed if the frozen local MoneyPuck cache is absent.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from hockey_moneyline.acquire import cache_path
from hockey_moneyline.advanced_stats import (
    XGOAL_FEATURE_COLUMNS,
    build_pregame_xgoal_share_features,
)
from hockey_moneyline.errors import AcquisitionError, EvaluationError
from hockey_moneyline.evaluate import (
    LOGISTIC_MAX_ITER,
    LOGISTIC_RANDOM_STATE,
    LOGISTIC_SOLVER,
    ForwardFoldSpec,
    _run_forward_folds,
    _validate_forward_specs,
    _xgoal_availability_summary,
    shared_logistic_model_record,
)
from hockey_moneyline.features import build_features
from hockey_moneyline.provenance import (
    DEFAULT_SEASONS,
    FINAL_TEST_SEASON,
    MIN_PRIOR_GAMES,
    provenance_record,
)
from hockey_moneyline.schema import FEATURE_COLUMNS, NormalizedGame
from hockey_moneyline.xgoal_share import (
    BASELINE_V1_PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE,
    PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE_FEATURE_COLUMNS,
    assert_prior_xgoal_share_contract,
    attach_pregame_xgoal_share_features,
    xgoal_features_valid,
)

MODEL_DEFINITION_FREEZE_COMMIT = "6742ae8552ea7728eb571f531338c24f24abead9"
FROZEN_FEATURE_FAMILY = BASELINE_V1_PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE
FROZEN_FEATURE_COLUMNS = PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE_FEATURE_COLUMNS
FROZEN_MIN_PRIOR_GAMES = 5
FROZEN_CANONICAL_SEASONS = (20212022, 20222023, 20232024, 20242025, 20252026)
FROZEN_TRAIN_SEASONS = (20212022, 20222023, 20232024, 20242025)
FROZEN_TEST_SEASON = 20252026
FROZEN_HELDOUT_FOLD_NAME = "frozen_heldout_test"
DEFAULT_FROZEN_HELDOUT_OUTPUT = Path(
    "data/derived/frozen_candidate_2025_26_heldout_report.json"
)
EXACT_SEASONS_CLI = "20212022,20222023,20232024,20242025,20252026"

FROZEN_FEATURE_COLUMN_ORDER = (
    "home_prior_games",
    "away_prior_games",
    "home_win_rate",
    "away_win_rate",
    "home_goals_for_per_game",
    "away_goals_for_per_game",
    "home_goals_against_per_game",
    "away_goals_against_per_game",
    "home_goal_diff_per_game",
    "away_goal_diff_per_game",
    "home_home_win_rate",
    "away_away_win_rate",
    "home_season_games",
    "away_season_games",
    "home_season_win_rate",
    "away_season_win_rate",
    "home_recent_win_rate",
    "away_recent_win_rate",
    "home_elo_minus_away_elo",
    "home_prior_xgoal_share",
    "away_prior_xgoal_share",
)


def frozen_heldout_fold_specs() -> list[ForwardFoldSpec]:
    """Return the one authorized frozen-candidate final fold."""

    return [
        ForwardFoldSpec(
            FROZEN_HELDOUT_FOLD_NAME,
            FROZEN_TRAIN_SEASONS,
            FROZEN_TEST_SEASON,
            "test",
        )
    ]


def assert_frozen_candidate_contract() -> None:
    """Guard the Issue #27 freeze: 21 columns, logistic config, eligibility."""

    assert_prior_xgoal_share_contract()
    if MODEL_DEFINITION_FREEZE_COMMIT != "6742ae8552ea7728eb571f531338c24f24abead9":
        raise RuntimeError("model-definition freeze commit constant was altered")
    if FROZEN_FEATURE_FAMILY != "baseline_v1_plus_elo_plus_prior_xgoal_share":
        raise RuntimeError("frozen feature family was altered")
    if tuple(FROZEN_FEATURE_COLUMNS) != FROZEN_FEATURE_COLUMN_ORDER:
        raise RuntimeError("frozen 21-column order was altered")
    if FROZEN_FEATURE_COLUMNS[:18] != FEATURE_COLUMNS:
        raise RuntimeError("frozen family must start with the 18 baseline columns")
    if XGOAL_FEATURE_COLUMNS != ("home_prior_xgoal_share", "away_prior_xgoal_share"):
        raise RuntimeError("frozen xGoal feature columns were altered")
    if FROZEN_MIN_PRIOR_GAMES != 5 or MIN_PRIOR_GAMES != 5:
        raise RuntimeError("frozen min-prior-games must remain 5")
    if FROZEN_MIN_PRIOR_GAMES != MIN_PRIOR_GAMES:
        raise RuntimeError("heldout runner cannot override min-prior-games")
    if LOGISTIC_SOLVER != "lbfgs":
        raise RuntimeError("frozen logistic solver was altered")
    if LOGISTIC_MAX_ITER != 1000:
        raise RuntimeError("frozen logistic max_iter was altered")
    if LOGISTIC_RANDOM_STATE != 42:
        raise RuntimeError("frozen logistic random_state was altered")
    if FROZEN_CANONICAL_SEASONS != DEFAULT_SEASONS:
        raise RuntimeError("frozen canonical seasons drifted from provenance.DEFAULT_SEASONS")
    if FROZEN_TEST_SEASON != FINAL_TEST_SEASON or FROZEN_TEST_SEASON != 20252026:
        raise RuntimeError("frozen final evaluation season must remain 20252026")
    if FROZEN_TRAIN_SEASONS != tuple(
        season for season in FROZEN_CANONICAL_SEASONS if season != FROZEN_TEST_SEASON
    ):
        raise RuntimeError("frozen training seasons must be 2021-22 through 2024-25")
    specs = frozen_heldout_fold_specs()
    if len(specs) != 1:
        raise RuntimeError("frozen heldout runner must expose exactly one fold")
    spec = specs[0]
    if (
        spec.train_seasons != FROZEN_TRAIN_SEASONS
        or spec.evaluation_season != FROZEN_TEST_SEASON
        or spec.role != "test"
    ):
        raise RuntimeError("frozen heldout fold contract was altered")
    _validate_forward_specs(specs)


def require_existing_frozen_cache(cache_dir: Path) -> Path:
    """Return the local frozen MoneyPuck cache, or fail closed without download."""

    path = cache_path(cache_dir)
    if not path.exists() or path.stat().st_size == 0:
        raise AcquisitionError(
            f"Frozen MoneyPuck cache is unavailable at {path}; "
            "dedicated heldout runner refuses to download"
        )
    return path


def sha256_path(path: Path) -> str:
    """Hash file bytes only. Does not parse CSV rows or 2025-26 values."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def assert_frozen_heldout_request(
    seasons: Sequence[int],
    *,
    confirm_consume_2025_26: bool,
) -> tuple[int, ...]:
    """Reject unauthorized or mistyped heldout requests before data evaluation."""

    selected = tuple(seasons)
    if not confirm_consume_2025_26:
        raise EvaluationError(
            "Frozen heldout runner requires --confirm-consume-2025-26"
        )
    if selected != FROZEN_CANONICAL_SEASONS:
        raise EvaluationError(
            "Frozen heldout runner requires exactly seasons "
            f"{EXACT_SEASONS_CLI}"
        )
    if selected[-1] != FROZEN_TEST_SEASON:
        raise EvaluationError(
            "Frozen heldout runner requires 2025-26 as the sole final evaluation season"
        )
    if selected.count(FROZEN_TEST_SEASON) != 1:
        raise EvaluationError(
            "Frozen heldout runner requires 2025-26 as the sole final evaluation season"
        )
    return selected


def evaluate_frozen_heldout_games(
    games: Sequence[NormalizedGame],
    records: Mapping[tuple[str, str], Any],
    *,
    confirm_consume_2025_26: bool,
) -> dict[str, Any]:
    """Score the frozen 21-column logistic candidate on the one authorized fold.

    No feature, model, hyperparameter, calibration, eligibility, or fold
    override is accepted. ``min_prior_games`` is fixed at 5. Finite prior xGoal
    share is required. Missing values are not imputed.
    """

    assert_frozen_candidate_contract()
    if not confirm_consume_2025_26:
        raise EvaluationError(
            "Frozen heldout runner requires confirm_consume_2025_26=True"
        )
    if not games:
        raise EvaluationError("No games supplied for frozen heldout evaluation")

    present = {game.season for game in games}
    missing = [season for season in FROZEN_CANONICAL_SEASONS if season not in present]
    if missing:
        raise EvaluationError(
            f"Frozen heldout evaluation is missing required seasons: {missing}"
        )
    extra = sorted(season for season in present if season not in FROZEN_CANONICAL_SEASONS)
    if extra:
        raise EvaluationError(
            f"Frozen heldout evaluation received non-canonical seasons: {extra}"
        )

    specs = frozen_heldout_fold_specs()
    _validate_forward_specs(specs)
    if len(specs) != 1 or specs[0].evaluation_season != FROZEN_TEST_SEASON:
        raise EvaluationError("Frozen heldout runner must use exactly one 2025-26 test fold")

    rows = build_features(games)
    xgoal_rows = build_pregame_xgoal_share_features(games, records)
    attached = attach_pregame_xgoal_share_features(rows, xgoal_rows)
    availability = _xgoal_availability_summary(attached, FROZEN_MIN_PRIOR_GAMES)
    folds = _run_forward_folds(
        attached,
        specs,
        FROZEN_MIN_PRIOR_GAMES,
        FROZEN_FEATURE_COLUMNS,
        extra_eligible=xgoal_features_valid,
        extra_exclusion_name="xgoal_unavailable",
    )
    if len(folds) != 1:
        raise EvaluationError("Frozen heldout runner produced more than one fold")
    fold = folds[0]
    train_columns = list(fold["logistic_regression"]["feature_columns"])
    if train_columns != list(FROZEN_FEATURE_COLUMNS):
        raise EvaluationError("Frozen heldout train columns drifted from the 21-column contract")
    metrics = fold["logistic_regression"]["metrics"]
    return {
        "issue": 32,
        "stage": "frozen_candidate_heldout_evaluation",
        "experiment": "issue_32_frozen_candidate_2025_26_heldout",
        "model_definition_freeze_commit": MODEL_DEFINITION_FREEZE_COMMIT,
        "source": provenance_record(),
        "feature_family": FROZEN_FEATURE_FAMILY,
        "feature_columns": list(FROZEN_FEATURE_COLUMNS),
        "shared_logistic_model": shared_logistic_model_record(),
        "min_prior_games": FROZEN_MIN_PRIOR_GAMES,
        "imputation": False,
        "feature_override": False,
        "model_override": False,
        "hyperparameter_override": False,
        "calibration_option": False,
        "eligibility_override": False,
        "alternate_fold": False,
        "refresh": False,
        "train_seasons": list(FROZEN_TRAIN_SEASONS),
        "evaluation_season": FROZEN_TEST_SEASON,
        "n_folds": 1,
        "fold": fold,
        "eligible_train_row_count": fold["train_window"]["n_eligible"],
        "eligible_test_row_count": fold["evaluation_window"]["n_eligible"],
        "accuracy": metrics["accuracy"],
        "roc_auc": metrics["roc_auc"],
        "log_loss": metrics["log_loss"],
        "brier": metrics["brier"],
        "ece": metrics["calibration"]["ece"],
        "calibration": metrics["calibration"],
        "mean_predicted_home_win_probability": metrics["mean_predicted_home_win"],
        "observed_home_win_rate": metrics["home_win_rate"],
        "pass_fail_threshold": None,
        "xgoal_availability": availability,
        "identical_train_prediction_column_order": True,
        "held_out_test_evaluated": True,
        "champion_promoted": False,
        "feature_timing": (
            "Features are built once in (game_date, game_id) order. Baseline, "
            "Elo, and prior xGoal-share snapshots for a calendar date are emitted "
            "before any same-day result updates team, Elo, or advanced state. "
            "Features use only history.game_date < target.game_date and reuse "
            "the Issue #19 primitive. This pathway evaluates exactly one fold: "
            "train 2021-22 through 2024-25, test 2025-26."
        ),
        "historical_data_window": {
            "seasons": sorted({game.season for game in games}),
            "start": min(game.game_date for game in games).isoformat(),
            "end": max(game.game_date for game in games).isoformat(),
            "n_normalized_games": len(games),
            "n_feature_rows": len(attached),
        },
        "notes": [
            "Issue #32 evaluates the frozen 21-column logistic candidate on 2025-26 once.",
            "2025-26 is a candidate-specific protected final evaluation period, not a globally untouched season.",
            "Issue #11 previously scored an older baseline on 2025-26; those metrics are not evidence for this candidate.",
            "No feature, model, hyperparameter, calibration, eligibility, or fold override is accepted.",
            "The dedicated runner never refreshes provider data.",
            "No champion model is promoted and no post-hoc pass/fail threshold is applied.",
        ],
    }
