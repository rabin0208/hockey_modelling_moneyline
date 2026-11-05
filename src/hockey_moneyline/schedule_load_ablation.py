"""Issue #38 Cycle-2 prior schedule-load compact feature ablation.

Dedicated Cycle-2 Modeling & Evaluation pathway. Does not weaken historical
Issue #21/#23/#25 validation-only guards that still reject 2025-26 as a fresh
heldout. Here 2025-26 is consumed Cycle-2 development evidence only; 2026-27
remains the protected future final evaluation period.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from hockey_moneyline.acquire import cache_path
from hockey_moneyline.advanced_stats import (
    AdvancedTeamGame,
    build_pregame_xgoal_share_features,
)
from hockey_moneyline.elo import ELO_CONFIG, ELO_FEATURE_COLUMN
from hockey_moneyline.errors import AcquisitionError, EvaluationError, FeatureError
from hockey_moneyline.evaluate import (
    LOGISTIC_MAX_ITER,
    LOGISTIC_RANDOM_STATE,
    LOGISTIC_SOLVER,
    ForwardFoldSpec,
    _array_sha256,
    _continuation_gate,
    _fold_eligible_rows,
    _macro_average_logistic,
    _row_identity_sha256,
    _run_forward_folds,
    _xgoal_availability_summary,
    shared_logistic_model_record,
)
from hockey_moneyline.features import build_features
from hockey_moneyline.provenance import MIN_PRIOR_GAMES, provenance_record
from hockey_moneyline.schedule_load import (
    CYCLE2_DEVELOPMENT_SEASONS,
    RESERVED_FUTURE_SEASON,
    SMALLEST_RECOMMENDED_FAMILY,
    ScheduleHistoryGame,
    build_prior_schedule_load_features,
)
from hockey_moneyline.schema import FEATURE_COLUMNS, NormalizedGame
from hockey_moneyline.xgoal_share import (
    BASELINE_V1_PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE,
    PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE_FEATURE_COLUMNS,
    assert_prior_xgoal_share_contract,
    attach_pregame_xgoal_share_features,
    xgoal_features_valid,
)

BASELINE_V1_PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE_PLUS_PRIOR_SCHEDULE_LOAD_COMPACT = (
    "baseline_v1_plus_elo_plus_prior_xgoal_share_plus_prior_schedule_load_compact"
)

SCHEDULE_LOAD_COMPACT_FEATURE_COLUMNS = SMALLEST_RECOMMENDED_FAMILY

PLUS_XGOAL_PLUS_PRIOR_SCHEDULE_LOAD_COMPACT_FEATURE_COLUMNS = (
    PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE_FEATURE_COLUMNS
    + SCHEDULE_LOAD_COMPACT_FEATURE_COLUMNS
)

FORBIDDEN_EXPERIMENTAL_COLUMNS = (
    "home_calendar_rest_days",
    "away_calendar_rest_days",
    "home_games_prior_3d",
    "away_games_prior_3d",
    "home_three_in_four",
    "away_three_in_four",
    "home_four_in_six",
    "away_four_in_six",
    "home_games_next_3d",
    "away_games_next_3d",
    "home_games_next_7d",
    "away_games_next_7d",
    "corsiPercentage",
    "fenwickPercentage",
    "xGoalsPercentage",
)

CYCLE2_EXACT_SEASONS_CLI = "20212022,20222023,20232024,20242025,20252026"
DEFAULT_CYCLE2_SCHEDULE_LOAD_OUTPUT = Path(
    "data/derived/cycle2_schedule_load_compact_ablation_report.json"
)
# Same MoneyPuck cache identity recorded by Issue #32 / Issue #34.
KNOWN_CYCLE2_MONEYPUCK_SHA256 = (
    "d4ca191ccc57e54c56f46497f09240b3e7f2d80f66dc2c41fe635825d035dd8f"
)

CARRY_FORWARD_DECISION = "carry forward Cycle-2 schedule-load candidate"
DO_NOT_CARRY_FORWARD_DECISION = "do not carry forward Cycle-2 schedule-load candidate"

SCHEDULE_LOAD_COMPACT_CONFIG = {
    "name": "prior_schedule_load_compact",
    "added_columns": list(SCHEDULE_LOAD_COMPACT_FEATURE_COLUMNS),
    "removed_from_incumbent": [],
    "incumbent_family": BASELINE_V1_PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE,
    "experimental_family": (
        BASELINE_V1_PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE_PLUS_PRIOR_SCHEDULE_LOAD_COMPACT
    ),
    "definitions": {
        "back_to_back": (
            "1 when the previous completed same-season schedule-history game was "
            "exactly one calendar day before target; otherwise 0"
        ),
        "games_prior_7d": (
            "completed same-season schedule-history games with date difference "
            "1 through 7 days"
        ),
    },
    "schedule_history_source": (
        "all complete regular-season MoneyPuck situation=all schedule-history "
        "games, including winner-unlabeled tied games"
    ),
    "timing": (
        "history.game_date < target.game_date; all target-date rows are "
        "snapshotted before any same-day schedule-history updates; season reset; "
        "no future-looking congestion"
    ),
    "minimum_history": (
        "Eligible rows require at least five prior canonical games for both "
        "teams and finite prior xGoal shares. Compact schedule-load features are "
        "fully defined and must not remove otherwise-eligible rows."
    ),
    "reuses_issue34_primitive": True,
    "reuses_issue19_primitive": True,
}


def cycle2_development_fold_specs() -> list[ForwardFoldSpec]:
    """Exact Issue #38 Cycle-2 development folds A/B/C/D.

    Fold D evaluates consumed 2025-26 development evidence. It is not a fresh
    heldout/test fold.
    """

    seasons = tuple(CYCLE2_DEVELOPMENT_SEASONS)
    return [
        ForwardFoldSpec("fold_a", seasons[:1], seasons[1], "cycle2_development"),
        ForwardFoldSpec("fold_b", seasons[:2], seasons[2], "cycle2_development"),
        ForwardFoldSpec("fold_c", seasons[:3], seasons[3], "cycle2_development"),
        ForwardFoldSpec("fold_d", seasons[:4], seasons[4], "cycle2_development"),
    ]


def assert_prior_schedule_load_compact_contract() -> None:
    """Guard incumbent 21 columns and the Issue #38 experimental 25-column family."""

    assert_prior_xgoal_share_contract()
    if list(PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE_FEATURE_COLUMNS) != [
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
    ]:
        raise RuntimeError("Issue #38 incumbent 21-column order was altered")
    if len(PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE_FEATURE_COLUMNS) != 21:
        raise RuntimeError("Issue #38 incumbent must remain exactly 21 columns")
    if SCHEDULE_LOAD_COMPACT_FEATURE_COLUMNS != (
        "home_back_to_back",
        "away_back_to_back",
        "home_games_prior_7d",
        "away_games_prior_7d",
    ):
        raise RuntimeError("Issue #38 compact schedule-load columns were altered")
    if PLUS_XGOAL_PLUS_PRIOR_SCHEDULE_LOAD_COMPACT_FEATURE_COLUMNS != (
        PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE_FEATURE_COLUMNS
        + SCHEDULE_LOAD_COMPACT_FEATURE_COLUMNS
    ):
        raise RuntimeError(
            "experimental family must equal incumbent plus the four compact "
            "schedule-load columns in order"
        )
    if len(PLUS_XGOAL_PLUS_PRIOR_SCHEDULE_LOAD_COMPACT_FEATURE_COLUMNS) != 25:
        raise RuntimeError("Issue #38 experimental family must be exactly 25 columns")
    forbidden = set(FORBIDDEN_EXPERIMENTAL_COLUMNS) & set(
        PLUS_XGOAL_PLUS_PRIOR_SCHEDULE_LOAD_COMPACT_FEATURE_COLUMNS
    )
    if forbidden:
        raise RuntimeError(
            "experimental family includes disallowed columns: "
            f"{sorted(forbidden)}"
        )
    if FEATURE_COLUMNS != PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE_FEATURE_COLUMNS[:18]:
        raise RuntimeError("baseline_v1 columns were altered")
    if ELO_FEATURE_COLUMN not in PLUS_XGOAL_PLUS_PRIOR_SCHEDULE_LOAD_COMPACT_FEATURE_COLUMNS:
        raise RuntimeError("experimental family must retain home_elo_minus_away_elo")


def schedule_load_compact_features_valid(row: Mapping[str, Any]) -> bool:
    """Return True when all four compact schedule-load columns are finite."""

    for column in SCHEDULE_LOAD_COMPACT_FEATURE_COLUMNS:
        if column not in row:
            return False
        value = row[column]
        if value is None:
            return False
        try:
            number = float(value)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(number):
            return False
    return True


def xgoal_and_schedule_load_compact_features_valid(row: Mapping[str, Any]) -> bool:
    """Return True when incumbent xGoal shares and compact schedule-load are finite."""

    return xgoal_features_valid(row) and schedule_load_compact_features_valid(row)


def attach_pregame_schedule_load_compact_features(
    modeling_rows: Sequence[Mapping[str, Any]],
    schedule_load_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Join Issue #34 compact schedule-load snapshots onto incumbent modeling rows."""

    if len(modeling_rows) != len(schedule_load_rows):
        raise FeatureError(
            "Issue #34 schedule-load primitive produced "
            f"{len(schedule_load_rows)} rows but incumbent modeling has "
            f"{len(modeling_rows)} rows"
        )

    by_id: dict[str, Mapping[str, Any]] = {}
    for schedule_row in schedule_load_rows:
        game_id = str(schedule_row["game_id"])
        if game_id in by_id:
            raise FeatureError(f"Duplicate schedule-load feature row for game {game_id}")
        by_id[game_id] = schedule_row

    modeling_ids = [str(row["game_id"]) for row in modeling_rows]
    if len(set(modeling_ids)) != len(modeling_ids):
        raise FeatureError("Incumbent modeling rows contain duplicate game_id values")
    if set(by_id) != set(modeling_ids):
        missing = sorted(set(modeling_ids) - set(by_id))
        extra = sorted(set(by_id) - set(modeling_ids))
        raise FeatureError(
            "Issue #34 schedule-load primitive cannot attach cleanly to incumbent "
            f"modeling rows; missing={missing[:10]} extra={extra[:10]}"
        )

    attached: list[dict[str, Any]] = []
    for row in modeling_rows:
        schedule_row = by_id[str(row["game_id"])]
        if (
            schedule_row["home_team_id"] != row["home_team_id"]
            or schedule_row["away_team_id"] != row["away_team_id"]
            or str(schedule_row["game_date"]) != str(row["game_date"])
            or int(schedule_row["season"]) != int(row["season"])
        ):
            raise FeatureError(
                "Issue #34 schedule-load identity/date/season disagrees with "
                f"modeling row {row['game_id']}"
            )
        merged = dict(row)
        for column in SCHEDULE_LOAD_COMPACT_FEATURE_COLUMNS:
            merged[column] = schedule_row[column]
        attached.append(merged)
    return attached


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def require_existing_cycle2_cache(cache_dir: Path) -> Path:
    """Return the local MoneyPuck cache, or fail closed without download."""

    path = cache_path(cache_dir)
    if not path.exists() or path.stat().st_size == 0:
        raise AcquisitionError(
            f"Cycle-2 MoneyPuck cache is unavailable at {path}; "
            "Issue #38 refuses to download or silently substitute new data"
        )
    return path


def _validate_cycle2_development_specs(
    specs: Sequence[ForwardFoldSpec],
    *,
    require_exact_abcd: bool = True,
) -> None:
    if not specs:
        raise EvaluationError("Cycle-2 schedule-load ablation requires at least one fold")
    expected = cycle2_development_fold_specs()
    if require_exact_abcd:
        if [spec.name for spec in specs] != [spec.name for spec in expected]:
            raise EvaluationError(
                "Cycle-2 schedule-load ablation requires exact folds fold_a/fold_b/"
                "fold_c/fold_d"
            )
    names: set[str] = set()
    previous_eval: int | None = None
    for index, spec in enumerate(specs):
        if spec.name in names:
            raise EvaluationError(f"Duplicate Cycle-2 fold name: {spec.name}")
        names.add(spec.name)
        if spec.role in {"test", "heldout", "held_out"}:
            raise EvaluationError(
                f"Cycle-2 fold {spec.name} must not be labeled heldout/test "
                f"(role={spec.role!r})"
            )
        if spec.role != "cycle2_development":
            raise EvaluationError(
                f"Cycle-2 fold {spec.name} must use role='cycle2_development'"
            )
        if require_exact_abcd:
            expected_spec = expected[index]
            if spec.train_seasons != expected_spec.train_seasons:
                raise EvaluationError(
                    f"Cycle-2 fold {spec.name} train seasons were altered"
                )
            if spec.evaluation_season != expected_spec.evaluation_season:
                raise EvaluationError(
                    f"Cycle-2 fold {spec.name} evaluation season was altered"
                )
        if RESERVED_FUTURE_SEASON in spec.train_seasons:
            raise EvaluationError("Cycle-2 folds cannot train on reserved 2026-27")
        if spec.evaluation_season == RESERVED_FUTURE_SEASON:
            raise EvaluationError("Cycle-2 folds cannot evaluate reserved 2026-27")
        if spec.evaluation_season >= RESERVED_FUTURE_SEASON:
            raise EvaluationError(
                f"Cycle-2 fold {spec.name} evaluation season "
                f"{spec.evaluation_season} is reserved/future"
            )
        if not spec.train_seasons:
            raise EvaluationError(f"Cycle-2 fold {spec.name} has no training seasons")
        if list(spec.train_seasons) != sorted(spec.train_seasons):
            raise EvaluationError(
                f"Cycle-2 fold {spec.name} training seasons must be ordered"
            )
        if len(set(spec.train_seasons)) != len(spec.train_seasons):
            raise EvaluationError(
                f"Cycle-2 fold {spec.name} training seasons must be unique"
            )
        if max(spec.train_seasons) >= spec.evaluation_season:
            raise EvaluationError(
                f"Cycle-2 fold {spec.name} is not strictly chronological"
            )
        if previous_eval is not None and spec.evaluation_season <= previous_eval:
            raise EvaluationError(
                "Cycle-2 fold evaluation seasons must increase strictly"
            )
        previous_eval = spec.evaluation_season


def _reject_reserved_future_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    if any(int(row["season"]) >= RESERVED_FUTURE_SEASON for row in rows):
        raise EvaluationError(
            "Cycle-2 schedule-load ablation rejects reserved future season "
            f"{RESERVED_FUTURE_SEASON}"
        )


def _schedule_load_availability_summary(
    rows: Sequence[dict[str, Any]],
    min_prior_games: int,
) -> dict[str, Any]:
    incumbent_eligible = [
        row
        for row in rows
        if (
            row["home_prior_games"] >= min_prior_games
            and row["away_prior_games"] >= min_prior_games
            and xgoal_features_valid(row)
        )
    ]
    experimental_eligible = [
        row for row in incumbent_eligible if schedule_load_compact_features_valid(row)
    ]
    return {
        "total_candidate_modeling_rows": len(rows),
        "eligible_incumbent_rows": len(incumbent_eligible),
        "rows_with_valid_schedule_load_compact_state": sum(
            1 for row in rows if schedule_load_compact_features_valid(row)
        ),
        "final_experimental_eligible_rows": len(experimental_eligible),
        "excluded_because_schedule_load_unavailable": (
            len(incumbent_eligible) - len(experimental_eligible)
        ),
        "xgoal_feature_available_rows": sum(
            1 for row in rows if xgoal_features_valid(row)
        ),
    }


def _identical_row_evidence(
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
            raise EvaluationError(
                "Issue #38 fold names do not align for identical-row check"
            )
        incumbent_train, incumbent_eval = _fold_eligible_rows(
            rows, spec, min_prior_games, xgoal_features_valid
        )
        experimental_train, experimental_eval = _fold_eligible_rows(
            rows, spec, min_prior_games, xgoal_and_schedule_load_compact_features_valid
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
            incumbent["train_window"]["n_eligible"]
            == experimental["train_window"]["n_eligible"]
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
                f"Issue #38 fold {spec.name} incumbent/experimental game IDs, "
                "targets, or logistic configuration are not identical. Stop and "
                "return the discrepancy to 00."
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
                "train_y_sha256": _array_sha256(
                    np.array(experimental_train_y, dtype=int)
                ),
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


def evaluate_schedule_load_compact_ablation_rows(
    rows: Sequence[dict[str, Any]],
    min_prior_games: int = MIN_PRIOR_GAMES,
    fold_specs: Sequence[ForwardFoldSpec] | None = None,
) -> dict[str, Any]:
    """Issue #38 Cycle-2 prior schedule-load compact ablation on folds A/B/C/D."""

    assert_prior_schedule_load_compact_contract()
    specs = list(fold_specs or cycle2_development_fold_specs())
    _validate_cycle2_development_specs(
        specs,
        require_exact_abcd=fold_specs is None,
    )
    _reject_reserved_future_rows(rows)

    required = PLUS_XGOAL_PLUS_PRIOR_SCHEDULE_LOAD_COMPACT_FEATURE_COLUMNS
    if any(column not in row for row in rows for column in required):
        raise EvaluationError(
            "Issue #38 schedule-load compact primitive was not attached to "
            "incumbent modeling rows"
        )

    xgoal_availability = _xgoal_availability_summary(rows, min_prior_games)
    if xgoal_availability["excluded_because_xgoal_unavailable"] > 0:
        raise EvaluationError(
            "Incumbent Issue #21 family unexpectedly lost otherwise eligible rows "
            "because xGoal state is unavailable: "
            f"{xgoal_availability}"
        )
    availability = _schedule_load_availability_summary(rows, min_prior_games)
    if availability["excluded_because_schedule_load_unavailable"] > 0:
        raise EvaluationError(
            "Experimental family unexpectedly lost otherwise eligible rows "
            "because compact schedule-load state is unavailable: "
            f"{availability}. Stop and return to 00."
        )

    families: dict[str, dict[str, Any]] = {}
    for name, columns in (
        (
            BASELINE_V1_PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE,
            PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE_FEATURE_COLUMNS,
        ),
        (
            BASELINE_V1_PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE_PLUS_PRIOR_SCHEDULE_LOAD_COMPACT,
            PLUS_XGOAL_PLUS_PRIOR_SCHEDULE_LOAD_COMPACT_FEATURE_COLUMNS,
        ),
    ):
        folds = _run_forward_folds(
            rows,
            specs,
            min_prior_games,
            columns,
            extra_eligible=xgoal_and_schedule_load_compact_features_valid,
            extra_exclusion_name="xgoal_or_schedule_load_unavailable",
        )
        families[name] = {
            "name": name,
            "feature_columns": list(columns),
            "removed_feature_columns": [],
            "added_feature_columns": (
                list(SCHEDULE_LOAD_COMPACT_CONFIG["added_columns"])
                if name
                == BASELINE_V1_PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE_PLUS_PRIOR_SCHEDULE_LOAD_COMPACT
                else []
            ),
            "folds": folds,
            "macro_average": _macro_average_logistic(folds),
        }

    identities = _identical_row_evidence(
        rows,
        specs,
        min_prior_games,
        families[BASELINE_V1_PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE]["folds"],
        families[
            BASELINE_V1_PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE_PLUS_PRIOR_SCHEDULE_LOAD_COMPACT
        ]["folds"],
    )

    gate = _continuation_gate(
        families[BASELINE_V1_PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE],
        families[
            BASELINE_V1_PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE_PLUS_PRIOR_SCHEDULE_LOAD_COMPACT
        ],
        issue_label="Issue #38",
        pass_decision=CARRY_FORWARD_DECISION,
        fail_decision=DO_NOT_CARRY_FORWARD_DECISION,
    )
    deltas = {
        key: (
            None
            if families[BASELINE_V1_PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE]["macro_average"][
                key
            ]
            is None
            or families[
                BASELINE_V1_PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE_PLUS_PRIOR_SCHEDULE_LOAD_COMPACT
            ]["macro_average"][key]
            is None
            else families[
                BASELINE_V1_PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE_PLUS_PRIOR_SCHEDULE_LOAD_COMPACT
            ]["macro_average"][key]
            - families[BASELINE_V1_PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE]["macro_average"][
                key
            ]
        )
        for key in ("accuracy", "roc_auc", "log_loss", "brier", "ece")
    }
    return {
        "source": provenance_record(),
        "experiment": "issue_38_cycle2_prior_schedule_load_compact_ablation",
        "evaluation_contract": "cycle2_development_forward_chaining",
        "consumed_cycle2_evidence_season": 20252026,
        "reserved_future_final_evaluation_season": RESERVED_FUTURE_SEASON,
        "reserved_future_season_accessed": False,
        "fold_d_is_fresh_heldout": False,
        "shared_logistic_model": shared_logistic_model_record(),
        "elo": dict(ELO_CONFIG),
        "prior_schedule_load_compact": dict(SCHEDULE_LOAD_COMPACT_CONFIG),
        "min_prior_games": min_prior_games,
        "schedule_load_availability": availability,
        "identical_rows_and_targets": identities,
        "feature_timing": (
            "Features are built once in (game_date, game_id) order. Baseline, "
            "Elo, prior xGoal-share, and compact prior schedule-load snapshots "
            "for a calendar date are emitted before any same-day updates. "
            "Schedule-load state is built from complete schedule history "
            "(including winner-unlabeled tied games) under "
            "history.game_date < target.game_date. Modeling targets remain "
            "winner-labeled only. Fold D evaluates consumed 2025-26 development "
            "evidence; 2026-27 is never accessed."
        ),
        "feature_families": families,
        "macro_deltas_schedule_load_compact_minus_incumbent": deltas,
        "continuation_gate": gate,
        "notes": [
            "Issue #38 compares baseline_v1_plus_elo_plus_prior_xgoal_share against "
            "baseline_v1_plus_elo_plus_prior_xgoal_share_plus_prior_schedule_load_compact "
            "using the same logistic regression.",
            "The experimental family adds only home_back_to_back, away_back_to_back, "
            "home_games_prior_7d, and away_games_prior_7d.",
            "Calendar rest, prior-3-day counts, 3-in-4, 4-in-6, and future-looking "
            "congestion are not included.",
            "2025-26 is consumed Cycle-2 development evidence, not a fresh heldout.",
            "2026-27 was not loaded, inspected, joined, featurized, scored, or evaluated.",
            "Passing the continuation gate retains only a Cycle-2 schedule-load "
            "candidate. It does not promote a champion model.",
            "No odds, roster/goalie context, deployment, Production, or worker/"
            "runtime work is included.",
        ],
    }


def evaluate_schedule_load_compact_ablation_games(
    games: Sequence[NormalizedGame],
    records: Mapping[tuple[str, str], AdvancedTeamGame],
    schedule_history: Sequence[ScheduleHistoryGame],
    min_prior_games: int = MIN_PRIOR_GAMES,
    fold_specs: Sequence[ForwardFoldSpec] | None = None,
) -> dict[str, Any]:
    if not games:
        raise EvaluationError(
            "No games supplied for Cycle-2 schedule-load compact ablation"
        )
    if any(game.season >= RESERVED_FUTURE_SEASON for game in games):
        raise EvaluationError(
            "Cycle-2 schedule-load ablation rejects reserved future season "
            f"{RESERVED_FUTURE_SEASON}"
        )
    if any(game.season >= RESERVED_FUTURE_SEASON for game in schedule_history):
        raise EvaluationError(
            "Cycle-2 schedule-history rejects reserved future season "
            f"{RESERVED_FUTURE_SEASON}"
        )
    present = {game.season for game in games}
    missing = [season for season in CYCLE2_DEVELOPMENT_SEASONS if season not in present]
    if fold_specs is None and missing:
        raise EvaluationError(
            f"Cycle-2 schedule-load ablation is missing required seasons: {missing}"
        )

    history_ids = {game.game_id for game in schedule_history}
    target_ids = {game.game_id for game in games}
    history_only = len(history_ids - target_ids)

    rows = build_features(games)
    xgoal_rows = build_pregame_xgoal_share_features(games, records)
    attached = attach_pregame_xgoal_share_features(rows, xgoal_rows)
    schedule_rows = build_prior_schedule_load_features(schedule_history, games)
    attached = attach_pregame_schedule_load_compact_features(attached, schedule_rows)

    report = evaluate_schedule_load_compact_ablation_rows(
        attached,
        min_prior_games=min_prior_games,
        fold_specs=fold_specs,
    )
    report["historical_data_window"] = {
        "seasons": sorted({game.season for game in games}),
        "start": min(game.game_date for game in games).isoformat(),
        "end": max(game.game_date for game in games).isoformat(),
        "n_normalized_modeling_targets": len(games),
        "n_complete_schedule_history_games": len(schedule_history),
        "n_history_only_tied_unlabeled_games": history_only,
        "n_feature_rows": len(attached),
    }
    report["history_target_contract"] = {
        "complete_schedule_history_games": len(schedule_history),
        "winner_labeled_modeling_targets": len(games),
        "history_only_tied_unlabeled_games": history_only,
        "schedule_history_includes_tied_unlabeled": True,
        "modeling_targets_require_winner_label": True,
        "attachment_rule": "exact game_id/date/season/home/away identity",
        "timing_rule": "history.game_date < target.game_date",
    }
    return report
