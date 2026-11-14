"""Issue #55 manual immutable pregame prediction-record execution.

Usage:
    python -m hockey_moneyline.issue55
    python -m hockey_moneyline.issue55 execute --target-json ... --nhl-snapshot-json ...
        --moneypuck-csv ... --cache-dir ... --store ... --observed-at-utc ...

This is a deliberately manual execution surface. It loads the committed Issue #52
frozen artifact, consumes an explicit target plus an Issue #54 current-season
prior-state package, and appends one auditable prediction record. It does not
refit, schedule scoring, place wagers, or introduce odds/staking.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from hockey_moneyline.advanced_stats import (
    AdvancedTeamGame,
    inspect_advanced_stats_csv_for_cycle2_development,
)
from hockey_moneyline.current_season_prior import (
    CURRENT_SEASON,
    CurrentSeasonPriorStatePackage,
    build_current_season_prior_state,
)
from hockey_moneyline.errors import (
    CurrentSeasonPriorSourceError,
    FeatureError,
    HockeyMoneylineError,
    PredictionError,
)
from hockey_moneyline.issue54 import (
    completed_transition_inputs,
    verified_empty_inputs,
)
from hockey_moneyline.nhl_api import NHLScheduleGame, NHLScheduleSnapshot
from hockey_moneyline.prediction_record_store import (
    DEFAULT_STORE_RELATIVE_PATH,
    PredictionRecordStore,
    prediction_record_content_hash,
)
from hockey_moneyline.prospective_artifact import (
    EXPECTED_FITTED_ARTIFACT_SHA256,
    FROZEN_FEATURE_COLUMNS,
    FrozenFittedArtifact,
    assert_frozen_prospective_contract,
    load_committed_frozen_artifact,
)
from hockey_moneyline.prospective_scoring import (
    CURRENT_SEASON_PRIOR_SOURCE_OBSERVED_AT_KEY,
    CURRENT_SEASON_PRIOR_SOURCE_SHA256_KEY,
    FORBIDDEN_MARKET_FIELDS,
    FORBIDDEN_TARGET_OUTCOME_FIELDS,
    PregamePredictionRecord,
    score_historical_replay_prediction,
    score_pregame_prediction,
)
from hockey_moneyline.schedule_load import (
    CYCLE2_DEVELOPMENT_SEASONS,
    RESERVED_FUTURE_SEASON,
    ScheduleHistoryGame,
    schedule_history_from_moneypuck_csv,
)
from hockey_moneyline.schedule_load_ablation import (
    KNOWN_CYCLE2_MONEYPUCK_SHA256,
    require_existing_cycle2_cache,
    sha256_path,
)
from hockey_moneyline.schema import NormalizedGame
from hockey_moneyline.unscored_target import UnscoredTarget

DEFAULT_OUTPUT = Path("data/derived/issue55_manual_pregame_prediction_record_report.json")
DEFAULT_PROOF_STORE = Path(
    "data/derived/prediction_records/issue55_validation_records.jsonl"
)
READY = "READY FOR FIRST PROTECTED 2026-27 PREGAME PREDICTION EXECUTION"
NOT_READY = "NOT READY — ONE MATERIAL PREDICTION-EXECUTION DEPENDENCY"
FORBIDDEN_TARGET_INPUT_FIELDS = FORBIDDEN_MARKET_FIELDS + FORBIDDEN_TARGET_OUTCOME_FIELDS


@dataclass(frozen=True, slots=True)
class ManualExecutionResult:
    written: bool
    idempotent_duplicate: bool
    record: PregamePredictionRecord
    record_sha256: str
    envelope: dict[str, Any]
    store_path: str
    artifact_sha256: str

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "written": self.written,
            "idempotent_duplicate": self.idempotent_duplicate,
            "record_sha256": self.record_sha256,
            "store_path": self.store_path,
            "artifact_sha256": self.artifact_sha256,
            "game_id": self.record.game_id,
            "season": self.record.season,
            "game_type": self.record.game_type,
            "prediction_timestamp_utc": self.record.prediction_timestamp_utc,
            "scheduled_start_time_utc": self.record.scheduled_start_time_utc,
            "model_home_win_probability": self.record.model_home_win_probability,
            "probability_kind": self.record.probability_kind,
            "feature_vector_sha256": self.record.feature_vector_sha256,
            "feature_columns": list(self.record.feature_columns),
        }


def require_committed_execution_artifact() -> FrozenFittedArtifact:
    """Load the committed freeze only. Never refit during prediction execution."""

    assert_frozen_prospective_contract()
    artifact = load_committed_frozen_artifact()
    if artifact.fitted_artifact_sha256 != EXPECTED_FITTED_ARTIFACT_SHA256:
        raise PredictionError(
            "Frozen artifact mismatch/tampering: "
            f"{artifact.fitted_artifact_sha256} != {EXPECTED_FITTED_ARTIFACT_SHA256}"
        )
    if tuple(artifact.feature_columns) != tuple(FROZEN_FEATURE_COLUMNS):
        raise PredictionError(
            "Feature-order drift: committed artifact columns are not the frozen 25-column order"
        )
    return artifact


def parse_unscored_target_input(payload: UnscoredTarget | Mapping[str, Any]) -> UnscoredTarget:
    """Require an unscored target. Outcome/market fields fail closed."""

    if isinstance(payload, UnscoredTarget):
        return payload
    if isinstance(payload, NormalizedGame):
        raise PredictionError(
            "Target input contains outcome fields; protected execution requires an "
            "unscored target without home_score/away_score/home_win"
        )
    if not isinstance(payload, Mapping):
        raise PredictionError("Target input must be an UnscoredTarget or JSON object")
    leaked = [field for field in FORBIDDEN_TARGET_INPUT_FIELDS if field in payload]
    if leaked:
        raise PredictionError(
            f"Target input contains forbidden outcome/market fields: {leaked}"
        )
    required = (
        "game_id",
        "game_date",
        "season",
        "game_type",
        "home_team_id",
        "away_team_id",
        "scheduled_start_time_utc",
        "observed_at_utc",
    )
    missing = [field for field in required if payload.get(field) in (None, "")]
    if missing:
        raise PredictionError(f"Target input is missing required identity/timing fields: {missing}")
    try:
        game_date = payload["game_date"]
        if isinstance(game_date, date) and not isinstance(game_date, datetime):
            parsed_date = game_date
        else:
            parsed_date = date.fromisoformat(str(game_date))
        start = payload["scheduled_start_time_utc"]
        observed = payload["observed_at_utc"]
        if not isinstance(start, datetime):
            start = datetime.fromisoformat(str(start).replace("Z", "+00:00"))
        if not isinstance(observed, datetime):
            observed = datetime.fromisoformat(str(observed).replace("Z", "+00:00"))
        return UnscoredTarget(
            game_id=str(payload["game_id"]),
            game_date=parsed_date,
            season=int(payload["season"]),
            game_type=str(payload["game_type"]),
            home_team_id=str(payload["home_team_id"]),
            away_team_id=str(payload["away_team_id"]),
            scheduled_start_time_utc=start,
            observed_at_utc=observed,
            nhl_game_id=(
                str(payload["nhl_game_id"]) if payload.get("nhl_game_id") else None
            ),
            source=str(payload.get("source") or "manual_explicit_target"),
        )
    except FeatureError as exc:
        raise PredictionError(str(exc)) from exc
    except (TypeError, ValueError) as exc:
        raise PredictionError(f"Target input is not a valid unscored target: {exc}") from exc


def _aware_utc(value: datetime, *, label: str) -> datetime:
    if value.tzinfo is None:
        raise PredictionError(f"{label} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _reject_reserved_future(
    games: Sequence[NormalizedGame] | Sequence[ScheduleHistoryGame],
    *,
    context: str,
) -> None:
    future = [game.game_id for game in games if game.season >= RESERVED_FUTURE_SEASON]
    if future:
        raise PredictionError(
            f"{context} cannot include 2026-27 outcomes outside the Issue #54 package; "
            f"ids={future[:10]}"
        )


def _reject_reserved_records(
    records: Mapping[tuple[str, str], AdvancedTeamGame],
    *,
    context: str,
) -> None:
    future = [
        game_id
        for (game_id, _team), record in records.items()
        if record.season >= RESERVED_FUTURE_SEASON
    ]
    if future:
        raise PredictionError(
            f"{context} cannot include 2026-27 advanced-stat outcomes outside the "
            f"Issue #54 package; ids={future[:10]}"
        )


def merge_frozen_history_with_current_season_package(
    frozen_completed_history: Sequence[NormalizedGame],
    frozen_advanced_records: Mapping[tuple[str, str], AdvancedTeamGame],
    frozen_schedule_history: Sequence[ScheduleHistoryGame],
    package: CurrentSeasonPriorStatePackage,
) -> tuple[
    list[NormalizedGame],
    dict[tuple[str, str], AdvancedTeamGame],
    list[ScheduleHistoryGame],
]:
    """Join frozen development history with approved Issue #54 current-season state."""

    _reject_reserved_future(
        frozen_completed_history,
        context="Frozen completed history",
    )
    _reject_reserved_future(
        frozen_schedule_history,
        context="Frozen schedule history",
    )
    _reject_reserved_records(
        frozen_advanced_records,
        context="Frozen advanced-stat records",
    )
    non_regular = [
        game.game_id for game in frozen_completed_history if game.game_type != "regular"
    ]
    if non_regular:
        raise PredictionError(
            "Pregame scoring rejects preseason/non-regular completed history; "
            f"ids={non_regular[:10]}"
        )
    overlap = set(game.game_id for game in frozen_completed_history) & set(
        game.game_id for game in package.winner_labeled_history
    )
    if overlap:
        raise PredictionError(
            "Frozen history and Issue #54 package share game identities: "
            f"{sorted(overlap)[:10]}"
        )
    merged_games = [*frozen_completed_history, *package.winner_labeled_history]
    merged_records = {**dict(frozen_advanced_records), **package.xgoal_records}
    merged_schedule = [*frozen_schedule_history, *package.schedule_history]
    if package.cutoff_game_id in {game.game_id for game in merged_games}:
        raise PredictionError(
            "Target outcome fields/history must not include the unscored cutoff "
            f"{package.cutoff_game_id}"
        )
    return merged_games, merged_records, merged_schedule


def _require_verified_package(
    package: CurrentSeasonPriorStatePackage | None,
    target: UnscoredTarget,
) -> CurrentSeasonPriorStatePackage:
    if package is None:
        raise PredictionError(
            "missing/unverified Issue #54 current-season source package; "
            "protected 2026-27 execution refuses to score without it"
        )
    if not isinstance(package, CurrentSeasonPriorStatePackage):
        raise PredictionError(
            "current-season source package is unverified; "
            "protected execution requires CurrentSeasonPriorStatePackage"
        )
    if package.season != CURRENT_SEASON:
        raise PredictionError(
            f"Issue #54 package season {package.season} is not the protected 2026-27 season"
        )
    if package.cutoff_game_id != target.game_id:
        raise PredictionError(
            "Issue #54 package cutoff disagrees with the explicit target: "
            f"{package.cutoff_game_id} != {target.game_id}"
        )
    if not package.source_sha256 or package.observed_at_utc.tzinfo is None:
        raise PredictionError(
            "Issue #54 package is missing source hash or observation timestamp"
        )
    return package


def _store_from(store: PredictionRecordStore | Path | str) -> PredictionRecordStore:
    if isinstance(store, PredictionRecordStore):
        return store
    return PredictionRecordStore(store)


def execute_manual_pregame_prediction(
    target: UnscoredTarget | Mapping[str, Any],
    current_season_package: CurrentSeasonPriorStatePackage | None,
    *,
    frozen_completed_history: Sequence[NormalizedGame],
    frozen_advanced_records: Mapping[tuple[str, str], AdvancedTeamGame],
    frozen_schedule_history: Sequence[ScheduleHistoryGame],
    store: PredictionRecordStore | Path | str,
) -> ManualExecutionResult:
    """Score one explicit 2026-27 target and append an immutable record."""

    artifact = require_committed_execution_artifact()
    parsed_target = parse_unscored_target_input(target)
    if parsed_target.season != CURRENT_SEASON or parsed_target.game_type != "regular":
        raise PredictionError(
            "Manual protected execution requires a 2026-27 regular-season target"
        )
    package = _require_verified_package(current_season_package, parsed_target)
    merged_games, merged_records, merged_schedule = (
        merge_frozen_history_with_current_season_package(
            frozen_completed_history,
            frozen_advanced_records,
            frozen_schedule_history,
            package,
        )
    )
    record = score_pregame_prediction(
        parsed_target,
        merged_games,
        merged_records,
        merged_schedule,
        artifact,
        **package.as_issue52_scoring_kwargs(),
    )
    if record.model_artifact_sha256 != EXPECTED_FITTED_ARTIFACT_SHA256:
        raise PredictionError("Scored record is not bound to the committed frozen artifact")
    if tuple(record.feature_columns) != tuple(FROZEN_FEATURE_COLUMNS):
        raise PredictionError("Scored record feature-order drifted from the frozen 25 columns")
    append = _store_from(store).append(record)
    return ManualExecutionResult(
        written=append.written,
        idempotent_duplicate=append.idempotent_duplicate,
        record=record,
        record_sha256=append.envelope["record_sha256"],
        envelope=append.envelope,
        store_path=append.store_path,
        artifact_sha256=artifact.fitted_artifact_sha256,
    )


def execute_historical_dry_run_prediction(
    target_game: NormalizedGame,
    all_completed_games: Sequence[NormalizedGame],
    advanced_records: Mapping[tuple[str, str], AdvancedTeamGame],
    schedule_history: Sequence[ScheduleHistoryGame],
    *,
    store: PredictionRecordStore | Path | str,
    scheduled_start_time_utc: datetime,
    prediction_timestamp_utc: datetime,
) -> ManualExecutionResult:
    """Score a historical/synthetic target through the same append-only store.

    Target outcomes are unread by the Issue #52 historical replay primitive.
    2026-27 games are rejected.
    """

    artifact = require_committed_execution_artifact()
    if target_game.season >= RESERVED_FUTURE_SEASON:
        raise PredictionError("Historical replay dry-run cannot select a 2026-27 target")
    if target_game.game_type != "regular":
        raise PredictionError(
            f"Frozen prospective target must be regular season, got {target_game.game_type!r}"
        )
    start = _aware_utc(scheduled_start_time_utc, label="scheduled_start_time_utc")
    observed = _aware_utc(prediction_timestamp_utc, label="prediction_timestamp_utc")
    if observed >= start:
        raise PredictionError(
            "Prediction/observation timestamp must occur before scheduled puck drop"
        )
    _reject_reserved_future(all_completed_games, context="Historical dry-run")
    _reject_reserved_future(schedule_history, context="Historical dry-run schedule history")
    _reject_reserved_records(advanced_records, context="Historical dry-run advanced-stat records")
    _snapshot, record = score_historical_replay_prediction(
        target_game,
        all_completed_games,
        advanced_records,
        schedule_history,
        artifact,
        scheduled_start_time_utc=start,
        prediction_timestamp_utc=observed,
        source_versions={"historical_replay_dry_run": True, "issue55_manual_execution": True},
    )
    append = _store_from(store).append(record)
    return ManualExecutionResult(
        written=append.written,
        idempotent_duplicate=append.idempotent_duplicate,
        record=record,
        record_sha256=append.envelope["record_sha256"],
        envelope=append.envelope,
        store_path=append.store_path,
        artifact_sha256=artifact.fitted_artifact_sha256,
    )


def load_moneypuck_csv_rows(path: Path) -> list[dict[str, str]]:
    """Load explicit caller-supplied MoneyPuck rows. Empty files are not verified-empty."""

    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise PredictionError(f"Current-season MoneyPuck file has no header: {path}")
            return [dict(row) for row in reader]
    except OSError as exc:
        raise PredictionError(f"Unreadable current-season MoneyPuck file {path}") from exc


def nhl_snapshot_from_dict(payload: Mapping[str, Any]) -> NHLScheduleSnapshot:
    """Rehydrate an explicit NHL schedule/status snapshot from caller-supplied JSON."""

    if not isinstance(payload, Mapping):
        raise PredictionError("NHL snapshot JSON must be an object")
    games_payload = payload.get("games")
    if not isinstance(games_payload, list):
        raise PredictionError("NHL snapshot JSON is missing games")
    games: list[NHLScheduleGame] = []
    for item in games_payload:
        if not isinstance(item, Mapping):
            raise PredictionError("NHL snapshot game entry must be an object")
        observed = datetime.fromisoformat(
            str(item["observed_at_utc"]).replace("Z", "+00:00")
        )
        start = datetime.fromisoformat(
            str(item["scheduled_start_time_utc"]).replace("Z", "+00:00")
        )
        games.append(
            NHLScheduleGame(
                nhl_game_id=str(item["nhl_game_id"]),
                season=int(item["season"]),
                game_type=str(item["game_type"]),
                game_date=date.fromisoformat(str(item["game_date"])),
                scheduled_start_time_utc=start,
                home_nhl_team_id=int(item["home_nhl_team_id"]),
                away_nhl_team_id=int(item["away_nhl_team_id"]),
                home_team_abbrev=str(item["home_team_abbrev"]),
                away_team_abbrev=str(item["away_team_abbrev"]),
                game_state=str(item["game_state"]),
                game_schedule_state=str(item["game_schedule_state"]),
                observed_at_utc=observed,
            )
        )
    covered_raw = payload.get("covered_dates") or []
    covered = frozenset(date.fromisoformat(str(day)) for day in covered_raw)
    observed_at = datetime.fromisoformat(
        str(payload["observed_at_utc"]).replace("Z", "+00:00")
    )
    return NHLScheduleSnapshot(
        endpoint_path=str(payload["endpoint_path"]),
        observed_at_utc=observed_at,
        games=tuple(games),
        covered_dates=covered,
    )


def load_frozen_development_history(cache_dir: Path) -> tuple[
    tuple[NormalizedGame, ...],
    dict[tuple[str, str], AdvancedTeamGame],
    tuple[ScheduleHistoryGame, ...],
]:
    """Load frozen 2021-22 through 2025-26 history. Does not open 2026-27 outcomes."""

    path = require_existing_cycle2_cache(cache_dir)
    actual_sha = sha256_path(path)
    if actual_sha != KNOWN_CYCLE2_MONEYPUCK_SHA256:
        raise PredictionError(
            "Frozen MoneyPuck cache SHA-256 mismatch: "
            f"{actual_sha} != {KNOWN_CYCLE2_MONEYPUCK_SHA256}"
        )
    inspection = inspect_advanced_stats_csv_for_cycle2_development(
        path,
        CYCLE2_DEVELOPMENT_SEASONS,
    )
    history = schedule_history_from_moneypuck_csv(path, seasons=CYCLE2_DEVELOPMENT_SEASONS)
    _reject_reserved_future(inspection.games, context="Frozen development cache")
    _reject_reserved_future(history, context="Frozen development schedule history")
    return tuple(inspection.games), dict(inspection.records), tuple(history)


def _game(
    game_id: str,
    day: str,
    season: int,
    home: str,
    away: str,
    *,
    home_won: bool = True,
    game_type: str = "regular",
) -> NormalizedGame:
    return NormalizedGame(
        game_id=game_id,
        game_date=date.fromisoformat(day),
        season=season,
        game_type=game_type,
        home_team_id=home,
        home_team_name=home,
        away_team_id=away,
        away_team_name=away,
        home_score=3 if home_won else 2,
        away_score=2 if home_won else 3,
        home_win=1 if home_won else 0,
        home_team_abbrev=home,
        away_team_abbrev=away,
        source="synthetic",
    )


def _record(
    game: NormalizedGame, team: str, opponent: str, side: str, seed: float
) -> AdvancedTeamGame:
    return AdvancedTeamGame(
        game_id=game.game_id,
        game_date=game.game_date,
        season=game.season,
        team_id=team,
        opponent_id=opponent,
        side=side,
        values={
            "xGoalsFor": 1.0 + seed,
            "xGoalsAgainst": 0.8 + seed / 2.0,
        },
    )


def historical_dry_run_fixture() -> tuple[
    list[NormalizedGame],
    dict[tuple[str, str], AdvancedTeamGame],
    list[ScheduleHistoryGame],
]:
    """Synthetic 2021-22/2022-23 history for manual-path validation. No 2026-27."""

    games: list[NormalizedGame] = []
    for offset, day in enumerate(
        (
            "2022-03-01",
            "2022-03-03",
            "2022-03-05",
            "2022-03-07",
            "2022-03-09",
            "2022-03-11",
        ),
        start=1,
    ):
        games.append(
            _game(
                f"20210210{offset:02d}",
                day,
                20212022,
                "BOS",
                "BUF",
                home_won=offset % 2 == 1,
            )
        )
        games.append(
            _game(
                f"20210220{offset:02d}",
                day,
                20212022,
                "CAR",
                "NJD",
                home_won=offset % 2 == 0,
            )
        )
    games.extend(
        [
            _game("2022020001", "2022-10-10", 20222023, "BOS", "BUF"),
            _game("2022020002", "2022-10-10", 20222023, "CAR", "NJD", home_won=False),
            _game("2022020004", "2022-10-12", 20222023, "BOS", "BUF", home_won=False),
        ]
    )
    records: dict[tuple[str, str], AdvancedTeamGame] = {}
    for index, game in enumerate(games, start=1):
        seed = index / 100.0
        records[(game.game_id, game.home_team_id)] = _record(
            game, game.home_team_id, game.away_team_id, "HOME", seed
        )
        records[(game.game_id, game.away_team_id)] = _record(
            game, game.away_team_id, game.home_team_id, "AWAY", seed + 0.01
        )
    history = [
        ScheduleHistoryGame(
            game_id=game.game_id,
            game_date=game.game_date,
            season=game.season,
            home_team_id=game.home_team_id,
            away_team_id=game.away_team_id,
        )
        for game in games
    ]
    history.append(
        ScheduleHistoryGame(
            game_id="2022020003",
            game_date=date(2022, 10, 11),
            season=20222023,
            home_team_id="BOS",
            away_team_id="CAR",
        )
    )
    return games, records, history


def _dry_run_timestamps(game_date: date) -> tuple[datetime, datetime]:
    scheduled = datetime.combine(game_date, time(23, 0), tzinfo=timezone.utc)
    observed = datetime.combine(game_date, time(12, 0), tzinfo=timezone.utc)
    return scheduled, observed


def _fail_closed_cases(
    games: Sequence[NormalizedGame],
    records: Mapping[tuple[str, str], AdvancedTeamGame],
    history: Sequence[ScheduleHistoryGame],
    store_dir: Path,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    target = games[-1]
    scheduled, observed = _dry_run_timestamps(target.game_date)

    def capture(name: str, callback) -> None:
        try:
            callback()
        except (PredictionError, FeatureError, CurrentSeasonPriorSourceError) as exc:
            cases.append({"case": name, "failed_closed": True, "message": str(exc)})
            return
        cases.append({"case": name, "failed_closed": False, "message": None})

    capture(
        "missing_issue54_source_package",
        lambda: execute_manual_pregame_prediction(
            verified_empty_inputs()[0],
            None,
            frozen_completed_history=games,
            frozen_advanced_records=records,
            frozen_schedule_history=history,
            store=store_dir / "missing_package.jsonl",
        ),
    )
    capture(
        "pregame_timestamp_guard",
        lambda: execute_historical_dry_run_prediction(
            target,
            games,
            records,
            history,
            store=store_dir / "late_timestamp.jsonl",
            scheduled_start_time_utc=scheduled,
            prediction_timestamp_utc=scheduled,
        ),
    )
    preseason = _game(
        "2022010001",
        "2022-09-20",
        20222023,
        "BOS",
        "BUF",
        game_type="preseason",
    )
    capture(
        "non_regular_target",
        lambda: execute_historical_dry_run_prediction(
            preseason,
            games,
            records,
            history,
            store=store_dir / "preseason.jsonl",
            scheduled_start_time_utc=scheduled,
            prediction_timestamp_utc=observed,
        ),
    )
    capture(
        "target_outcome_fields_in_input",
        lambda: parse_unscored_target_input(target),
    )
    capture(
        "target_outcome_fields_in_mapping",
        lambda: parse_unscored_target_input(
            {
                "game_id": "2026020001",
                "game_date": "2026-10-08",
                "season": CURRENT_SEASON,
                "game_type": "regular",
                "home_team_id": "BOS",
                "away_team_id": "BUF",
                "scheduled_start_time_utc": "2026-10-08T23:00:00+00:00",
                "observed_at_utc": "2026-10-08T16:00:00+00:00",
                "home_score": 3,
                "home_win": 1,
            }
        ),
    )

    conflict_store = PredictionRecordStore(store_dir / "conflict.jsonl")
    first = execute_historical_dry_run_prediction(
        target,
        games,
        records,
        history,
        store=conflict_store,
        scheduled_start_time_utc=scheduled,
        prediction_timestamp_utc=observed,
    )

    def conflict() -> None:
        earlier = datetime.combine(target.game_date, time(11, 0), tzinfo=timezone.utc)
        execute_historical_dry_run_prediction(
            target,
            games,
            records,
            history,
            store=conflict_store,
            scheduled_start_time_utc=scheduled,
            prediction_timestamp_utc=earlier,
        )

    capture("conflicting_duplicate_record", conflict)
    after_conflict = conflict_store.get_by_game_id(target.game_id)
    if after_conflict is None or after_conflict["record_sha256"] != first.record_sha256:
        cases.append(
            {
                "case": "append_only_unchanged_after_conflict",
                "failed_closed": False,
                "message": "conflict write mutated the existing protected record",
            }
        )
    else:
        cases.append(
            {
                "case": "append_only_unchanged_after_conflict",
                "failed_closed": True,
                "message": "store retained the original protected record",
            }
        )
    return cases


def build_issue55_execution_report(
    *,
    store_dir: Path | None = None,
) -> dict[str, Any]:
    blockers: list[str] = []
    artifact = require_committed_execution_artifact()
    if artifact.fitted_artifact_sha256 != EXPECTED_FITTED_ARTIFACT_SHA256:
        blockers.append("committed frozen artifact hash drifted")
    if tuple(artifact.feature_columns) != tuple(FROZEN_FEATURE_COLUMNS):
        blockers.append("frozen 25-column order drifted")
    if len(FROZEN_FEATURE_COLUMNS) != 25:
        blockers.append("frozen feature count is not 25")

    workdir = Path(store_dir or Path("data/derived/prediction_records/issue55_proof"))
    workdir.mkdir(parents=True, exist_ok=True)
    games, records, history = historical_dry_run_fixture()
    target = games[-1]
    scheduled, observed = _dry_run_timestamps(target.game_date)
    hist_store = PredictionRecordStore(workdir / "historical_dry_run.jsonl")
    first = execute_historical_dry_run_prediction(
        target,
        games,
        records,
        history,
        store=hist_store,
        scheduled_start_time_utc=scheduled,
        prediction_timestamp_utc=observed,
    )
    second = execute_historical_dry_run_prediction(
        target,
        games,
        records,
        history,
        store=hist_store,
        scheduled_start_time_utc=scheduled,
        prediction_timestamp_utc=observed,
    )
    if first.record_sha256 != second.record_sha256:
        blockers.append("identical-input historical records are not deterministic")
    if prediction_record_content_hash(first.record) != first.record_sha256:
        blockers.append("record-level content hash is not reproducible from the Issue #52 payload")
    if second.written or not second.idempotent_duplicate:
        blockers.append("duplicate exact write was not an idempotent no-op")
    if tuple(first.record.feature_columns) != tuple(FROZEN_FEATURE_COLUMNS):
        blockers.append("historical dry-run feature-order drifted")
    leaked = [
        field
        for field in (*FORBIDDEN_MARKET_FIELDS, *FORBIDDEN_TARGET_OUTCOME_FIELDS)
        if field in first.record.to_dict() or field in first.envelope
    ]
    if leaked:
        blockers.append(f"historical dry-run leaked fields: {leaked}")

    protected_store = PredictionRecordStore(workdir / "protected_synthetic.jsonl")
    empty_target, empty_snapshot, empty_rows, empty_observed = verified_empty_inputs()
    empty_package = build_current_season_prior_state(
        empty_target,
        empty_snapshot,
        empty_rows,
        observed_at_utc=empty_observed,
    )
    protected = execute_manual_pregame_prediction(
        empty_target,
        empty_package,
        frozen_completed_history=games,
        frozen_advanced_records=records,
        frozen_schedule_history=history,
        store=protected_store,
    )
    protected_again = execute_manual_pregame_prediction(
        empty_target,
        empty_package,
        frozen_completed_history=games,
        frozen_advanced_records=records,
        frozen_schedule_history=history,
        store=protected_store,
    )
    if protected.record_sha256 != protected_again.record_sha256:
        blockers.append("identical-input protected records are not deterministic")
    if protected_again.written or not protected_again.idempotent_duplicate:
        blockers.append("protected duplicate exact write was not an idempotent no-op")
    versions = protected.record.source_versions
    if versions.get(CURRENT_SEASON_PRIOR_SOURCE_SHA256_KEY) != empty_package.source_sha256:
        blockers.append("protected record did not preserve Issue #54 source SHA-256")
    if versions.get(CURRENT_SEASON_PRIOR_SOURCE_OBSERVED_AT_KEY) != (
        empty_package.observed_at_utc.isoformat()
    ):
        blockers.append("protected record did not preserve Issue #54 observation timestamp")
    if protected.artifact_sha256 != EXPECTED_FITTED_ARTIFACT_SHA256:
        blockers.append("protected execution did not use the committed frozen artifact")

    trans_target, trans_snapshot, trans_rows, trans_observed = completed_transition_inputs()
    trans_package = build_current_season_prior_state(
        trans_target,
        trans_snapshot,
        trans_rows,
        observed_at_utc=trans_observed,
    )
    trans = execute_manual_pregame_prediction(
        trans_target,
        trans_package,
        frozen_completed_history=games,
        frozen_advanced_records=records,
        frozen_schedule_history=history,
        store=PredictionRecordStore(workdir / "protected_transition.jsonl"),
    )
    feature_row = dict(
        zip(trans.record.feature_columns, trans.record.feature_values, strict=True)
    )
    if feature_row.get("home_season_games") != 2.0:
        blockers.append("completed-state Issue #54 prior games did not update frozen feature state")

    fail_closed = _fail_closed_cases(games, records, history, workdir / "fail_closed")
    if any(not case["failed_closed"] for case in fail_closed):
        leaked_cases = [case["case"] for case in fail_closed if not case["failed_closed"]]
        blockers.append(f"fail-closed cases did not raise: {leaked_cases}")

    outcome = READY if not blockers else NOT_READY
    report: dict[str, Any] = {
        "issue": 55,
        "outcome": outcome,
        "blockers": blockers,
        "frozen_contract_unchanged": {
            "model_parameters_changed": False,
            "feature_protocol_changed": False,
            "fitted_artifact_sha256": artifact.fitted_artifact_sha256,
            "expected_fitted_artifact_sha256": EXPECTED_FITTED_ARTIFACT_SHA256,
            "matches_expected_fitted_artifact_sha256": (
                artifact.fitted_artifact_sha256 == EXPECTED_FITTED_ARTIFACT_SHA256
            ),
            "feature_columns": list(FROZEN_FEATURE_COLUMNS),
            "feature_count": len(FROZEN_FEATURE_COLUMNS),
            "2026_27_outcomes_used_adaptively": False,
            "refit_during_prediction_execution": False,
        },
        "execution_surface": {
            "manual_explicit_only": True,
            "requires_explicit_target_and_source_package": True,
            "recurring_schedule": False,
            "daemon": False,
            "webhook_driven_scoring": False,
            "automatic_retry_loop": False,
            "production_deployed": False,
            "odds_staking_recommendations": False,
        },
        "record_store": {
            "layout": DEFAULT_STORE_RELATIVE_PATH,
            "gitignored_derived": True,
            "raw_provider_file_committed": False,
            "append_only": True,
            "duplicate_exact_write": "idempotent_no_op",
            "conflicting_duplicate": "fail_closed",
            "committed_live_2026_27_records": False,
            "storage_note": (
                "Prediction records are written to gitignored data/derived/prediction_records/. "
                "They are not committed because derived feature vectors are built from "
                "MoneyPuck/NHL source material under the existing derived-artifact policy. "
                "This issue does not invent a new Git-commit policy for prospective records."
            ),
        },
        "historical_synthetic_dry_run": first.to_audit_dict()
        | {
            "deterministic_identical_input_hash": first.record_sha256 == second.record_sha256,
            "duplicate_exact_write_idempotent": second.idempotent_duplicate and not second.written,
            "prediction_before_scheduled_start": (
                first.record.prediction_timestamp_utc < first.record.scheduled_start_time_utc
            ),
            "target_outcome_fields_in_prediction_record": False,
            "market_fields_in_prediction_record": False,
            "2026_27_target": False,
        },
        "protected_synthetic_empty_state": protected.to_audit_dict()
        | {
            "current_season_prior_source_sha256": versions.get(
                CURRENT_SEASON_PRIOR_SOURCE_SHA256_KEY
            ),
            "current_season_prior_source_observed_at_utc": versions.get(
                CURRENT_SEASON_PRIOR_SOURCE_OBSERVED_AT_KEY
            ),
            "empty_state_verified": empty_package.empty_state_verified,
            "duplicate_exact_write_idempotent": (
                protected_again.idempotent_duplicate and not protected_again.written
            ),
        },
        "protected_synthetic_completed_state": trans.to_audit_dict()
        | {
            "home_season_games": feature_row.get("home_season_games"),
            "current_season_prior_source_sha256": trans.record.source_versions.get(
                CURRENT_SEASON_PRIOR_SOURCE_SHA256_KEY
            ),
        },
        "fail_closed": fail_closed,
        "remaining_dependency": (
            None
            if not blockers
            else {
                "owner": "03 — Prediction, Odds & Recommendation Engine",
                "dependency": blockers[0],
                "unblock_signal": (
                    "Manual execution must load the committed frozen artifact, require an "
                    "Issue #54 source package for 2026-27 targets, append immutable records, "
                    "and fail closed on the documented prediction-execution guards"
                ),
            }
        ),
        "notes": [
            "Issue #54 / PR #56 is merged READY FOR 03 CURRENT-SEASON PRIOR-STATE CONSUMPTION.",
            "Issue #55 is the active 03 manual execution runway.",
            "Prediction execution loaded the committed frozen artifact and did not refit.",
            "Validation used historical/synthetic dry-run plus synthetic Issue #54 packages.",
            "No 2026-27 outcomes were used adaptively for fitting, tuning, or artifact mutation.",
            "No odds, market, ROI, staking, recommendation, Production, or schedule automation is included.",
        ],
    }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Issue #55 manual immutable pregame prediction-record execution"
    )
    sub = parser.add_subparsers(dest="command")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--proof-store-dir",
        type=Path,
        default=DEFAULT_PROOF_STORE.parent / "issue55_proof",
    )

    execute = sub.add_parser(
        "execute",
        help="Manually score one explicit 2026-27 target with an Issue #54 source package",
    )
    execute.add_argument("--target-json", type=Path, required=True)
    execute.add_argument("--nhl-snapshot-json", type=Path, required=True)
    execute.add_argument("--moneypuck-csv", type=Path, required=True)
    execute.add_argument("--cache-dir", type=Path, default=Path("data/cache/moneypuck"))
    execute.add_argument("--store", type=Path, default=Path(DEFAULT_STORE_RELATIVE_PATH))
    execute.add_argument("--observed-at-utc", required=True)
    execute.add_argument(
        "--result-json",
        dest="execute_output",
        type=Path,
        default=None,
        help="Optional path for the manual execution audit JSON",
    )
    return parser


def cmd_execute(args: argparse.Namespace) -> ManualExecutionResult:
    try:
        target_payload = json.loads(args.target_json.read_text(encoding="utf-8"))
        snapshot_payload = json.loads(args.nhl_snapshot_json.read_text(encoding="utf-8"))
        target = parse_unscored_target_input(target_payload)
        snapshot = nhl_snapshot_from_dict(snapshot_payload)
        observed = datetime.fromisoformat(str(args.observed_at_utc).replace("Z", "+00:00"))
        rows = load_moneypuck_csv_rows(args.moneypuck_csv)
        package = build_current_season_prior_state(
            target,
            snapshot,
            rows,
            observed_at_utc=observed,
        )
        games, records, history = load_frozen_development_history(args.cache_dir)
        return execute_manual_pregame_prediction(
            target,
            package,
            frozen_completed_history=games,
            frozen_advanced_records=records,
            frozen_schedule_history=history,
            store=args.store,
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, HockeyMoneylineError):
            raise
        raise PredictionError(f"Manual execution input is invalid: {exc}") from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "execute":
            result = cmd_execute(args)
            payload = result.to_audit_dict()
            payload["record_sha256"] = result.record_sha256
            payload["idempotent_duplicate"] = result.idempotent_duplicate
            print(json.dumps(payload, indent=2, sort_keys=True))
            if getattr(args, "execute_output", None) is not None:
                args.execute_output.parent.mkdir(parents=True, exist_ok=True)
                args.execute_output.write_text(
                    json.dumps(payload, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                print(f"Wrote {args.execute_output}")
            return 0
        report = build_issue55_execution_report(store_dir=args.proof_store_dir)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        print(f"Wrote {args.output}")
        return 0 if report["outcome"] == READY else 1
    except HockeyMoneylineError as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

