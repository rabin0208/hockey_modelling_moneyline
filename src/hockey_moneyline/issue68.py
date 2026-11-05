"""Issue #68 manual opening-day multi-game prediction execution.

Usage:
    python -m hockey_moneyline.issue68
    python -m hockey_moneyline.issue68 inventory --game-date 2026-09-29 \
        --observed-at-utc TIMESTAMP --write-snapshot PATH
    python -m hockey_moneyline.issue68 execute --game-ids ID1,ID2 \
        --nhl-snapshot-json PATH --moneypuck-csv PATH --cache-dir PATH \
        --store PATH --observed-at-utc TIMESTAMP

Issue #55 one-target execution remains the safe single-game path. Opening day
has five regular-season games, three of which start after midnight UTC while
keeping the NHL game date 2026-09-29. Repeating hand-built target JSON plus a
separate receipt command for each game is a material operator-error risk, so
this module adds one explicit manual batch over the existing frozen scorer,
Issue #54 source package, append-only store, and Issue #63 receipt.

The operator must invoke it. It does not schedule itself, post to GitHub, or
score unless ``execute`` is called with an explicit finite game-id list.
"""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from hockey_moneyline.acquire import cache_path
from hockey_moneyline.current_season_prior import (
    CURRENT_SEASON,
    assemble_current_season_nhl_coverage,
    build_current_season_prior_state,
    required_coverage_dates,
)
from hockey_moneyline.errors import (
    CurrentSeasonPriorSourceError,
    FeatureError,
    HockeyMoneylineError,
    PredictionError,
)
from hockey_moneyline.issue55 import (
    execute_manual_pregame_prediction,
    historical_dry_run_fixture,
    load_frozen_development_history,
    load_moneypuck_csv_rows,
    nhl_snapshot_from_dict,
    require_committed_execution_artifact,
)
from hockey_moneyline.nhl_api import NHLScheduleGame, NHLScheduleSnapshot
from hockey_moneyline.normalize import canonical_moneypuck_team_code
from hockey_moneyline.prediction_record_store import PredictionRecordStore
from hockey_moneyline.pregame_audit_receipt import build_pregame_audit_receipt
from hockey_moneyline.prospective_artifact import (
    EXPECTED_FITTED_ARTIFACT_SHA256,
    FROZEN_FEATURE_COLUMNS,
)
from hockey_moneyline.prospective_scoring import (
    FORBIDDEN_MARKET_FIELDS,
    FORBIDDEN_TARGET_OUTCOME_FIELDS,
)
from hockey_moneyline.unscored_target import UnscoredTarget

DEFAULT_OUTPUT = Path("data/derived/issue68_opening_day_manual_batch_report.json")
DEFAULT_STORE = Path("data/derived/prediction_records/pregame_prediction_records.jsonl")
OPENING_DAY = date(2026, 9, 29)
OPENING_DAY_INVENTORY_OBSERVED_AT_UTC = "2026-09-24T17:38:07.994672+00:00"
OPENING_DAY_INVENTORY_ENDPOINT = "assembled:/v1/schedule/{date}"
FUT_GAME_STATE = "FUT"
READY = "READY FOR OPENING-DAY MANUAL MULTI-GAME EXECUTION"
NOT_READY = "NOT READY — ONE MATERIAL OPENING-DAY EXECUTION DEPENDENCY"
SOURCE_REFRESH_FAILURE = "source refresh failure:"
NO_LONGER_FUT = "is no longer FUT"
AT_OR_AFTER_PUCK_DROP = "observation/prediction timestamp is at or after puck drop"
REAL_OPENING_DAY_GAME_IDS = (
    "2026020001",
    "2026020002",
    "2026020003",
    "2026020004",
    "2026020005",
)
# Schedule observation only. These ids are not passed to the rehearsal scorer.
OPENING_DAY_OBSERVED_TARGETS = (
    {
        "execution_order": 1,
        "game_id": "2026020001",
        "away_team": "FLA",
        "home_team": "CAR",
        "scheduled_start_time_utc": "2026-09-29T21:00:00+00:00",
        "game_type": "regular",
        "game_state_at_observation": "FUT",
        "season": CURRENT_SEASON,
        "game_date": "2026-09-29",
    },
    {
        "execution_order": 2,
        "game_id": "2026020002",
        "away_team": "MTL",
        "home_team": "TOR",
        "scheduled_start_time_utc": "2026-09-29T23:00:00+00:00",
        "game_type": "regular",
        "game_state_at_observation": "FUT",
        "season": CURRENT_SEASON,
        "game_date": "2026-09-29",
    },
    {
        "execution_order": 3,
        "game_id": "2026020003",
        "away_team": "NYR",
        "home_team": "BOS",
        "scheduled_start_time_utc": "2026-09-30T00:00:00+00:00",
        "game_type": "regular",
        "game_state_at_observation": "FUT",
        "season": CURRENT_SEASON,
        "game_date": "2026-09-29",
    },
    {
        "execution_order": 4,
        "game_id": "2026020004",
        "away_team": "VAN",
        "home_team": "EDM",
        "scheduled_start_time_utc": "2026-09-30T02:00:00+00:00",
        "game_type": "regular",
        "game_state_at_observation": "FUT",
        "season": CURRENT_SEASON,
        "game_date": "2026-09-29",
    },
    {
        "execution_order": 5,
        "game_id": "2026020005",
        "away_team": "CHI",
        "home_team": "VGK",
        "scheduled_start_time_utc": "2026-09-30T02:30:00+00:00",
        "game_type": "regular",
        "game_state_at_observation": "FUT",
        "season": CURRENT_SEASON,
        "game_date": "2026-09-29",
    },
)
SYNTHETIC_GAME_IDS = (
    "SYNTHETIC-ISSUE68-001",
    "SYNTHETIC-ISSUE68-002",
    "SYNTHETIC-ISSUE68-003",
)
INVALID_SYNTHETIC_GAME_ID = "SYNTHETIC-ISSUE68-MISSING"
_SYNTHETIC_OBSERVED = datetime(2026, 9, 29, 16, tzinfo=timezone.utc)
_PRODUCT_MODULE = "issue68.py"


@dataclass(frozen=True, slots=True)
class ManualBatchGameResult:
    requested_order: int
    game_id: str
    status: str
    written: bool
    idempotent_duplicate: bool
    record_preserved: bool
    record_sha256: str | None
    existing_record_sha256: str | None
    receipt_sha256: str | None
    receipt: dict[str, Any] | None
    error: str | None
    scheduled_start_time_utc: str | None
    home_team_id: str | None
    away_team_id: str | None
    model_home_win_probability: float | None
    artifact_sha256: str | None

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "requested_order": self.requested_order,
            "game_id": self.game_id,
            "status": self.status,
            "written": self.written,
            "idempotent_duplicate": self.idempotent_duplicate,
            "record_preserved": self.record_preserved,
            "record_sha256": self.record_sha256,
            "existing_record_sha256": self.existing_record_sha256,
            "receipt_sha256": self.receipt_sha256,
            "receipt": self.receipt,
            "error": self.error,
            "scheduled_start_time_utc": self.scheduled_start_time_utc,
            "home_team_id": self.home_team_id,
            "away_team_id": self.away_team_id,
            "model_home_win_probability": self.model_home_win_probability,
            "artifact_sha256": self.artifact_sha256,
        }


@dataclass(frozen=True, slots=True)
class ManualBatchResult:
    game_ids: tuple[str, ...]
    results: tuple[ManualBatchGameResult, ...]
    store_path: str
    artifact_sha256: str
    partial_failure: bool

    def to_audit_dict(self) -> dict[str, Any]:
        succeeded = [
            item.game_id
            for item in self.results
            if item.status in {"written", "idempotent_duplicate"}
        ]
        failed = [item.game_id for item in self.results if item.status == "failed"]
        return {
            "issue": 68,
            "command": "execute",
            "manual_explicit": True,
            "unattended": False,
            "recurring_schedule": False,
            "github_posting": False,
            "partial_failure": self.partial_failure,
            "preserved_existing_records": True,
            "game_ids": list(self.game_ids),
            "succeeded_game_ids": succeeded,
            "failed_game_ids": failed,
            "succeeded_count": len(succeeded),
            "failed_count": len(failed),
            "store_path": self.store_path,
            "artifact_sha256": self.artifact_sha256,
            "results": [item.to_audit_dict() for item in self.results],
        }


def batch_process_exit_code(result: ManualBatchResult) -> int:
    """Non-zero when any requested target failed. Successes stay in the summary."""

    return 1 if result.partial_failure else 0


def _utc(value: datetime, *, label: str) -> datetime:
    if value.tzinfo is None:
        raise PredictionError(f"{label} must be timezone-aware")
    return value.astimezone(timezone.utc)


def parse_explicit_game_ids(raw: str) -> list[str]:
    """Parse one explicit finite list. Duplicates and blanks are invalid invocations."""

    parts = [part.strip() for part in str(raw).split(",")]
    if not parts or any(part == "" for part in parts):
        raise PredictionError("explicit target list is empty or contains a blank game id")
    if len(parts) != len(set(parts)):
        raise PredictionError("explicit target list contains duplicate game ids")
    return parts


def _require_explicit_game_ids(game_ids: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(str(game_id).strip() for game_id in game_ids)
    if not normalized or any(game_id == "" for game_id in normalized):
        raise PredictionError("explicit target list is empty or contains a blank game id")
    if len(normalized) != len(set(normalized)):
        raise PredictionError("explicit target list contains duplicate game ids")
    return normalized


def nhl_snapshot_to_dict(snapshot: NHLScheduleSnapshot) -> dict[str, Any]:
    """Serialize a snapshot for the existing Issue #55 loader."""

    return {
        "endpoint_path": snapshot.endpoint_path,
        "observed_at_utc": _utc(
            snapshot.observed_at_utc, label="snapshot.observed_at_utc"
        ).isoformat(),
        "covered_dates": [day.isoformat() for day in sorted(snapshot.covered_dates)],
        "games": [game.to_dict() for game in snapshot.games],
    }


def list_regular_season_targets(
    snapshot: NHLScheduleSnapshot,
    game_date: date,
) -> list[dict[str, Any]]:
    """List regular-season targets for one NHL game date. Does not score."""

    selected = [
        game
        for game in snapshot.games
        if game.game_date == game_date
        and game.season == CURRENT_SEASON
        and game.game_type == "regular"
    ]
    selected.sort(key=lambda game: (game.scheduled_start_time_utc, game.nhl_game_id))
    rows: list[dict[str, Any]] = []
    for index, game in enumerate(selected, start=1):
        rows.append(
            {
                "execution_order": index,
                "game_id": game.nhl_game_id,
                "away_team": game.away_team_abbrev,
                "home_team": game.home_team_abbrev,
                "scheduled_start_time_utc": game.scheduled_start_time_utc.isoformat(),
                "game_type": game.game_type,
                "game_state": game.game_state,
                "game_date": game.game_date.isoformat(),
                "season": game.season,
                "scored": False,
            }
        )
    return rows


def compare_recorded_opening_day_inventory(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    recorded = [item["game_id"] for item in OPENING_DAY_OBSERVED_TARGETS]
    observed = [str(row["game_id"]) for row in rows]
    return {
        "matches_recorded_opening_day_game_ids": observed == recorded,
        "recorded_game_ids": recorded,
        "observed_game_ids": observed,
        "recorded_observed_at_utc": OPENING_DAY_INVENTORY_OBSERVED_AT_UTC,
    }


def refresh_nhl_schedule_snapshot(
    game_date: date,
    observed_at_utc: datetime,
    *,
    opener: Any | None = None,
) -> NHLScheduleSnapshot:
    """Explicit NHL coverage refresh. A fetch failure is not an empty slate."""

    kwargs: dict[str, Any] = {"observed_at_utc": observed_at_utc}
    if opener is not None:
        kwargs["opener"] = opener
    try:
        return assemble_current_season_nhl_coverage(game_date, **kwargs)
    except (HockeyMoneylineError, OSError, TimeoutError) as exc:
        raise PredictionError(f"{SOURCE_REFRESH_FAILURE} {exc}") from exc


def _resolve_snapshot_game(snapshot: NHLScheduleSnapshot, game_id: str) -> NHLScheduleGame:
    matches = [game for game in snapshot.games if game.nhl_game_id == game_id]
    if len(matches) != 1:
        raise PredictionError(
            f"invalid target {game_id}: not uniquely present in the NHL schedule snapshot"
        )
    return matches[0]


def _target_for_batch_game(game: NHLScheduleGame, observed: datetime) -> UnscoredTarget:
    if game.season != CURRENT_SEASON or game.game_type != "regular":
        raise PredictionError(
            f"invalid target {game.nhl_game_id}: manual batch requires a 2026-27 "
            f"regular-season game, got season {game.season} game_type {game.game_type!r}"
        )
    if game.game_state != FUT_GAME_STATE:
        raise PredictionError(
            f"Target {game.nhl_game_id} {NO_LONGER_FUT}; "
            f"NHL game_state is {game.game_state!r}"
        )
    if game.scheduled_start_time_utc.tzinfo is None:
        raise PredictionError(
            f"scheduled start for {game.nhl_game_id} must be timezone-aware"
        )
    start = game.scheduled_start_time_utc.astimezone(timezone.utc)
    if observed >= start:
        raise PredictionError(
            f"{AT_OR_AFTER_PUCK_DROP} for {game.nhl_game_id}"
        )
    try:
        home = canonical_moneypuck_team_code(game.home_team_abbrev, game.season)
        away = canonical_moneypuck_team_code(game.away_team_abbrev, game.season)
        return UnscoredTarget(
            game_id=game.nhl_game_id,
            game_date=game.game_date,
            season=game.season,
            game_type=game.game_type,
            home_team_id=home,
            away_team_id=away,
            scheduled_start_time_utc=start,
            observed_at_utc=observed,
            nhl_game_id=game.nhl_game_id,
            home_nhl_team_id=game.home_nhl_team_id,
            away_nhl_team_id=game.away_nhl_team_id,
            source="manual_explicit_batch",
        )
    except FeatureError as exc:
        raise PredictionError(str(exc)) from exc


def _failed_game(
    *,
    requested_order: int,
    game_id: str,
    error: str,
    store: PredictionRecordStore,
    record_sha256: str | None = None,
    written: bool = False,
    idempotent_duplicate: bool = False,
    scheduled_start_time_utc: str | None = None,
    home_team_id: str | None = None,
    away_team_id: str | None = None,
    artifact_sha256: str | None = None,
) -> ManualBatchGameResult:
    existing_sha256 = None
    preserved = False
    try:
        existing = store.get_by_game_id(game_id)
    except HockeyMoneylineError:
        existing = None
    if existing is not None:
        preserved = True
        existing_sha256 = str(existing["record_sha256"])
    if record_sha256 is not None:
        preserved = True
    return ManualBatchGameResult(
        requested_order=requested_order,
        game_id=game_id,
        status="failed",
        written=written,
        idempotent_duplicate=idempotent_duplicate,
        record_preserved=preserved,
        record_sha256=record_sha256 or existing_sha256,
        existing_record_sha256=existing_sha256,
        receipt_sha256=None,
        receipt=None,
        error=error,
        scheduled_start_time_utc=scheduled_start_time_utc,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        model_home_win_probability=None,
        artifact_sha256=artifact_sha256,
    )


def execute_manual_opening_day_batch(
    game_ids: Sequence[str],
    nhl_snapshot: NHLScheduleSnapshot | None,
    moneypuck_rows: Sequence[Mapping[str, Any]] | None,
    *,
    frozen_completed_history: Sequence[Any],
    frozen_advanced_records: Mapping[tuple[str, str], Any],
    frozen_schedule_history: Sequence[Any],
    store: PredictionRecordStore | Path | str,
    observed_at_utc: datetime,
) -> ManualBatchResult:
    """Score an explicit game-id list. Per-target failures do not roll back writes."""

    ids = _require_explicit_game_ids(game_ids)
    if not isinstance(nhl_snapshot, NHLScheduleSnapshot) or moneypuck_rows is None:
        raise PredictionError(
            f"{SOURCE_REFRESH_FAILURE} NHL snapshot or current-season MoneyPuck "
            "material is missing"
        )
    observed = _utc(observed_at_utc, label="observed_at_utc")
    snapshot_observed = _utc(
        nhl_snapshot.observed_at_utc, label="nhl_snapshot.observed_at_utc"
    )
    if snapshot_observed != observed:
        raise PredictionError(
            f"{SOURCE_REFRESH_FAILURE} NHL snapshot observation timestamp "
            f"{snapshot_observed.isoformat()} disagrees with the explicit observation "
            f"timestamp {observed.isoformat()}"
        )
    artifact = require_committed_execution_artifact()
    rows = list(moneypuck_rows)
    record_store = store if isinstance(store, PredictionRecordStore) else PredictionRecordStore(store)
    results: list[ManualBatchGameResult] = []
    for index, game_id in enumerate(ids, start=1):
        try:
            game = _resolve_snapshot_game(nhl_snapshot, game_id)
            target = _target_for_batch_game(game, observed)
            package = build_current_season_prior_state(
                target,
                nhl_snapshot,
                rows,
                observed_at_utc=observed,
            )
            execution = execute_manual_pregame_prediction(
                target,
                package,
                frozen_completed_history=frozen_completed_history,
                frozen_advanced_records=frozen_advanced_records,
                frozen_schedule_history=frozen_schedule_history,
                store=record_store,
            )
        except HockeyMoneylineError as exc:
            results.append(
                _failed_game(
                    requested_order=index,
                    game_id=game_id,
                    error=str(exc),
                    store=record_store,
                    artifact_sha256=artifact.fitted_artifact_sha256,
                )
            )
            continue
        try:
            receipt = build_pregame_audit_receipt(execution.envelope)
        except PredictionError as exc:
            results.append(
                _failed_game(
                    requested_order=index,
                    game_id=game_id,
                    error=(
                        "receipt generation failed after the immutable record was "
                        f"preserved: {exc}"
                    ),
                    store=record_store,
                    record_sha256=execution.record_sha256,
                    written=execution.written,
                    idempotent_duplicate=execution.idempotent_duplicate,
                    scheduled_start_time_utc=execution.record.scheduled_start_time_utc,
                    home_team_id=execution.record.home_team_id,
                    away_team_id=execution.record.away_team_id,
                    artifact_sha256=execution.artifact_sha256,
                )
            )
            continue
        status = "written" if execution.written else "idempotent_duplicate"
        if not execution.written and not execution.idempotent_duplicate:
            results.append(
                _failed_game(
                    requested_order=index,
                    game_id=game_id,
                    error="append result was neither a write nor an idempotent duplicate",
                    store=record_store,
                    record_sha256=execution.record_sha256,
                    artifact_sha256=execution.artifact_sha256,
                )
            )
            continue
        results.append(
            ManualBatchGameResult(
                requested_order=index,
                game_id=game_id,
                status=status,
                written=execution.written,
                idempotent_duplicate=execution.idempotent_duplicate,
                record_preserved=True,
                record_sha256=execution.record_sha256,
                existing_record_sha256=execution.record_sha256,
                receipt_sha256=receipt.receipt_sha256,
                receipt=receipt.to_dict(),
                error=None,
                scheduled_start_time_utc=execution.record.scheduled_start_time_utc,
                home_team_id=execution.record.home_team_id,
                away_team_id=execution.record.away_team_id,
                model_home_win_probability=float(execution.record.model_home_win_probability),
                artifact_sha256=execution.artifact_sha256,
            )
        )
    partial = any(item.status == "failed" for item in results)
    return ManualBatchResult(
        game_ids=ids,
        results=tuple(results),
        store_path=str(record_store.path),
        artifact_sha256=artifact.fitted_artifact_sha256,
        partial_failure=partial,
    )


def _source_failure(exc: Exception) -> PredictionError:
    message = str(exc)
    if message.startswith(SOURCE_REFRESH_FAILURE):
        return exc if isinstance(exc, PredictionError) else PredictionError(message)
    return PredictionError(f"{SOURCE_REFRESH_FAILURE} {message}")


def assert_current_season_csv_is_not_frozen_cache(csv_path: Path, cache_dir: Path) -> None:
    frozen = cache_path(cache_dir)
    if csv_path.resolve() == frozen.resolve():
        raise PredictionError(
            f"{SOURCE_REFRESH_FAILURE} current-season MoneyPuck path is the frozen "
            "development cache; pass a separate current-season file and leave the "
            "frozen cache unchanged"
        )


def _synthetic_game(
    game_id: str,
    hour: int,
    minute: int,
    *,
    home: str,
    away: str,
    home_id: int,
    away_id: int,
    state: str = FUT_GAME_STATE,
    game_type: str = "regular",
    observed_at_utc: datetime = _SYNTHETIC_OBSERVED,
) -> NHLScheduleGame:
    return NHLScheduleGame(
        nhl_game_id=game_id,
        season=CURRENT_SEASON,
        game_type=game_type,
        game_date=OPENING_DAY,
        scheduled_start_time_utc=datetime(
            2026, 9, 29, hour, minute, tzinfo=timezone.utc
        ),
        home_nhl_team_id=home_id,
        away_nhl_team_id=away_id,
        home_team_abbrev=home,
        away_team_abbrev=away,
        game_state=state,
        game_schedule_state="OK",
        observed_at_utc=observed_at_utc,
    )


def synthetic_opening_day_games(
    *,
    observed_at_utc: datetime = _SYNTHETIC_OBSERVED,
    state_overrides: Mapping[str, str] | None = None,
) -> tuple[NHLScheduleGame, ...]:
    """Three synthetic regular-season targets. Not real 2026-27 games."""

    specs = (
        (SYNTHETIC_GAME_IDS[0], 21, 0, "BOS", "BUF", 6, 7),
        (SYNTHETIC_GAME_IDS[1], 23, 0, "CAR", "NJD", 12, 1),
        (SYNTHETIC_GAME_IDS[2], 23, 30, "NJD", "BUF", 1, 7),
    )
    overrides = dict(state_overrides or {})
    games = []
    for game_id, hour, minute, home, away, home_id, away_id in specs:
        games.append(
            _synthetic_game(
                game_id,
                hour,
                minute,
                home=home,
                away=away,
                home_id=home_id,
                away_id=away_id,
                state=overrides.get(game_id, FUT_GAME_STATE),
                observed_at_utc=observed_at_utc,
            )
        )
    return tuple(games)


def synthetic_opening_day_snapshot(
    *,
    observed_at_utc: datetime = _SYNTHETIC_OBSERVED,
    games: Sequence[NHLScheduleGame] | None = None,
) -> NHLScheduleSnapshot:
    selected = tuple(games) if games is not None else synthetic_opening_day_games(
        observed_at_utc=observed_at_utc
    )
    return NHLScheduleSnapshot(
        endpoint_path="synthetic://issue68-opening-day-rehearsal",
        observed_at_utc=observed_at_utc,
        games=selected,
        covered_dates=frozenset(required_coverage_dates(OPENING_DAY)),
    )


def _mapping_keys(payload: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(payload, Mapping):
        keys.update(str(key) for key in payload)
        for value in payload.values():
            keys.update(_mapping_keys(value))
    elif isinstance(payload, list):
        for item in payload:
            keys.update(_mapping_keys(item))
    return keys


def _store_hashes(store: PredictionRecordStore) -> dict[str, str]:
    return {
        str(envelope["record"]["game_id"]): str(envelope["record_sha256"])
        for envelope in store.iter_envelopes()
    }


def _assert_product_surface_is_local() -> None:
    path = Path(__file__).resolve()
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    banned_roots = {"requests", "github", "ghapi", "httpx", "apscheduler", "schedule"}
    for module_name in modules:
        root = module_name.split(".", 1)[0]
        if root in banned_roots:
            raise PredictionError(f"{_PRODUCT_MODULE} imports forbidden client {module_name}")


def build_issue68_opening_day_report(*, store_dir: Path | None = None) -> dict[str, Any]:
    """Synthetic multi-game rehearsal. Does not score a real 2026-27 game."""

    blockers: list[str] = []
    try:
        _assert_product_surface_is_local()
    except PredictionError as exc:
        blockers.append(str(exc))

    artifact = require_committed_execution_artifact()
    if artifact.fitted_artifact_sha256 != EXPECTED_FITTED_ARTIFACT_SHA256:
        blockers.append("committed frozen artifact hash drifted")
    if tuple(artifact.feature_columns) != tuple(FROZEN_FEATURE_COLUMNS):
        blockers.append("frozen 25-column order drifted")
    if len(FROZEN_FEATURE_COLUMNS) != 25:
        blockers.append("frozen feature count is not 25")
    if [item["game_id"] for item in OPENING_DAY_OBSERVED_TARGETS] != list(
        REAL_OPENING_DAY_GAME_IDS
    ):
        blockers.append("recorded opening-day inventory ids drifted")
    recorded_starts = [
        item["scheduled_start_time_utc"] for item in OPENING_DAY_OBSERVED_TARGETS
    ]
    if recorded_starts != sorted(recorded_starts):
        blockers.append("recorded opening-day inventory is not in scheduled-start order")

    workdir = Path(store_dir or Path("data/derived/prediction_records/issue68_proof"))
    workdir.mkdir(parents=True, exist_ok=True)
    games, records, history = historical_dry_run_fixture()
    snapshot = synthetic_opening_day_snapshot()
    rows: list[dict[str, str]] = []
    primary = PredictionRecordStore(workdir / "synthetic_batch.jsonl")
    first = execute_manual_opening_day_batch(
        SYNTHETIC_GAME_IDS,
        snapshot,
        rows,
        frozen_completed_history=games,
        frozen_advanced_records=records,
        frozen_schedule_history=history,
        store=primary,
        observed_at_utc=_SYNTHETIC_OBSERVED,
    )
    second = execute_manual_opening_day_batch(
        SYNTHETIC_GAME_IDS,
        snapshot,
        rows,
        frozen_completed_history=games,
        frozen_advanced_records=records,
        frozen_schedule_history=history,
        store=primary,
        observed_at_utc=_SYNTHETIC_OBSERVED,
    )
    if len(first.results) != 3 or any(item.status != "written" for item in first.results):
        blockers.append("three-target synthetic batch did not write every target")
    if any(item.receipt_sha256 is None for item in first.results):
        blockers.append("three-target synthetic batch did not emit a receipt per write")
    if any(item.status != "idempotent_duplicate" or item.written for item in second.results):
        blockers.append("exact duplicate batch was not an idempotent no-op")
    for left, right in zip(first.results, second.results, strict=True):
        if left.record_sha256 != right.record_sha256 or left.receipt_sha256 != right.receipt_sha256:
            blockers.append("duplicate batch did not reproduce record and receipt hashes")
        try:
            stored = primary.get_by_game_id(left.game_id)
            regenerated = build_pregame_audit_receipt(stored or {})
        except (PredictionError, CurrentSeasonPriorSourceError, FeatureError) as exc:
            blockers.append(f"receipt regeneration failed: {exc}")
            continue
        if regenerated.receipt_sha256 != left.receipt_sha256:
            blockers.append("receipt regeneration was not deterministic")
    if _store_hashes(primary) != {
        item.game_id: item.record_sha256 for item in first.results
    }:
        blockers.append("idempotent duplicate changed the stored records")

    mixed = execute_manual_opening_day_batch(
        (SYNTHETIC_GAME_IDS[0], INVALID_SYNTHETIC_GAME_ID, SYNTHETIC_GAME_IDS[1]),
        snapshot,
        rows,
        frozen_completed_history=games,
        frozen_advanced_records=records,
        frozen_schedule_history=history,
        store=primary,
        observed_at_utc=_SYNTHETIC_OBSERVED,
    )
    if not mixed.partial_failure or mixed.results[1].status != "failed":
        blockers.append("invalid target did not fail visibly")
    if mixed.results[1].error is None or "invalid target" not in mixed.results[1].error:
        blockers.append("invalid target failure was not explicit")
    if mixed.results[0].status != "idempotent_duplicate" or mixed.results[2].status != (
        "idempotent_duplicate"
    ):
        blockers.append("valid targets were not auditable after an invalid sibling failed")
    if SYNTHETIC_GAME_IDS[2] not in _store_hashes(primary):
        blockers.append("a valid target absent from the failing invocation was dropped")
    if _store_hashes(primary) != {
        item.game_id: item.record_sha256 for item in first.results
    }:
        blockers.append("partial failure mutated already-valid immutable records")

    later_observed = datetime(2026, 9, 29, 18, tzinfo=timezone.utc)
    conflict_game_id = "SYNTHETIC-ISSUE68-004"
    conflict_games = (
        *synthetic_opening_day_games(observed_at_utc=later_observed),
        _synthetic_game(
            conflict_game_id,
            23,
            45,
            home="BOS",
            away="CAR",
            home_id=6,
            away_id=12,
            observed_at_utc=later_observed,
        ),
    )
    conflict_snapshot = synthetic_opening_day_snapshot(
        observed_at_utc=later_observed,
        games=conflict_games,
    )
    before_conflict = _store_hashes(primary)
    conflict = execute_manual_opening_day_batch(
        (SYNTHETIC_GAME_IDS[0], conflict_game_id),
        conflict_snapshot,
        rows,
        frozen_completed_history=games,
        frozen_advanced_records=records,
        frozen_schedule_history=history,
        store=primary,
        observed_at_utc=later_observed,
    )
    after_conflict = _store_hashes(primary)
    if conflict.results[0].status != "failed" or (
        conflict.results[0].error is None
        or "Duplicate conflicting" not in conflict.results[0].error
    ):
        blockers.append("conflicting duplicate did not fail visibly")
    if after_conflict.get(SYNTHETIC_GAME_IDS[0]) != before_conflict.get(SYNTHETIC_GAME_IDS[0]):
        blockers.append("conflicting duplicate overwrote the existing record")
    if conflict.results[1].status != "written":
        blockers.append("a new valid target did not remain auditable beside a conflict")

    missing_store = workdir / "missing_source.jsonl"
    try:
        execute_manual_opening_day_batch(
            SYNTHETIC_GAME_IDS,
            None,
            rows,
            frozen_completed_history=games,
            frozen_advanced_records=records,
            frozen_schedule_history=history,
            store=missing_store,
            observed_at_utc=_SYNTHETIC_OBSERVED,
        )
    except PredictionError as exc:
        source_message = str(exc)
    else:
        source_message = ""
        blockers.append("missing source did not fail the invocation")
    if not source_message.startswith(SOURCE_REFRESH_FAILURE):
        blockers.append("source refresh failure was not explicit")
    if missing_store.exists():
        blockers.append("source refresh failure wrote a prediction record")

    stale_store = workdir / "stale_source.jsonl"
    try:
        execute_manual_opening_day_batch(
            SYNTHETIC_GAME_IDS,
            snapshot,
            rows,
            frozen_completed_history=games,
            frozen_advanced_records=records,
            frozen_schedule_history=history,
            store=stale_store,
            observed_at_utc=later_observed,
        )
    except PredictionError as exc:
        stale_message = str(exc)
    else:
        stale_message = ""
        blockers.append("disagreeing source observation did not fail the invocation")
    if not stale_message.startswith(SOURCE_REFRESH_FAILURE) or stale_store.exists():
        blockers.append("stale source refresh was not a visible no-write failure")

    non_fut_store = PredictionRecordStore(workdir / "non_fut.jsonl")
    live_games = synthetic_opening_day_games(
        state_overrides={SYNTHETIC_GAME_IDS[1]: "LIVE"}
    )
    non_fut = execute_manual_opening_day_batch(
        SYNTHETIC_GAME_IDS,
        synthetic_opening_day_snapshot(games=live_games),
        rows,
        frozen_completed_history=games,
        frozen_advanced_records=records,
        frozen_schedule_history=history,
        store=non_fut_store,
        observed_at_utc=_SYNTHETIC_OBSERVED,
    )
    if non_fut.results[1].status != "failed" or (
        non_fut.results[1].error is None or NO_LONGER_FUT not in non_fut.results[1].error
    ):
        blockers.append("target no longer FUT did not fail visibly")
    if [item.status for item in non_fut.results] != ["written", "failed", "written"]:
        blockers.append("non-FUT failure did not preserve the other valid targets")
    if SYNTHETIC_GAME_IDS[1] in _store_hashes(non_fut_store):
        blockers.append("a non-FUT target was written")

    late_observed = datetime(2026, 9, 29, 21, tzinfo=timezone.utc)
    late_store = PredictionRecordStore(workdir / "late_timestamp.jsonl")
    late_games = synthetic_opening_day_games(observed_at_utc=late_observed)
    late = execute_manual_opening_day_batch(
        SYNTHETIC_GAME_IDS,
        synthetic_opening_day_snapshot(
            observed_at_utc=late_observed,
            games=late_games,
        ),
        rows,
        frozen_completed_history=games,
        frozen_advanced_records=records,
        frozen_schedule_history=history,
        store=late_store,
        observed_at_utc=late_observed,
    )
    if late.results[0].status != "failed" or (
        late.results[0].error is None or AT_OR_AFTER_PUCK_DROP not in late.results[0].error
    ):
        blockers.append("timestamp at puck drop did not fail that target")
    if [item.status for item in late.results] != ["failed", "written", "written"]:
        blockers.append("a late target aborted later games that were still before puck drop")

    after_drop_observed = datetime(2026, 9, 29, 21, 5, tzinfo=timezone.utc)
    after_store = PredictionRecordStore(workdir / "after_drop.jsonl")
    after_games = synthetic_opening_day_games(observed_at_utc=after_drop_observed)
    after = execute_manual_opening_day_batch(
        (SYNTHETIC_GAME_IDS[0],),
        synthetic_opening_day_snapshot(
            observed_at_utc=after_drop_observed,
            games=after_games,
        ),
        rows,
        frozen_completed_history=games,
        frozen_advanced_records=records,
        frozen_schedule_history=history,
        store=after_store,
        observed_at_utc=after_drop_observed,
    )
    if after.results[0].status != "failed" or (
        after.results[0].error is None or AT_OR_AFTER_PUCK_DROP not in after.results[0].error
    ):
        blockers.append("timestamp after puck drop did not fail that target")
    if after_store.path.exists():
        blockers.append("a post-drop target wrote a prediction record")

    touched_ids = {
        item.game_id
        for batch in (first, second, mixed, conflict, non_fut, late, after)
        for item in batch.results
    }
    real_emitted = sorted(touched_ids & set(REAL_OPENING_DAY_GAME_IDS))
    if real_emitted:
        blockers.append(f"real 2026-27 game ids were scored: {real_emitted}")
    leaked = sorted(
        _mapping_keys(first.to_audit_dict())
        & set(FORBIDDEN_MARKET_FIELDS + FORBIDDEN_TARGET_OUTCOME_FIELDS)
    )
    if leaked:
        blockers.append(f"batch summary leaked fields: {leaked}")

    outcome = READY if not blockers else NOT_READY
    return {
        "issue": 68,
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
            "feature_count": len(FROZEN_FEATURE_COLUMNS),
            "real_2026_27_protected_prediction_emitted": bool(real_emitted),
            "refit_during_prediction_execution": False,
        },
        "opening_day_inventory": {
            "game_date": OPENING_DAY.isoformat(),
            "observed_at_utc": OPENING_DAY_INVENTORY_OBSERVED_AT_UTC,
            "endpoint": OPENING_DAY_INVENTORY_ENDPOINT,
            "coverage_start": "2026-09-15",
            "coverage_end": "2026-10-05",
            "required_coverage_missing": [],
            "early_2026_27_regular_season_games": [],
            "completed_2026_27_regular_season_games": [],
            "scored": False,
            "targets": [dict(item) for item in OPENING_DAY_OBSERVED_TARGETS],
            "execution_order_rule": "scheduled_start_time_utc, then canonical game id",
        },
        "assessment": {
            "issue55_single_target_safe": True,
            "issue55_single_target_low_error_for_this_slate": False,
            "batch_wrapper_added": True,
            "reason": (
                "Five 2026-09-29 regular-season games require five hand-built target "
                "payloads and five separate receipt commands. Three starts fall on "
                "2026-09-30 UTC while the NHL game date stays 2026-09-29. The batch "
                "resolves explicit game ids from the approved snapshot, checks puck "
                "drop per game, writes the existing append-only store, and emits one "
                "Issue #63 receipt per successful record."
            ),
        },
        "execution_surface": {
            "manual_explicit_only": True,
            "requires_explicit_finite_game_id_list": True,
            "uses_committed_frozen_artifact_only": True,
            "recurring_schedule": False,
            "daemon": False,
            "webhook": False,
            "unattended": False,
            "github_api_in_product_code": False,
            "automatic_posting": False,
            "production_deployed": False,
            "odds_staking_recommendations": False,
        },
        "failure_semantics": {
            "one_target_failing_while_others_valid": (
                "That target is status=failed in the summary, the process exits 1, "
                "and every other target still runs. Already-written immutable records "
                "are not deleted or rolled back."
            ),
            "duplicate_exact_record": (
                "status=idempotent_duplicate, written=false, the stored record is "
                "unchanged, and the deterministic receipt is regenerated."
            ),
            "conflicting_duplicate": (
                "status=failed with Duplicate conflicting, the existing record hash "
                "is unchanged, and other targets in the same invocation still run."
            ),
            "source_refresh_failure": (
                "The invocation raises source refresh failure before any target is "
                "scored. This invocation writes nothing. Records already in the store "
                "stay in place."
            ),
            "target_no_longer_fut": (
                "status=failed, error contains 'is no longer FUT', and that game is "
                "not written. Other FUT targets still run."
            ),
            "observation_at_or_after_puck_drop": (
                "status=failed, error contains 'observation/prediction timestamp is "
                "at or after puck drop', and that game is not written. Later games "
                "whose starts are still after the observation still run."
            ),
            "duplicate_ids_in_one_invocation": (
                "The invocation raises before scoring. No new record is written."
            ),
        },
        "synthetic_rehearsal": {
            "first_batch": first.to_audit_dict(),
            "duplicate_batch_idempotent": second.to_audit_dict(),
            "invalid_target_batch": mixed.to_audit_dict(),
            "conflicting_duplicate_batch": conflict.to_audit_dict(),
            "non_fut_batch": non_fut.to_audit_dict(),
            "at_puck_drop_batch": late.to_audit_dict(),
            "after_puck_drop_batch": after.to_audit_dict(),
            "source_refresh_failure": source_message,
            "stale_source_failure": stale_message,
            "real_opening_day_ids_scored": real_emitted,
        },
        "remaining_dependency": (
            None
            if not blockers
            else {
                "owner": "03 — Prediction, Odds & Recommendation Engine",
                "dependency": blockers[0],
                "unblock_signal": (
                    "Manual batch execution must use the committed frozen artifact, "
                    "fail visibly per target without hiding partial results, and "
                    "rehearse only on synthetic games"
                ),
            }
        ),
        "notes": [
            "Opening-day inventory was read from the approved NHL schedule contract and was not scored.",
            "Rehearsal used synthetic game ids only. No real protected 2026-27 prediction was emitted.",
            "Issue #55 single-target execution is unchanged and remains available.",
            "The operator posts each receipt manually to Issue #65. Product code does not call GitHub.",
            "No odds, staking, recommendation, Production, or unattended schedule is included.",
        ],
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Issue #68 manual opening-day multi-game prediction execution"
    )
    sub = parser.add_subparsers(dest="command")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--proof-store-dir", type=Path, default=None)

    inventory = sub.add_parser(
        "inventory",
        help="Refresh the NHL schedule and list one date. Does not score.",
    )
    inventory.add_argument("--game-date", required=True)
    inventory.add_argument("--observed-at-utc", required=True)
    inventory.add_argument("--write-snapshot", type=Path, required=True)

    execute = sub.add_parser(
        "execute",
        help="Manually score an explicit finite list of target game ids",
    )
    execute.add_argument("--game-ids", required=True)
    execute.add_argument("--nhl-snapshot-json", type=Path, required=True)
    execute.add_argument("--moneypuck-csv", type=Path, required=True)
    execute.add_argument("--cache-dir", type=Path, default=Path("data/cache/moneypuck"))
    execute.add_argument("--store", type=Path, default=DEFAULT_STORE)
    execute.add_argument("--observed-at-utc", required=True)
    execute.add_argument("--result-json", type=Path, default=None)
    return parser


def cmd_inventory(args: argparse.Namespace) -> dict[str, Any]:
    try:
        game_date = date.fromisoformat(str(args.game_date))
        observed = datetime.fromisoformat(str(args.observed_at_utc).replace("Z", "+00:00"))
    except ValueError as exc:
        raise PredictionError(f"inventory timestamp or game date is invalid: {exc}") from exc
    snapshot = refresh_nhl_schedule_snapshot(game_date, observed)
    rows = list_regular_season_targets(snapshot, game_date)
    payload: dict[str, Any] = {
        "issue": 68,
        "command": "inventory",
        "scored": False,
        "prediction_records_written": 0,
        "game_date": game_date.isoformat(),
        "observed_at_utc": snapshot.observed_at_utc.isoformat(),
        "endpoint_path": snapshot.endpoint_path,
        "covered_date_count": len(snapshot.covered_dates),
        "targets": rows,
        "execution_order_rule": "scheduled_start_time_utc, then canonical game id",
        "snapshot_path": str(args.write_snapshot),
    }
    if game_date == OPENING_DAY:
        payload["recorded_inventory_comparison"] = compare_recorded_opening_day_inventory(rows)
    _write_json(args.write_snapshot, nhl_snapshot_to_dict(snapshot))
    return payload


def cmd_execute(args: argparse.Namespace) -> ManualBatchResult:
    game_ids = parse_explicit_game_ids(args.game_ids)
    try:
        observed = datetime.fromisoformat(str(args.observed_at_utc).replace("Z", "+00:00"))
    except ValueError as exc:
        raise PredictionError(f"explicit observation timestamp is invalid: {exc}") from exc
    if observed.tzinfo is None:
        raise PredictionError("observed_at_utc must be timezone-aware")
    try:
        assert_current_season_csv_is_not_frozen_cache(args.moneypuck_csv, args.cache_dir)
        snapshot_payload = json.loads(args.nhl_snapshot_json.read_text(encoding="utf-8"))
        snapshot = nhl_snapshot_from_dict(snapshot_payload)
        rows = load_moneypuck_csv_rows(args.moneypuck_csv)
        frozen_games, frozen_records, frozen_history = load_frozen_development_history(
            args.cache_dir
        )
    except HockeyMoneylineError as exc:
        raise _source_failure(exc) from exc
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise _source_failure(exc) from exc
    return execute_manual_opening_day_batch(
        game_ids,
        snapshot,
        rows,
        frozen_completed_history=frozen_games,
        frozen_advanced_records=frozen_records,
        frozen_schedule_history=frozen_history,
        store=args.store,
        observed_at_utc=observed,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "inventory":
            payload = cmd_inventory(args)
            print(json.dumps(payload, indent=2, sort_keys=True))
            print(f"Wrote {args.write_snapshot}")
            return 0
        if args.command == "execute":
            result = cmd_execute(args)
            payload = result.to_audit_dict()
            rendered = json.dumps(payload, indent=2, sort_keys=True)
            print(rendered)
            if args.result_json is not None:
                _write_json(args.result_json, payload)
                print(f"Wrote {args.result_json}")
            if result.partial_failure:
                print(
                    "PARTIAL FAILURE: "
                    f"{payload['failed_count']} of {len(result.game_ids)} targets failed. "
                    "Already-written immutable records were preserved."
                )
            return batch_process_exit_code(result)
        report = build_issue68_opening_day_report(store_dir=args.proof_store_dir)
        _write_json(args.output, report)
        print(json.dumps({"issue": 68, "outcome": report["outcome"], "blockers": report["blockers"]}, indent=2, sort_keys=True))
        print(f"Wrote {args.output}")
        return 0 if report["outcome"] == READY else 1
    except HockeyMoneylineError as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
