"""Issue #54 current-season prior-state acquisition and provenance.

Supplies the smallest auditable 2026-27 prior-state package required by the
frozen Issue #52 scorer. It does not change the model, features, artifact,
prospective protocol, or provider strategy.

Approved contracts only:
- MoneyPuck published game-level CSV for completed regular-season
  result / xGoal / schedule-history state;
- NHL Web API ``/v1/schedule/{date}`` for schedule/game identity/timing/status.

Verified-empty same-season state is distinct from missing provider data.
Preseason outcomes never enter result, form, Elo, xGoal, or schedule-load state.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.request import urlopen

from hockey_moneyline.advanced_stats import AdvancedTeamGame, XGOAL_RAW_FIELDS
from hockey_moneyline.errors import CurrentSeasonPriorSourceError, NHLApiError
from hockey_moneyline.nhl_api import (
    FINAL_GAME_STATES,
    NHL_WEB_BASE_URL,
    NHLScheduleGame,
    NHLScheduleSnapshot,
    SCHEDULE_DATE_PATH,
    fetch_schedule,
    merge_schedule_snapshots,
)
from hockey_moneyline.normalize import (
    TARGET_ERA_TEAM_CODES,
    _home_row_to_mapping,
    _int_score,
    _validate_away_row,
    canonical_moneypuck_team_code,
)
from hockey_moneyline.provenance import (
    ACCESS_METHOD,
    MONEYPUCK_SITUATION,
    SOURCE_FILE_URL,
    SOURCE_ID,
    SOURCE_PAGE,
    season_from_moneypuck_year,
)
from hockey_moneyline.prospective_scoring import (
    CURRENT_SEASON_PRIOR_SOURCE_OBSERVED_AT_KEY,
    CURRENT_SEASON_PRIOR_SOURCE_SHA256_KEY,
)
from hockey_moneyline.schedule_load import (
    RESERVED_FUTURE_SEASON,
    SCHEDULE_HISTORY_REQUIRED_COLUMNS,
    ScheduleHistoryGame,
    _history_game_from_home_row,
    _is_regular_season_row,
    _validate_history_away_row,
    validate_schedule_history,
)
from hockey_moneyline.schema import NormalizedGame, game_from_mapping, validate_games
from hockey_moneyline.unscored_target import UnscoredTarget

CURRENT_SEASON = RESERVED_FUTURE_SEASON
CURRENT_SEASON_MONEYPUCK_YEAR = 2026
# Documented 2026-27 regular-season lower bound from live NHL Web API evidence
# (Issue #57 / #59): first scheduled regular-season game 2026020001 CAR vs FLA
# on 2026-09-29. Schedule evidence of any earlier 2026-27 regular-season game
# still fails closed.
CURRENT_SEASON_REGULAR_WINDOW_START = date(2026, 9, 29)
PREWINDOW_LOOKBACK_DAYS = 14
READY = "READY FOR 03 CURRENT-SEASON PRIOR-STATE CONSUMPTION"
NOT_READY = "NOT READY — ONE MATERIAL CURRENT-SEASON SOURCE DEPENDENCY"
SOURCE_CONTRACTS = (
    "moneypuck_published_game_level_all_teams_csv",
    "nhl_web_api_schedule_date",
)
MONEYPUCK_CURRENT_SEASON_COLUMNS = SCHEDULE_HISTORY_REQUIRED_COLUMNS + (
    "goalsFor",
    "goalsAgainst",
    *XGOAL_RAW_FIELDS,
)


def _aware_utc(value: datetime, *, label: str) -> datetime:
    if value.tzinfo is None:
        raise CurrentSeasonPriorSourceError(f"{label} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _date_range(start: date, end: date) -> list[date]:
    if end < start:
        return []
    return [start + timedelta(days=offset) for offset in range((end - start).days + 1)]


def current_season_coverage_start() -> date:
    return CURRENT_SEASON_REGULAR_WINDOW_START - timedelta(days=PREWINDOW_LOOKBACK_DAYS)


def required_coverage_dates(target_game_date: date) -> tuple[date, ...]:
    """Calendar dates whose NHL schedule/status evidence must be present."""

    return tuple(_date_range(current_season_coverage_start(), target_game_date))


def _canonical_team(code: Any, *, season: int, label: str) -> str:
    canonical = canonical_moneypuck_team_code(code, season)
    if canonical not in TARGET_ERA_TEAM_CODES:
        raise CurrentSeasonPriorSourceError(
            f"Unknown {label} team identity {code!r} in current season {season}; "
            "do not fuzzy-match or invent a mapping"
        )
    return canonical


def _row_game_type(row: Mapping[str, Any]) -> str:
    game_id = str(row.get("gameId") or "")
    if len(game_id) >= 6 and game_id[4:6] == "01":
        return "preseason"
    if _is_regular_season_row(row):
        return "regular"
    if len(game_id) >= 6 and game_id[4:6] == "03":
        return "playoff"
    return "other"


def _finite_metric(value: Any, *, field: str, game_id: str, team: str) -> float:
    if value is None or str(value).strip() == "":
        raise CurrentSeasonPriorSourceError(
            f"Current-season {field} is unavailable for game {game_id} team {team}"
        )
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CurrentSeasonPriorSourceError(
            f"Current-season {field} is non-numeric for game {game_id} team {team}: "
            f"{value!r}"
        ) from exc
    if not math.isfinite(number):
        raise CurrentSeasonPriorSourceError(
            f"Current-season {field} is non-finite for game {game_id} team {team}"
        )
    return number


@dataclass(frozen=True, slots=True)
class CurrentSeasonMoneyPuckParse:
    """Parsed MoneyPuck current-season regular-season material. May be empty."""

    winner_labeled: tuple[NormalizedGame, ...]
    schedule_history: tuple[ScheduleHistoryGame, ...]
    xgoal_records: dict[tuple[str, str], AdvancedTeamGame]
    excluded_preseason_game_ids: tuple[str, ...]
    history_only_tied_game_ids: tuple[str, ...]


def parse_current_season_moneypuck_rows(
    rows: Iterable[Mapping[str, Any]] | None,
    *,
    season: int = CURRENT_SEASON,
    shootout_winners: Mapping[str, int] | None = None,
) -> CurrentSeasonMoneyPuckParse:
    """Parse approved MoneyPuck current-season rows. Empty material is allowed.

    Emptiness here is not proof of a valid empty prior state; NHL schedule/status
    evidence must still verify that no eligible completed regular-season games
    existed before the cutoff.

    Tied MoneyPuck situation=all rows become winner-labeled when
    ``shootout_winners`` supplies the NHL final home_win for that game_id.
    """

    if rows is None:
        raise CurrentSeasonPriorSourceError(
            "Current-season MoneyPuck material is missing; refuse to treat an "
            "absent provider payload as a verified empty state"
        )

    so_wins = dict(shootout_winners or {})
    selected_year = str(CURRENT_SEASON_MONEYPUCK_YEAR)
    homes: dict[str, Mapping[str, Any]] = {}
    aways: dict[str, Mapping[str, Any]] = {}
    team_rows: dict[tuple[str, str], Mapping[str, Any]] = {}
    preseason_ids: set[str] = set()

    for row in rows:
        raw_season = str(row.get("season") or "").strip()
        if raw_season != selected_year:
            continue
        if str(row.get("situation") or "").strip() != MONEYPUCK_SITUATION:
            continue
        game_id = str(row.get("gameId") or "").strip()
        if not game_id:
            raise CurrentSeasonPriorSourceError(
                "Current-season MoneyPuck row is missing gameId"
            )
        game_type = _row_game_type(row)
        if game_type == "preseason":
            preseason_ids.add(game_id)
            continue
        if game_type != "regular":
            continue
        row_season = season_from_moneypuck_year(int(raw_season))
        if row_season != season:
            raise CurrentSeasonPriorSourceError(
                f"Current-season MoneyPuck season {row_season} disagrees with {season}"
            )
        side = str(row.get("home_or_away") or "").strip().upper()
        team = _canonical_team(row.get("team"), season=row_season, label="MoneyPuck")
        key = (game_id, team)
        existing_team = team_rows.get(key)
        if existing_team is not None and dict(existing_team) != dict(row):
            raise CurrentSeasonPriorSourceError(
                f"Conflicting MoneyPuck current-season row for {key}"
            )
        team_rows[key] = row
        if side == "HOME":
            existing = homes.get(game_id)
            if existing is not None and dict(existing) != dict(row):
                raise CurrentSeasonPriorSourceError(
                    f"Conflicting MoneyPuck HOME row for game_id {game_id}"
                )
            homes[game_id] = row
        elif side == "AWAY":
            existing = aways.get(game_id)
            if existing is not None and dict(existing) != dict(row):
                raise CurrentSeasonPriorSourceError(
                    f"Conflicting MoneyPuck AWAY row for game_id {game_id}"
                )
            aways[game_id] = row
        else:
            raise CurrentSeasonPriorSourceError(
                f"Current-season MoneyPuck game {game_id} has invalid home_or_away {side!r}"
            )

    winner_labeled: list[NormalizedGame] = []
    schedule_history: list[ScheduleHistoryGame] = []
    xgoal_records: dict[tuple[str, str], AdvancedTeamGame] = {}
    tied_ids: list[str] = []

    orphan_aways = sorted(set(aways) - set(homes))
    if orphan_aways:
        raise CurrentSeasonPriorSourceError(
            "Current-season MoneyPuck AWAY rows have no HOME pair: "
            f"{orphan_aways[:10]}"
        )

    for game_id, home_row in sorted(homes.items(), key=lambda item: item[0]):
        away_row = aways.get(game_id)
        if away_row is None:
            raise CurrentSeasonPriorSourceError(
                f"Current-season MoneyPuck game {game_id} is missing its AWAY row"
            )
        history_game = _history_game_from_home_row(home_row)
        if history_game.season != season:
            raise CurrentSeasonPriorSourceError(
                f"Current-season schedule-history game {game_id} has season "
                f"{history_game.season}, expected {season}"
            )
        _canonical_team(history_game.home_team_id, season=season, label="MoneyPuck home")
        _canonical_team(history_game.away_team_id, season=season, label="MoneyPuck away")
        _validate_history_away_row(history_game, away_row)
        schedule_history.append(history_game)

        home_score = _int_score(home_row.get("goalsFor"), game_id=game_id, field="goalsFor")
        away_score = _int_score(
            home_row.get("goalsAgainst"), game_id=game_id, field="goalsAgainst"
        )
        if home_score == away_score:
            away_for = _int_score(
                away_row.get("goalsFor"), game_id=game_id, field="away goalsFor"
            )
            away_against = _int_score(
                away_row.get("goalsAgainst"), game_id=game_id, field="away goalsAgainst"
            )
            if away_for != away_score or away_against != home_score:
                raise CurrentSeasonPriorSourceError(
                    f"Conflicting HOME/AWAY scores for tied current-season game {game_id}"
                )
            so_home_win = so_wins.get(game_id)
            if so_home_win is None:
                tied_ids.append(game_id)
                continue
            mapped = _home_row_to_mapping(
                home_row,
                SOURCE_ID,
                home_win=int(so_home_win),
                decided_in_shootout=True,
            )
        else:
            mapped = _home_row_to_mapping(home_row, SOURCE_ID)
        _validate_away_row(game_id, mapped, away_row)
        game = game_from_mapping(mapped)
        if game.game_type != "regular":
            raise CurrentSeasonPriorSourceError(
                f"Current-season winner-labeled game {game_id} is not regular season"
            )
        winner_labeled.append(game)
        for team_id, opponent_id, side, source_row in (
            (game.home_team_id, game.away_team_id, "HOME", home_row),
            (game.away_team_id, game.home_team_id, "AWAY", away_row),
        ):
            values = {
                field: _finite_metric(
                    source_row.get(field),
                    field=field,
                    game_id=game_id,
                    team=team_id,
                )
                for field in XGOAL_RAW_FIELDS
            }
            xgoal_records[(game_id, team_id)] = AdvancedTeamGame(
                game_id=game_id,
                game_date=game.game_date,
                season=game.season,
                team_id=team_id,
                opponent_id=opponent_id,
                side=side,
                values=values,
            )

    ordered_history = (
        tuple(validate_schedule_history(schedule_history)) if schedule_history else ()
    )
    ordered_winners = tuple(validate_games(winner_labeled)) if winner_labeled else ()
    return CurrentSeasonMoneyPuckParse(
        winner_labeled=ordered_winners,
        schedule_history=ordered_history,
        xgoal_records=xgoal_records,
        excluded_preseason_game_ids=tuple(sorted(preseason_ids)),
        history_only_tied_game_ids=tuple(tied_ids),
    )


def parse_current_season_moneypuck_csv(
    path: Path,
    *,
    season: int = CURRENT_SEASON,
    shootout_winners: Mapping[str, int] | None = None,
) -> CurrentSeasonMoneyPuckParse:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise CurrentSeasonPriorSourceError(
                    f"Current-season MoneyPuck file has no header: {path}"
                )
            missing = [
                column
                for column in MONEYPUCK_CURRENT_SEASON_COLUMNS
                if column not in reader.fieldnames
            ]
            if missing:
                raise CurrentSeasonPriorSourceError(
                    f"Current-season MoneyPuck file is missing columns {missing}"
                )
            return parse_current_season_moneypuck_rows(
                reader,
                season=season,
                shootout_winners=shootout_winners,
            )
    except OSError as exc:
        raise CurrentSeasonPriorSourceError(
            f"Unreadable current-season MoneyPuck file {path}"
        ) from exc


def assemble_current_season_nhl_coverage(
    target_game_date: date,
    *,
    observed_at_utc: datetime,
    coverage_start: date | None = None,
    timeout: int = 30,
    opener: Callable[..., Any] = urlopen,
) -> NHLScheduleSnapshot:
    """Assemble explicit NHL ``/v1/schedule/{date}`` coverage through the cutoff date."""

    observed = _aware_utc(observed_at_utc, label="observed_at_utc")
    start = coverage_start or current_season_coverage_start()
    end = target_game_date
    if end < start:
        raise CurrentSeasonPriorSourceError(
            "Target cutoff date is before the documented current-season coverage start; "
            "the cutoff cannot be reproduced"
        )
    snapshots: list[NHLScheduleSnapshot] = []
    cursor = start
    while cursor <= end:
        snapshots.append(
            fetch_schedule(
                cursor,
                observed_at_utc=observed,
                timeout=timeout,
                opener=opener,
            )
        )
        assembled = merge_schedule_snapshots(snapshots)
        cursor += timedelta(days=1)
        while cursor <= end and cursor in assembled.covered_dates:
            cursor += timedelta(days=1)
    try:
        return merge_schedule_snapshots(snapshots)
    except NHLApiError as exc:
        raise CurrentSeasonPriorSourceError(str(exc)) from exc


@dataclass(frozen=True, slots=True)
class CurrentSeasonPriorStatePackage:
    """Durable current-season prior-state snapshot for Issue #52 scoring."""

    season: int
    observed_at_utc: datetime
    cutoff_game_id: str
    cutoff_game_date: date
    cutoff_scheduled_start_time_utc: datetime
    source_sha256: str
    source_contracts: tuple[str, ...]
    nhl_endpoint_paths: tuple[str, ...]
    moneypuck_source_id: str
    moneypuck_source_file_url: str
    covered_dates: tuple[date, ...]
    eligible_prior_completed_game_ids: tuple[str, ...]
    winner_labeled_history: tuple[NormalizedGame, ...]
    schedule_history: tuple[ScheduleHistoryGame, ...]
    xgoal_records: dict[tuple[str, str], AdvancedTeamGame]
    empty_state_verified: bool
    empty_state_proof: dict[str, Any] | None
    excluded_preseason_game_ids: tuple[str, ...]
    history_only_tied_game_ids: tuple[str, ...]

    def issue52_source_versions(self) -> dict[str, Any]:
        """Keys already required by Issue #52 protected scoring, plus audit fields."""

        return {
            CURRENT_SEASON_PRIOR_SOURCE_SHA256_KEY: self.source_sha256,
            CURRENT_SEASON_PRIOR_SOURCE_OBSERVED_AT_KEY: (
                self.observed_at_utc.isoformat()
            ),
            "current_season_prior_empty_verified": self.empty_state_verified,
            "current_season_prior_eligible_completed_count": len(
                self.eligible_prior_completed_game_ids
            ),
            "current_season_prior_season": self.season,
            "current_season_prior_cutoff_game_id": self.cutoff_game_id,
            "current_season_prior_source_contracts": list(self.source_contracts),
        }

    def as_issue52_scoring_kwargs(self) -> dict[str, Any]:
        """Verified packages, including verified-empty, are available source state."""

        return {
            "current_season_prior_source_available": True,
            "source_versions": self.issue52_source_versions(),
        }

    def to_audit_dict(self) -> dict[str, Any]:
        proof = self.empty_state_proof
        return {
            "season": self.season,
            "observed_at_utc": self.observed_at_utc.isoformat(),
            "cutoff_game_id": self.cutoff_game_id,
            "cutoff_game_date": self.cutoff_game_date.isoformat(),
            "cutoff_scheduled_start_time_utc": (
                self.cutoff_scheduled_start_time_utc.isoformat()
            ),
            "current_season_prior_source_sha256": self.source_sha256,
            "source_contracts": list(self.source_contracts),
            "nhl_endpoint_paths": list(self.nhl_endpoint_paths),
            "nhl_base_url": NHL_WEB_BASE_URL,
            "nhl_schedule_date_path": SCHEDULE_DATE_PATH,
            "moneypuck_source_id": self.moneypuck_source_id,
            "moneypuck_source_page": SOURCE_PAGE,
            "moneypuck_source_file_url": self.moneypuck_source_file_url,
            "moneypuck_access_method": ACCESS_METHOD,
            "covered_dates": [day.isoformat() for day in self.covered_dates],
            "eligible_prior_completed_game_ids": list(
                self.eligible_prior_completed_game_ids
            ),
            "eligible_prior_completed_count": len(
                self.eligible_prior_completed_game_ids
            ),
            "winner_labeled_count": len(self.winner_labeled_history),
            "schedule_history_count": len(self.schedule_history),
            "xgoal_record_count": len(self.xgoal_records),
            "history_only_tied_game_ids": list(self.history_only_tied_game_ids),
            "excluded_preseason_game_ids": list(self.excluded_preseason_game_ids),
            "empty_state_verified": self.empty_state_verified,
            "empty_state_proof": proof,
            "timing_rule": "history.game_date < target.game_date",
            "raw_provider_file_committed": False,
        }


def _canonical_source_payload(package_fields: Mapping[str, Any]) -> str:
    return json.dumps(package_fields, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _source_sha256(package_fields: Mapping[str, Any]) -> str:
    payload = _canonical_source_payload(package_fields)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _nhl_identity(game: NHLScheduleGame) -> dict[str, Any]:
    return {
        "game_id": game.nhl_game_id,
        "game_date": game.game_date.isoformat(),
        "season": game.season,
        "game_type": game.game_type,
        "home_team_id": game.home_team_abbrev,
        "away_team_id": game.away_team_abbrev,
        "scheduled_start_time_utc": game.scheduled_start_time_utc.isoformat(),
        "game_state": game.game_state,
        "game_schedule_state": game.game_schedule_state,
    }


def _target_from_snapshot(
    snapshot: NHLScheduleSnapshot,
    target: UnscoredTarget,
) -> NHLScheduleGame:
    matches = [game for game in snapshot.games if game.nhl_game_id == target.game_id]
    if len(matches) != 1:
        raise CurrentSeasonPriorSourceError(
            "Target cutoff cannot be reproduced: NHL schedule/status snapshot does "
            f"not uniquely contain target {target.game_id!r}"
        )
    nhl_target = matches[0]
    expected = (
        target.game_date,
        target.season,
        target.game_type,
        target.home_team_id,
        target.away_team_id,
    )
    observed = (
        nhl_target.game_date,
        nhl_target.season,
        nhl_target.game_type,
        nhl_target.home_team_abbrev,
        nhl_target.away_team_abbrev,
    )
    if expected != observed:
        raise CurrentSeasonPriorSourceError(
            "Source identities conflict for target cutoff "
            f"{target.game_id}: target {expected} != NHL {observed}"
        )
    if nhl_target.season != CURRENT_SEASON or nhl_target.game_type != "regular":
        raise CurrentSeasonPriorSourceError(
            "Target cutoff must be a 2026-27 regular-season NHL game"
        )
    if target.scheduled_start_time_utc is None:
        raise CurrentSeasonPriorSourceError(
            "Target cutoff cannot be reproduced without scheduled_start_time_utc"
        )
    target_start = _aware_utc(
        target.scheduled_start_time_utc, label="scheduled_start_time_utc"
    )
    if nhl_target.scheduled_start_time_utc != target_start:
        raise CurrentSeasonPriorSourceError(
            "Source identities conflict: target scheduled start "
            f"{target_start.isoformat()} != NHL {nhl_target.scheduled_start_time_utc.isoformat()}"
        )
    return nhl_target


def _content_fields(
    *,
    cutoff: UnscoredTarget,
    nhl_eligible: Sequence[NHLScheduleGame],
    winner_labeled: Sequence[NormalizedGame],
    schedule_history: Sequence[ScheduleHistoryGame],
    xgoal_records: Mapping[tuple[str, str], AdvancedTeamGame],
    empty_state_verified: bool,
    covered_dates: Sequence[date],
    nhl_endpoint_paths: Sequence[str],
) -> dict[str, Any]:
    xgoal_rows = [
        {
            "game_id": record.game_id,
            "team_id": record.team_id,
            "xGoalsFor": record.values["xGoalsFor"],
            "xGoalsAgainst": record.values["xGoalsAgainst"],
        }
        for _, record in sorted(xgoal_records.items(), key=lambda item: (item[0][0], item[0][1]))
    ]
    return {
        "season": CURRENT_SEASON,
        "cutoff": {
            "game_id": cutoff.game_id,
            "game_date": cutoff.game_date.isoformat(),
            "scheduled_start_time_utc": cutoff.scheduled_start_time_utc.isoformat()
            if cutoff.scheduled_start_time_utc is not None
            else None,
            "home_team_id": cutoff.home_team_id,
            "away_team_id": cutoff.away_team_id,
        },
        "source_contracts": list(SOURCE_CONTRACTS),
        "moneypuck_source_id": SOURCE_ID,
        "moneypuck_source_file_url": SOURCE_FILE_URL,
        "nhl_base_url": NHL_WEB_BASE_URL,
        "nhl_endpoint_paths": list(nhl_endpoint_paths),
        "covered_dates": [day.isoformat() for day in covered_dates],
        "nhl_eligible_completed": [_nhl_identity(game) for game in nhl_eligible],
        "winner_labeled": [
            {
                "game_id": game.game_id,
                "game_date": game.game_date.isoformat(),
                "home_team_id": game.home_team_id,
                "away_team_id": game.away_team_id,
                "home_score": game.home_score,
                "away_score": game.away_score,
            }
            for game in winner_labeled
        ],
        "schedule_history": [
            {
                "game_id": game.game_id,
                "game_date": game.game_date.isoformat(),
                "home_team_id": game.home_team_id,
                "away_team_id": game.away_team_id,
            }
            for game in schedule_history
        ],
        "xgoal": xgoal_rows,
        "empty_state_verified": empty_state_verified,
        "timing_rule": "history.game_date < target.game_date",
    }


def build_current_season_prior_state(
    target: UnscoredTarget,
    nhl_snapshot: NHLScheduleSnapshot | None,
    moneypuck_rows: Iterable[Mapping[str, Any]] | None,
    *,
    observed_at_utc: datetime,
    moneypuck_retrieved_at_utc: datetime | None = None,
) -> CurrentSeasonPriorStatePackage:
    """Build the Issue #54 package or fail closed. Never imputes or substitutes."""

    observed = _aware_utc(observed_at_utc, label="observed_at_utc")
    retrieved = _aware_utc(
        moneypuck_retrieved_at_utc or observed,
        label="moneypuck_retrieved_at_utc",
    )
    if target.season != CURRENT_SEASON or target.game_type != "regular":
        raise CurrentSeasonPriorSourceError(
            "Current-season prior-state package requires a 2026-27 regular-season target"
        )
    if target.observed_at_utc is not None:
        target_observed = _aware_utc(target.observed_at_utc, label="target.observed_at_utc")
        if target_observed != observed:
            raise CurrentSeasonPriorSourceError(
                "Source identities conflict: package observation timestamp disagrees "
                "with the target observation timestamp"
            )
    if nhl_snapshot is None:
        raise CurrentSeasonPriorSourceError(
            "NHL schedule/status snapshot is missing; refuse to treat absent schedule "
            "evidence as a verified empty current-season state"
        )
    nhl_observed = _aware_utc(nhl_snapshot.observed_at_utc, label="nhl_snapshot.observed_at_utc")
    if nhl_observed != observed:
        raise CurrentSeasonPriorSourceError(
            "Provider material is stale relative to the target observation: NHL "
            f"snapshot observed_at {nhl_observed.isoformat()} != {observed.isoformat()}"
        )
    if retrieved != observed:
        raise CurrentSeasonPriorSourceError(
            "Provider material is stale relative to the target observation: MoneyPuck "
            f"retrieved_at {retrieved.isoformat()} != {observed.isoformat()}"
        )

    required_dates = required_coverage_dates(target.game_date)
    missing_dates = [day for day in required_dates if day not in nhl_snapshot.covered_dates]
    if missing_dates:
        raise CurrentSeasonPriorSourceError(
            "Target cutoff cannot be reproduced: NHL schedule/status coverage is "
            f"incomplete for dates {missing_dates[0].isoformat()}"
            + (f"... ({len(missing_dates)} missing)" if len(missing_dates) > 1 else "")
        )

    nhl_target = _target_from_snapshot(nhl_snapshot, target)
    if observed >= nhl_target.scheduled_start_time_utc:
        raise CurrentSeasonPriorSourceError(
            "Current-season prior-state observation must occur before target puck drop"
        )

    early_regular = [
        game
        for game in nhl_snapshot.games
        if game.season == CURRENT_SEASON
        and game.game_type == "regular"
        and game.game_date < CURRENT_SEASON_REGULAR_WINDOW_START
    ]
    if early_regular:
        raise CurrentSeasonPriorSourceError(
            "NHL schedule/status shows 2026-27 regular-season games before the "
            "documented window start; return this calendar-bound conflict to 00 "
            f"({early_regular[0].nhl_game_id} on {early_regular[0].game_date.isoformat()})"
        )

    excluded_preseason = tuple(
        sorted(
            game.nhl_game_id
            for game in nhl_snapshot.games
            if game.game_type == "preseason" or (game.season == CURRENT_SEASON and game.game_type != "regular")
        )
    )
    nhl_preseason_ids = tuple(
        sorted(
            game.nhl_game_id
            for game in nhl_snapshot.games
            if game.game_type == "preseason"
        )
    )

    nhl_eligible = tuple(
        sorted(
            (
                game
                for game in nhl_snapshot.games
                if game.season == CURRENT_SEASON
                and game.game_type == "regular"
                and game.game_state in FINAL_GAME_STATES
                and game.game_date < target.game_date
            ),
            key=lambda game: (game.game_date, game.nhl_game_id),
        )
    )
    nhl_eligible_ids = {game.nhl_game_id for game in nhl_eligible}
    parsed = parse_current_season_moneypuck_rows(moneypuck_rows, season=CURRENT_SEASON)

    moneypuck_by_id = {game.game_id: game for game in parsed.schedule_history}
    winners_by_id = {game.game_id: game for game in parsed.winner_labeled}
    if target.game_id in moneypuck_by_id or target.game_id in winners_by_id:
        raise CurrentSeasonPriorSourceError(
            "Source identities conflict: MoneyPuck current-season material includes "
            f"the unscored target {target.game_id} as completed state"
        )

    # Same-day / future MoneyPuck rows are ignored for this cutoff, not used as
    # prior state. Completed NHL games missing from MoneyPuck fail closed.
    extra_on_covered_dates = []
    for history_game in parsed.schedule_history:
        if history_game.game_date >= target.game_date:
            continue
        nhl_game = next(
            (game for game in nhl_snapshot.games if game.nhl_game_id == history_game.game_id),
            None,
        )
        if nhl_game is None:
            extra_on_covered_dates.append(history_game.game_id)
            continue
        if nhl_game.game_type != "regular" or nhl_game.season != CURRENT_SEASON:
            raise CurrentSeasonPriorSourceError(
                f"Source identities conflict for {history_game.game_id}: MoneyPuck "
                "regular-season row disagrees with NHL game type/season"
            )
        if nhl_game.game_state not in FINAL_GAME_STATES:
            raise CurrentSeasonPriorSourceError(
                f"Source identities conflict for {history_game.game_id}: MoneyPuck has "
                f"a completed current-season row while NHL state is {nhl_game.game_state!r}"
            )
    if extra_on_covered_dates:
        raise CurrentSeasonPriorSourceError(
            "Source identities conflict: MoneyPuck current-season regular-season games "
            f"are absent from NHL schedule/status evidence: {extra_on_covered_dates[:10]}"
        )

    prior_winners: list[NormalizedGame] = []
    prior_history: list[ScheduleHistoryGame] = []
    prior_xgoal: dict[tuple[str, str], AdvancedTeamGame] = {}
    prior_tied: list[str] = []

    for nhl_game in nhl_eligible:
        history_game = moneypuck_by_id.get(nhl_game.nhl_game_id)
        if history_game is None:
            raise CurrentSeasonPriorSourceError(
                "Schedule/status evidence implies prior completed regular-season "
                f"game {nhl_game.nhl_game_id} on {nhl_game.game_date.isoformat()} but "
                "required MoneyPuck completed-state records are missing"
            )
        expected_identity = (
            nhl_game.game_date,
            nhl_game.home_team_abbrev,
            nhl_game.away_team_abbrev,
            nhl_game.season,
        )
        observed_identity = (
            history_game.game_date,
            history_game.home_team_id,
            history_game.away_team_id,
            history_game.season,
        )
        if expected_identity != observed_identity:
            raise CurrentSeasonPriorSourceError(
                f"Source identities conflict for {nhl_game.nhl_game_id}: "
                f"NHL {expected_identity} != MoneyPuck {observed_identity}"
            )
        if nhl_game.scheduled_start_time_utc > retrieved:
            raise CurrentSeasonPriorSourceError(
                "Provider material is stale relative to the target observation: "
                f"MoneyPuck retrieved_at {retrieved.isoformat()} is before NHL "
                f"completed game {nhl_game.nhl_game_id} start "
                f"{nhl_game.scheduled_start_time_utc.isoformat()}"
            )
        prior_history.append(history_game)
        winner = winners_by_id.get(nhl_game.nhl_game_id)
        if winner is None:
            if nhl_game.nhl_game_id not in parsed.history_only_tied_game_ids:
                raise CurrentSeasonPriorSourceError(
                    "Required winner-labeled completed result is missing for NHL "
                    f"final game {nhl_game.nhl_game_id} and the MoneyPuck row is "
                    "not a documented tied/unlabeled schedule-history game"
                )
            prior_tied.append(nhl_game.nhl_game_id)
            continue
        for team_id in (winner.home_team_id, winner.away_team_id):
            record = parsed.xgoal_records.get((winner.game_id, team_id))
            if record is None:
                raise CurrentSeasonPriorSourceError(
                    "Prior xGoal data required by the frozen state is unavailable "
                    f"for game {winner.game_id} team {team_id}"
                )
            prior_xgoal[(winner.game_id, team_id)] = record
        prior_winners.append(winner)

    empty_state_verified = not nhl_eligible
    if empty_state_verified:
        leftover_prior = [
            game.game_id
            for game in parsed.schedule_history
            if game.game_date < target.game_date
        ]
        if leftover_prior:
            raise CurrentSeasonPriorSourceError(
                "Source identities conflict: NHL schedule/status shows no eligible "
                "prior completed 2026-27 regular-season games but MoneyPuck still "
                f"has current-season rows before cutoff: {leftover_prior[:10]}"
            )
        empty_state_proof: dict[str, Any] | None = {
            "kind": "verified_empty_current_season_prior_state",
            "eligible_prior_completed_regular_season_game_count": 0,
            "required_coverage_start": required_dates[0].isoformat() if required_dates else None,
            "required_coverage_end": target.game_date.isoformat(),
            "covered_date_count": len(required_dates),
            "nhl_regular_season_final_games_before_cutoff": [],
            "nhl_preseason_games_excluded": list(nhl_preseason_ids),
            "moneypuck_current_season_regular_rows_before_cutoff": 0,
            "note": (
                "NHL schedule/status coverage of every required date before the "
                "target cutoff shows no completed 2026-27 regular-season games, "
                "and approved MoneyPuck current-season regular-season material "
                "before the cutoff is also empty. This is not inferred from a "
                "missing provider payload."
            ),
        }
        prior_winners = []
        prior_history = []
        prior_xgoal = {}
        prior_tied = []
    else:
        empty_state_proof = None

    ordered_winners = tuple(validate_games(prior_winners)) if prior_winners else ()
    ordered_history = (
        tuple(validate_schedule_history(prior_history)) if prior_history else ()
    )
    # Include known NHL calendar games that MoneyPuck has not completed yet so
    # B2B / prior-7d fatigue features follow the schedule for live scoring.
    if not empty_state_verified:
        known_ids = {game.game_id for game in ordered_history}
        calendar_extra: list[ScheduleHistoryGame] = []
        for nhl_game in nhl_snapshot.games:
            if (
                nhl_game.season != CURRENT_SEASON
                or nhl_game.game_type != "regular"
                or nhl_game.game_date >= target.game_date
                or nhl_game.nhl_game_id in known_ids
            ):
                continue
            calendar_extra.append(
                ScheduleHistoryGame(
                    game_id=nhl_game.nhl_game_id,
                    game_date=nhl_game.game_date,
                    season=nhl_game.season,
                    home_team_id=_canonical_team(
                        nhl_game.home_team_abbrev,
                        season=nhl_game.season,
                        label="home",
                    ),
                    away_team_id=_canonical_team(
                        nhl_game.away_team_abbrev,
                        season=nhl_game.season,
                        label="away",
                    ),
                )
            )
        if calendar_extra:
            ordered_history = tuple(
                validate_schedule_history(list(ordered_history) + calendar_extra)
            )
    endpoint_paths = (nhl_snapshot.endpoint_path,)
    covered = tuple(sorted(nhl_snapshot.covered_dates))
    content = _content_fields(
        cutoff=target,
        nhl_eligible=nhl_eligible,
        winner_labeled=ordered_winners,
        schedule_history=ordered_history,
        xgoal_records=prior_xgoal,
        empty_state_verified=empty_state_verified,
        covered_dates=covered,
        nhl_endpoint_paths=endpoint_paths,
    )
    package = CurrentSeasonPriorStatePackage(
        season=CURRENT_SEASON,
        observed_at_utc=observed,
        cutoff_game_id=target.game_id,
        cutoff_game_date=target.game_date,
        cutoff_scheduled_start_time_utc=nhl_target.scheduled_start_time_utc,
        source_sha256=_source_sha256(content),
        source_contracts=SOURCE_CONTRACTS,
        nhl_endpoint_paths=endpoint_paths,
        moneypuck_source_id=SOURCE_ID,
        moneypuck_source_file_url=SOURCE_FILE_URL,
        covered_dates=covered,
        eligible_prior_completed_game_ids=tuple(game.nhl_game_id for game in nhl_eligible),
        winner_labeled_history=ordered_winners,
        schedule_history=ordered_history,
        xgoal_records=prior_xgoal,
        empty_state_verified=empty_state_verified,
        empty_state_proof=empty_state_proof,
        excluded_preseason_game_ids=tuple(
            sorted(set(parsed.excluded_preseason_game_ids) | set(excluded_preseason))
        ),
        history_only_tied_game_ids=tuple(prior_tied),
    )
    recomputed = _source_sha256(
        _content_fields(
            cutoff=target,
            nhl_eligible=nhl_eligible,
            winner_labeled=package.winner_labeled_history,
            schedule_history=package.schedule_history,
            xgoal_records=package.xgoal_records,
            empty_state_verified=package.empty_state_verified,
            covered_dates=package.covered_dates,
            nhl_endpoint_paths=package.nhl_endpoint_paths,
        )
    )
    if recomputed != package.source_sha256:
        raise CurrentSeasonPriorSourceError(
            "Source identities conflict: current-season prior-state SHA-256 is not "
            "reproducible from the packaged content"
        )
    return package
