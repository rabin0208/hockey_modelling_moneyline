"""Cycle-2 leakage-safe prior-only schedule-load feature contract.

Schedule-load history and modeling targets are deliberately separate:
- schedule history contains every regular-season MoneyPuck situation=all game
  needed for date/team workload state, including games whose winner target is
  unavailable because the source remains tied after regulation/overtime;
- target games remain the existing winner-labeled canonical NormalizedGame set.

No schedule observed today is used as proof of historical future availability.
"""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from hockey_moneyline.errors import FeatureError, SchemaError
from hockey_moneyline.normalize import canonical_moneypuck_team_code
from hockey_moneyline.provenance import MONEYPUCK_SITUATION, season_from_moneypuck_year
from hockey_moneyline.schema import NormalizedGame, parse_game_date, validate_games

CYCLE2_DEVELOPMENT_SEASONS = (20212022, 20222023, 20232024, 20242025, 20252026)
RESERVED_FUTURE_SEASON = 20262027

SCHEDULE_HISTORY_REQUIRED_COLUMNS = (
    "team",
    "season",
    "gameId",
    "opposingTeam",
    "home_or_away",
    "gameDate",
    "situation",
)

PRIOR_SCHEDULE_LOAD_COLUMNS = (
    "home_calendar_rest_days",
    "away_calendar_rest_days",
    "home_back_to_back",
    "away_back_to_back",
    "home_games_prior_3d",
    "away_games_prior_3d",
    "home_games_prior_7d",
    "away_games_prior_7d",
    "home_three_in_four",
    "away_three_in_four",
    "home_four_in_six",
    "away_four_in_six",
)

SMALLEST_RECOMMENDED_FAMILY = (
    "home_back_to_back",
    "away_back_to_back",
    "home_games_prior_7d",
    "away_games_prior_7d",
)

FUTURE_LOOKING_EXCLUDED = (
    "home_games_next_3d",
    "away_games_next_3d",
    "home_games_next_7d",
    "away_games_next_7d",
)


@dataclass(frozen=True, slots=True)
class ScheduleHistoryGame:
    """Regular-season game identity needed for workload state.

    MoneyPuck rows are completed games. Live scoring may also include known
    NHL calendar games that have not finished yet so fatigue features reflect
    the scheduled slate (back-to-backs, games in the prior 7 days).
    """

    game_id: str
    game_date: date
    season: int
    home_team_id: str
    away_team_id: str


def merge_calendar_schedule_for_fatigue(
    completed_schedule: Sequence[ScheduleHistoryGame],
    calendar_games: Sequence[ScheduleHistoryGame],
    *,
    target_date: date,
    season: int,
) -> list[ScheduleHistoryGame]:
    """Union completed history with known scheduled games before ``target_date``.

    Completed MoneyPuck rows win on id conflicts when identities match.
    Conflicting identities raise FeatureError. Returns an empty list when neither
    source contributes a same-season prior game.
    """

    by_id: dict[str, ScheduleHistoryGame] = {}
    for game in completed_schedule:
        if game.season == season and game.game_date < target_date:
            by_id[game.game_id] = game
    for game in calendar_games:
        if game.season != season or game.game_date >= target_date:
            continue
        existing = by_id.get(game.game_id)
        if existing is None:
            by_id[game.game_id] = game
            continue
        if (
            existing.game_date != game.game_date
            or existing.season != game.season
            or existing.home_team_id != game.home_team_id
            or existing.away_team_id != game.away_team_id
        ):
            raise FeatureError(
                f"Calendar schedule game {game.game_id} disagrees with completed "
                "schedule-history identity"
            )
    if not by_id:
        return []
    return validate_schedule_history(list(by_id.values()))


def _truthy_playoff(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "1.0", "true", "yes", "y"}


def _is_regular_season_row(row: Mapping[str, Any]) -> bool:
    game_id = str(row.get("gameId") or "")
    return (
        len(game_id) >= 6
        and game_id[4:6] == "02"
        and not _truthy_playoff(row.get("playoffGame"))
    )


def _row_season(row: Mapping[str, Any]) -> int:
    try:
        return season_from_moneypuck_year(int(row["season"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise SchemaError(
            f"MoneyPuck schedule-history row has invalid season: {row.get('season')!r}"
        ) from exc


def _history_game_from_home_row(row: Mapping[str, Any]) -> ScheduleHistoryGame:
    season = _row_season(row)
    game_id = str(row.get("gameId") or "").strip()
    if not game_id:
        raise SchemaError("MoneyPuck schedule-history row is missing gameId")
    home = canonical_moneypuck_team_code(row.get("team"), season)
    away = canonical_moneypuck_team_code(row.get("opposingTeam"), season)
    if home == away:
        raise SchemaError(
            f"Schedule-history game {game_id} maps home and away to the same team {home}"
        )
    return ScheduleHistoryGame(
        game_id=game_id,
        game_date=parse_game_date(row.get("gameDate") or ""),
        season=season,
        home_team_id=home,
        away_team_id=away,
    )


def _validate_history_away_row(
    game: ScheduleHistoryGame,
    away_row: Mapping[str, Any],
) -> None:
    season = _row_season(away_row)
    if season != game.season:
        raise SchemaError(
            f"Conflicting HOME/AWAY season for schedule-history game {game.game_id}"
        )
    if parse_game_date(away_row.get("gameDate") or "") != game.game_date:
        raise SchemaError(
            f"Conflicting HOME/AWAY date for schedule-history game {game.game_id}"
        )
    away = canonical_moneypuck_team_code(away_row.get("team"), season)
    opponent = canonical_moneypuck_team_code(away_row.get("opposingTeam"), season)
    if away != game.away_team_id or opponent != game.home_team_id:
        raise SchemaError(
            f"Conflicting HOME/AWAY team identity for schedule-history game {game.game_id}"
        )


def validate_schedule_history(
    games: Sequence[ScheduleHistoryGame],
) -> list[ScheduleHistoryGame]:
    if not games:
        raise SchemaError("No schedule-history games to validate")
    ordered = sorted(games, key=lambda game: (game.game_date, game.game_id))
    seen: set[str] = set()
    for game in ordered:
        if game.game_id in seen:
            raise SchemaError(f"Duplicate schedule-history game_id: {game.game_id}")
        seen.add(game.game_id)
        if game.home_team_id == game.away_team_id:
            raise SchemaError(
                f"Schedule-history game {game.game_id} has identical home/away identity"
            )
    return ordered


def schedule_history_from_moneypuck_rows(
    rows: Iterable[Mapping[str, Any]],
    seasons: Iterable[int] | None = None,
) -> list[ScheduleHistoryGame]:
    """Build complete regular-season schedule chronology without requiring a winner."""

    allowed = set(seasons) if seasons is not None else None
    homes: dict[str, ScheduleHistoryGame] = {}
    aways: dict[str, Mapping[str, Any]] = {}

    for row in rows:
        if str(row.get("situation") or "").strip() != MONEYPUCK_SITUATION:
            continue
        if not _is_regular_season_row(row):
            continue
        season = _row_season(row)
        if allowed is not None and season not in allowed:
            continue
        side = str(row.get("home_or_away") or "").strip().upper()
        game_id = str(row.get("gameId") or "").strip()
        if not game_id:
            raise SchemaError("MoneyPuck schedule-history row is missing gameId")

        if side == "HOME":
            candidate = _history_game_from_home_row(row)
            existing = homes.get(game_id)
            if existing is not None and existing != candidate:
                raise SchemaError(
                    f"Conflicting duplicate HOME schedule-history row for game_id {game_id}"
                )
            homes[game_id] = candidate
        elif side == "AWAY":
            existing = aways.get(game_id)
            if existing is not None and dict(existing) != dict(row):
                raise SchemaError(
                    f"Conflicting duplicate AWAY schedule-history row for game_id {game_id}"
                )
            aways[game_id] = row
        else:
            raise SchemaError(
                f"Schedule-history game {game_id} has invalid home_or_away {side!r}"
            )

    if not homes:
        raise SchemaError("No regular-season MoneyPuck HOME schedule-history rows were usable")

    games: list[ScheduleHistoryGame] = []
    for game_id, game in homes.items():
        away = aways.get(game_id)
        if away is None:
            raise SchemaError(
                f"Schedule-history game {game_id} is missing its AWAY situation=all row"
            )
        _validate_history_away_row(game, away)
        games.append(game)

    orphan_aways = sorted(set(aways) - set(homes))
    if orphan_aways:
        raise SchemaError(
            f"{len(orphan_aways)} AWAY schedule-history rows have no HOME row"
        )
    return validate_schedule_history(games)


def schedule_history_from_moneypuck_csv(
    path: Path,
    seasons: Iterable[int] | None = None,
) -> list[ScheduleHistoryGame]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise SchemaError(f"MoneyPuck file has no header: {path}")
            missing = [
                field
                for field in SCHEDULE_HISTORY_REQUIRED_COLUMNS
                if field not in reader.fieldnames
            ]
            if missing:
                raise SchemaError(
                    f"MoneyPuck schedule-history source is missing columns {missing}"
                )
            return schedule_history_from_moneypuck_rows(reader, seasons=seasons)
    except OSError as exc:
        raise SchemaError(f"Unreadable MoneyPuck file {path}") from exc


def _prior_dates(history: Sequence[date], target: date, days: int) -> list[date]:
    return [day for day in history if 1 <= (target - day).days <= days]


def _team_state(history: Sequence[date], target: date) -> dict[str, float | None]:
    previous = history[-1] if history else None
    rest_days = float((target - previous).days) if previous is not None else None
    prior_3d = len(_prior_dates(history, target, 3))
    prior_5d = len(_prior_dates(history, target, 5))
    prior_7d = len(_prior_dates(history, target, 7))
    return {
        "calendar_rest_days": rest_days,
        "back_to_back": float(previous is not None and (target - previous).days == 1),
        "games_prior_3d": float(prior_3d),
        "games_prior_7d": float(prior_7d),
        "three_in_four": float(prior_3d >= 2),
        "four_in_six": float(prior_5d >= 3),
    }


def _reject_reserved_future_seasons(seasons: Iterable[int]) -> None:
    forbidden = sorted({season for season in seasons if season >= RESERVED_FUTURE_SEASON})
    if forbidden:
        raise FeatureError(
            "Cycle-2 schedule-load development path rejects reserved future season(s): "
            + ", ".join(str(season) for season in forbidden)
        )


def _validate_target_attachment(
    schedule_history: Sequence[ScheduleHistoryGame],
    targets: Sequence[NormalizedGame],
) -> None:
    by_id = {game.game_id: game for game in schedule_history}
    for target in targets:
        history_game = by_id.get(target.game_id)
        if history_game is None:
            raise FeatureError(
                f"Modeling target {target.game_id} is missing from complete schedule history"
            )
        expected = (
            target.game_date,
            target.season,
            target.home_team_id,
            target.away_team_id,
        )
        observed = (
            history_game.game_date,
            history_game.season,
            history_game.home_team_id,
            history_game.away_team_id,
        )
        if observed != expected:
            raise FeatureError(
                f"Modeling target {target.game_id} disagrees with schedule-history identity"
            )


def build_prior_schedule_load_features(
    schedule_history: Sequence[ScheduleHistoryGame],
    target_games: Sequence[NormalizedGame],
) -> list[dict[str, Any]]:
    """Build state from complete schedule history and emit rows only for targets.

    All schedule-history games, including winner-unlabeled tied situation=all
    games, update later workload state. Only labeled NormalizedGame targets
    receive feature rows.
    """

    history_games = validate_schedule_history(schedule_history)
    targets = validate_games(target_games)
    _reject_reserved_future_seasons(
        [game.season for game in history_games] + [game.season for game in targets]
    )
    _validate_target_attachment(history_games, targets)

    targets_by_date: dict[date, list[NormalizedGame]] = defaultdict(list)
    for target in targets:
        targets_by_date[target.game_date].append(target)

    history_by_date: dict[date, list[ScheduleHistoryGame]] = defaultdict(list)
    for game in history_games:
        history_by_date[game.game_date].append(game)

    history_state: dict[tuple[int, str], list[date]] = defaultdict(list)
    rows: list[dict[str, Any]] = []

    # Every target is known to exist in schedule history, so iterating complete
    # schedule dates is sufficient. Emit targets first, then update all games on
    # the date, including target-unlabeled tied games.
    for current_date in sorted(history_by_date):
        day_targets = sorted(
            targets_by_date.get(current_date, []),
            key=lambda game: (game.game_date, game.game_id),
        )
        for game in day_targets:
            home = _team_state(
                history_state[(game.season, game.home_team_id)],
                game.game_date,
            )
            away = _team_state(
                history_state[(game.season, game.away_team_id)],
                game.game_date,
            )
            row: dict[str, Any] = {
                "game_id": game.game_id,
                "game_date": game.game_date.isoformat(),
                "season": game.season,
                "home_team_id": game.home_team_id,
                "away_team_id": game.away_team_id,
            }
            for side, payload in (("home", home), ("away", away)):
                for key, value in payload.items():
                    row[f"{side}_{key}"] = value
            missing = [field for field in PRIOR_SCHEDULE_LOAD_COLUMNS if field not in row]
            if missing:
                raise FeatureError(
                    f"Prior schedule-load row {game.game_id} omitted required fields {missing}"
                )
            rows.append(row)

        for game in sorted(
            history_by_date[current_date],
            key=lambda item: (item.game_date, item.game_id),
        ):
            history_state[(game.season, game.home_team_id)].append(game.game_date)
            history_state[(game.season, game.away_team_id)].append(game.game_date)

    if len(rows) != len(targets):
        raise FeatureError(
            f"Prior schedule-load cardinality mismatch: {len(rows)} rows for {len(targets)} targets"
        )
    return rows


def _coverage(values: Iterable[float | None]) -> dict[str, Any]:
    materialized = list(values)
    non_null = [value for value in materialized if value is not None]
    invalid = sum(
        1
        for value in non_null
        if not isinstance(value, (int, float)) or not math.isfinite(float(value))
    )
    return {
        "eligible_team_side_values": len(materialized),
        "non_null": len(non_null),
        "coverage": (len(non_null) / len(materialized)) if materialized else None,
        "invalid_non_finite_or_non_numeric": invalid,
        "null_or_undefined": len(materialized) - len(non_null),
    }


def inspect_prior_schedule_load_contract(
    schedule_history: Sequence[ScheduleHistoryGame],
    target_games: Sequence[NormalizedGame],
) -> dict[str, Any]:
    """Build corrected Issue #34 source/timing/cardinality/coverage report."""

    history_games = validate_schedule_history(schedule_history)
    targets = validate_games(target_games)
    seasons_present = sorted(
        {game.season for game in history_games} | {game.season for game in targets}
    )
    unexpected = [
        season for season in seasons_present if season not in CYCLE2_DEVELOPMENT_SEASONS
    ]
    if unexpected:
        raise FeatureError(
            "Issue #34 accepts only Cycle-2 development seasons 2021-22 through 2025-26; "
            f"got {unexpected}"
        )
    _validate_target_attachment(history_games, targets)

    rows = build_prior_schedule_load_features(history_games, targets)
    history_by_season = Counter(game.season for game in history_games)
    targets_by_season = Counter(game.season for game in targets)
    rows_by_season = Counter(int(row["season"]) for row in rows)
    history_ids = {game.game_id for game in history_games}
    target_ids = {game.game_id for game in targets}
    history_only_by_season = Counter(
        game.season for game in history_games if game.game_id not in target_ids
    )

    blockers: list[str] = []
    coverage: dict[str, dict[str, Any]] = {}

    for season in CYCLE2_DEVELOPMENT_SEASONS:
        season_rows = [row for row in rows if int(row["season"]) == season]
        season_coverage: dict[str, Any] = {}
        for concept in (
            "calendar_rest_days",
            "back_to_back",
            "games_prior_3d",
            "games_prior_7d",
            "three_in_four",
            "four_in_six",
        ):
            values: list[float | None] = []
            for row in season_rows:
                values.extend([row[f"home_{concept}"], row[f"away_{concept}"]])
            season_coverage[concept] = _coverage(values)
        coverage[str(season)] = season_coverage

        if rows_by_season.get(season, 0) != targets_by_season.get(season, 0):
            blockers.append(
                f"{season} feature rows {rows_by_season.get(season, 0)} "
                f"!= modeling targets {targets_by_season.get(season, 0)}"
            )
        if history_by_season.get(season, 0) < targets_by_season.get(season, 0):
            blockers.append(
                f"{season} schedule history has fewer games than modeling targets"
            )
        for concept in ("back_to_back", "games_prior_7d"):
            stat = season_coverage[concept]
            if stat["coverage"] != 1.0 or stat["invalid_non_finite_or_non_numeric"] != 0:
                blockers.append(
                    f"{season} {concept} is not complete/finite on all target team-side values"
                )

    missing_target_ids = target_ids - history_ids
    if missing_target_ids:
        blockers.append(
            f"{len(missing_target_ids)} modeling targets are absent from schedule history"
        )

    report: dict[str, Any] = {
        "issue": 34,
        "development_seasons": list(CYCLE2_DEVELOPMENT_SEASONS),
        "consumed_cycle2_evidence_season": 20252026,
        "reserved_future_final_evaluation_season": RESERVED_FUTURE_SEASON,
        "source_contract": {
            "source": "MoneyPuck situation=all regular-season team-game chronology",
            "schedule_history_requires_winner_label": False,
            "modeling_target_requires_winner_label": True,
            "join_required": False,
            "raw_schedule_history_inputs": [
                "gameId",
                "gameDate",
                "season",
                "team",
                "opposingTeam",
                "home_or_away",
                "situation",
            ],
            "feature_row_cardinality": "exactly one schedule-load row per labeled modeling target",
            "schedule_history_cardinality": (
                "exactly one game identity per regular-season game, including winner-unlabeled tied games"
            ),
            "duplicate_identity_behavior": "hard SchemaError on conflicting/duplicate game identity",
            "target_attachment": "exact game_id/date/season/home/away identity match",
            "nhl_schedule_join": "not required for the Cycle-2 prior-only calendar-date family",
        },
        "timing_contract": {
            "rule": "history.game_date < target.game_date",
            "calendar_date_semantics": True,
            "exact_time_semantics": False,
            "same_day": (
                "all labeled target rows for a date are snapshotted before any same-day "
                "schedule-history game updates state"
            ),
            "future_mutation": "later completed games cannot change already-emitted earlier rows",
            "season_boundary": "team workload state resets at each season boundary",
            "first_same_season_game": (
                "calendar_rest_days is null; counts and indicators are defined as zero"
            ),
            "home_away": "team-relative state only; no side-specific workload history",
        },
        "candidate_definitions": {
            "calendar_rest_days": (
                "calendar-date difference between target game and the team's previous "
                "completed same-season schedule-history game; null if none exists"
            ),
            "back_to_back": (
                "1 when the previous completed same-season schedule-history game was exactly "
                "one calendar day before target; otherwise 0, including empty history"
            ),
            "games_prior_3d": (
                "completed same-season schedule-history games with date difference 1 through 3 days"
            ),
            "games_prior_7d": (
                "completed same-season schedule-history games with date difference 1 through 7 days"
            ),
            "three_in_four": (
                "1 when at least two completed same-season schedule-history games occurred "
                "1 through 3 days before target"
            ),
            "four_in_six": (
                "1 when at least three completed same-season schedule-history games occurred "
                "1 through 5 days before target"
            ),
        },
        "existing_issue6_9_audit": {
            "safe_backward_looking_concepts": [
                "rest from prior completed game",
                "back-to-back",
                "games prior 3 days",
                "games prior 7 days",
                "3-in-4",
                "4-in-6",
            ],
            "not_reused_exact_time_field": (
                "Issue #6/#9 rest_hours/rest_days are exact scheduled-start-time semantics; "
                "Issue #34 deliberately uses MoneyPuck calendar-date semantics instead"
            ),
            "excluded_future_looking_fields": list(FUTURE_LOOKING_EXCLUDED),
            "reason_future_excluded": (
                "a schedule retrieved today does not prove the same future schedule state "
                "was available before a historical target"
            ),
        },
        "coverage": coverage,
        "cardinality": {
            "schedule_history_games_by_season": {
                str(season): history_by_season.get(season, 0)
                for season in CYCLE2_DEVELOPMENT_SEASONS
            },
            "modeling_target_games_by_season": {
                str(season): targets_by_season.get(season, 0)
                for season in CYCLE2_DEVELOPMENT_SEASONS
            },
            "feature_rows_by_season": {
                str(season): rows_by_season.get(season, 0)
                for season in CYCLE2_DEVELOPMENT_SEASONS
            },
            "schedule_history_only_games_by_season": {
                str(season): history_only_by_season.get(season, 0)
                for season in CYCLE2_DEVELOPMENT_SEASONS
            },
            "unmatched_modeling_targets": len(missing_target_ids),
            "separate_provider_join_duplicates": 0,
        },
        "smallest_recommended_family": {
            "name": "prior_schedule_load_compact",
            "feature_columns": list(SMALLEST_RECOMMENDED_FAMILY),
            "concepts": ["back_to_back", "games_prior_7d"],
            "reason": (
                "captures acute one-day turnaround plus broader seven-day workload with "
                "full empty-history definition, avoiding nullable rest and redundant "
                "3-in-4/4-in-6 threshold expansions in the first ablation"
            ),
        },
        "blockers": blockers,
        "outcome": (
            "READY FOR 02 SCHEDULE-LOAD ABLATION"
            if not blockers
            else "NOT READY — SCHEDULE-LOAD CONTRACT BLOCKED"
        ),
    }
    return report
