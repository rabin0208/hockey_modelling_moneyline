"""Leakage-safe schedule-only pre-game context from NHL schedule snapshots."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from hockey_moneyline.errors import FeatureError
from hockey_moneyline.nhl_api import (
    DEFAULT_CONTEXT_LOOKAHEAD_DAYS,
    DEFAULT_CONTEXT_LOOKBACK_DAYS,
    FINAL_GAME_STATES,
    NHLScheduleGame,
    NHLScheduleSnapshot,
)

SCHEDULE_CONTEXT_COLUMNS = (
    "home_rest_hours",
    "away_rest_hours",
    "home_rest_days",
    "away_rest_days",
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
    "home_games_next_3d",
    "away_games_next_3d",
    "home_games_next_7d",
    "away_games_next_7d",
)


def _date_range(start: date, end: date) -> set[date]:
    if end < start:
        return set()
    return {
        start + timedelta(days=offset)
        for offset in range((end - start).days + 1)
    }


def _team_games(snapshot: NHLScheduleSnapshot, team_id: int) -> list[NHLScheduleGame]:
    return sorted(
        [
            game
            for game in snapshot.games
            if team_id in (game.home_nhl_team_id, game.away_nhl_team_id)
        ],
        key=lambda game: (game.scheduled_start_time_utc, game.nhl_game_id),
    )


def _previous_final_game(
    snapshot: NHLScheduleSnapshot,
    target: NHLScheduleGame,
    team_id: int,
) -> NHLScheduleGame | None:
    prior = [
        game
        for game in _team_games(snapshot, team_id)
        if game.scheduled_start_time_utc < target.scheduled_start_time_utc
        and game.game_state in FINAL_GAME_STATES
    ]
    return prior[-1] if prior else None


def _require_schedule_coverage(
    snapshot: NHLScheduleSnapshot,
    target: NHLScheduleGame,
) -> None:
    required_context_dates = _date_range(
        target.game_date - timedelta(days=DEFAULT_CONTEXT_LOOKBACK_DAYS),
        target.game_date + timedelta(days=DEFAULT_CONTEXT_LOOKAHEAD_DAYS),
    )
    missing = sorted(required_context_dates - set(snapshot.covered_dates))
    if missing:
        raise FeatureError(
            "Schedule coverage incomplete for required ±7-day context; missing dates: "
            + ", ".join(day.isoformat() for day in missing)
        )

    for label, team_id in (
        ("home", target.home_nhl_team_id),
        ("away", target.away_nhl_team_id),
    ):
        previous = _previous_final_game(snapshot, target, team_id)
        if previous is None:
            raise FeatureError(
                f"Schedule coverage does not establish a previous final game for {label} team {team_id}"
            )
        # Rest is exact only when every calendar day between the established
        # previous game and the target has been covered. Otherwise an unseen
        # intermediate game could make the rest value wrong.
        rest_dates = _date_range(previous.game_date, target.game_date)
        missing_rest = sorted(rest_dates - set(snapshot.covered_dates))
        if missing_rest:
            raise FeatureError(
                f"Schedule coverage incomplete for {label} rest window; missing dates: "
                + ", ".join(day.isoformat() for day in missing_rest)
            )


def _side_context(
    snapshot: NHLScheduleSnapshot,
    target: NHLScheduleGame,
    team_id: int,
) -> dict[str, float]:
    games = _team_games(snapshot, team_id)
    prior = [
        game
        for game in games
        if game.scheduled_start_time_utc < target.scheduled_start_time_utc
        and game.game_state in FINAL_GAME_STATES
    ]
    future = [
        game
        for game in games
        if game.scheduled_start_time_utc > target.scheduled_start_time_utc
    ]
    previous = prior[-1]

    rest_hours = (
        target.scheduled_start_time_utc - previous.scheduled_start_time_utc
    ).total_seconds() / 3600.0
    back_to_back = float((target.game_date - previous.game_date).days == 1)

    def prior_count(days: int) -> int:
        lower = target.scheduled_start_time_utc - timedelta(days=days)
        return sum(
            lower <= game.scheduled_start_time_utc < target.scheduled_start_time_utc
            for game in prior
        )

    def future_count(days: int) -> int:
        upper = target.scheduled_start_time_utc + timedelta(days=days)
        return sum(
            target.scheduled_start_time_utc < game.scheduled_start_time_utc <= upper
            for game in future
        )

    prior_3d = prior_count(3)
    prior_5d = prior_count(5)
    return {
        "rest_hours": rest_hours,
        "rest_days": rest_hours / 24.0,
        "back_to_back": back_to_back,
        "games_prior_3d": float(prior_3d),
        "games_prior_7d": float(prior_count(7)),
        "three_in_four": float(prior_3d >= 2),
        "four_in_six": float(prior_5d >= 3),
        "games_next_3d": float(future_count(3)),
        "games_next_7d": float(future_count(7)),
    }


def build_schedule_context(
    snapshot: NHLScheduleSnapshot,
    target_nhl_game_id: str,
) -> dict[str, Any]:
    by_id = {game.nhl_game_id: game for game in snapshot.games}
    target = by_id.get(str(target_nhl_game_id))
    if target is None:
        raise FeatureError(
            f"Target NHL game {target_nhl_game_id!r} not present in schedule snapshot"
        )

    # Future schedule density is valid only from a snapshot actually observed
    # before the target. The snapshot observation time is therefore the
    # availability time for every schedule-only feature in this row.
    if snapshot.observed_at_utc >= target.scheduled_start_time_utc:
        raise FeatureError(
            "Schedule snapshot must be observed before target start for pre-game context"
        )

    _require_schedule_coverage(snapshot, target)

    home = _side_context(snapshot, target, target.home_nhl_team_id)
    away = _side_context(snapshot, target, target.away_nhl_team_id)
    row: dict[str, Any] = {
        "nhl_game_id": target.nhl_game_id,
        "target_start_time_utc": target.scheduled_start_time_utc.isoformat(),
        "available_at_utc": snapshot.observed_at_utc.isoformat(),
    }
    for side, values in (("home", home), ("away", away)):
        for key, value in values.items():
            row[f"{side}_{key}"] = value

    missing = [column for column in SCHEDULE_CONTEXT_COLUMNS if column not in row]
    if missing:
        raise FeatureError(f"Schedule context omitted required columns: {missing}")
    return row
