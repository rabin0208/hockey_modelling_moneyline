"""Issue #45 unscored-target frozen feature parity primitive.

This module does not change frozen training-time feature semantics. It provides
one target representation and one prior-history replay path that can construct
the Issue #40 frozen 25-column vector without target outcomes or a target
advanced-stat record.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Mapping, Sequence

from hockey_moneyline.advanced_stats import AdvancedTeamGame
from hockey_moneyline.elo import ELO_FEATURE_COLUMN, ELO_INITIAL, updated_ratings
from hockey_moneyline.errors import FeatureError
from hockey_moneyline.features import TeamState
from hockey_moneyline.nhl_api import NHLScheduleGame
from hockey_moneyline.normalize import canonical_moneypuck_team_code
from hockey_moneyline.provenance import MIN_PRIOR_GAMES
from hockey_moneyline.schedule_load import (
    ScheduleHistoryGame,
    merge_calendar_schedule_for_fatigue,
    validate_schedule_history,
)
from hockey_moneyline.schedule_load_ablation import (
    PLUS_XGOAL_PLUS_PRIOR_SCHEDULE_LOAD_COMPACT_FEATURE_COLUMNS,
    assert_prior_schedule_load_compact_contract,
)
from hockey_moneyline.schema import FEATURE_COLUMNS, NormalizedGame, validate_games

FROZEN_ISSUE40_FEATURE_COLUMNS = (
    PLUS_XGOAL_PLUS_PRIOR_SCHEDULE_LOAD_COMPACT_FEATURE_COLUMNS
)


@dataclass(frozen=True, slots=True)
class UnscoredTarget:
    """Pregame target identity with no score, home_win, or advanced-stat fields."""

    game_id: str
    game_date: date
    season: int
    game_type: str
    home_team_id: str
    away_team_id: str
    scheduled_start_time_utc: datetime | None = None
    observed_at_utc: datetime | None = None
    nhl_game_id: str | None = None
    home_nhl_team_id: int | None = None
    away_nhl_team_id: int | None = None
    source: str = "historical_replay"

    def __post_init__(self) -> None:
        if not self.game_id:
            raise FeatureError("Unscored target requires game identity")
        if self.game_type != "regular":
            raise FeatureError(
                f"Frozen prospective target must be regular season, got {self.game_type!r}"
            )
        if not self.home_team_id or not self.away_team_id:
            raise FeatureError("Unscored target requires home/away team identity")
        if self.home_team_id == self.away_team_id:
            raise FeatureError("Unscored target home and away teams must differ")
        if (self.scheduled_start_time_utc is None) != (self.observed_at_utc is None):
            raise FeatureError(
                "Target timing requires both scheduled_start_time_utc and observed_at_utc"
            )
        if self.scheduled_start_time_utc is not None:
            if self.scheduled_start_time_utc.tzinfo is None:
                raise FeatureError("Target scheduled start must be timezone-aware")
            if self.observed_at_utc is None or self.observed_at_utc.tzinfo is None:
                raise FeatureError("Target observation time must be timezone-aware")
            if self.observed_at_utc >= self.scheduled_start_time_utc:
                raise FeatureError(
                    "Target observation must occur before scheduled target start"
                )

    def to_metadata(self) -> dict[str, Any]:
        """Return auditable target metadata; intentionally contains no outcomes."""

        return {
            "game_id": self.game_id,
            "game_date": self.game_date.isoformat(),
            "season": self.season,
            "game_type": self.game_type,
            "home_team_id": self.home_team_id,
            "away_team_id": self.away_team_id,
            "scheduled_start_time_utc": (
                self.scheduled_start_time_utc.isoformat()
                if self.scheduled_start_time_utc is not None
                else None
            ),
            "observed_at_utc": (
                self.observed_at_utc.isoformat()
                if self.observed_at_utc is not None
                else None
            ),
            "nhl_game_id": self.nhl_game_id,
            "home_nhl_team_id": self.home_nhl_team_id,
            "away_nhl_team_id": self.away_nhl_team_id,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class FrozenFeatureSnapshot:
    target: UnscoredTarget
    feature_columns: tuple[str, ...]
    feature_values: tuple[float, ...]

    def as_feature_row(self) -> dict[str, float]:
        return dict(zip(self.feature_columns, self.feature_values, strict=True))

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target.to_metadata(),
            "feature_columns": list(self.feature_columns),
            "feature_values": list(self.feature_values),
        }


def unscored_target_from_nhl_schedule(game: NHLScheduleGame) -> UnscoredTarget:
    """Adapt an observed pregame NHL schedule row to canonical target identity."""

    if game.observed_at_utc >= game.scheduled_start_time_utc:
        raise FeatureError(
            "NHL target schedule observation must precede scheduled target start"
        )
    home = canonical_moneypuck_team_code(game.home_team_abbrev, game.season)
    away = canonical_moneypuck_team_code(game.away_team_abbrev, game.season)
    return UnscoredTarget(
        game_id=game.nhl_game_id,
        game_date=game.game_date,
        season=game.season,
        game_type=game.game_type,
        home_team_id=home,
        away_team_id=away,
        scheduled_start_time_utc=game.scheduled_start_time_utc,
        observed_at_utc=game.observed_at_utc,
        nhl_game_id=game.nhl_game_id,
        home_nhl_team_id=game.home_nhl_team_id,
        away_nhl_team_id=game.away_nhl_team_id,
        source="nhl_web_api_schedule",
    )


def unscored_target_from_historical_game(game: NormalizedGame) -> UnscoredTarget:
    """Copy historical target identity only for replay; outcome fields are unread."""

    return UnscoredTarget(
        game_id=game.game_id,
        game_date=game.game_date,
        season=game.season,
        game_type=game.game_type,
        home_team_id=game.home_team_id,
        away_team_id=game.away_team_id,
        source="historical_replay_identity_only",
    )


def _validate_completed_prior_history(
    target: UnscoredTarget,
    completed_history: Sequence[NormalizedGame],
) -> list[NormalizedGame]:
    if not completed_history:
        raise FeatureError("Frozen snapshot requires completed prior result history")
    ordered = validate_games(completed_history)
    non_prior = [
        game.game_id for game in ordered if game.game_date >= target.game_date
    ]
    if non_prior:
        raise FeatureError(
            "Completed result history must be cut strictly before target game_date; "
            f"non-prior game ids={non_prior[:10]}"
        )
    return ordered


def _base_and_elo_snapshot(
    target: UnscoredTarget,
    completed_history: Sequence[NormalizedGame],
) -> dict[str, float]:
    states: dict[str, TeamState] = defaultdict(TeamState)
    ratings: dict[str, float] = defaultdict(lambda: ELO_INITIAL)

    for game in completed_history:
        states[game.home_team_id].update(
            won=bool(game.home_win),
            goals_for=game.home_score,
            goals_against=game.away_score,
            is_home=True,
            season=game.season,
        )
        states[game.away_team_id].update(
            won=not bool(game.home_win),
            goals_for=game.away_score,
            goals_against=game.home_score,
            is_home=False,
            season=game.season,
        )
        new_home, new_away = updated_ratings(
            ratings[game.home_team_id],
            ratings[game.away_team_id],
            bool(game.home_win),
            decided_in_shootout=bool(game.decided_in_shootout),
        )
        ratings[game.home_team_id] = new_home
        ratings[game.away_team_id] = new_away

    home = states[target.home_team_id].snapshot(target.season)
    away = states[target.away_team_id].snapshot(target.season)
    row = {
        "home_prior_games": float(home["prior_games"]),
        "away_prior_games": float(away["prior_games"]),
        "home_win_rate": float(home["win_rate"]),
        "away_win_rate": float(away["win_rate"]),
        "home_goals_for_per_game": float(home["goals_for_per_game"]),
        "away_goals_for_per_game": float(away["goals_for_per_game"]),
        "home_goals_against_per_game": float(home["goals_against_per_game"]),
        "away_goals_against_per_game": float(away["goals_against_per_game"]),
        "home_goal_diff_per_game": float(home["goal_diff_per_game"]),
        "away_goal_diff_per_game": float(away["goal_diff_per_game"]),
        "home_home_win_rate": float(home["home_win_rate"]),
        "away_away_win_rate": float(away["away_win_rate"]),
        "home_season_games": float(home["season_games"]),
        "away_season_games": float(away["season_games"]),
        "home_season_win_rate": float(home["season_win_rate"]),
        "away_season_win_rate": float(away["season_win_rate"]),
        "home_recent_win_rate": float(home["recent_win_rate"]),
        "away_recent_win_rate": float(away["recent_win_rate"]),
        ELO_FEATURE_COLUMN: (
            float(ratings[target.home_team_id]) - float(ratings[target.away_team_id])
        ),
    }
    if tuple(row) != FEATURE_COLUMNS + (ELO_FEATURE_COLUMN,):
        raise FeatureError("Unscored base/Elo snapshot altered frozen feature order")
    return row


def _record_for_prior_game(
    records: Mapping[tuple[str, str], AdvancedTeamGame],
    game: NormalizedGame,
    team_id: str,
    opponent_id: str,
) -> AdvancedTeamGame:
    record = records.get((game.game_id, team_id))
    if record is None:
        raise FeatureError(
            f"Missing prior advanced-stat record for game {game.game_id} team {team_id}"
        )
    if (
        record.game_id != game.game_id
        or record.game_date != game.game_date
        or record.season != game.season
        or record.team_id != team_id
        or record.opponent_id != opponent_id
    ):
        raise FeatureError(
            f"Prior advanced-stat identity disagrees for game {game.game_id} team {team_id}"
        )
    return record


def _xgoal_snapshot(
    target: UnscoredTarget,
    completed_history: Sequence[NormalizedGame],
    records: Mapping[tuple[str, str], AdvancedTeamGame],
) -> dict[str, float]:
    state: dict[str, dict[str, float]] = defaultdict(
        lambda: {"for": 0.0, "against": 0.0, "games": 0.0}
    )
    for game in completed_history:
        for team_id, opponent_id in (
            (game.home_team_id, game.away_team_id),
            (game.away_team_id, game.home_team_id),
        ):
            record = _record_for_prior_game(records, game, team_id, opponent_id)
            for_value = record.values.get("xGoalsFor")
            against_value = record.values.get("xGoalsAgainst")
            if for_value is None or against_value is None:
                raise FeatureError(
                    f"Prior game {game.game_id} team {team_id} lacks xGoalsFor/xGoalsAgainst"
                )
            try:
                for_number = float(for_value)
                against_number = float(against_value)
            except (TypeError, ValueError) as exc:
                raise FeatureError(
                    f"Prior game {game.game_id} team {team_id} has non-numeric xGoal values"
                ) from exc
            if not math.isfinite(for_number) or not math.isfinite(against_number):
                raise FeatureError(
                    f"Prior game {game.game_id} team {team_id} has non-finite xGoal values"
                )
            team = state[team_id]
            team["for"] += for_number
            team["against"] += against_number
            team["games"] += 1.0

    def share(team_id: str) -> float:
        payload = state[team_id]
        denominator = payload["for"] + payload["against"]
        if payload["games"] <= 0 or denominator <= 0:
            raise FeatureError(
                f"Prior xGoal share unavailable for target team {team_id}; no finite prior state"
            )
        value = payload["for"] / denominator
        if not math.isfinite(value):
            raise FeatureError(f"Prior xGoal share is non-finite for target team {team_id}")
        return value

    return {
        "home_prior_xgoal_share": share(target.home_team_id),
        "away_prior_xgoal_share": share(target.away_team_id),
    }


def _validate_labeled_history_in_schedule(
    completed_history: Sequence[NormalizedGame],
    schedule_history: Sequence[ScheduleHistoryGame],
) -> None:
    by_id = {game.game_id: game for game in schedule_history}
    for completed in completed_history:
        schedule = by_id.get(completed.game_id)
        if schedule is None:
            raise FeatureError(
                f"Completed prior game {completed.game_id} is missing from complete schedule history"
            )
        if (
            schedule.game_date != completed.game_date
            or schedule.season != completed.season
            or schedule.home_team_id != completed.home_team_id
            or schedule.away_team_id != completed.away_team_id
        ):
            raise FeatureError(
                f"Completed prior game {completed.game_id} disagrees with schedule history"
            )


def _schedule_load_snapshot(
    target: UnscoredTarget,
    schedule_history: Sequence[ScheduleHistoryGame],
    calendar_schedule: Sequence[ScheduleHistoryGame] = (),
) -> dict[str, float]:
    """Build B2B / prior-7d from completed history plus known calendar games."""

    ordered = merge_calendar_schedule_for_fatigue(
        schedule_history,
        calendar_schedule,
        target_date=target.game_date,
        season=target.season,
    )

    def team_dates(team_id: str) -> list[date]:
        return [
            game.game_date
            for game in ordered
            if team_id in (game.home_team_id, game.away_team_id)
        ]

    def side(team_id: str) -> tuple[float, float]:
        dates = team_dates(team_id)
        previous = dates[-1] if dates else None
        back_to_back = float(
            previous is not None and (target.game_date - previous).days == 1
        )
        prior_7d = float(
            sum(1 <= (target.game_date - day).days <= 7 for day in dates)
        )
        return back_to_back, prior_7d

    home_b2b, home_7d = side(target.home_team_id)
    away_b2b, away_7d = side(target.away_team_id)
    return {
        "home_back_to_back": home_b2b,
        "away_back_to_back": away_b2b,
        "home_games_prior_7d": home_7d,
        "away_games_prior_7d": away_7d,
    }


def build_unscored_frozen_feature_snapshot(
    target: UnscoredTarget,
    completed_history: Sequence[NormalizedGame],
    advanced_records: Mapping[tuple[str, str], AdvancedTeamGame],
    schedule_history: Sequence[ScheduleHistoryGame],
    *,
    calendar_schedule: Sequence[ScheduleHistoryGame] = (),
    min_prior_games: int = MIN_PRIOR_GAMES,
) -> FrozenFeatureSnapshot:
    """Emit the frozen 25 values from completed priors plus optional calendar fatigue."""

    assert_prior_schedule_load_compact_contract()
    if len(FROZEN_ISSUE40_FEATURE_COLUMNS) != 25:
        raise FeatureError("Issue #40 frozen feature family is no longer exactly 25 columns")

    ordered = _validate_completed_prior_history(target, completed_history)
    schedule_ordered = validate_schedule_history(schedule_history)
    _validate_labeled_history_in_schedule(ordered, schedule_ordered)

    row = _base_and_elo_snapshot(target, ordered)
    if (
        row["home_prior_games"] < min_prior_games
        or row["away_prior_games"] < min_prior_games
    ):
        raise FeatureError(
            "Unscored target is ineligible under the frozen minimum-prior-games gate"
        )

    row.update(_xgoal_snapshot(target, ordered, advanced_records))
    row.update(
        _schedule_load_snapshot(
            target,
            schedule_ordered,
            calendar_schedule=calendar_schedule,
        )
    )

    missing = [
        column for column in FROZEN_ISSUE40_FEATURE_COLUMNS if column not in row
    ]
    if missing:
        raise FeatureError(f"Unscored frozen snapshot omitted columns: {missing}")

    values: list[float] = []
    for column in FROZEN_ISSUE40_FEATURE_COLUMNS:
        value = row[column]
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise FeatureError(
                f"Frozen feature {column} is not numeric for target {target.game_id}"
            ) from exc
        if not math.isfinite(number):
            raise FeatureError(
                f"Frozen feature {column} is not finite for target {target.game_id}"
            )
        values.append(number)

    return FrozenFeatureSnapshot(
        target=target,
        feature_columns=tuple(FROZEN_ISSUE40_FEATURE_COLUMNS),
        feature_values=tuple(values),
    )


def _validate_historical_replay_target_in_schedule(
    target_game: NormalizedGame,
    schedule_history: Sequence[ScheduleHistoryGame],
) -> None:
    matches = [
        game for game in schedule_history if game.game_id == target_game.game_id
    ]
    if len(matches) != 1:
        raise FeatureError(
            f"Historical replay target {target_game.game_id} is not uniquely present "
            "in schedule history"
        )
    schedule = matches[0]
    if (
        schedule.game_date != target_game.game_date
        or schedule.season != target_game.season
        or schedule.home_team_id != target_game.home_team_id
        or schedule.away_team_id != target_game.away_team_id
    ):
        raise FeatureError(
            f"Historical replay target {target_game.game_id} disagrees with schedule history"
        )


def build_historical_replay_snapshot(
    target_game: NormalizedGame,
    all_completed_games: Sequence[NormalizedGame],
    advanced_records: Mapping[tuple[str, str], AdvancedTeamGame],
    schedule_history: Sequence[ScheduleHistoryGame],
    *,
    min_prior_games: int = MIN_PRIOR_GAMES,
) -> FrozenFeatureSnapshot:
    """Replay one historical target using only games from earlier calendar dates.

    The replay harness may receive a larger historical game collection, but it
    cuts result history at history.game_date < target.game_date before calling
    the unscored production primitive. Target score/home_win fields are never
    copied into UnscoredTarget.
    """

    _validate_historical_replay_target_in_schedule(target_game, schedule_history)
    target = unscored_target_from_historical_game(target_game)
    prior = [
        game
        for game in validate_games(all_completed_games)
        if game.game_date < target.game_date
    ]
    return build_unscored_frozen_feature_snapshot(
        target,
        prior,
        advanced_records,
        schedule_history,
        min_prior_games=min_prior_games,
    )
