"""Leakage-safe pre-game feature construction.

A single chronological path is used for training and future prediction.
Until a trustworthy start timestamp is available, features use only games
with `history.game_date < target.game_date`. Same-day results are excluded.
Baseline, Elo, ordered-lag, and linear-recency snapshots for a date are
emitted before any same-day result updates.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from hockey_moneyline.elo import ELO_FEATURE_COLUMN, ELO_INITIAL, updated_ratings
from hockey_moneyline.errors import FeatureError
from hockey_moneyline.linear_recency import LINEAR_RECENCY_COLUMNS, linear_recency_win_rate
from hockey_moneyline.ordered_lags import ORDERED_LAG_COLUMNS, ORDERED_LAG_COUNT
from hockey_moneyline.provenance import RECENT_FORM_WINDOW
from hockey_moneyline.schema import FEATURE_COLUMNS, NormalizedGame, validate_games


@dataclass
class TeamState:
    games: int = 0
    wins: int = 0
    goals_for: int = 0
    goals_against: int = 0
    home_games: int = 0
    home_wins: int = 0
    away_games: int = 0
    away_wins: int = 0
    season: int | None = None
    season_games: int = 0
    season_wins: int = 0
    recent_results: deque[int] = field(default_factory=lambda: deque(maxlen=RECENT_FORM_WINDOW))

    def snapshot(self, current_season: int) -> dict[str, float | None]:
        win_rate = (self.wins / self.games) if self.games else 0.5
        gf = (self.goals_for / self.games) if self.games else 0.0
        ga = (self.goals_against / self.games) if self.games else 0.0
        same_season = self.season == current_season
        season_games = self.season_games if same_season else 0
        season_wins = self.season_wins if same_season else 0
        recent = (
            sum(self.recent_results) / len(self.recent_results)
            if self.recent_results
            else 0.5
        )
        payload: dict[str, float | None] = {
            "prior_games": float(self.games),
            "win_rate": win_rate,
            "goals_for_per_game": gf,
            "goals_against_per_game": ga,
            "goal_diff_per_game": gf - ga,
            "home_win_rate": (self.home_wins / self.home_games) if self.home_games else 0.5,
            "away_win_rate": (self.away_wins / self.away_games) if self.away_games else 0.5,
            "season_games": float(season_games),
            "season_win_rate": (season_wins / season_games) if season_games else 0.5,
            "recent_win_rate": recent,
            "recent_win_rate_linear10": linear_recency_win_rate(self.recent_results),
        }
        for lag in range(1, ORDERED_LAG_COUNT + 1):
            if len(self.recent_results) >= lag:
                # lag_1 is the immediately preceding completed game.
                payload[f"result_lag_{lag}"] = float(self.recent_results[-lag])
            else:
                payload[f"result_lag_{lag}"] = None
        return payload

    def update(self, *, won: bool, goals_for: int, goals_against: int, is_home: bool, season: int) -> None:
        if self.season != season:
            self.season = season
            self.season_games = 0
            self.season_wins = 0
        self.games += 1
        self.wins += int(won)
        self.goals_for += goals_for
        self.goals_against += goals_against
        self.season_games += 1
        self.season_wins += int(won)
        self.recent_results.append(int(won))
        if is_home:
            self.home_games += 1
            self.home_wins += int(won)
        else:
            self.away_games += 1
            self.away_wins += int(won)


def _feature_row(
    game: NormalizedGame,
    home: Mapping[str, float | None],
    away: Mapping[str, float | None],
    home_elo: float,
    away_elo: float,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "game_id": game.game_id,
        "game_date": game.game_date.isoformat(),
        "start_time_utc": game.start_time_utc.isoformat() if game.start_time_utc else None,
        "season": game.season,
        "game_type": game.game_type,
        "home_team_id": game.home_team_id,
        "home_team_name": game.home_team_name,
        "away_team_id": game.away_team_id,
        "away_team_name": game.away_team_name,
        "home_score": game.home_score,
        "away_score": game.away_score,
        "home_win": game.home_win,
        "home_prior_games": home["prior_games"],
        "away_prior_games": away["prior_games"],
        "home_win_rate": home["win_rate"],
        "away_win_rate": away["win_rate"],
        "home_goals_for_per_game": home["goals_for_per_game"],
        "away_goals_for_per_game": away["goals_for_per_game"],
        "home_goals_against_per_game": home["goals_against_per_game"],
        "away_goals_against_per_game": away["goals_against_per_game"],
        "home_goal_diff_per_game": home["goal_diff_per_game"],
        "away_goal_diff_per_game": away["goal_diff_per_game"],
        "home_home_win_rate": home["home_win_rate"],
        "away_away_win_rate": away["away_win_rate"],
        "home_season_games": home["season_games"],
        "away_season_games": away["season_games"],
        "home_season_win_rate": home["season_win_rate"],
        "away_season_win_rate": away["season_win_rate"],
        "home_recent_win_rate": home["recent_win_rate"],
        "away_recent_win_rate": away["recent_win_rate"],
        "home_recent_win_rate_linear10": home["recent_win_rate_linear10"],
        "away_recent_win_rate_linear10": away["recent_win_rate_linear10"],
        ELO_FEATURE_COLUMN: home_elo - away_elo,
    }
    for lag in range(1, ORDERED_LAG_COUNT + 1):
        row[f"home_result_lag_{lag}"] = home[f"result_lag_{lag}"]
        row[f"away_result_lag_{lag}"] = away[f"result_lag_{lag}"]
    return row


def assert_ordered5_lags_complete(rows: Sequence[Mapping[str, Any]]) -> None:
    """Require every selected ordered5 row to have all ten lag values."""
    for row in rows:
        missing = [
            column
            for column in ORDERED_LAG_COLUMNS
            if column not in row or row[column] is None
        ]
        if missing:
            raise FeatureError(
                f"Ordered5 row {row.get('game_id')!r} is missing required lag "
                f"values: {missing}"
            )


def build_features(games: Sequence[NormalizedGame]) -> list[dict[str, Any]]:
    ordered = validate_games(games)
    states: dict[str, TeamState] = defaultdict(TeamState)
    elo_ratings: dict[str, float] = defaultdict(lambda: ELO_INITIAL)
    rows: list[dict[str, Any]] = []
    required_columns = (
        FEATURE_COLUMNS
        + (ELO_FEATURE_COLUMN,)
        + ORDERED_LAG_COLUMNS
        + LINEAR_RECENCY_COLUMNS
    )

    index = 0
    while index < len(ordered):
        current_date = ordered[index].game_date
        day_games: list[NormalizedGame] = []
        while index < len(ordered) and ordered[index].game_date == current_date:
            day_games.append(ordered[index])
            index += 1

        for game in day_games:
            home_state = states[game.home_team_id]
            away_state = states[game.away_team_id]
            row = _feature_row(
                game,
                home_state.snapshot(game.season),
                away_state.snapshot(game.season),
                elo_ratings[game.home_team_id],
                elo_ratings[game.away_team_id],
            )
            for column in required_columns:
                if column not in row:
                    raise FeatureError(f"Feature builder omitted required column {column}")
            rows.append(row)

        for game in day_games:
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
            new_home_elo, new_away_elo = updated_ratings(
                elo_ratings[game.home_team_id],
                elo_ratings[game.away_team_id],
                bool(game.home_win),
                decided_in_shootout=bool(game.decided_in_shootout),
            )
            elo_ratings[game.home_team_id] = new_home_elo
            elo_ratings[game.away_team_id] = new_away_elo
    return rows


def feature_matrix(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str] | None = None,
) -> list[list[float]]:
    selected = tuple(columns) if columns is not None else FEATURE_COLUMNS
    if any(column in ORDERED_LAG_COLUMNS for column in selected):
        assert_ordered5_lags_complete(rows)
    matrix: list[list[float]] = []
    for row in rows:
        missing = [column for column in selected if column not in row]
        if missing:
            raise FeatureError(f"Row {row.get('game_id')!r} missing feature columns: {missing}")
        values: list[float] = []
        for column in selected:
            value = row[column]
            if value is None:
                raise FeatureError(
                    f"Row {row.get('game_id')!r} has missing feature column {column}"
                )
            values.append(float(value))
        matrix.append(values)
    return matrix
