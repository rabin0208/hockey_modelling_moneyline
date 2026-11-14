"""Deterministic MoneyPuck ↔ NHL schedule identity mapping."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

from hockey_moneyline.errors import IdentityError
from hockey_moneyline.nhl_api import NHLScheduleGame
from hockey_moneyline.schema import NormalizedGame


@dataclass(frozen=True, slots=True)
class TeamAlias:
    source_code: str
    nhl_abbrev: str
    first_season: int = 19001901
    last_season: int = 99999999

    def applies(self, code: str, season: int) -> bool:
        return (
            self.source_code == code.upper()
            and self.first_season <= season <= self.last_season
        )


# MoneyPuck currently uses dotted forms for these abbreviations. The validity
# window is explicit so any future franchise/code change is deliberate rather
# than hidden in fuzzy matching.
TEAM_ALIASES = (
    TeamAlias("L.A", "LAK"),
    TeamAlias("N.J", "NJD"),
    TeamAlias("S.J", "SJS"),
    TeamAlias("T.B", "TBL"),
)


def nhl_abbrev_for_source(code: str, season: int) -> str:
    normalized = str(code).strip().upper()
    for alias in TEAM_ALIASES:
        if alias.applies(normalized, season):
            return alias.nhl_abbrev
    return normalized


@dataclass(frozen=True, slots=True)
class IdentityMismatch:
    game_id: str
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class IdentityJoinReport:
    total: int
    matched: tuple[NormalizedGame, ...]
    unmatched_game_ids: tuple[str, ...]
    mismatches: tuple[IdentityMismatch, ...]

    @property
    def matched_count(self) -> int:
        return len(self.matched)

    @property
    def coverage(self) -> float:
        return (self.matched_count / self.total) if self.total else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "total": self.total,
            "matched": self.matched_count,
            "coverage": self.coverage,
            "unmatched_game_ids": list(self.unmatched_game_ids),
            "mismatches": [
                {"game_id": mismatch.game_id, "reasons": list(mismatch.reasons)}
                for mismatch in self.mismatches
            ],
        }


def _mismatch_reasons(game: NormalizedGame, nhl: NHLScheduleGame) -> list[str]:
    reasons: list[str] = []
    if game.season != nhl.season:
        reasons.append(f"season {game.season} != {nhl.season}")
    if game.game_type != nhl.game_type:
        reasons.append(f"game_type {game.game_type!r} != {nhl.game_type!r}")
    if game.game_date != nhl.game_date:
        reasons.append(f"game_date {game.game_date} != {nhl.game_date}")

    home = nhl_abbrev_for_source(
        game.home_team_abbrev or game.home_team_id,
        game.season,
    )
    away = nhl_abbrev_for_source(
        game.away_team_abbrev or game.away_team_id,
        game.season,
    )
    if home != nhl.home_team_abbrev:
        reasons.append(f"home team {home!r} != {nhl.home_team_abbrev!r}")
    if away != nhl.away_team_abbrev:
        reasons.append(f"away team {away!r} != {nhl.away_team_abbrev!r}")
    return reasons


def join_moneypuck_to_nhl(
    games: Sequence[NormalizedGame],
    nhl_games: Sequence[NHLScheduleGame],
) -> IdentityJoinReport:
    by_id: dict[str, NHLScheduleGame] = {}
    for nhl_game in nhl_games:
        if nhl_game.nhl_game_id in by_id:
            raise IdentityError(f"Duplicate NHL game id: {nhl_game.nhl_game_id}")
        by_id[nhl_game.nhl_game_id] = nhl_game

    matched: list[NormalizedGame] = []
    unmatched: list[str] = []
    mismatches: list[IdentityMismatch] = []

    for game in games:
        # Exact ID equality is the only candidate lookup. There is no fuzzy or
        # date-only fallback; date/team/type agreement is then independently
        # validated before any NHL fields are attached.
        nhl = by_id.get(str(game.game_id))
        if nhl is None:
            unmatched.append(game.game_id)
            continue
        reasons = _mismatch_reasons(game, nhl)
        if reasons:
            mismatches.append(IdentityMismatch(game.game_id, tuple(reasons)))
            continue
        matched.append(
            replace(
                game,
                nhl_game_id=nhl.nhl_game_id,
                home_nhl_team_id=nhl.home_nhl_team_id,
                away_nhl_team_id=nhl.away_nhl_team_id,
                start_time_utc=nhl.scheduled_start_time_utc,
                scheduled_start_time_utc=nhl.scheduled_start_time_utc,
                game_state=nhl.game_state,
                game_schedule_state=nhl.game_schedule_state,
            )
        )

    return IdentityJoinReport(
        total=len(games),
        matched=tuple(matched),
        unmatched_game_ids=tuple(unmatched),
        mismatches=tuple(mismatches),
    )


def require_clean_join(
    report: IdentityJoinReport,
    *,
    require_full_coverage: bool = False,
) -> IdentityJoinReport:
    if report.mismatches:
        raise IdentityError(f"NHL identity mismatches: {report.to_dict()['mismatches']}")
    if require_full_coverage and report.unmatched_game_ids:
        raise IdentityError(
            f"NHL identity coverage incomplete: {list(report.unmatched_game_ids)}"
        )
    return report
