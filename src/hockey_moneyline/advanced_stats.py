"""Issue #19 MoneyPuck advanced team-stat source/join/timing contract.

This module is deliberately not part of the incumbent model feature matrix. It
validates the already-approved MoneyPuck ``all_teams.csv`` source on development
seasons only and exposes one leakage-safe rolling primitive for a later,
separately governed 02 ablation.
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
from hockey_moneyline.normalize import canonical_moneypuck_team_code, games_from_moneypuck_csv
from hockey_moneyline.provenance import season_from_moneypuck_year
from hockey_moneyline.schema import NormalizedGame, parse_game_date, validate_games

ISSUE19_SEASONS = (20212022, 20222023, 20232024, 20242025)
HELD_OUT_SEASON = 20252026

ADVANCED_IDENTITY_COLUMNS = (
    "team",
    "season",
    "gameId",
    "opposingTeam",
    "home_or_away",
    "gameDate",
    "situation",
)

ADVANCED_CANDIDATE_FIELDS = (
    "shotAttemptsFor",
    "shotAttemptsAgainst",
    "corsiPercentage",
    "unblockedShotAttemptsFor",
    "unblockedShotAttemptsAgainst",
    "fenwickPercentage",
    "xGoalsFor",
    "xGoalsAgainst",
    "xGoalsPercentage",
    "lowDangerShotsFor",
    "lowDangerShotsAgainst",
    "mediumDangerShotsFor",
    "mediumDangerShotsAgainst",
    "highDangerShotsFor",
    "highDangerShotsAgainst",
)

ADVANCED_FIELD_DEFINITIONS = {
    "shotAttemptsFor": "Team-perspective Corsi count: all shot attempts for.",
    "shotAttemptsAgainst": "Opponent-perspective Corsi count: all shot attempts against.",
    "corsiPercentage": "Team share of all shot attempts in the game/situation.",
    "unblockedShotAttemptsFor": "Team-perspective Fenwick count: unblocked shot attempts for.",
    "unblockedShotAttemptsAgainst": "Opponent-perspective Fenwick count: unblocked shot attempts against.",
    "fenwickPercentage": "Team share of all unblocked shot attempts in the game/situation.",
    "xGoalsFor": "Sum of MoneyPuck expected-goal values for the team's unblocked attempts.",
    "xGoalsAgainst": "Sum of MoneyPuck expected-goal values for the opponent's unblocked attempts.",
    "xGoalsPercentage": "Team share of expected goals in the game/situation.",
    "lowDangerShotsFor": "Unblocked team attempts with MoneyPuck goal probability below 8%.",
    "lowDangerShotsAgainst": "Opponent unblocked attempts with MoneyPuck goal probability below 8%.",
    "mediumDangerShotsFor": "Unblocked team attempts with goal probability at least 8% and below 20%.",
    "mediumDangerShotsAgainst": "Opponent unblocked attempts with goal probability at least 8% and below 20%.",
    "highDangerShotsFor": "Unblocked team attempts with MoneyPuck goal probability at least 20%.",
    "highDangerShotsAgainst": "Opponent unblocked attempts with MoneyPuck goal probability at least 20%.",
}

PAIR_BASES = (
    "shotAttempts",
    "unblockedShotAttempts",
    "xGoals",
    "lowDangerShots",
    "mediumDangerShots",
    "highDangerShots",
)

XGOAL_RAW_FIELDS = ("xGoalsFor", "xGoalsAgainst")
XGOAL_FEATURE_COLUMNS = ("home_prior_xgoal_share", "away_prior_xgoal_share")
CORSI_RAW_FIELDS = ("shotAttemptsFor", "shotAttemptsAgainst")
CORSI_FEATURE_COLUMNS = ("home_prior_corsi_share", "away_prior_corsi_share")


@dataclass(frozen=True)
class AdvancedTeamGame:
    game_id: str
    game_date: date
    season: int
    team_id: str
    opponent_id: str
    side: str
    values: Mapping[str, float | None]
    invalid_fields: tuple[str, ...] = ()


@dataclass
class AdvancedInspection:
    report: dict[str, Any]
    games: list[NormalizedGame]
    records: dict[tuple[str, str], AdvancedTeamGame]


def _is_regular_season(row: Mapping[str, Any]) -> bool:
    game_id = str(row.get("gameId") or "")
    playoff = str(row.get("playoffGame") or "").strip().lower() in {
        "1", "1.0", "true", "yes", "y"
    }
    return len(game_id) >= 6 and game_id[4:6] == "02" and not playoff


def _parse_metric(value: Any) -> tuple[float | None, bool]:
    if value is None or str(value).strip() == "":
        return None, False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None, True
    if not math.isfinite(number):
        return None, True
    return number, False


def _record_equivalent(left: AdvancedTeamGame, right: AdvancedTeamGame) -> bool:
    return (
        left.game_id == right.game_id
        and left.game_date == right.game_date
        and left.season == right.season
        and left.team_id == right.team_id
        and left.opponent_id == right.opponent_id
        and left.side == right.side
        and dict(left.values) == dict(right.values)
        and left.invalid_fields == right.invalid_fields
    )


def _expected_team_keys(
    games: Sequence[NormalizedGame],
) -> dict[tuple[str, str], tuple[NormalizedGame, str, str]]:
    expected: dict[tuple[str, str], tuple[NormalizedGame, str, str]] = {}
    for game in games:
        expected[(game.game_id, game.home_team_id)] = (game, "HOME", game.away_team_id)
        expected[(game.game_id, game.away_team_id)] = (game, "AWAY", game.home_team_id)
    return expected


def _coverage_report(
    records: Iterable[AdvancedTeamGame],
    present_fields: Sequence[str],
    seasons: Sequence[int],
) -> dict[str, dict[str, dict[str, float | int | None]]]:
    grouped: dict[int, list[AdvancedTeamGame]] = defaultdict(list)
    for record in records:
        grouped[record.season].append(record)

    output: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for season in seasons:
        season_records = grouped.get(season, [])
        season_payload: dict[str, dict[str, float | int | None]] = {}
        for field in ADVANCED_CANDIDATE_FIELDS:
            values = [
                record.values.get(field)
                for record in season_records
                if field in present_fields
            ]
            numeric = [float(value) for value in values if value is not None]
            invalid = sum(field in record.invalid_fields for record in season_records)
            eligible = len(season_records)
            non_null = len(numeric)
            season_payload[field] = {
                "eligible_team_game_rows": eligible,
                "non_null": non_null,
                "coverage": (non_null / eligible) if eligible else None,
                "invalid_non_finite_or_non_numeric": invalid,
                "min": min(numeric) if numeric else None,
                "max": max(numeric) if numeric else None,
            }
        output[str(season)] = season_payload
    return output


def inspect_advanced_stats_csv(
    path: Path,
    seasons: Sequence[int] = ISSUE19_SEASONS,
    shootout_winners: Mapping[str, int] | None = None,
) -> AdvancedInspection:
    """Inspect only Issue #19 development seasons from the canonical CSV.

    The provider's combined file is streamed, but rows outside the requested
    development seasons are discarded immediately after the ``season`` cell is
    checked; candidate fields from 2025-26 are never parsed, profiled, joined,
    featurized, or evaluated.
    """

    selected = tuple(seasons)
    if not selected:
        raise SchemaError("Issue #19 requires at least one development season")
    if HELD_OUT_SEASON in selected or any(season not in ISSUE19_SEASONS for season in selected):
        raise SchemaError(
            "Issue #19 advanced-stat inspection is restricted to 2021-22 through 2024-25; "
            "2025-26 and other seasons are not permitted"
        )
    return _inspect_advanced_stats_selected(
        path, selected, shootout_winners=shootout_winners
    )


def inspect_advanced_stats_csv_for_authorized_heldout(
    path: Path,
    seasons: Sequence[int],
    *,
    confirm_consume_2025_26: bool,
    shootout_winners: Mapping[str, int] | None = None,
) -> AdvancedInspection:
    """Gated Issue #19 join/timing inspection for the frozen heldout pathway.

    Public validation-only ``inspect_advanced_stats_csv`` still rejects 2025-26.
    This wrapper reuses the same parse/join/timing implementation and requires
    an explicit confirmation flag plus the exact six-season canonical set.
    """

    if not confirm_consume_2025_26:
        raise SchemaError(
            "Authorized heldout advanced-stat inspection requires confirm_consume_2025_26=True"
        )
    selected = tuple(seasons)
    if selected != ISSUE19_SEASONS + (HELD_OUT_SEASON,):
        raise SchemaError(
            "Authorized heldout advanced-stat inspection requires exactly "
            "20212022,20222023,20232024,20242025,20252026"
        )
    return _inspect_advanced_stats_selected(
        path, selected, shootout_winners=shootout_winners
    )


CYCLE2_DEVELOPMENT_SEASONS = ISSUE19_SEASONS + (HELD_OUT_SEASON,)
RESERVED_FUTURE_SEASON = 20262027


def inspect_advanced_stats_csv_for_cycle2_development(
    path: Path,
    seasons: Sequence[int],
    shootout_winners: Mapping[str, int] | None = None,
) -> AdvancedInspection:
    """Cycle-2 development wrapper over the Issue #19 parse/join/timing primitive.

    Permits consumed 2025-26 development evidence while hard-rejecting 2026-27.
    Does not weaken the public validation-only path that still rejects 2025-26.
    """

    selected = tuple(seasons)
    if not selected:
        raise SchemaError("Cycle-2 advanced-stat inspection requires at least one season")
    forbidden = sorted(
        {season for season in selected if season >= RESERVED_FUTURE_SEASON}
    )
    if forbidden:
        raise SchemaError(
            "Cycle-2 advanced-stat inspection rejects reserved future season(s): "
            + ", ".join(str(season) for season in forbidden)
        )
    unexpected = [
        season for season in selected if season not in CYCLE2_DEVELOPMENT_SEASONS
    ]
    if unexpected:
        raise SchemaError(
            "Cycle-2 advanced-stat inspection accepts only 2021-22 through "
            f"2025-26; got {unexpected}"
        )
    if selected != CYCLE2_DEVELOPMENT_SEASONS:
        raise SchemaError(
            "Cycle-2 advanced-stat inspection requires exactly "
            "20212022,20222023,20232024,20242025,20252026"
        )
    return _inspect_advanced_stats_selected(
        path, selected, shootout_winners=shootout_winners
    )


def _inspect_advanced_stats_selected(
    path: Path,
    selected: Sequence[int],
    shootout_winners: Mapping[str, int] | None = None,
) -> AdvancedInspection:
    selected = tuple(selected)
    normalized = games_from_moneypuck_csv(
        path, seasons=selected, shootout_winners=shootout_winners
    )
    games = normalized.games
    expected = _expected_team_keys(games)
    selected_raw = {str(int(str(season)[:4])) for season in selected}

    records: dict[tuple[str, str], AdvancedTeamGame] = {}
    equivalent_duplicates = Counter()
    raw_unmatched = Counter()
    present_fields: list[str] = []

    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise SchemaError(f"MoneyPuck file has no header: {path}")
            missing_identity = [
                field for field in ADVANCED_IDENTITY_COLUMNS if field not in reader.fieldnames
            ]
            if missing_identity:
                raise SchemaError(
                    f"MoneyPuck advanced-stat source is missing identity columns {missing_identity}"
                )
            present_fields = [
                field for field in ADVANCED_CANDIDATE_FIELDS if field in reader.fieldnames
            ]

            for row in reader:
                raw_season = str(row.get("season") or "").strip()
                if raw_season not in selected_raw:
                    continue
                if str(row.get("situation") or "").strip() != "all":
                    continue
                if not _is_regular_season(row):
                    continue

                season = season_from_moneypuck_year(int(raw_season))
                game_id = str(row.get("gameId") or "").strip()
                if not game_id:
                    raise SchemaError("MoneyPuck advanced-stat row is missing gameId")
                team = canonical_moneypuck_team_code(row.get("team"), season)
                opponent = canonical_moneypuck_team_code(row.get("opposingTeam"), season)
                side = str(row.get("home_or_away") or "").strip().upper()
                if side not in {"HOME", "AWAY"}:
                    raise SchemaError(
                        f"Game {game_id} team {team} has invalid home_or_away {side!r}"
                    )
                key = (game_id, team)
                expected_row = expected.get(key)
                if expected_row is None:
                    raw_unmatched[season] += 1
                    continue

                game, expected_side, expected_opponent = expected_row
                row_date = parse_game_date(row.get("gameDate") or "")
                if row_date != game.game_date:
                    raise SchemaError(
                        f"Advanced row {key} date {row_date} disagrees with normalized date {game.game_date}"
                    )
                if side != expected_side:
                    raise SchemaError(
                        f"Advanced row {key} side {side} disagrees with expected {expected_side}"
                    )
                if opponent != expected_opponent:
                    raise SchemaError(
                        f"Advanced row {key} opponent {opponent} disagrees with expected {expected_opponent}"
                    )

                values: dict[str, float | None] = {}
                invalid_fields: list[str] = []
                for field in present_fields:
                    parsed, invalid = _parse_metric(row.get(field))
                    values[field] = parsed
                    if invalid:
                        invalid_fields.append(field)

                record = AdvancedTeamGame(
                    game_id=game_id,
                    game_date=row_date,
                    season=season,
                    team_id=team,
                    opponent_id=opponent,
                    side=side,
                    values=values,
                    invalid_fields=tuple(sorted(invalid_fields)),
                )
                existing = records.get(key)
                if existing is not None:
                    if _record_equivalent(existing, record):
                        equivalent_duplicates[season] += 1
                        continue
                    raise SchemaError(
                        f"Conflicting duplicate MoneyPuck advanced-stat row for {key}"
                    )
                records[key] = record
    except OSError as exc:
        raise SchemaError(f"Unreadable MoneyPuck file {path}") from exc

    normalized_unmatched = sorted(set(expected) - set(records))
    pair_conflicts: list[str] = []
    for game in games:
        home = records.get((game.game_id, game.home_team_id))
        away = records.get((game.game_id, game.away_team_id))
        if home is None or away is None:
            continue
        for base in PAIR_BASES:
            for_field = f"{base}For"
            against_field = f"{base}Against"
            if for_field not in present_fields or against_field not in present_fields:
                continue
            comparisons = (
                (home.values.get(for_field), away.values.get(against_field), "homeFor/awayAgainst"),
                (away.values.get(for_field), home.values.get(against_field), "awayFor/homeAgainst"),
            )
            for left, right, label in comparisons:
                if left is None or right is None:
                    continue
                if not math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-9):
                    pair_conflicts.append(f"{game.game_id}:{base}:{label}:{left}!={right}")

    coverage = _coverage_report(records.values(), present_fields, selected)
    normalized_games_by_season = Counter(game.season for game in games)
    matched_by_season = Counter(record.season for record in records.values())
    duplicate_total = sum(equivalent_duplicates.values())
    expected_target_excluded = {
        str(season): 2 * normalized.skipped_tied_scores_by_season.get(season, 0)
        for season in selected
    }

    blockers: list[str] = []
    missing_candidate_fields = [
        field for field in ADVANCED_CANDIDATE_FIELDS if field not in present_fields
    ]
    if any(field in missing_candidate_fields for field in XGOAL_RAW_FIELDS):
        blockers.append("required xGoal candidate fields are missing from the source header")
    if normalized_unmatched:
        blockers.append(
            f"{len(normalized_unmatched)} normalized game-team keys have no matched advanced row"
        )
    if duplicate_total:
        blockers.append(
            f"{duplicate_total} equivalent duplicate advanced rows violate expected 1:1 team-game cardinality"
        )
    if pair_conflicts:
        blockers.append(
            f"{len(pair_conflicts)} HOME/AWAY For-vs-Against pair consistency conflicts"
        )
    for season in selected:
        actual_unmatched = raw_unmatched.get(season, 0)
        expected_unmatched = expected_target_excluded[str(season)]
        if actual_unmatched != expected_unmatched:
            blockers.append(
                f"{season} raw unmatched rows {actual_unmatched} != expected target-excluded rows {expected_unmatched}"
            )
        for field in XGOAL_RAW_FIELDS:
            stat = coverage[str(season)][field]
            if stat["eligible_team_game_rows"] and (
                stat["coverage"] != 1.0
                or stat["invalid_non_finite_or_non_numeric"] != 0
            ):
                blockers.append(
                    f"{season} {field} is not complete/finite for matched team-game rows"
                )

    ready = not blockers
    report: dict[str, Any] = {
        "issue": 19,
        "seasons_profiled": list(selected),
        "held_out_2025_26_profiled": HELD_OUT_SEASON in selected,
        "source_contract": {
            "situation": "all",
            "game_type": "regular",
            "raw_identity_key": ["gameId", "team"],
            "canonical_team_game_key": ["game_id", "canonical_team_id"],
            "expected_cardinality": (
                "one matched situation=all row per canonical game-team key; "
                "two matched team rows per normalized game"
            ),
            "candidate_fields_present": present_fields,
            "candidate_fields_missing": missing_candidate_fields,
            "field_definitions": ADVANCED_FIELD_DEFINITIONS,
        },
        "join": {
            "normalized_games_by_season": {
                str(season): normalized_games_by_season.get(season, 0)
                for season in selected
            },
            "expected_team_game_rows_by_season": {
                str(season): 2 * normalized_games_by_season.get(season, 0)
                for season in selected
            },
            "matched_team_game_rows_by_season": {
                str(season): matched_by_season.get(season, 0)
                for season in selected
            },
            "equivalent_duplicate_rows_by_season": {
                str(season): equivalent_duplicates.get(season, 0)
                for season in selected
            },
            "conflicting_duplicate_policy": "hard SchemaError; never guessed or silently dropped",
            "normalized_game_team_unmatched": len(normalized_unmatched),
            "raw_unmatched_rows_by_season": {
                str(season): raw_unmatched.get(season, 0) for season in selected
            },
            "expected_target_excluded_raw_rows_by_season": expected_target_excluded,
            "home_away_pair_conflicts": len(pair_conflicts),
            "pair_conflict_examples": pair_conflicts[:10],
        },
        "coverage": coverage,
        "timing_contract": {
            "classification": "postgame team-game outcomes; usable only for later target dates",
            "rule": "history.game_date < target.game_date",
            "same_day": "all target rows for a date are emitted before any same-day advanced result updates",
            "future_mutation": "future advanced outcomes cannot alter already-emitted earlier target features",
            "team_identity": "explicit canonical team codes; ARI and UTA remain distinct",
        },
        "smallest_candidate_family": {
            "name": "prior_xgoal_share",
            "raw_fields": list(XGOAL_RAW_FIELDS),
            "feature_columns": list(XGOAL_FEATURE_COLUMNS),
            "aggregation": (
                "For each team, sum xGoalsFor and xGoalsAgainst over canonical prior completed games only, "
                "then compute xGoalsFor/(xGoalsFor+xGoalsAgainst). No same-day result is visible. "
                "Rows require the existing minimum prior-game gate before a future 02 ablation; "
                "no ad-hoc missing-value imputation."
            ),
            "reason": (
                "Two raw fields provide one compact shot-quality share per team and avoid adding "
                "correlated Corsi/Fenwick/danger families in the first ablation."
            ),
        },
        "deferred_candidates": {
            "corsi": ["shotAttemptsFor", "shotAttemptsAgainst", "corsiPercentage"],
            "fenwick": [
                "unblockedShotAttemptsFor",
                "unblockedShotAttemptsAgainst",
                "fenwickPercentage",
            ],
            "danger": [
                "lowDangerShotsFor",
                "lowDangerShotsAgainst",
                "mediumDangerShotsFor",
                "mediumDangerShotsAgainst",
                "highDangerShotsFor",
                "highDangerShotsAgainst",
            ],
            "special_teams": (
                "not proposed from situation=all in Issue #19; a future contract would need "
                "explicit strength-state denominators/semantics"
            ),
        },
        "blockers": blockers,
        "outcome": "READY FOR 02 ABLATION" if ready else "NOT READY",
    }
    return AdvancedInspection(report=report, games=games, records=records)


def _for_against_share(payload: Mapping[str, float]) -> float | None:
    denominator = payload["for"] + payload["against"]
    if payload["games"] <= 0 or denominator <= 0:
        return None
    return payload["for"] / denominator


def corsi_source_readiness(inspection: AdvancedInspection) -> dict[str, Any]:
    """Explicit Corsi completeness check independent of Issue #19 READY.

    Issue #19's READY/FOR-02 blocker logic is xGoal-oriented. A READY outcome is
    not sufficient proof that ``shotAttemptsFor`` / ``shotAttemptsAgainst`` are
    complete and finite on matched development-season rows.
    """

    selected = list(inspection.report["seasons_profiled"])
    present_fields = list(inspection.report["source_contract"]["candidate_fields_present"])
    coverage = inspection.report["coverage"]
    blockers: list[str] = []
    seasons: dict[str, dict[str, Any]] = {}

    for season in selected:
        season_key = str(season)
        season_coverage = coverage.get(season_key, {})
        season_blockers: list[str] = []
        field_stats: dict[str, dict[str, Any]] = {}
        for field in CORSI_RAW_FIELDS:
            present = field in present_fields
            stat = season_coverage.get(field, {})
            eligible = int(stat.get("eligible_team_game_rows") or 0)
            field_coverage = stat.get("coverage")
            invalid = int(stat.get("invalid_non_finite_or_non_numeric") or 0)
            complete = (
                present
                and eligible > 0
                and field_coverage == 1.0
                and invalid == 0
            )
            field_stats[field] = {
                "present_in_header": present,
                "eligible_team_game_rows": eligible,
                "coverage": field_coverage,
                "invalid_non_finite_or_non_numeric": invalid,
                "complete_and_finite": complete,
            }
            if not present:
                season_blockers.append(f"{season_key} {field} is missing from the source header")
            elif eligible <= 0:
                season_blockers.append(
                    f"{season_key} {field} has no matched target-usable team-game rows"
                )
            elif field_coverage != 1.0 or invalid != 0:
                season_blockers.append(
                    f"{season_key} {field} is not complete/finite for matched team-game rows "
                    f"(coverage={field_coverage}, invalid={invalid})"
                )
        seasons[season_key] = {
            "fields": field_stats,
            "ready": not season_blockers,
            "blockers": season_blockers,
        }
        blockers.extend(season_blockers)

    return {
        "ready": not blockers,
        "blockers": blockers,
        "fields_required": list(CORSI_RAW_FIELDS),
        "unused_row_level_percentage": "corsiPercentage",
        "uses_corsiPercentage": False,
        "issue19_outcome": inspection.report["outcome"],
        "issue19_outcome_is_not_sufficient": True,
        "held_out_2025_26_profiled": inspection.report["held_out_2025_26_profiled"],
        "seasons_profiled": selected,
        "seasons": seasons,
        "note": (
            "Issue #19 READY FOR 02 ABLATION is xGoal-oriented and is not treated "
            "as proof of Corsi readiness. Issue #25 requires explicit 100% finite "
            "coverage of shotAttemptsFor and shotAttemptsAgainst on matched "
            "target-usable team-game rows in every selected development season."
        ),
    }


def assert_corsi_source_ready(inspection: AdvancedInspection) -> dict[str, Any]:
    """Fail visibly if Corsi source fields are incomplete; do not impute."""

    payload = corsi_source_readiness(inspection)
    if not payload["ready"]:
        raise SchemaError(
            "Issue #25 Corsi source fields are not complete/finite on development "
            "seasons; return this discrepancy to 01 / 00 and do not impute or "
            "substitute corsiPercentage: "
            f"{payload['blockers']}"
        )
    return payload


def _build_pregame_for_against_share_features(
    games: Sequence[NormalizedGame],
    records: Mapping[tuple[str, str], AdvancedTeamGame],
    *,
    for_field: str,
    against_field: str,
    home_column: str,
    away_column: str,
) -> list[dict[str, Any]]:
    """Emit leakage-safe prior For/(For+Against) shares from one field pair.

    Same-day snapshots are emitted before any same-day state update. Only
    ``history.game_date < target.game_date`` contributes. Row-level percentage
    fields are never read.
    """

    ordered = validate_games(games)
    state: dict[str, dict[str, float]] = defaultdict(
        lambda: {"for": 0.0, "against": 0.0, "games": 0.0}
    )
    rows: list[dict[str, Any]] = []

    index = 0
    while index < len(ordered):
        current_date = ordered[index].game_date
        day_games: list[NormalizedGame] = []
        while index < len(ordered) and ordered[index].game_date == current_date:
            day_games.append(ordered[index])
            index += 1

        for game in day_games:
            home = state[game.home_team_id]
            away = state[game.away_team_id]
            rows.append(
                {
                    "game_id": game.game_id,
                    "game_date": game.game_date.isoformat(),
                    "season": game.season,
                    "home_team_id": game.home_team_id,
                    "away_team_id": game.away_team_id,
                    "home_advanced_prior_games": int(home["games"]),
                    "away_advanced_prior_games": int(away["games"]),
                    home_column: _for_against_share(home),
                    away_column: _for_against_share(away),
                }
            )

        for game in day_games:
            for team_id in (game.home_team_id, game.away_team_id):
                record = records.get((game.game_id, team_id))
                if record is None:
                    raise FeatureError(
                        f"Missing advanced-stat record for game {game.game_id} team {team_id}"
                    )
                for_value = record.values.get(for_field)
                against_value = record.values.get(against_field)
                if for_value is None or against_value is None:
                    raise FeatureError(
                        f"Game {game.game_id} team {team_id} lacks finite "
                        f"{for_field}/{against_field}"
                    )
                team_state = state[team_id]
                team_state["for"] += float(for_value)
                team_state["against"] += float(against_value)
                team_state["games"] += 1.0

    return rows


def build_pregame_xgoal_share_features(
    games: Sequence[NormalizedGame],
    records: Mapping[tuple[str, str], AdvancedTeamGame],
) -> list[dict[str, Any]]:
    """Emit leakage-safe prior xGoal shares without changing incumbent features."""

    return _build_pregame_for_against_share_features(
        games,
        records,
        for_field="xGoalsFor",
        against_field="xGoalsAgainst",
        home_column="home_prior_xgoal_share",
        away_column="away_prior_xgoal_share",
    )


def build_pregame_corsi_share_features(
    games: Sequence[NormalizedGame],
    records: Mapping[tuple[str, str], AdvancedTeamGame],
) -> list[dict[str, Any]]:
    """Emit leakage-safe prior Corsi shares from shot-attempt counts only.

    ``corsiPercentage`` is intentionally unread. The share is
    ``sum(prior shotAttemptsFor) / (sum(prior shotAttemptsFor) + sum(prior shotAttemptsAgainst))``.
    """

    return _build_pregame_for_against_share_features(
        games,
        records,
        for_field="shotAttemptsFor",
        against_field="shotAttemptsAgainst",
        home_column="home_prior_corsi_share",
        away_column="away_prior_corsi_share",
    )
