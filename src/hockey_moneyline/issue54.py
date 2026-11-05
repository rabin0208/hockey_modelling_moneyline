"""Issue #54 current-season prior-state source-readiness proof.

Usage:
    python -m hockey_moneyline.issue54

The default proof uses synthetic NHL schedule/status fixtures and synthetic
MoneyPuck current-season rows only. It does not fit, tune, or otherwise adapt
from 2026-27 outcomes. Optional ``--live-nhl`` records a bounded schedule
observation and never treats a network failure as verified-empty state.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from hockey_moneyline.current_season_prior import (
    CURRENT_SEASON,
    NOT_READY,
    READY,
    assemble_current_season_nhl_coverage,
    build_current_season_prior_state,
    parse_current_season_moneypuck_rows,
    required_coverage_dates,
)
from hockey_moneyline.errors import CurrentSeasonPriorSourceError, HockeyMoneylineError
from hockey_moneyline.nhl_api import NHLScheduleGame, NHLScheduleSnapshot
from hockey_moneyline.prospective_scoring import (
    CURRENT_SEASON_PRIOR_SOURCE_OBSERVED_AT_KEY,
    CURRENT_SEASON_PRIOR_SOURCE_SHA256_KEY,
    require_current_season_prior_source_identity,
)
from hockey_moneyline.unscored_target import UnscoredTarget, unscored_target_from_nhl_schedule

DEFAULT_OUTPUT = Path("data/derived/issue54_current_season_prior_state_report.json")
FIRST_GAME_OBSERVED = datetime(2026, 10, 8, 16, tzinfo=timezone.utc)
FIRST_GAME_START = datetime(2026, 10, 8, 23, tzinfo=timezone.utc)
TRANSITION_OBSERVED = datetime(2026, 10, 12, 16, tzinfo=timezone.utc)
TRANSITION_START = datetime(2026, 10, 12, 23, tzinfo=timezone.utc)


def _nhl_game(
    game_id: str,
    day: str,
    *,
    home: str,
    away: str,
    state: str,
    observed_at_utc: datetime,
    game_type: int = 2,
    start: datetime | None = None,
    home_id: int = 6,
    away_id: int = 7,
) -> NHLScheduleGame:
    game_date = date.fromisoformat(day)
    scheduled = start or datetime(
        game_date.year, game_date.month, game_date.day, 23, tzinfo=timezone.utc
    )
    type_name = {1: "preseason", 2: "regular", 3: "playoff"}[game_type]
    return NHLScheduleGame(
        nhl_game_id=game_id,
        season=CURRENT_SEASON,
        game_type=type_name,
        game_date=game_date,
        scheduled_start_time_utc=scheduled,
        home_nhl_team_id=home_id,
        away_nhl_team_id=away_id,
        home_team_abbrev=home,
        away_team_abbrev=away,
        game_state=state,
        game_schedule_state="OK",
        observed_at_utc=observed_at_utc,
    )


def _snapshot(
    games: Sequence[NHLScheduleGame],
    *,
    observed_at_utc: datetime,
    target_date: date,
    endpoint_path: str | None = None,
) -> NHLScheduleSnapshot:
    return NHLScheduleSnapshot(
        endpoint_path=endpoint_path or f"/v1/schedule/{target_date.isoformat()}",
        observed_at_utc=observed_at_utc,
        games=tuple(games),
        covered_dates=frozenset(required_coverage_dates(target_date)),
    )


def _mp_row(
    *,
    team: str,
    opponent: str,
    side: str,
    game_id: str,
    game_date: str,
    goals_for: int,
    goals_against: int,
    xgf: str = "1.5",
    xga: str = "1.1",
    playoff: str = "0",
) -> dict[str, str]:
    compact = game_date.replace("-", "")
    return {
        "team": team,
        "season": "2026",
        "gameId": game_id,
        "opposingTeam": opponent,
        "home_or_away": side,
        "gameDate": compact,
        "situation": "all",
        "goalsFor": str(goals_for),
        "goalsAgainst": str(goals_against),
        "playoffGame": playoff,
        "xGoalsFor": xgf,
        "xGoalsAgainst": xga,
    }


def _mp_game(
    game_id: str,
    game_date: str,
    home: str,
    away: str,
    home_score: int,
    away_score: int,
    *,
    home_xgf: str = "1.5",
    away_xgf: str = "1.2",
) -> list[dict[str, str]]:
    return [
        _mp_row(
            team=home,
            opponent=away,
            side="HOME",
            game_id=game_id,
            game_date=game_date,
            goals_for=home_score,
            goals_against=away_score,
            xgf=home_xgf,
            xga=away_xgf,
        ),
        _mp_row(
            team=away,
            opponent=home,
            side="AWAY",
            game_id=game_id,
            game_date=game_date,
            goals_for=away_score,
            goals_against=home_score,
            xgf=away_xgf,
            xga=home_xgf,
        ),
    ]


def first_game_target() -> UnscoredTarget:
    nhl = _nhl_game(
        "2026020001",
        "2026-10-08",
        home="BOS",
        away="BUF",
        state="FUT",
        observed_at_utc=FIRST_GAME_OBSERVED,
        start=FIRST_GAME_START,
    )
    return unscored_target_from_nhl_schedule(nhl)


def verified_empty_inputs() -> tuple[
    UnscoredTarget,
    NHLScheduleSnapshot,
    list[dict[str, str]],
    datetime,
]:
    target = first_game_target()
    preseason = _nhl_game(
        "2026010099",
        "2026-09-20",
        home="BOS",
        away="CAR",
        state="OFF",
        observed_at_utc=FIRST_GAME_OBSERVED,
        game_type=1,
        home_id=6,
        away_id=12,
    )
    snapshot = _snapshot(
        (
            preseason,
            _nhl_game(
                target.game_id,
                target.game_date.isoformat(),
                home=target.home_team_id,
                away=target.away_team_id,
                state="FUT",
                observed_at_utc=FIRST_GAME_OBSERVED,
                start=FIRST_GAME_START,
            ),
        ),
        observed_at_utc=FIRST_GAME_OBSERVED,
        target_date=target.game_date,
    )
    preseason_rows = _mp_game("2026010099", "2026-09-20", "BOS", "CAR", 4, 1)
    return target, snapshot, preseason_rows, FIRST_GAME_OBSERVED


def completed_transition_target() -> UnscoredTarget:
    nhl = _nhl_game(
        "2026020010",
        "2026-10-12",
        home="BOS",
        away="BUF",
        state="FUT",
        observed_at_utc=TRANSITION_OBSERVED,
        start=TRANSITION_START,
    )
    return unscored_target_from_nhl_schedule(nhl)


def completed_transition_inputs() -> tuple[
    UnscoredTarget,
    NHLScheduleSnapshot,
    list[dict[str, str]],
    datetime,
]:
    target = completed_transition_target()
    games = (
        _nhl_game(
            "2026010099",
            "2026-09-20",
            home="BOS",
            away="CAR",
            state="OFF",
            observed_at_utc=TRANSITION_OBSERVED,
            game_type=1,
            home_id=6,
            away_id=12,
        ),
        _nhl_game(
            "2026020002",
            "2026-10-08",
            home="BOS",
            away="BUF",
            state="OFF",
            observed_at_utc=TRANSITION_OBSERVED,
            start=datetime(2026, 10, 8, 23, tzinfo=timezone.utc),
        ),
        _nhl_game(
            "2026020003",
            "2026-10-09",
            home="CAR",
            away="NJD",
            state="FINAL",
            observed_at_utc=TRANSITION_OBSERVED,
            start=datetime(2026, 10, 9, 23, tzinfo=timezone.utc),
            home_id=12,
            away_id=1,
        ),
        _nhl_game(
            "2026020004",
            "2026-10-10",
            home="BOS",
            away="CAR",
            state="OFF",
            observed_at_utc=TRANSITION_OBSERVED,
            start=datetime(2026, 10, 10, 23, tzinfo=timezone.utc),
            home_id=6,
            away_id=12,
        ),
        _nhl_game(
            target.game_id,
            target.game_date.isoformat(),
            home=target.home_team_id,
            away=target.away_team_id,
            state="FUT",
            observed_at_utc=TRANSITION_OBSERVED,
            start=TRANSITION_START,
        ),
    )
    snapshot = _snapshot(
        games,
        observed_at_utc=TRANSITION_OBSERVED,
        target_date=target.game_date,
    )
    rows: list[dict[str, str]] = []
    rows.extend(_mp_game("2026010099", "2026-09-20", "BOS", "CAR", 4, 1))
    rows.extend(_mp_game("2026020002", "2026-10-08", "BOS", "BUF", 3, 2))
    rows.extend(_mp_game("2026020003", "2026-10-09", "CAR", "NJD", 2, 2))
    rows.extend(
        _mp_game(
            "2026020004",
            "2026-10-10",
            "BOS",
            "CAR",
            1,
            0,
            home_xgf="2.1",
            away_xgf="0.9",
        )
    )
    return target, snapshot, rows, TRANSITION_OBSERVED


def _fail_closed_cases() -> list[dict[str, Any]]:
    target, snapshot, rows, observed = verified_empty_inputs()
    cases: list[dict[str, Any]] = []

    def capture(name: str, callback) -> None:
        try:
            callback()
        except CurrentSeasonPriorSourceError as exc:
            cases.append({"case": name, "failed_closed": True, "message": str(exc)})
            return
        cases.append({"case": name, "failed_closed": False, "message": None})

    capture(
        "missing_nhl_snapshot",
        lambda: build_current_season_prior_state(
            target, None, rows, observed_at_utc=observed
        ),
    )
    capture(
        "missing_moneypuck_material",
        lambda: build_current_season_prior_state(
            target, snapshot, None, observed_at_utc=observed
        ),
    )
    incomplete = NHLScheduleSnapshot(
        endpoint_path=snapshot.endpoint_path,
        observed_at_utc=observed,
        games=snapshot.games,
        covered_dates=frozenset({target.game_date}),
    )
    capture(
        "incomplete_schedule_coverage",
        lambda: build_current_season_prior_state(
            target, incomplete, rows, observed_at_utc=observed
        ),
    )
    stale_nhl = NHLScheduleSnapshot(
        endpoint_path=snapshot.endpoint_path,
        observed_at_utc=datetime(2026, 10, 7, 16, tzinfo=timezone.utc),
        games=snapshot.games,
        covered_dates=snapshot.covered_dates,
    )
    capture(
        "stale_nhl_observation",
        lambda: build_current_season_prior_state(
            target, stale_nhl, rows, observed_at_utc=observed
        ),
    )
    capture(
        "stale_moneypuck_retrieval",
        lambda: build_current_season_prior_state(
            target,
            snapshot,
            rows,
            observed_at_utc=observed,
            moneypuck_retrieved_at_utc=datetime(2026, 10, 7, 16, tzinfo=timezone.utc),
        ),
    )

    trans_target, trans_snapshot, trans_rows, trans_observed = completed_transition_inputs()
    without_completed = [
        row for row in trans_rows if str(row.get("gameId")) != "2026020002"
    ]
    capture(
        "missing_completed_moneypuck_for_nhl_final",
        lambda: build_current_season_prior_state(
            trans_target,
            trans_snapshot,
            without_completed,
            observed_at_utc=trans_observed,
        ),
    )
    missing_xgoal = []
    for row in trans_rows:
        copied = dict(row)
        if copied["gameId"] == "2026020004" and copied["home_or_away"] == "HOME":
            copied["xGoalsFor"] = ""
        missing_xgoal.append(copied)
    capture(
        "missing_prior_xgoal",
        lambda: build_current_season_prior_state(
            trans_target,
            trans_snapshot,
            missing_xgoal,
            observed_at_utc=trans_observed,
        ),
    )
    conflicting = []
    for row in trans_rows:
        copied = dict(row)
        if copied["gameId"] == "2026020002" and copied["home_or_away"] == "HOME":
            copied["team"] = "TOR"
        if copied["gameId"] == "2026020002" and copied["home_or_away"] == "AWAY":
            copied["opposingTeam"] = "TOR"
        conflicting.append(copied)
    capture(
        "conflicting_source_identities",
        lambda: build_current_season_prior_state(
            trans_target,
            trans_snapshot,
            conflicting,
            observed_at_utc=trans_observed,
        ),
    )
    return cases


def _optional_live_nhl_observation() -> dict[str, Any]:
    target = first_game_target()
    try:
        snapshot = assemble_current_season_nhl_coverage(
            target.game_date,
            observed_at_utc=FIRST_GAME_OBSERVED,
            timeout=20,
        )
    except (HockeyMoneylineError, OSError, TimeoutError) as exc:
        return {
            "attempted": True,
            "available": False,
            "used_as_verified_empty_proof": False,
            "error": str(exc),
            "note": (
                "Live NHL observation is supplemental evidence only. A fetch "
                "failure is not treated as verified-empty current-season state."
            ),
        }
    regular_final_before_cutoff = [
        game.nhl_game_id
        for game in snapshot.games
        if game.season == CURRENT_SEASON
        and game.game_type == "regular"
        and game.game_state in {"OFF", "FINAL"}
        and game.game_date < target.game_date
    ]
    preseason_ids = [
        game.nhl_game_id for game in snapshot.games if game.game_type == "preseason"
    ]
    return {
        "attempted": True,
        "available": True,
        "used_as_verified_empty_proof": False,
        "observed_at_utc": snapshot.observed_at_utc.isoformat(),
        "endpoint_path": snapshot.endpoint_path,
        "covered_date_count": len(snapshot.covered_dates),
        "game_count": len(snapshot.games),
        "regular_final_game_ids_before_synthetic_cutoff": regular_final_before_cutoff,
        "preseason_game_ids_present": preseason_ids[:20],
        "note": (
            "Live NHL schedule/status was observed under the approved Issue #6/#9 "
            "contract. Verified-empty proof still requires matching MoneyPuck "
            "material and is recorded from explicit synthetic fixtures in this issue."
        ),
    }


def build_issue54_readiness_report(
    *,
    include_live_nhl: bool = False,
) -> dict[str, Any]:
    blockers: list[str] = []
    empty_target, empty_snapshot, empty_rows, empty_observed = verified_empty_inputs()
    empty_package = build_current_season_prior_state(
        empty_target,
        empty_snapshot,
        empty_rows,
        observed_at_utc=empty_observed,
    )
    empty_again = build_current_season_prior_state(
        empty_target,
        empty_snapshot,
        empty_rows,
        observed_at_utc=empty_observed,
    )
    trans_target, trans_snapshot, trans_rows, trans_observed = completed_transition_inputs()
    trans_package = build_current_season_prior_state(
        trans_target,
        trans_snapshot,
        trans_rows,
        observed_at_utc=trans_observed,
    )
    parsed_transition = parse_current_season_moneypuck_rows(trans_rows)

    if not empty_package.empty_state_verified or empty_package.empty_state_proof is None:
        blockers.append("first-game package was not a verified empty current-season state")
    if empty_package.eligible_prior_completed_game_ids:
        blockers.append("verified-empty package included eligible completed games")
    if empty_package.winner_labeled_history or empty_package.schedule_history:
        blockers.append("verified-empty package leaked current-season result/schedule state")
    if "2026010099" not in empty_package.excluded_preseason_game_ids:
        blockers.append("verified-empty package did not record excluded preseason identity")
    if empty_package.source_sha256 != empty_again.source_sha256:
        blockers.append("verified-empty source SHA-256 is not deterministic")
    try:
        require_current_season_prior_source_identity(empty_package.issue52_source_versions())
    except Exception as exc:  # pragma: no cover - report path
        blockers.append(f"Issue #52 identity rejected verified-empty package: {exc}")

    if trans_package.empty_state_verified:
        blockers.append("completed-state package was incorrectly marked verified empty")
    if trans_package.eligible_prior_completed_game_ids != (
        "2026020002",
        "2026020003",
        "2026020004",
    ):
        blockers.append(
            "completed-state eligible ids drifted: "
            f"{trans_package.eligible_prior_completed_game_ids}"
        )
    if [game.game_id for game in trans_package.winner_labeled_history] != [
        "2026020002",
        "2026020004",
    ]:
        blockers.append("winner-labeled history does not match completed non-tied games")
    if [game.game_id for game in trans_package.schedule_history] != [
        "2026020002",
        "2026020003",
        "2026020004",
    ]:
        blockers.append("schedule history omitted tied/unlabeled completed games")
    if trans_package.history_only_tied_game_ids != ("2026020003",):
        blockers.append("tied completed game was not preserved as schedule-history-only")
    expected_xgoal_keys = {
        ("2026020002", "BOS"),
        ("2026020002", "BUF"),
        ("2026020004", "BOS"),
        ("2026020004", "CAR"),
    }
    if set(trans_package.xgoal_records) != expected_xgoal_keys:
        blockers.append("completed-state xGoal records are incomplete")
    if "2026010099" not in trans_package.excluded_preseason_game_ids:
        blockers.append("completed-state package consumed preseason identity")
    if trans_package.source_sha256 == empty_package.source_sha256:
        blockers.append("empty and completed-state packages share a content hash")
    try:
        require_current_season_prior_source_identity(trans_package.issue52_source_versions())
    except Exception as exc:  # pragma: no cover - report path
        blockers.append(f"Issue #52 identity rejected completed-state package: {exc}")
    if parsed_transition.excluded_preseason_game_ids != ("2026010099",):
        blockers.append("MoneyPuck parser failed to exclude preseason rows")

    fail_closed = _fail_closed_cases()
    if any(not case["failed_closed"] for case in fail_closed):
        leaked = [case["case"] for case in fail_closed if not case["failed_closed"]]
        blockers.append(f"fail-closed cases did not raise: {leaked}")

    live = (
        _optional_live_nhl_observation()
        if include_live_nhl
        else {
            "attempted": False,
            "available": False,
            "used_as_verified_empty_proof": False,
            "note": "Live NHL observation was not requested for this proof run.",
        }
    )

    outcome = READY if not blockers else NOT_READY
    report: dict[str, Any] = {
        "issue": 54,
        "outcome": outcome,
        "blockers": blockers,
        "frozen_contract_unchanged": {
            "model_parameters_changed": False,
            "feature_protocol_changed": False,
            "fitted_artifact_sha256": (
                "1739c7edf14ecd23ab3e94b54ef9194a2d868026ceb320663a85fb9a292aac28"
            ),
            "training_row_identity_sha256": (
                "743373bdbb31e22cc5f188402cc736808f7cac9515bf53f5ee88a058dbff168c"
            ),
            "2026_27_outcomes_used_for_fitting_tuning_or_selection": False,
        },
        "source_boundary": {
            "moneypuck_published_game_level": True,
            "nhl_web_api_schedule": True,
            "new_provider_introduced": False,
            "credentials_expanded": False,
        },
        "verified_empty_first_game": empty_package.to_audit_dict(),
        "completed_state_transition": trans_package.to_audit_dict(),
        "issue52_identity": {
            "verified_empty": empty_package.issue52_source_versions(),
            "completed_state": trans_package.issue52_source_versions(),
            "required_keys": [
                CURRENT_SEASON_PRIOR_SOURCE_SHA256_KEY,
                CURRENT_SEASON_PRIOR_SOURCE_OBSERVED_AT_KEY,
            ],
        },
        "fail_closed": fail_closed,
        "live_nhl_observation": live,
        "remaining_dependency": (
            None
            if not blockers
            else {
                "owner": "00 — Hockey Moneyline HQ",
                "dependency": blockers[0],
                "unblock_signal": (
                    "Approved MoneyPuck + NHL contracts must supply a reproducible "
                    "current-season prior-state package, including verified-empty "
                    "first-game semantics, and fail closed on missing/stale/"
                    "incomplete/conflicting source material"
                ),
            }
        ),
        "notes": [
            "Issue #54 does not change the frozen 25-column protocol or fitted artifact.",
            "2026-27 outcomes are not used for fitting, tuning, recalibration, or selection.",
            "Verified-empty state requires NHL schedule/status coverage plus explicit MoneyPuck material, not a missing payload.",
            "Issue #55 must not begin live 2026-27 scoring until this gate is merged READY.",
        ],
    }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Issue #54 current-season prior-state source-readiness proof"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--live-nhl",
        action="store_true",
        help=(
            "Optionally observe approved NHL schedule/status endpoints; never "
            "treat fetch failure as empty state"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = build_issue54_readiness_report(include_live_nhl=args.live_nhl)
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
