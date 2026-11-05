"""Issue #91 on-demand refresh proof.

Usage:
    python -m hockey_moneyline.issue91

The proof uses synthetic NHL schedule payloads and synthetic MoneyPuck rows.
It does not fit or score, and it does not call the network.
"""

from __future__ import annotations

import csv
import io
import json
from datetime import date, datetime, timezone
from typing import Any, Callable, Mapping, Sequence

from hockey_moneyline.current_season_prior import MONEYPUCK_CURRENT_SEASON_COLUMNS
from hockey_moneyline.errors import HostedRefreshError
from hockey_moneyline.hosted_refresh import (
    BLOCKED,
    ON_DEMAND_REFRESH_CONSUMPTION_OUTCOME,
    PUBLICATION_LAG_MESSAGE,
    READY,
    HostedRefreshBundle,
    assert_hosted_refresh_does_not_score,
    refresh_hosted_slate,
)

PROOF_SLATE = date(2026, 10, 8)
PROOF_OBSERVED = datetime(2026, 10, 8, 16, tzinfo=timezone.utc)
PROOF_PRIOR_ID = "2026020001"
PROOF_TARGET_ID = "2026020010"
CSV_COLUMNS = tuple(MONEYPUCK_CURRENT_SEASON_COLUMNS) + ("playoffGame",)


class BytesResponse(io.BytesIO):
    status = 200

    def __enter__(self) -> "BytesResponse":
        return self

    def __exit__(self, *_args: object) -> bool:
        self.close()
        return False


def nhl_game_payload(
    game_id: str,
    game_date: str,
    *,
    home: str,
    away: str,
    state: str,
    start: str,
    game_type: int = 2,
    home_id: int = 6,
    away_id: int = 7,
) -> dict[str, Any]:
    return {
        "id": int(game_id),
        "season": 20262027,
        "gameType": game_type,
        "gameDate": game_date,
        "startTimeUTC": start,
        "gameState": state,
        "gameScheduleState": "OK",
        "homeTeam": {"id": home_id, "abbrev": home},
        "awayTeam": {"id": away_id, "abbrev": away},
    }


def moneypuck_game_rows(
    game_id: str,
    game_date: str,
    home: str,
    away: str,
    home_score: int,
    away_score: int,
    *,
    home_xgf: str = "1.5",
    away_xgf: str = "1.2",
    season: str = "2026",
    playoff: str = "0",
) -> list[dict[str, str]]:
    compact = game_date.replace("-", "")

    def row(
        team: str,
        opponent: str,
        side: str,
        goals_for: int,
        goals_against: int,
        xgf: str,
        xga: str,
    ) -> dict[str, str]:
        return {
            "team": team,
            "season": season,
            "gameId": game_id,
            "opposingTeam": opponent,
            "home_or_away": side,
            "gameDate": compact,
            "situation": "all",
            "goalsFor": str(goals_for),
            "goalsAgainst": str(goals_against),
            "xGoalsFor": xgf,
            "xGoalsAgainst": xga,
            "playoffGame": playoff,
        }

    return [
        row(home, away, "HOME", home_score, away_score, home_xgf, away_xgf),
        row(away, home, "AWAY", away_score, home_score, away_xgf, home_xgf),
    ]


def csv_bytes(rows: Sequence[Mapping[str, str]]) -> bytes:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})
    return buffer.getvalue().encode("utf-8")


def schedule_opener(
    games_by_date: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[Callable[..., BytesResponse], list[Any]]:
    calls: list[Any] = []

    def opener(request: Any, timeout: int = 30) -> BytesResponse:
        del timeout
        calls.append(request)
        day = request.full_url.rsplit("/", 1)[-1]
        payload = {"gameWeek": [{"date": day, "games": list(games_by_date.get(day, ()))}]}
        return BytesResponse(json.dumps(payload).encode("utf-8"))

    return opener, calls


def moneypuck_opener(payload: bytes) -> tuple[Callable[..., BytesResponse], list[Any]]:
    calls: list[Any] = []

    def opener(request: Any, timeout: int = 30) -> BytesResponse:
        del timeout
        calls.append(request)
        return BytesResponse(payload)

    return opener, calls


def index_games(games: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for game in games:
        grouped.setdefault(str(game["gameDate"]), []).append(game)
    return grouped


def proof_nhl_games(*, include_prior: bool = True) -> list[dict[str, Any]]:
    games = [
        nhl_game_payload(
            PROOF_TARGET_ID,
            PROOF_SLATE.isoformat(),
            home="CAR",
            away="FLA",
            state="FUT",
            start="2026-10-08T23:00:00Z",
            home_id=12,
            away_id=13,
        )
    ]
    if include_prior:
        games.append(
            nhl_game_payload(
                PROOF_PRIOR_ID,
                "2026-10-07",
                home="BOS",
                away="BUF",
                state="OFF",
                start="2026-10-07T23:00:00Z",
            )
        )
    return games


def proof_moneypuck_rows(*, include_prior: bool = True) -> list[dict[str, str]]:
    if not include_prior:
        return []
    return moneypuck_game_rows(PROOF_PRIOR_ID, "2026-10-07", "BOS", "BUF", 3, 2)


def refresh_proof_slate(*, include_prior_in_moneypuck: bool = True) -> HostedRefreshBundle:
    nhl_open, _nhl_calls = schedule_opener(index_games(proof_nhl_games(include_prior=True)))
    csv_open, _csv_calls = moneypuck_opener(
        csv_bytes(proof_moneypuck_rows(include_prior=include_prior_in_moneypuck))
    )
    return refresh_hosted_slate(
        PROOF_SLATE,
        PROOF_OBSERVED,
        nhl_opener=nhl_open,
        moneypuck_opener=csv_open,
    )


def build_issue91_report() -> dict[str, Any]:
    """Run the synthetic ready and blocked proofs. No network and no scoring."""

    assert_hosted_refresh_does_not_score()
    ready = refresh_proof_slate(include_prior_in_moneypuck=True)
    blocked = refresh_proof_slate(include_prior_in_moneypuck=False)
    again = refresh_proof_slate(include_prior_in_moneypuck=True)
    if ready.readiness != READY or ready.blocker is not None:
        raise HostedRefreshError("synthetic complete refresh was not READY")
    if blocked.readiness != BLOCKED or blocked.blocker is None:
        raise HostedRefreshError("synthetic missing-final refresh was not blocked")
    if blocked.blocker.message != PUBLICATION_LAG_MESSAGE:
        raise HostedRefreshError("blocked refresh did not return the publication-lag message")
    if ready.to_dict() != again.to_dict():
        raise HostedRefreshError("synthetic refresh was not deterministic")
    if ready.to_dict()["model_scored"] is not False:
        raise HostedRefreshError("refresh bundle claims a model score")
    return {
        "outcome": ON_DEMAND_REFRESH_CONSUMPTION_OUTCOME,
        "synthetic_ready": True,
        "blocked_message": blocked.blocker.message,
        "moneypuck_sha256": ready.moneypuck_provenance.sha256 if ready.moneypuck_provenance else None,
        "eligible_target_ids": [item.target.game_id for item in ready.scoring_inputs()],
        "model_scored": False,
        "protected_record_written": False,
        "stale_state_used": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    del argv
    report = build_issue91_report()
    print(
        json.dumps(
            {
                "outcome": report["outcome"],
                "synthetic_ready": report["synthetic_ready"],
                "blocked_message": report["blocked_message"],
                "eligible_target_ids": report["eligible_target_ids"],
            },
            sort_keys=True,
        )
    )
    print(report["outcome"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
