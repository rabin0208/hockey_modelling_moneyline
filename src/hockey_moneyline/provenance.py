"""Bounded MoneyPuck development-source metadata.

MoneyPuck's intentionally published game-level download remains the canonical
historical training/evaluation source for the current private-development
baseline. Issue #11 extends the evaluated historical window; it does not change
the provider or the commercial/redistribution posture. See docs/data-provenance.md.
"""

from __future__ import annotations

from typing import Any

SOURCE_ID = "moneypuck_game_level"
SOURCE_NAME = "MoneyPuck published game-level data"
SOURCE_HOST = "moneypuck.com"
SOURCE_PAGE = "https://moneypuck.com/data.htm"
SOURCE_FILE_URL = "https://moneypuck.com/moneypuck/playerData/careers/gameByGame/all_teams.csv"
ACCESS_METHOD = (
    "Unauthenticated HTTPS GET of the single game-by-game all-teams CSV that "
    "MoneyPuck publishes on its public data page. No unlisted endpoints are used."
)
REGULAR_SEASON_GAME_TYPE = "regular"
MONEYPUCK_SITUATION = "all"

# Issue #11 canonical historical evaluation window. MoneyPuck season labels are
# start years (2021 == 2021-22); canonical season IDs use YYYYYYYY.
DEFAULT_SEASONS = (20212022, 20222023, 20232024, 20242025, 20252026)
MONEYPUCK_SEASON_YEARS = (2021, 2022, 2023, 2024, 2025)

# Strictly forward-chaining canonical Issue #11 evaluation contract.
FORWARD_VALIDATION_SEASONS = (20222023, 20232024, 20242025)
FINAL_TEST_SEASON = 20252026

# Legacy Issue #2 constants remain available so the original regression helpers
# and tests stay reproducible. They are not the Issue #11 canonical evaluation.
TRAIN_SEASONS = (20242025,)
VALIDATION_START = "2025-10-01"
VALIDATION_END_EXCLUSIVE = "2026-02-01"
TEST_START = "2026-02-01"
TEST_END_INCLUSIVE = "2026-06-30"

RECENT_FORM_WINDOW = 10
MIN_PRIOR_GAMES = 5

USER_AGENT = "hockey-moneyline-issue-11-dev/0.4 (+https://github.com/EduardoRasanmar/hockey_moneyline)"

PROVENANCE_NOTES = [
    "MoneyPuck was approved by Eduardo in Hockey Issue #2 as the bounded historical development source.",
    "Issue #11 extends the same approved source window to 2021-22 through 2025-26 where published data is available.",
    "Use is limited to this private, non-commercial development baseline.",
    "Credit: MoneyPuck.com. Listed data is stated as free for non-commercial use with clear attribution.",
    "Do not scrape MoneyPuck pages or unlisted endpoints.",
    "Raw MoneyPuck/provider files must not be committed to Git.",
    "Not a final production, publication, commercial-use, or provider-of-record commitment.",
    "Automated tests use synthetic fixtures only.",
]


def season_from_moneypuck_year(year: int) -> int:
    return int(f"{year}{year + 1}")


def moneypuck_year_from_season(season: int) -> int:
    text = str(season)
    if len(text) != 8:
        raise ValueError(f"Unsupported season id {season!r}")
    return int(text[:4])


def provenance_record() -> dict[str, Any]:
    return {
        "source_id": SOURCE_ID,
        "source_name": SOURCE_NAME,
        "source_host": SOURCE_HOST,
        "source_page": SOURCE_PAGE,
        "source_file_url": SOURCE_FILE_URL,
        "access_method": ACCESS_METHOD,
        "credit": "MoneyPuck.com",
        "exact_data": (
            "Completed regular-season team-game rows from the published "
            "all_teams.csv file, filtered to situation=all, for MoneyPuck "
            f"season years {list(MONEYPUCK_SEASON_YEARS)} "
            f"(canonical seasons {list(DEFAULT_SEASONS)}). "
            "Only game id, game date, season, game type, home/away team "
            "identifiers, and final scores are retained."
        ),
        "identifiers": {
            "game_id": "MoneyPuck gameId (NHL-style game identifier)",
            "team_id": (
                "MoneyPuck team / opposingTeam abbreviation, with only explicit "
                "punctuation aliases normalized to stable NHL tricodes"
            ),
            "season": "Canonical YYYYYYYY season id derived from MoneyPuck season start year",
            "game_type": "Derived from gameId (02=regular) and playoffGame",
            "game_date": "MoneyPuck gameDate (YYYYMMDD)",
            "start_time_utc": "Not provided by this MoneyPuck file; left unset",
        },
        "historical_window": {
            "season_years": list(MONEYPUCK_SEASON_YEARS),
            "canonical_seasons": list(DEFAULT_SEASONS),
            "evaluation": "strict forward chaining; 2025-26 remains final held-out test",
        },
        "raw_data_may_be_committed": False,
        "update_behavior": (
            "Re-run acquisition. The published CSV is cached locally unless "
            "--refresh is passed. HTTP/schema failures raise AcquisitionError "
            "and do not invent games, scores, or teams."
        ),
        "failure_behavior": "Visible exception; no silent defaults or fabricated rows.",
        "use_constraints": PROVENANCE_NOTES,
        "final_provider_commitment": False,
    }
