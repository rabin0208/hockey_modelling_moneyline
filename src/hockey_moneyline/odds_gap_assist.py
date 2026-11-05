"""Manual, bounded assist for The Odds Gap free current NHL moneyline export.

This is not a scraper, cron, daemon, or paid client. One invocation performs at
most one HTTP GET of the documented free export, or reads a CSV the operator
already downloaded. It does not send an API key, follow pagination, call the
paid archive, or download MoneyPuck ``oddsHistory``.

Rows become canonical snapshots only when the operator supplies the NHL game
id, home/away identity, and puck drop. Team names are not turned into NHL ids.
``devig_fair_prob`` and ``best_price_flag`` are dropped and never stored.
"""

from __future__ import annotations

import csv
import io
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from hockey_moneyline.errors import ForwardOddsArchiveError
from hockey_moneyline.forward_odds_archive import (
    decimal_text_from_american,
    format_utc,
    parse_aware_utc,
    parse_nhl_game_id,
    parse_team_identity,
    validate_forward_odds_snapshot,
    write_forward_odds_snapshot,
)

ODDS_GAP_EXPORT_URL = (
    "https://theoddsgap.com/api/odds-export.csv?sport=icehockey_nhl&market=ml"
)
ODDS_GAP_ATTRIBUTION_URL = "https://theoddsgap.com/data"
ODDS_GAP_USER_AGENT = (
    "hockey-moneyline-issue-87-manual-assist/1.0 "
    "(+https://github.com/EduardoRasanmar/hockey_moneyline)"
)
ODDS_GAP_CAPTURE_METHOD = "oddsgap_free_csv_assist"
ODDS_GAP_ATTRIBUTION = (
    "The Odds Gap free current NHL moneyline export "
    f"({ODDS_GAP_ATTRIBUTION_URL}). Attribution: The Odds Gap. "
    "Research and modeling use only. devig_fair_prob and best_price_flag "
    "were not stored. No API key, paid archive, pagination, or schedule was used."
)
MAX_RESPONSE_BYTES = 16 * 1024 * 1024
REQUIRED_COLUMNS = (
    "snapshot_ts",
    "sport",
    "away",
    "home",
    "commence_time",
    "market",
    "side",
    "line",
    "book",
    "american_odds",
)
DROPPED_COLUMNS = ("devig_fair_prob", "best_price_flag")
_FORBIDDEN_REQUEST_HEADERS = frozenset(
    {"authorization", "x-oddsgap-data-key", "x-api-key", "api-key"}
)

# Exact display names from the free NHL export. Aliases below are explicit
# spellings, not fuzzy matching. An unknown name stays unmapped.
ODDS_GAP_TEAM_TO_ABBREV = {
    "Anaheim Ducks": "ANA",
    "Boston Bruins": "BOS",
    "Buffalo Sabres": "BUF",
    "Calgary Flames": "CGY",
    "Carolina Hurricanes": "CAR",
    "Chicago Blackhawks": "CHI",
    "Colorado Avalanche": "COL",
    "Columbus Blue Jackets": "CBJ",
    "Dallas Stars": "DAL",
    "Detroit Red Wings": "DET",
    "Edmonton Oilers": "EDM",
    "Florida Panthers": "FLA",
    "Los Angeles Kings": "LAK",
    "Minnesota Wild": "MIN",
    "Montréal Canadiens": "MTL",
    "Montreal Canadiens": "MTL",
    "Nashville Predators": "NSH",
    "New Jersey Devils": "NJD",
    "New York Islanders": "NYI",
    "New York Rangers": "NYR",
    "Ottawa Senators": "OTT",
    "Philadelphia Flyers": "PHI",
    "Pittsburgh Penguins": "PIT",
    "San Jose Sharks": "SJS",
    "Seattle Kraken": "SEA",
    "St Louis Blues": "STL",
    "St. Louis Blues": "STL",
    "Tampa Bay Lightning": "TBL",
    "Toronto Maple Leafs": "TOR",
    "Utah Mammoth": "UTA",
    "Vancouver Canucks": "VAN",
    "Vegas Golden Knights": "VGK",
    "Washington Capitals": "WSH",
    "Winnipeg Jets": "WPG",
}


@dataclass(frozen=True, slots=True)
class SuppliedNhlGame:
    nhl_game_id: str
    home_team: str
    away_team: str
    scheduled_puck_drop_utc: str


@dataclass(frozen=True, slots=True)
class OddsGapFetch:
    text: str
    access_tier: str | None
    rate_limit_limit: str | None
    rate_limit_remaining: str | None
    rate_limit_reset: str | None
    window_start: str | None
    window_end: str | None
    request_count: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_count": self.request_count,
            "access_tier": self.access_tier,
            "rate_limit_limit": self.rate_limit_limit,
            "rate_limit_remaining": self.rate_limit_remaining,
            "rate_limit_reset": self.rate_limit_reset,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "credentials_sent": False,
            "pagination_followed": False,
            "paid_tier_used": False,
        }


@dataclass(frozen=True, slots=True)
class OddsGapProposal:
    source_row_count: int
    ready: tuple[dict[str, Any], ...]
    unmapped_count: int
    rejected_count: int
    collapsed_earlier_observation_count: int
    filtered_other_book_count: int
    unknown_team_names: tuple[str, ...]
    rejected_reason_counts: tuple[tuple[str, int], ...]
    bounded_to_latest_per_game_book: bool
    book_filter: str | None
    fetch: OddsGapFetch | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": "the_odds_gap_free_current_export",
            "attribution": ODDS_GAP_ATTRIBUTION,
            "export_url": ODDS_GAP_EXPORT_URL,
            "paid_tier_used": False,
            "credentials_used": False,
            "pagination_followed": False,
            "moneypuck_oddshistory_adopted": False,
            "capture_method": ODDS_GAP_CAPTURE_METHOD,
            "source_row_count": self.source_row_count,
            "bounded_to_latest_per_game_book": self.bounded_to_latest_per_game_book,
            "book_filter": self.book_filter,
            "ready_count": len(self.ready),
            "ready": [dict(payload) for payload in self.ready],
            "unmapped_count": self.unmapped_count,
            "rejected_count": self.rejected_count,
            "collapsed_earlier_observation_count": (
                self.collapsed_earlier_observation_count
            ),
            "filtered_other_book_count": self.filtered_other_book_count,
            "unknown_team_names": list(self.unknown_team_names),
            "rejected_reason_counts": [
                {"reason": reason, "count": count}
                for reason, count in self.rejected_reason_counts
            ],
            "fetch": None if self.fetch is None else self.fetch.to_dict(),
            "written": [],
        }


def parse_supplied_schedule(
    games: Sequence[Mapping[str, Any]],
) -> tuple[SuppliedNhlGame, ...]:
    """Accept operator-supplied NHL identity. Missing ids are not invented."""

    if isinstance(games, (str, bytes)) or not isinstance(games, Sequence):
        raise ForwardOddsArchiveError(
            "NHL schedule identity must be a list of games supplied by the operator"
        )
    parsed: list[SuppliedNhlGame] = []
    seen: set[str] = set()
    for game in games:
        if not isinstance(game, Mapping) or "nhl_game_id" not in game:
            raise ForwardOddsArchiveError(
                "NHL schedule identity is missing nhl_game_id; refusing to invent one"
            )
        nhl_game_id = parse_nhl_game_id(game.get("nhl_game_id"))
        if nhl_game_id in seen:
            raise ForwardOddsArchiveError(
                f"NHL schedule identity repeats game id {nhl_game_id}"
            )
        seen.add(nhl_game_id)
        home_team = parse_team_identity(game.get("home_team"), label="home_team")
        away_team = parse_team_identity(game.get("away_team"), label="away_team")
        if home_team == away_team:
            raise ForwardOddsArchiveError(
                "home/away identities must be present and distinct"
            )
        puck_drop = format_utc(
            parse_aware_utc(
                game.get("scheduled_puck_drop_utc"),
                label="scheduled_puck_drop_utc",
            )
        )
        parsed.append(
            SuppliedNhlGame(
                nhl_game_id=nhl_game_id,
                home_team=home_team,
                away_team=away_team,
                scheduled_puck_drop_utc=puck_drop,
            )
        )
    return tuple(parsed)


def _header(headers: Any, name: str) -> str | None:
    if headers is None or not hasattr(headers, "get"):
        return None
    value = headers.get(name)
    if value in (None, ""):
        value = headers.get(name.lower())
    if value in (None, ""):
        return None
    return str(value)


def _assert_request_has_no_credential(request: Request) -> None:
    for key in request.header_items():
        name = key[0].lower()
        if name in _FORBIDDEN_REQUEST_HEADERS or name.endswith("-key") or name == "key":
            raise ForwardOddsArchiveError(
                "Odds Gap assist refused to send a credential"
            )


def fetch_odds_gap_csv(
    *,
    opener: Callable[..., Any] = urlopen,
    timeout: float = 60,
) -> OddsGapFetch:
    """Perform one GET of the free current NHL moneyline export.

    HTTP 429, a non-free access tier, or an ``X-Next-Start`` pagination cursor
    fails closed. This function does not retry and does not follow that cursor.
    """

    request = Request(
        ODDS_GAP_EXPORT_URL,
        headers={
            "Accept": "text/csv",
            "User-Agent": ODDS_GAP_USER_AGENT,
        },
        method="GET",
    )
    if request.full_url != ODDS_GAP_EXPORT_URL:
        raise ForwardOddsArchiveError("Odds Gap assist refused a non-approved URL")
    _assert_request_has_no_credential(request)
    headers: Any = None
    try:
        with opener(request, timeout=timeout) as response:
            status = getattr(response, "status", None) or getattr(response, "code", 200)
            headers = getattr(response, "headers", None)
            if int(status) == 429:
                raise ForwardOddsArchiveError(
                    "The Odds Gap rate limit refused the request; "
                    f"retry-after={_header(headers, 'Retry-After')}. "
                    "No retry is attempted."
                )
            if int(status) != 200:
                raise ForwardOddsArchiveError(
                    f"The Odds Gap export failed with HTTP {status}"
                )
            payload = response.read(MAX_RESPONSE_BYTES + 1)
    except ForwardOddsArchiveError:
        raise
    except HTTPError as exc:
        retry = _header(getattr(exc, "headers", None), "Retry-After")
        if exc.code == 429:
            raise ForwardOddsArchiveError(
                "The Odds Gap rate limit refused the request; "
                f"retry-after={retry}. No retry is attempted."
            ) from exc
        raise ForwardOddsArchiveError(
            f"The Odds Gap export failed with HTTP {exc.code}"
        ) from exc
    except URLError as exc:
        raise ForwardOddsArchiveError(
            f"The Odds Gap export could not be read: {exc.reason}"
        ) from exc
    except OSError as exc:
        raise ForwardOddsArchiveError(
            f"The Odds Gap export could not be read: {exc}"
        ) from exc
    if not isinstance(payload, (bytes, bytearray)):
        raise ForwardOddsArchiveError("The Odds Gap export did not return bytes")
    if len(payload) > MAX_RESPONSE_BYTES:
        raise ForwardOddsArchiveError(
            "The Odds Gap response exceeded the bounded size cap"
        )
    # ``headers`` is from the successful response. Re-read it only if present.
    response_headers = headers
    if _header(response_headers, "X-Next-Start"):
        raise ForwardOddsArchiveError(
            "The Odds Gap response is truncated (X-Next-Start). "
            "This assist does not follow pagination."
        )
    tier = _header(response_headers, "X-Access-Tier")
    if tier is None or tier.lower() != "free":
        raise ForwardOddsArchiveError(
            f"The Odds Gap access tier {tier!r} is not the approved no-cost free tier"
        )
    try:
        text = bytes(payload).decode("utf-8-sig")
    except UnicodeError as exc:
        raise ForwardOddsArchiveError(
            "The Odds Gap export was not valid UTF-8"
        ) from exc
    return OddsGapFetch(
        text=text,
        access_tier=tier,
        rate_limit_limit=_header(response_headers, "X-RateLimit-Limit"),
        rate_limit_remaining=_header(response_headers, "X-RateLimit-Remaining"),
        rate_limit_reset=_header(response_headers, "X-RateLimit-Reset"),
        window_start=_header(response_headers, "X-Window-Start"),
        window_end=_header(response_headers, "X-Window-End"),
        request_count=1,
    )


def _match_game(
    schedule: Sequence[SuppliedNhlGame],
    *,
    home_team: str,
    away_team: str,
    scheduled_puck_drop_utc: str,
) -> SuppliedNhlGame | None:
    matches = [
        game
        for game in schedule
        if game.home_team == home_team
        and game.away_team == away_team
        and game.scheduled_puck_drop_utc == scheduled_puck_drop_utc
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def _row_reason(message: str) -> str:
    text = message.lower()
    if "missing one side" in text:
        return "missing_side"
    if "conflicting" in text:
        return "conflicting_side"
    if "timezone-aware" in text:
        return "naive_timestamp"
    if "strictly before" in text:
        return "post_puck"
    if "decimal odds" in text or "american odds" in text:
        return "invalid_odds"
    if "not a two-sided moneyline" in text:
        return "not_moneyline"
    if "sport" in text:
        return "wrong_sport"
    if "spread or total" in text:
        return "spread_line_present"
    if "bookmaker mixing" in text:
        return "mixed_book"
    if "book/source" in text:
        return "missing_book"
    return "invalid_row"


def prepare_odds_gap_snapshots(
    csv_text: str,
    schedule: Sequence[Mapping[str, Any]] | Sequence[SuppliedNhlGame],
    *,
    book: str | None = None,
    all_pregame_observations: bool = False,
    fetch: OddsGapFetch | None = None,
) -> OddsGapProposal:
    """Turn one CSV into validated snapshot payloads. This does not write."""

    if isinstance(schedule, Sequence) and schedule and isinstance(schedule[0], SuppliedNhlGame):
        supplied = tuple(schedule)  # type: ignore[arg-type]
    else:
        supplied = parse_supplied_schedule(schedule)  # type: ignore[arg-type]
    book_filter = book.strip() if isinstance(book, str) and book.strip() else None
    if book is not None and book_filter is None:
        raise ForwardOddsArchiveError("book filter is empty")

    reader = csv.DictReader(io.StringIO(csv_text))
    fieldnames = reader.fieldnames or []
    missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
    if missing:
        raise ForwardOddsArchiveError(
            "The Odds Gap CSV is missing required columns: " + ", ".join(missing)
        )

    groups: dict[tuple[str, ...], dict[str, str]] = {}
    conflicts: set[tuple[str, ...]] = set()
    source_row_count = 0
    for row in reader:
        source_row_count += 1
        key = tuple(str(row.get(column) or "") for column in (
            "snapshot_ts",
            "sport",
            "away",
            "home",
            "commence_time",
            "market",
            "book",
            "line",
        ))
        side = str(row.get("side") or "").strip().lower()
        american = str(row.get("american_odds") or "").strip()
        sides = groups.setdefault(key, {})
        previous = sides.get(side)
        if previous is not None and previous != american:
            conflicts.add(key)
        else:
            sides[side] = american

    rejected: Counter[str] = Counter()
    unknown_names: set[str] = set()
    unmapped_count = 0
    candidates: list[dict[str, Any]] = []

    for key, sides in groups.items():
        snapshot_ts, sport, away_name, home_name, commence_time, market, book_name, line = key
        if key in conflicts:
            rejected["conflicting_side"] += 1
            continue
        if sport != "icehockey_nhl":
            rejected["wrong_sport"] += 1
            continue
        if market != "ml":
            rejected["not_moneyline"] += 1
            continue
        if line.strip():
            rejected["spread_line_present"] += 1
            continue
        if set(sides) != {"home", "away"}:
            if "home" not in sides or "away" not in sides:
                rejected["missing_side"] += 1
            else:
                rejected["not_moneyline"] += 1
            continue
        try:
            observed = format_utc(parse_aware_utc(snapshot_ts, label="observation_utc"))
            puck_drop = format_utc(
                parse_aware_utc(commence_time, label="scheduled_puck_drop_utc")
            )
        except ForwardOddsArchiveError as exc:
            rejected[_row_reason(str(exc))] += 1
            continue
        if observed >= puck_drop:
            rejected["post_puck"] += 1
            continue
        home_abbrev = ODDS_GAP_TEAM_TO_ABBREV.get(home_name)
        away_abbrev = ODDS_GAP_TEAM_TO_ABBREV.get(away_name)
        if home_abbrev is None or away_abbrev is None:
            if home_abbrev is None and home_name:
                unknown_names.add(home_name)
            if away_abbrev is None and away_name:
                unknown_names.add(away_name)
            unmapped_count += 1
            continue
        game = _match_game(
            supplied,
            home_team=home_abbrev,
            away_team=away_abbrev,
            scheduled_puck_drop_utc=puck_drop,
        )
        if game is None:
            unmapped_count += 1
            continue
        try:
            home_decimal, home_american = decimal_text_from_american(
                sides["home"],
                label="home",
            )
            away_decimal, away_american = decimal_text_from_american(
                sides["away"],
                label="away",
            )
            payload = {
                "nhl_game_id": game.nhl_game_id,
                "home_team": game.home_team,
                "away_team": game.away_team,
                "scheduled_puck_drop_utc": game.scheduled_puck_drop_utc,
                "book": book_name.strip(),
                "home_decimal_odds": home_decimal,
                "away_decimal_odds": away_decimal,
                "home_american_odds": home_american,
                "away_american_odds": away_american,
                "observation_utc": observed,
                "capture_method": ODDS_GAP_CAPTURE_METHOD,
                "provenance": (
                    f"{ODDS_GAP_ATTRIBUTION} book={book_name.strip()}. "
                    f"snapshot_ts={observed}."
                ),
            }
            validate_forward_odds_snapshot(payload)
        except ForwardOddsArchiveError as exc:
            rejected[_row_reason(str(exc))] += 1
            continue
        for dropped in DROPPED_COLUMNS:
            if dropped in payload:
                raise ForwardOddsArchiveError(
                    "Odds Gap assist tried to store a dropped market field"
                )
        candidates.append(payload)

    filtered_other_book = 0
    if book_filter is not None:
        kept = [payload for payload in candidates if payload["book"] == book_filter]
        filtered_other_book = len(candidates) - len(kept)
        candidates = kept

    unique: dict[tuple[str, str, str], dict[str, Any]] = {}
    duplicate_keys: set[tuple[str, str, str]] = set()
    for payload in candidates:
        logical = (payload["nhl_game_id"], payload["book"], payload["observation_utc"])
        previous = unique.get(logical)
        if previous is not None and previous != payload:
            duplicate_keys.add(logical)
        else:
            unique[logical] = payload
    if duplicate_keys:
        rejected["conflicting_side"] += len(duplicate_keys)
        candidates = [
            payload
            for payload in unique.values()
            if (payload["nhl_game_id"], payload["book"], payload["observation_utc"])
            not in duplicate_keys
        ]
    else:
        candidates = list(unique.values())

    collapsed = 0
    if not all_pregame_observations:
        latest: dict[tuple[str, str], dict[str, Any]] = {}
        for payload in candidates:
            logical = (payload["nhl_game_id"], payload["book"])
            current = latest.get(logical)
            if current is None or payload["observation_utc"] > current["observation_utc"]:
                if current is not None:
                    collapsed += 1
                latest[logical] = payload
            else:
                collapsed += 1
        candidates = list(latest.values())

    candidates.sort(
        key=lambda payload: (
            payload["observation_utc"],
            payload["nhl_game_id"],
            payload["book"],
        )
    )
    return OddsGapProposal(
        source_row_count=source_row_count,
        ready=tuple(candidates),
        unmapped_count=unmapped_count,
        rejected_count=sum(rejected.values()),
        collapsed_earlier_observation_count=collapsed,
        filtered_other_book_count=filtered_other_book,
        unknown_team_names=tuple(sorted(unknown_names)),
        rejected_reason_counts=tuple(sorted(rejected.items())),
        bounded_to_latest_per_game_book=not all_pregame_observations,
        book_filter=book_filter,
        fetch=fetch,
    )


def prepare_odds_gap_csv_file(
    path: Path | str,
    schedule: Sequence[Mapping[str, Any]],
    *,
    book: str | None = None,
    all_pregame_observations: bool = False,
) -> OddsGapProposal:
    """Read a local CSV. This does not contact The Odds Gap."""

    csv_path = Path(path)
    try:
        text = csv_path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise ForwardOddsArchiveError(
            f"Unreadable Odds Gap CSV {csv_path}"
        ) from exc
    return prepare_odds_gap_snapshots(
        text,
        schedule,
        book=book,
        all_pregame_observations=all_pregame_observations,
    )


def commit_odds_gap_proposal(
    proposal: OddsGapProposal,
    *,
    store: Any = None,
    confirm: bool = False,
) -> list[dict[str, Any]]:
    """Write ready snapshots only when the operator passes ``confirm=True``."""

    if not confirm:
        return []
    written: list[dict[str, Any]] = []
    for payload in proposal.ready:
        result = write_forward_odds_snapshot(payload, store=store)
        written.append(
            {
                "status": result.status,
                "written": result.written,
                "snapshot_sha256": result.snapshot_sha256,
                "nhl_game_id": result.envelope["snapshot"]["nhl_game_id"],
                "book": result.envelope["snapshot"]["book"],
                "observation_utc": result.envelope["snapshot"]["observation_utc"],
            }
        )
    return written
