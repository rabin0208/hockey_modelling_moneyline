"""Current NHL PlayNow moneylines from The Odds API v4.

This is the Issue #98 no-secret adapter. It requests one current
``icehockey_nhl`` head-to-head market from the Canadian region and accepts
only bookmaker key ``playnow_ca``. Another book is never renamed to PlayNow
or to Bet365. Bet365 stays unresolved.

The credential is read from the environment variable ``THE_ODDS_API_KEY``.
It is not committed, logged, or copied into snapshot provenance. Parsing a
saved response does not need a credential and does not open a network
connection.

NHL game ids come only from the operator schedule. Home and away must match
exactly one supplied game. Provider ``commence_time`` may differ from that
game's canonical ``scheduled_puck_drop_utc`` by at most
``COMMENCE_TIME_MATCH_TOLERANCE`` (±15 minutes, inclusive). The snapshot
stores the NHL scheduled time. Pre-puck eligibility uses that NHL time, not
the provider clock. Provider ``commence_time`` stays in provenance.

A quote is written through ``validate_forward_odds_snapshot`` /
``write_forward_odds_snapshot``. The archive still stores one book per snapshot.

``python -m hockey_moneyline.the_odds_api_playnow smoke`` performs exactly one
provider GET once that variable is set. This module does not retry, paginate,
call the historical endpoint, activate a paid plan, or schedule itself.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from hockey_moneyline.errors import ForwardOddsArchiveError
from hockey_moneyline.forward_odds_archive import (
    format_utc,
    parse_aware_utc,
    parse_decimal_odds,
    validate_forward_odds_snapshot,
    write_forward_odds_snapshot,
)
from hockey_moneyline.odds_gap_assist import (
    ODDS_GAP_TEAM_TO_ABBREV,
    SuppliedNhlGame,
    parse_supplied_schedule,
)

THE_ODDS_API_KEY_ENV = "THE_ODDS_API_KEY"
ODDS_API_HOST = "api.the-odds-api.com"
ODDS_API_PATH = "/v4/sports/icehockey_nhl/odds/"
ODDS_API_ENDPOINT = f"https://{ODDS_API_HOST}{ODDS_API_PATH}"
SPORT_KEY = "icehockey_nhl"
REGION = "ca"
MARKET = "h2h"
ODDS_FORMAT = "decimal"
PLAYNOW_BOOKMAKER_KEY = "playnow_ca"
# First free-key trial. A later book is a separate named snapshot and needs
# an explicit allowlist change. This request builder does not take a book argument.
TRIAL_BOOKMAKER_KEYS = frozenset({PLAYNOW_BOOKMAKER_KEY})
CAPTURE_METHOD = "the_odds_api_v4_current_h2h"
# Inclusive bound. The approved free-key smoke on head 98c478d saw common
# ~10-minute Odds API commence_time drift against the NHL scheduled start.
# ±10 minutes would sit on that observed magnitude and could reject the same
# drift. ±15 minutes covers it and stays far inside the gap between distinct
# NHL starts. The snapshot time remains the NHL scheduled puck drop.
COMMENCE_TIME_MATCH_TOLERANCE = timedelta(minutes=15)
COMMENCE_TIME_MATCH_TOLERANCE_LABEL = "PT15M"
USER_AGENT = (
    "hockey-moneyline-issue-98-odds-api/1.0 "
    "(+https://github.com/EduardoRasanmar/hockey_moneyline)"
)
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_EVENTS = 4096
READY = "READY"
NO_DATA = "NO_DATA"
BLOCKED = "BLOCKED"

# Same explicit display-name table as the Odds Gap assist. Unknown names stay
# unmapped. Matching is exact after stripping surrounding whitespace.
NHL_DISPLAY_NAME_TO_ABBREV = ODDS_GAP_TEAM_TO_ABBREV

_USAGE_HEADER_NAMES = (
    ("x-requests-remaining", "requests_remaining"),
    ("x-requests-used", "requests_used"),
    ("x-requests-last", "requests_last"),
)
_APPROVED_QUERY_KEYS = frozenset(
    {"regions", "markets", "oddsFormat", "bookmakers", "apiKey"}
)


class _RefuseRedirect(HTTPRedirectHandler):
    """Stop before a redirect can carry the query string anywhere else."""

    def http_error_302(self, req, fp, code, msg, headers):  # noqa: ANN001
        raise ForwardOddsArchiveError(
            f"The Odds API returned HTTP {code}; redirects are not followed"
        )

    http_error_301 = http_error_303 = http_error_307 = http_error_308 = http_error_302


@dataclass(frozen=True, slots=True)
class OddsApiUsage:
    """Response usage headers. The credential is not a field."""

    requests_remaining: str | None
    requests_used: str | None
    requests_last: str | None
    http_status: int
    request_count: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_count": self.request_count,
            "http_status": self.http_status,
            "x-requests-remaining": self.requests_remaining,
            "x-requests-used": self.requests_used,
            "x-requests-last": self.requests_last,
            "credential_logged": False,
            "paid_plan_activated": False,
            "historical_endpoint_called": False,
        }


@dataclass(frozen=True, slots=True)
class CurrentBookQuote:
    """One provider event after PlayNow / NHL identity checks."""

    status: str
    reason: str | None
    nhl_game_id: str | None = None
    home_team: str | None = None
    away_team: str | None = None
    scheduled_puck_drop_utc: str | None = None
    book: str | None = None
    observation_utc: str | None = None
    home_decimal_odds: str | None = None
    away_decimal_odds: str | None = None
    capture_method: str | None = None
    provenance: str | None = None
    snapshot_sha256: str | None = None
    provider_event_id: str | None = None
    ignored_bookmaker_keys: tuple[str, ...] = ()

    def operator_payload(self) -> dict[str, Any]:
        if self.status != READY:
            raise ForwardOddsArchiveError("Only a READY PlayNow quote can be written")
        return {
            "nhl_game_id": self.nhl_game_id,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "scheduled_puck_drop_utc": self.scheduled_puck_drop_utc,
            "book": self.book,
            "home_decimal_odds": self.home_decimal_odds,
            "away_decimal_odds": self.away_decimal_odds,
            "observation_utc": self.observation_utc,
            "capture_method": self.capture_method,
            "provenance": self.provenance,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "nhl_game_id": self.nhl_game_id,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "scheduled_puck_drop_utc": self.scheduled_puck_drop_utc,
            "book": self.book,
            "observation_utc": self.observation_utc,
            "home_decimal_odds": self.home_decimal_odds,
            "away_decimal_odds": self.away_decimal_odds,
            "capture_method": self.capture_method,
            "provenance": self.provenance,
            "snapshot_sha256": self.snapshot_sha256,
            "provider_event_id": self.provider_event_id,
            "ignored_bookmaker_keys": list(self.ignored_bookmaker_keys),
        }


@dataclass(frozen=True, slots=True)
class PlayNowAcquisition:
    quotes: tuple[CurrentBookQuote, ...]
    schedule_game_ids_without_ready_quote: tuple[str, ...]
    usage: OddsApiUsage | None = None
    request_count: int = 0

    @property
    def ready(self) -> tuple[CurrentBookQuote, ...]:
        return tuple(quote for quote in self.quotes if quote.status == READY)

    def to_dict(self) -> dict[str, Any]:
        reason_counts: dict[str, int] = {}
        for quote in self.quotes:
            if quote.reason:
                reason_counts[quote.reason] = reason_counts.get(quote.reason, 0) + 1
        return {
            "provider": "the_odds_api_v4",
            "sport": SPORT_KEY,
            "region": REGION,
            "market": MARKET,
            "odds_format": ODDS_FORMAT,
            "accepted_bookmaker_keys": sorted(TRIAL_BOOKMAKER_KEYS),
            "bet365_accepted": False,
            "paid_plan_activated": False,
            "historical_endpoint_called": False,
            "scheduled_capture": False,
            "credential_logged": False,
            "request_count": self.request_count,
            "capture_method": CAPTURE_METHOD,
            "commence_time_match_tolerance": COMMENCE_TIME_MATCH_TOLERANCE_LABEL,
            "ready_count": len(self.ready),
            "quote_count": len(self.quotes),
            "reason_counts": [
                {"reason": reason, "count": reason_counts[reason]}
                for reason in sorted(reason_counts)
            ],
            "schedule_game_ids_without_ready_quote": list(
                self.schedule_game_ids_without_ready_quote
            ),
            "quotes": [quote.to_dict() for quote in self.quotes],
            "usage": None if self.usage is None else self.usage.to_dict(),
            "written": [],
        }


def redact_secret(text: str, secret: str | None) -> str:
    """Remove a credential from text that might be printed."""

    if not secret or not text:
        return text
    encoded = quote(secret, safe="")
    redacted = text.replace(secret, "REDACTED")
    if encoded and encoded != secret:
        redacted = redacted.replace(encoded, "REDACTED")
    return redacted


def read_the_odds_api_key(
    environ: Mapping[str, str] | None = None,
) -> str:
    """Return the env credential. Empty and missing both fail closed."""

    source = os.environ if environ is None else environ
    if THE_ODDS_API_KEY_ENV not in source:
        raise ForwardOddsArchiveError(
            "THE_ODDS_API_KEY is absent; refusing to call The Odds API"
        )
    raw = source.get(THE_ODDS_API_KEY_ENV)
    if not isinstance(raw, str) or not raw.strip():
        raise ForwardOddsArchiveError(
            "THE_ODDS_API_KEY is absent; refusing to call The Odds API"
        )
    key = raw.strip()
    if any(character.isspace() for character in key):
        raise ForwardOddsArchiveError(
            "THE_ODDS_API_KEY is malformed; refusing to call The Odds API"
        )
    return key


def build_playnow_odds_url(api_key: str) -> str:
    """Build the single approved current-odds URL. The key stays in the query."""

    if not api_key or any(character.isspace() for character in api_key):
        raise ForwardOddsArchiveError(
            "THE_ODDS_API_KEY is malformed; refusing to call The Odds API"
        )
    query = "&".join(
        (
            f"regions={quote(REGION, safe='')}",
            f"markets={quote(MARKET, safe='')}",
            f"oddsFormat={quote(ODDS_FORMAT, safe='')}",
            f"bookmakers={quote(PLAYNOW_BOOKMAKER_KEY, safe='')}",
            f"apiKey={quote(api_key, safe='')}",
        )
    )
    return f"{ODDS_API_ENDPOINT}?{query}"


def public_odds_url() -> str:
    """Approved endpoint with the credential parameter removed."""

    return (
        f"{ODDS_API_ENDPOINT}?regions={REGION}&markets={MARKET}"
        f"&oddsFormat={ODDS_FORMAT}&bookmakers={PLAYNOW_BOOKMAKER_KEY}"
    )


def _assert_approved_odds_url(url: str, api_key: str) -> None:
    parts = urlsplit(url)
    if parts.scheme != "https" or parts.hostname != ODDS_API_HOST:
        raise ForwardOddsArchiveError("The Odds API request host is not approved")
    if parts.path not in {ODDS_API_PATH, ODDS_API_PATH.rstrip("/")}:
        raise ForwardOddsArchiveError("The Odds API request path is not approved")
    if "historical" in parts.path:
        raise ForwardOddsArchiveError(
            "The Odds API historical endpoint is not approved"
        )
    pairs = []
    if parts.query:
        for item in parts.query.split("&"):
            if "=" not in item:
                raise ForwardOddsArchiveError("The Odds API query is not approved")
            name, value = item.split("=", 1)
            pairs.append((name, value))
    names = [name for name, _value in pairs]
    if set(names) != _APPROVED_QUERY_KEYS or len(names) != len(_APPROVED_QUERY_KEYS):
        raise ForwardOddsArchiveError("The Odds API query is not approved")
    values = dict(pairs)
    if values["regions"] != REGION or values["markets"] != MARKET:
        raise ForwardOddsArchiveError("The Odds API query is not approved")
    if values["oddsFormat"] != ODDS_FORMAT:
        raise ForwardOddsArchiveError("The Odds API query is not approved")
    if values["bookmakers"] != PLAYNOW_BOOKMAKER_KEY:
        raise ForwardOddsArchiveError(
            "The Odds API request may only ask for bookmaker playnow_ca"
        )
    if values["apiKey"] != quote(api_key, safe=""):
        raise ForwardOddsArchiveError("The Odds API query is not approved")


def _header(headers: Any, name: str) -> str | None:
    if headers is None or not hasattr(headers, "get"):
        return None
    value = headers.get(name)
    if value in (None, ""):
        value = headers.get(name.lower())
    if value in (None, ""):
        return None
    return str(value)


def _usage_from_headers(headers: Any, *, http_status: int) -> OddsApiUsage:
    values = {
        field: _header(headers, header_name)
        for header_name, field in _USAGE_HEADER_NAMES
    }
    return OddsApiUsage(
        requests_remaining=values["requests_remaining"],
        requests_used=values["requests_used"],
        requests_last=values["requests_last"],
        http_status=http_status,
        request_count=1,
    )


def _default_open(request: Request, timeout: float) -> Any:
    return build_opener(_RefuseRedirect).open(request, timeout=timeout)


def fetch_playnow_current_odds(
    *,
    opener: Callable[..., Any] | None = None,
    timeout: float = 60,
    environ: Mapping[str, str] | None = None,
) -> tuple[Any, OddsApiUsage]:
    """Perform exactly one current-odds GET. Absent credentials do not call out.

    ``opener`` exists so tests can supply a fake. The command-line smoke path
    leaves it unset and uses the no-redirect opener.
    """

    api_key = read_the_odds_api_key(environ)
    url = build_playnow_odds_url(api_key)
    _assert_approved_odds_url(url, api_key)
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        },
        method="GET",
    )
    for name, value in request.header_items():
        lowered = name.lower()
        if lowered in {"authorization", "x-api-key"} or lowered.endswith("-key"):
            raise ForwardOddsArchiveError(
                "The Odds API credential must not be sent as a header"
            )
        if api_key and api_key in value:
            raise ForwardOddsArchiveError(
                "The Odds API credential must not be copied into a header"
            )
    open_request = _default_open if opener is None else opener
    headers: Any = None
    status = 0
    try:
        with open_request(request, timeout=timeout) as response:
            status = int(getattr(response, "status", None) or getattr(response, "code", 0) or 0)
            headers = getattr(response, "headers", None)
            if status != 200:
                usage = _usage_from_headers(headers, http_status=status)
                raise ForwardOddsArchiveError(
                    f"The Odds API request failed with HTTP {status}. "
                    f"x-requests-remaining={usage.requests_remaining}. "
                    f"x-requests-used={usage.requests_used}. "
                    f"x-requests-last={usage.requests_last}. "
                    "No retry is attempted."
                )
            payload = response.read(MAX_RESPONSE_BYTES + 1)
    except ForwardOddsArchiveError:
        raise
    except HTTPError as exc:
        usage = _usage_from_headers(getattr(exc, "headers", None), http_status=int(exc.code))
        raise ForwardOddsArchiveError(
            f"The Odds API request failed with HTTP {exc.code}. "
            f"x-requests-remaining={usage.requests_remaining}. "
            f"x-requests-used={usage.requests_used}. "
            f"x-requests-last={usage.requests_last}. "
            "No retry is attempted."
        ) from None
    except URLError:
        raise ForwardOddsArchiveError(
            "The Odds API request could not be read. No retry is attempted."
        ) from None
    except OSError:
        raise ForwardOddsArchiveError(
            "The Odds API request could not be read. No retry is attempted."
        ) from None
    if status != 200:
        raise ForwardOddsArchiveError(
            f"The Odds API request failed with HTTP {status}. No retry is attempted."
        )
    if not isinstance(payload, (bytes, bytearray)):
        raise ForwardOddsArchiveError("The Odds API response did not return bytes")
    if len(payload) > MAX_RESPONSE_BYTES:
        raise ForwardOddsArchiveError(
            "The Odds API response exceeded the bounded size cap"
        )
    usage = _usage_from_headers(headers, http_status=status)
    try:
        text = bytes(payload).decode("utf-8")
    except UnicodeError:
        raise ForwardOddsArchiveError("The Odds API response was not valid UTF-8") from None
    return _load_json(text), usage


def _load_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise ForwardOddsArchiveError(
            "The Odds API response was not valid JSON"
        ) from None


def _is_bet365(key: str, title: str | None) -> bool:
    blob = f"{key} {title or ''}"
    compact = "".join(character for character in blob.casefold() if character.isalnum())
    return "bet365" in compact


def _quote(
    *,
    status: str,
    reason: str | None,
    provider_event_id: str | None = None,
    ignored_bookmaker_keys: tuple[str, ...] = (),
    **fields: Any,
) -> CurrentBookQuote:
    return CurrentBookQuote(
        status=status,
        reason=reason,
        provider_event_id=provider_event_id,
        ignored_bookmaker_keys=tuple(sorted(ignored_bookmaker_keys)),
        **fields,
    )


def _event_id(event: Mapping[str, Any]) -> str | None:
    value = event.get("id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _parse_timestamp(value: Any, *, label: str) -> tuple[str | None, str | None]:
    """Return ``(formatted_utc, failure_reason)``."""

    try:
        return format_utc(parse_aware_utc(value, label=label)), None
    except ForwardOddsArchiveError as exc:
        text = str(exc).lower()
        if "timezone-aware" in text:
            if isinstance(value, str) and value.strip() and "T" in value and "+" not in value and not value.endswith(("Z", "z")):
                return None, "naive_timestamp"
            return None, "malformed_event"
        return None, "malformed_event"


def _within_commence_tolerance(commence_time: str, scheduled_puck_drop_utc: str) -> bool:
    """True when the provider clock is within the inclusive NHL match bound."""

    commence = parse_aware_utc(commence_time, label="commence_time")
    scheduled = parse_aware_utc(
        scheduled_puck_drop_utc,
        label="scheduled_puck_drop_utc",
    )
    return abs(commence - scheduled) <= COMMENCE_TIME_MATCH_TOLERANCE


def _match_games(
    schedule: Sequence[SuppliedNhlGame],
    *,
    home_team: str,
    away_team: str,
    commence_time: str,
) -> tuple[SuppliedNhlGame, ...]:
    """Exact home/away identity, plus the documented commence-time bound.

    More than one supplied game inside the window is returned so the caller
    can fail closed. A swapped home/away pair does not match.
    """

    return tuple(
        game
        for game in schedule
        if game.home_team == home_team
        and game.away_team == away_team
        and _within_commence_tolerance(commence_time, game.scheduled_puck_drop_utc)
    )


def _provenance(
    *,
    title: str,
    provider_event_id: str,
    commence_time: str,
    market_last_update: str,
    bookmaker_last_update: str,
    usage: OddsApiUsage | None,
) -> str:
    parts = [
        "The Odds API v4 current NHL h2h.",
        f"sport={SPORT_KEY}",
        f"region={REGION}",
        f"market={MARKET}",
        f"odds_format={ODDS_FORMAT}",
        f"bookmaker_key={PLAYNOW_BOOKMAKER_KEY}",
        f"bookmaker_title={title}",
        f"provider_event_id={provider_event_id}",
        f"commence_time={commence_time}",
        f"commence_time_match_tolerance={COMMENCE_TIME_MATCH_TOLERANCE_LABEL}",
        f"market_last_update={market_last_update}",
        f"bookmaker_last_update={bookmaker_last_update}",
        "endpoint=/v4/sports/icehockey_nhl/odds/",
        "historical_endpoint=false",
        "bet365_relabel=false",
    ]
    if usage is not None:
        parts.append(f"x_requests_remaining={usage.requests_remaining}")
        parts.append(f"x_requests_used={usage.requests_used}")
        parts.append(f"x_requests_last={usage.requests_last}")
    return " ".join(parts)


def _playnow_bookmaker(
    event: Mapping[str, Any],
) -> tuple[Mapping[str, Any] | None, tuple[str, ...], str | None]:
    """Return the single PlayNow book, ignored keys, and a failure reason."""

    bookmakers = event.get("bookmakers")
    if bookmakers is None:
        return None, (), "absent_playnow_book"
    if not isinstance(bookmakers, list):
        return None, (), "malformed_event"
    playnow: list[Mapping[str, Any]] = []
    ignored: list[str] = []
    saw_bet365 = False
    for entry in bookmakers:
        if not isinstance(entry, Mapping):
            return None, (), "malformed_event"
        key = entry.get("key")
        title = entry.get("title")
        title_text = title if isinstance(title, str) else None
        if not isinstance(key, str) or not key.strip():
            return None, (), "malformed_event"
        book_key = key.strip()
        if _is_bet365(book_key, title_text):
            saw_bet365 = True
            ignored.append(book_key)
            continue
        if book_key == PLAYNOW_BOOKMAKER_KEY:
            playnow.append(entry)
            continue
        ignored.append(book_key)
    ignored_keys = tuple(sorted(set(ignored)))
    if len(playnow) > 1:
        return None, ignored_keys, "ambiguous_playnow_book"
    if not playnow:
        if saw_bet365:
            return None, ignored_keys, "bet365_not_accepted"
        return None, ignored_keys, "absent_playnow_book"
    title = playnow[0].get("title")
    title_text = title.strip() if isinstance(title, str) else None
    if _is_bet365(PLAYNOW_BOOKMAKER_KEY, title_text):
        return None, ignored_keys, "bet365_not_accepted"
    return playnow[0], ignored_keys, None


def _h2h_market(
    bookmaker: Mapping[str, Any],
) -> tuple[Mapping[str, Any] | None, str | None]:
    markets = bookmaker.get("markets")
    if not isinstance(markets, list):
        return None, "malformed_event"
    h2h = [
        market
        for market in markets
        if isinstance(market, Mapping) and market.get("key") == MARKET
    ]
    if any(not isinstance(market, Mapping) for market in markets):
        return None, "malformed_event"
    if len(h2h) > 1:
        return None, "ambiguous_h2h_market"
    if not h2h:
        return None, "absent_h2h_market"
    return h2h[0], None


def _outcome_prices(
    market: Mapping[str, Any],
    *,
    home_name: str,
    away_name: str,
) -> tuple[dict[str, Any] | None, str | None]:
    outcomes = market.get("outcomes")
    if not isinstance(outcomes, list):
        return None, "malformed_event"
    prices: dict[str, Any] = {}
    for outcome in outcomes:
        if not isinstance(outcome, Mapping):
            return None, "malformed_event"
        if "point" in outcome and outcome.get("point") is not None:
            return None, "not_two_sided_moneyline"
        name = outcome.get("name")
        if not isinstance(name, str) or not name.strip():
            return None, "malformed_event"
        label = name.strip()
        if label in prices:
            return None, "not_two_sided_moneyline"
        if "price" not in outcome or outcome.get("price") in (None, ""):
            return None, "one_sided_market"
        prices[label] = outcome.get("price")
    if not prices:
        return None, "one_sided_market"
    wanted = {home_name, away_name}
    if set(prices) != wanted:
        if len(prices) < 2 and (home_name in prices or away_name in prices):
            return None, "one_sided_market"
        if home_name in prices and away_name in prices:
            return None, "not_two_sided_moneyline"
        if len(prices) == 1:
            return None, "one_sided_market"
        return None, "not_two_sided_moneyline"
    return prices, None


def _quote_from_event(
    event: Any,
    schedule: Sequence[SuppliedNhlGame],
    *,
    usage: OddsApiUsage | None,
) -> CurrentBookQuote:
    if not isinstance(event, Mapping):
        return _quote(status=BLOCKED, reason="malformed_event")
    provider_event_id = _event_id(event)
    if provider_event_id is None:
        return _quote(status=BLOCKED, reason="malformed_event")
    sport = event.get("sport_key")
    if sport != SPORT_KEY:
        return _quote(
            status=BLOCKED,
            reason="wrong_sport",
            provider_event_id=provider_event_id,
        )
    home_name = event.get("home_team")
    away_name = event.get("away_team")
    if not isinstance(home_name, str) or not isinstance(away_name, str):
        return _quote(
            status=BLOCKED,
            reason="malformed_event",
            provider_event_id=provider_event_id,
        )
    home_name = home_name.strip()
    away_name = away_name.strip()
    if not home_name or not away_name or home_name == away_name:
        return _quote(
            status=BLOCKED,
            reason="malformed_event",
            provider_event_id=provider_event_id,
        )
    bookmaker, ignored, book_reason = _playnow_bookmaker(event)
    if book_reason is not None or bookmaker is None:
        status = BLOCKED if book_reason == "malformed_event" or book_reason == "ambiguous_playnow_book" else NO_DATA
        if book_reason == "bet365_not_accepted":
            status = NO_DATA
        return _quote(
            status=status,
            reason=book_reason or "absent_playnow_book",
            provider_event_id=provider_event_id,
            ignored_bookmaker_keys=ignored,
        )
    title_value = bookmaker.get("title")
    if title_value is None:
        title = ""
    elif isinstance(title_value, str):
        title = title_value.strip()
    else:
        return _quote(
            status=BLOCKED,
            reason="malformed_event",
            provider_event_id=provider_event_id,
            ignored_bookmaker_keys=ignored,
        )
    market, market_reason = _h2h_market(bookmaker)
    if market_reason is not None or market is None:
        status = BLOCKED
        return _quote(
            status=status,
            reason=market_reason,
            provider_event_id=provider_event_id,
            ignored_bookmaker_keys=ignored,
        )
    prices, price_reason = _outcome_prices(
        market,
        home_name=home_name,
        away_name=away_name,
    )
    if price_reason is not None or prices is None:
        return _quote(
            status=BLOCKED,
            reason=price_reason,
            provider_event_id=provider_event_id,
            ignored_bookmaker_keys=ignored,
        )
    home_abbrev = NHL_DISPLAY_NAME_TO_ABBREV.get(home_name)
    away_abbrev = NHL_DISPLAY_NAME_TO_ABBREV.get(away_name)
    if home_abbrev is None or away_abbrev is None or home_abbrev == away_abbrev:
        return _quote(
            status=NO_DATA,
            reason="unknown_team",
            provider_event_id=provider_event_id,
            ignored_bookmaker_keys=ignored,
        )
    commence, commence_reason = _parse_timestamp(
        event.get("commence_time"),
        label="commence_time",
    )
    if commence_reason is not None or commence is None:
        return _quote(
            status=BLOCKED,
            reason=commence_reason,
            provider_event_id=provider_event_id,
            ignored_bookmaker_keys=ignored,
        )
    bookmaker_last_update_raw = bookmaker.get("last_update")
    bookmaker_last_update = ""
    if bookmaker_last_update_raw not in (None, ""):
        formatted, book_time_reason = _parse_timestamp(
            bookmaker_last_update_raw,
            label="bookmaker_last_update",
        )
        if book_time_reason is not None or formatted is None:
            return _quote(
                status=BLOCKED,
                reason=book_time_reason,
                provider_event_id=provider_event_id,
                ignored_bookmaker_keys=ignored,
            )
        bookmaker_last_update = formatted
    observed, observed_reason = _parse_timestamp(
        market.get("last_update"),
        label="observation_utc",
    )
    if observed_reason is not None or observed is None:
        raw_observation = market.get("last_update")
        observation_missing = raw_observation in (None, "") or (
            isinstance(raw_observation, str) and not raw_observation.strip()
        )
        reason = "missing_observation" if observation_missing else observed_reason
        return _quote(
            status=BLOCKED,
            reason=reason,
            provider_event_id=provider_event_id,
            ignored_bookmaker_keys=ignored,
        )
    matches = _match_games(
        schedule,
        home_team=home_abbrev,
        away_team=away_abbrev,
        commence_time=commence,
    )
    if len(matches) > 1:
        return _quote(
            status=BLOCKED,
            reason="ambiguous_game_identity",
            provider_event_id=provider_event_id,
            ignored_bookmaker_keys=ignored,
            home_team=home_abbrev,
            away_team=away_abbrev,
            observation_utc=observed,
        )
    if len(matches) == 0:
        return _quote(
            status=NO_DATA,
            reason="unmapped_game",
            provider_event_id=provider_event_id,
            ignored_bookmaker_keys=ignored,
            home_team=home_abbrev,
            away_team=away_abbrev,
            observation_utc=observed,
        )
    game = matches[0]
    # Eligibility uses the NHL scheduled start. A later provider commence_time
    # must not accept an observation that is already at or after puck drop.
    if observed >= game.scheduled_puck_drop_utc:
        return _quote(
            status=BLOCKED,
            reason="post_puck_observation",
            provider_event_id=provider_event_id,
            ignored_bookmaker_keys=ignored,
            nhl_game_id=game.nhl_game_id,
            home_team=game.home_team,
            away_team=game.away_team,
            observation_utc=observed,
            scheduled_puck_drop_utc=game.scheduled_puck_drop_utc,
        )
    try:
        home_decimal = parse_decimal_odds(prices[home_name], label="home")
        away_decimal = parse_decimal_odds(prices[away_name], label="away")
    except ForwardOddsArchiveError:
        return _quote(
            status=BLOCKED,
            reason="invalid_odds",
            provider_event_id=provider_event_id,
            ignored_bookmaker_keys=ignored,
            nhl_game_id=game.nhl_game_id,
            home_team=game.home_team,
            away_team=game.away_team,
            scheduled_puck_drop_utc=game.scheduled_puck_drop_utc,
            observation_utc=observed,
        )
    provenance = _provenance(
        title=title,
        provider_event_id=provider_event_id,
        commence_time=commence,
        market_last_update=observed,
        bookmaker_last_update=bookmaker_last_update,
        usage=usage,
    )
    payload = {
        "nhl_game_id": game.nhl_game_id,
        "home_team": game.home_team,
        "away_team": game.away_team,
        "scheduled_puck_drop_utc": game.scheduled_puck_drop_utc,
        "book": PLAYNOW_BOOKMAKER_KEY,
        "home_decimal_odds": home_decimal,
        "away_decimal_odds": away_decimal,
        "observation_utc": observed,
        "capture_method": CAPTURE_METHOD,
        "provenance": provenance,
    }
    try:
        validated = validate_forward_odds_snapshot(payload)
    except ForwardOddsArchiveError:
        return _quote(
            status=BLOCKED,
            reason="invalid_snapshot",
            provider_event_id=provider_event_id,
            ignored_bookmaker_keys=ignored,
        )
    return _quote(
        status=READY,
        reason=None,
        provider_event_id=provider_event_id,
        ignored_bookmaker_keys=ignored,
        nhl_game_id=validated.nhl_game_id,
        home_team=validated.home_team,
        away_team=validated.away_team,
        scheduled_puck_drop_utc=validated.scheduled_puck_drop_utc,
        book=validated.book,
        observation_utc=validated.observation_utc,
        home_decimal_odds=validated.home_decimal_odds,
        away_decimal_odds=validated.away_decimal_odds,
        capture_method=validated.capture_method,
        provenance=validated.provenance,
        snapshot_sha256=validated.snapshot_sha256,
    )


def _collapse_ambiguous_games(
    quotes: list[CurrentBookQuote],
) -> tuple[CurrentBookQuote, ...]:
    """Two different current quotes for one NHL game are not a choice."""

    grouped: dict[str, list[int]] = {}
    for index, quote in enumerate(quotes):
        if quote.status == READY and quote.nhl_game_id:
            grouped.setdefault(quote.nhl_game_id, []).append(index)
    for game_id, indexes in grouped.items():
        payloads = [quotes[index].operator_payload() for index in indexes]
        if len(payloads) == 1:
            continue
        if all(payload == payloads[0] for payload in payloads[1:]):
            for index in indexes[1:]:
                quotes[index] = _quote(
                    status=NO_DATA,
                    reason="duplicate_provider_event",
                    provider_event_id=quotes[index].provider_event_id,
                    ignored_bookmaker_keys=quotes[index].ignored_bookmaker_keys,
                    nhl_game_id=game_id,
                )
            continue
        for index in indexes:
            current = quotes[index]
            quotes[index] = _quote(
                status=BLOCKED,
                reason="ambiguous_provider_event",
                provider_event_id=current.provider_event_id,
                ignored_bookmaker_keys=current.ignored_bookmaker_keys,
                nhl_game_id=game_id,
                home_team=current.home_team,
                away_team=current.away_team,
                scheduled_puck_drop_utc=current.scheduled_puck_drop_utc,
                observation_utc=current.observation_utc,
            )
    return tuple(quotes)


def prepare_playnow_snapshots(
    payload: Any,
    schedule: Sequence[Mapping[str, Any]] | Sequence[SuppliedNhlGame],
    *,
    usage: OddsApiUsage | None = None,
) -> PlayNowAcquisition:
    """Map one decoded Odds API response. This does not write and does not fetch."""

    if isinstance(schedule, Sequence) and schedule and isinstance(schedule[0], SuppliedNhlGame):
        supplied = tuple(schedule)  # type: ignore[arg-type]
    else:
        supplied = parse_supplied_schedule(schedule)  # type: ignore[arg-type]
    if isinstance(payload, (str, bytes)) or not isinstance(payload, list):
        raise ForwardOddsArchiveError(
            "The Odds API response must be a JSON list of current events"
        )
    if len(payload) > MAX_EVENTS:
        raise ForwardOddsArchiveError(
            "The Odds API response listed more events than the current-NHL bound"
        )
    quotes = [
        _quote_from_event(event, supplied, usage=usage)
        for event in payload
    ]
    collapsed = _collapse_ambiguous_games(list(quotes))
    ready_ids = {quote.nhl_game_id for quote in collapsed if quote.status == READY}
    missing = tuple(
        game.nhl_game_id for game in supplied if game.nhl_game_id not in ready_ids
    )
    request_count = 0 if usage is None else usage.request_count
    return PlayNowAcquisition(
        quotes=collapsed,
        schedule_game_ids_without_ready_quote=missing,
        usage=usage,
        request_count=request_count,
    )


def commit_playnow_quotes(
    acquisition: PlayNowAcquisition,
    *,
    store: Any = None,
    confirm: bool = False,
) -> list[dict[str, Any]]:
    """Write READY quotes only when the operator passes ``confirm=True``."""

    if not confirm:
        return []
    written: list[dict[str, Any]] = []
    for quote in acquisition.ready:
        result = write_forward_odds_snapshot(quote.operator_payload(), store=store)
        if result.snapshot_sha256 != quote.snapshot_sha256:
            raise ForwardOddsArchiveError(
                "Written forward-odds snapshot hash did not match the validated quote"
            )
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


def _load_schedule(path: Path) -> list[Mapping[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ForwardOddsArchiveError(
            f"Unreadable NHL schedule identity file {path.name}"
        ) from exc
    if not isinstance(payload, list):
        raise ForwardOddsArchiveError(
            "NHL schedule identity file must be a JSON list; refusing to invent NHL ids"
        )
    return payload


def _report(acquisition: PlayNowAcquisition, written: list[dict[str, Any]]) -> dict[str, Any]:
    report = acquisition.to_dict()
    report["written"] = written
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Issue #98 The Odds API v4 PlayNow adapter. "
            "parse reads a local fixture. smoke performs one current-odds GET "
            "when THE_ODDS_API_KEY is set. Nothing is written unless --confirm is present."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    parse = sub.add_parser("parse", help="Map a local Odds API JSON fixture. No network.")
    parse.add_argument("--fixture", type=Path, required=True)
    parse.add_argument("--schedule", type=Path, required=True)
    parse.add_argument("--confirm", action="store_true")
    parse.add_argument("--store", type=Path, default=None)

    smoke = sub.add_parser(
        "smoke",
        help=(
            "One current NHL PlayNow request when THE_ODDS_API_KEY is set. "
            "Does not retry, schedule, or write unless --confirm is present."
        ),
    )
    smoke.add_argument("--schedule", type=Path, required=True)
    smoke.add_argument("--confirm", action="store_true")
    smoke.add_argument("--store", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    secret: str | None = None
    try:
        schedule = _load_schedule(args.schedule)
        if args.command == "smoke" and not schedule:
            raise ForwardOddsArchiveError(
                "NHL schedule identity is empty; refusing the provider request"
            )
        if args.command == "parse":
            try:
                raw = args.fixture.read_text(encoding="utf-8")
            except OSError as exc:
                raise ForwardOddsArchiveError(
                    f"Unreadable Odds API fixture {args.fixture.name}"
                ) from exc
            acquisition = prepare_playnow_snapshots(_load_json(raw), schedule)
        elif args.command == "smoke":
            secret = read_the_odds_api_key()
            payload, usage = fetch_playnow_current_odds()
            acquisition = prepare_playnow_snapshots(payload, schedule, usage=usage)
        else:
            raise ForwardOddsArchiveError("Unknown Odds API command")
        written = commit_playnow_quotes(
            acquisition,
            store=args.store,
            confirm=bool(args.confirm),
        )
        report = _report(acquisition, written)
        rendered = redact_secret(json.dumps(report, indent=2, sort_keys=True), secret)
        print(rendered)
        return 0
    except ForwardOddsArchiveError as exc:
        print("ERROR: " + redact_secret(str(exc), secret))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
