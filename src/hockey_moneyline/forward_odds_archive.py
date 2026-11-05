"""Append-only local store for forward odds snapshots.

One immutable snapshot is exactly one NHL game, one sportsbook/source, and one
observation. The store is a sibling of the Issue #55 prediction-record file.
It does not read or write ``pregame_prediction_records.jsonl`` and it does not
store model probabilities, market-implied probabilities, edge, recommendation
state, threshold, stake, or game results.

Issue #84 introduced the manual store. Issue #87 is the canonical owner of the
same path and the same snapshot semantics. Issue #85 (the daily decision
cockpit) should consume ``read_validated_snapshots``,
``validate_forward_odds_snapshot``, and ``write_forward_odds_snapshot`` from
this module instead of reimplementing odds validation.

This module has no sportsbook client, scraper, credential, schedule, or wager
placement. The optional Odds Gap assist lives in ``odds_gap_assist`` and still
writes through this validator.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterator, Mapping

from hockey_moneyline.errors import ForwardOddsArchiveError

SCHEMA_VERSION = 1
SNAPSHOT_KIND = "forward_odds_snapshot"
CAPTURE_MODE = "manual_explicit"
STATUS_WRITTEN = "written"
STATUS_EXISTING_IDENTICAL = "existing_identical"
DEFAULT_STORE_RELATIVE_PATH = (
    "data/derived/odds_archive/forward/forward_odds_snapshots.jsonl"
)
PREDICTION_RECORD_FILENAME = "pregame_prediction_records.jsonl"
PREDICTION_RECORD_DIRECTORY = "prediction_records"

OPERATOR_FIELDS = (
    "nhl_game_id",
    "home_team",
    "away_team",
    "scheduled_puck_drop_utc",
    "book",
    "home_decimal_odds",
    "away_decimal_odds",
    "observation_utc",
    "capture_method",
    "provenance",
)
OPTIONAL_OPERATOR_FIELDS = (
    "home_american_odds",
    "away_american_odds",
)
SNAPSHOT_BODY_FIELDS = (
    "schema_version",
    "kind",
    "nhl_game_id",
    "home_team",
    "away_team",
    "scheduled_puck_drop_utc",
    "book",
    "home_decimal_odds",
    "away_decimal_odds",
    "observation_utc",
    "capture_method",
    "provenance",
)
ENVELOPE_FIELDS = (
    "schema_version",
    "kind",
    "capture_mode",
    "append_only",
    "logical_duplicate_key",
    "snapshot_sha256",
    "snapshot",
)
ODDS_FIELDS = ("home_decimal_odds", "away_decimal_odds")
AMERICAN_ODDS_FIELDS = ("home_american_odds", "away_american_odds")
# Raw American prices are optional provenance. When both are absent they are
# omitted from the hashed body so Issue #84 snapshots still validate.
# Half-up two-decimal conversion of an American price is at most 0.005 away
# from the exact conversion, so 0.01 accepts that rounding and rejects a
# genuinely different posted decimal.
AMERICAN_DECIMAL_TOLERANCE = Decimal("0.01")
# No quote-age maximum is authorized. Any finite observation strictly before
# puck drop is eligible.
# Issue #85 imports these three callables plus ValidatedForwardOddsSnapshot.
ISSUE_85_READER_SURFACE = (
    "hockey_moneyline.forward_odds_archive.read_validated_snapshots",
    "hockey_moneyline.forward_odds_archive.validate_forward_odds_snapshot",
    "hockey_moneyline.forward_odds_archive.write_forward_odds_snapshot",
    "hockey_moneyline.forward_odds_archive.ValidatedForwardOddsSnapshot",
    "hockey_moneyline.forward_odds_archive.decimal_odds_for_market_math",
)
FORBIDDEN_SNAPSHOT_FIELDS = frozenset(
    {
        "model_probability",
        "model_home_win_probability",
        "model_away_win_probability",
        "model_home_win",
        "model_away_win",
        "probability",
        "probability_kind",
        "market_probability",
        "implied_probability",
        "fair_probability",
        "fair_prob",
        "devig_fair_prob",
        "devig_fair_probability",
        "market_implied_probability",
        "no_vig_probability",
        "edge",
        "recommendation",
        "recommendation_state",
        "threshold",
        "stake",
        "kelly",
        "home_score",
        "away_score",
        "home_win",
        "away_win",
        "outcome",
        "result",
        "game_outcome",
        "game_result",
        "roi",
    }
)
_SECOND_BOOK_FIELDS = frozenset(
    {
        "home_book",
        "away_book",
        "books",
        "bookmakers",
        "sportsbooks",
        "source_book",
    }
)
_TEAM_RE = re.compile(r"^[A-Z]{2,3}$")
_MIXED_BOOK_RE = re.compile(r"[,/;+]|\band\b", re.IGNORECASE)
_BLANK = (None, "")


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _walk_mapping_keys(payload: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(payload, Mapping):
        keys.update(str(key) for key in payload)
        for value in payload.values():
            keys.update(_walk_mapping_keys(value))
    elif isinstance(payload, list):
        for item in payload:
            keys.update(_walk_mapping_keys(item))
    return keys


def _forbidden_keys(payload: Mapping[str, Any]) -> list[str]:
    found = [
        key
        for key in _walk_mapping_keys(payload)
        if key.lower() in FORBIDDEN_SNAPSHOT_FIELDS
    ]
    return sorted(set(found))


def assert_store_path_separate(path: Path) -> None:
    """Keep odds snapshots off the protected prediction-record file."""

    if (
        path.name == PREDICTION_RECORD_FILENAME
        or PREDICTION_RECORD_DIRECTORY in path.parts
    ):
        raise ForwardOddsArchiveError(
            "Forward odds archive path must stay separate from "
            "pregame_prediction_records.jsonl"
        )


def logical_duplicate_key(nhl_game_id: str, book: str, observation_utc: str) -> str:
    """Deterministic key from NHL game id, book/source, and observation UTC."""

    return _canonical_json([nhl_game_id, book, observation_utc])


def snapshot_content_hash(body: Mapping[str, Any]) -> str:
    """SHA-256 of the canonical immutable snapshot body. The hash is not inside the body."""

    return hashlib.sha256(_canonical_json(dict(body)).encode("utf-8")).hexdigest()


def _require_mapping(payload: Mapping[str, Any] | Any) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ForwardOddsArchiveError("Forward odds snapshot input must be a JSON object")
    return payload


def _reject_contract_keys(payload: Mapping[str, Any], *, allowed: tuple[str, ...]) -> None:
    forbidden = _forbidden_keys(payload)
    if forbidden:
        raise ForwardOddsArchiveError(
            "Forward odds snapshot must not store model probability, outcome, "
            f"recommendation, or stake fields: {forbidden}"
        )
    unexpected = sorted(str(key) for key in payload if str(key) not in allowed)
    if unexpected:
        raise ForwardOddsArchiveError(
            "Forward odds snapshot contains fields outside the capture contract: "
            f"{unexpected}"
        )


def parse_nhl_game_id(value: Any) -> str:
    """Accept the 10-digit NHL game-id syntax used in this repository."""

    if isinstance(value, bool) or isinstance(value, float) or value is None:
        raise ForwardOddsArchiveError("NHL game ID is missing or invalid")
    if isinstance(value, int):
        text = str(value)
    elif isinstance(value, str):
        text = value.strip()
    else:
        raise ForwardOddsArchiveError("NHL game ID is missing or invalid")
    if not re.fullmatch(r"[0-9]{10}", text):
        raise ForwardOddsArchiveError("NHL game ID is missing or invalid")
    year = int(text[:4])
    game_type = text[4:6]
    if year < 2000 or year > 2100 or game_type not in {"01", "02", "03"}:
        raise ForwardOddsArchiveError("NHL game ID is missing or invalid")
    return text


def parse_team_identity(value: Any, *, label: str) -> str:
    if not isinstance(value, str):
        raise ForwardOddsArchiveError("home/away identities must be present and distinct")
    text = value.strip().upper()
    if not _TEAM_RE.fullmatch(text):
        raise ForwardOddsArchiveError(
            f"home/away identities must be present and distinct ({label})"
        )
    return text


def parse_decimal_odds(value: Any, *, label: str) -> str:
    """Return canonical decimal text. Finite and strictly greater than 1.0."""

    if isinstance(value, bool) or value is None:
        raise ForwardOddsArchiveError(f"{label} decimal odds are malformed")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ForwardOddsArchiveError(
                f"{label} decimal odds must be finite and greater than 1.0"
            )
        raw = str(value).strip()
    elif isinstance(value, int):
        raw = str(value)
    elif isinstance(value, str):
        raw = value.strip()
        if not raw:
            raise ForwardOddsArchiveError(f"{label} decimal odds are malformed")
    elif isinstance(value, Decimal):
        if not value.is_finite():
            raise ForwardOddsArchiveError(
                f"{label} decimal odds must be finite and greater than 1.0"
            )
        raw = format(value, "f")
    else:
        raise ForwardOddsArchiveError(f"{label} decimal odds are malformed")
    try:
        number = Decimal(raw)
    except InvalidOperation as exc:
        raise ForwardOddsArchiveError(f"{label} decimal odds are malformed") from exc
    if not number.is_finite() or number <= 1:
        raise ForwardOddsArchiveError(
            f"{label} decimal odds must be finite and greater than 1.0"
        )
    text = format(number, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if Decimal(text) <= 1:
        raise ForwardOddsArchiveError(
            f"{label} decimal odds must be finite and greater than 1.0"
        )
    return text


def _is_blank(value: Any) -> bool:
    return value in _BLANK or (isinstance(value, str) and not value.strip())


def parse_american_odds(value: Any, *, label: str) -> str:
    """Return canonical American text such as ``+105`` or ``-125``.

    Valid prices are integers at or outside ±100. Prices between -99 and +99
    are not American moneylines.
    """

    if isinstance(value, bool) or value is None:
        raise ForwardOddsArchiveError(f"{label} American odds are malformed")
    if isinstance(value, int):
        number = value
    elif isinstance(value, str):
        text = value.strip()
        if not text or not re.fullmatch(r"[+-]?[0-9]+", text):
            raise ForwardOddsArchiveError(f"{label} American odds are malformed")
        number = int(text)
    else:
        raise ForwardOddsArchiveError(f"{label} American odds are malformed")
    if number > -100 and number < 100:
        raise ForwardOddsArchiveError(
            f"{label} American odds must be at or outside ±100"
        )
    return f"{number:+d}"


def _exact_decimal_from_american(canonical_american: str) -> Decimal:
    number = int(canonical_american)
    if number >= 100:
        return Decimal(number) / Decimal(100) + 1
    return Decimal(100) / Decimal(abs(number)) + 1


def decimal_text_from_american(value: Any, *, label: str) -> tuple[str, str]:
    """Convert a raw American price to canonical decimal text.

    The decimal is the standard conversion, rounded half-up to two decimal
    places. The returned American text is the canonical ``+105`` / ``-125`` form.
    """

    canonical = parse_american_odds(value, label=label)
    rounded = _exact_decimal_from_american(canonical).quantize(
        Decimal("0.01"),
        rounding=ROUND_HALF_UP,
    )
    return parse_decimal_odds(format(rounded, "f"), label=label), canonical


def _assert_american_matches_decimal(
    canonical_american: str,
    decimal_text: str,
    *,
    label: str,
) -> None:
    derived = _exact_decimal_from_american(canonical_american)
    posted = Decimal(decimal_text)
    if abs(derived - posted) > AMERICAN_DECIMAL_TOLERANCE:
        raise ForwardOddsArchiveError(
            f"{label} raw American price does not match the decimal moneyline"
        )


def _optional_american_prices(
    source: Mapping[str, Any],
    *,
    home_decimal_odds: str,
    away_decimal_odds: str,
) -> dict[str, str]:
    home_present = "home_american_odds" in source and not _is_blank(source["home_american_odds"])
    away_present = "away_american_odds" in source and not _is_blank(source["away_american_odds"])
    if home_present != away_present:
        raise ForwardOddsArchiveError(
            "optional raw American prices must include both offered sides"
        )
    if not home_present:
        return {}
    home_american = parse_american_odds(source["home_american_odds"], label="home")
    away_american = parse_american_odds(source["away_american_odds"], label="away")
    _assert_american_matches_decimal(
        home_american,
        home_decimal_odds,
        label="home",
    )
    _assert_american_matches_decimal(
        away_american,
        away_decimal_odds,
        label="away",
    )
    return {
        "home_american_odds": home_american,
        "away_american_odds": away_american,
    }


def _reject_mixed_book_inputs(source: Mapping[str, Any]) -> None:
    mixed = sorted(
        str(key) for key in source if str(key).lower() in _SECOND_BOOK_FIELDS
    )
    if mixed:
        raise ForwardOddsArchiveError(
            "no bookmaker mixing inside one two-sided market snapshot: "
            + ", ".join(mixed)
        )


def _reject_mixed_book_name(book: str) -> str:
    if _MIXED_BOOK_RE.search(book):
        raise ForwardOddsArchiveError(
            "no bookmaker mixing inside one two-sided market snapshot"
        )
    return book


def parse_aware_utc(value: Any, *, label: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ForwardOddsArchiveError(f"{label} must be timezone-aware")
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ForwardOddsArchiveError(f"{label} must be timezone-aware") from exc
    if parsed.tzinfo is None or parsed.tzinfo.utcoffset(parsed) is None:
        raise ForwardOddsArchiveError(f"{label} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def format_utc(value: datetime) -> str:
    return value.isoformat()


def _require_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ForwardOddsArchiveError(f"{label} is required")
    return value.strip()


def build_snapshot_body(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate operator input and return the immutable snapshot body."""

    source = _require_mapping(payload)
    _reject_mixed_book_inputs(source)
    _reject_contract_keys(source, allowed=OPERATOR_FIELDS + OPTIONAL_OPERATOR_FIELDS)
    missing_odds = [
        field
        for field in ODDS_FIELDS
        if field not in source or source[field] in _BLANK or (
            isinstance(source[field], str) and not source[field].strip()
        )
    ]
    if missing_odds:
        raise ForwardOddsArchiveError(
            "both decimal sides are required; missing one side: " + ", ".join(missing_odds)
        )
    identity_missing = [
        field
        for field in ("nhl_game_id", "home_team", "away_team")
        if field not in source
        or source[field] in _BLANK
        or (isinstance(source[field], str) and not source[field].strip())
    ]
    if identity_missing:
        raise ForwardOddsArchiveError(
            "home/away identities or NHL game ID is missing or invalid: "
            + ", ".join(identity_missing)
        )
    text_missing = [
        field
        for field in ("book", "capture_method", "provenance")
        if field not in source
        or not isinstance(source[field], str)
        or not source[field].strip()
    ]
    if text_missing:
        raise ForwardOddsArchiveError(
            "book/source, capture method, and provenance are required; missing: "
            + ", ".join(text_missing)
        )
    for label in ("scheduled_puck_drop_utc", "observation_utc"):
        if label not in source or source[label] in _BLANK:
            raise ForwardOddsArchiveError(f"{label} must be timezone-aware")
    nhl_game_id = parse_nhl_game_id(source["nhl_game_id"])
    home_team = parse_team_identity(source["home_team"], label="home_team")
    away_team = parse_team_identity(source["away_team"], label="away_team")
    if home_team == away_team:
        raise ForwardOddsArchiveError("home/away identities must be present and distinct")
    home_decimal_odds = parse_decimal_odds(source["home_decimal_odds"], label="home")
    away_decimal_odds = parse_decimal_odds(source["away_decimal_odds"], label="away")
    puck_drop = parse_aware_utc(
        source["scheduled_puck_drop_utc"],
        label="scheduled_puck_drop_utc",
    )
    observed = parse_aware_utc(source["observation_utc"], label="observation_utc")
    if observed >= puck_drop:
        raise ForwardOddsArchiveError(
            "observation_utc must be strictly before scheduled_puck_drop_utc"
        )
    book = _reject_mixed_book_name(_require_text(source["book"], label="book/source"))
    capture_method = _require_text(source["capture_method"], label="capture method")
    provenance = _require_text(source["provenance"], label="provenance")
    american = _optional_american_prices(
        source,
        home_decimal_odds=home_decimal_odds,
        away_decimal_odds=away_decimal_odds,
    )
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": SNAPSHOT_KIND,
        "nhl_game_id": nhl_game_id,
        "home_team": home_team,
        "away_team": away_team,
        "scheduled_puck_drop_utc": format_utc(puck_drop),
        "book": book,
        "home_decimal_odds": home_decimal_odds,
        "away_decimal_odds": away_decimal_odds,
        "observation_utc": format_utc(observed),
        "capture_method": capture_method,
        "provenance": provenance,
        **american,
    }
    _reject_contract_keys(body, allowed=SNAPSHOT_BODY_FIELDS + tuple(american))
    return body


def build_envelope(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Wrap a validated snapshot body with its logical key and checksum."""

    body = build_snapshot_body(payload)
    digest = snapshot_content_hash(body)
    duplicate_key = logical_duplicate_key(
        body["nhl_game_id"],
        body["book"],
        body["observation_utc"],
    )
    envelope = {
        "schema_version": SCHEMA_VERSION,
        "kind": SNAPSHOT_KIND,
        "capture_mode": CAPTURE_MODE,
        "append_only": True,
        "logical_duplicate_key": duplicate_key,
        "snapshot_sha256": digest,
        "snapshot": body,
    }
    return _validate_envelope(envelope)


def _validate_envelope(payload: Mapping[str, Any]) -> dict[str, Any]:
    source = _require_mapping(payload)
    forbidden = _forbidden_keys(source)
    if forbidden:
        raise ForwardOddsArchiveError(
            "Forward odds snapshot must not store model probability, outcome, "
            f"recommendation, or stake fields: {forbidden}"
        )
    unexpected = sorted(str(key) for key in source if str(key) not in ENVELOPE_FIELDS)
    if unexpected:
        raise ForwardOddsArchiveError(
            "Persisted forward-odds envelope contains fields outside the capture contract: "
            f"{unexpected}"
        )
    if int(source.get("schema_version", 0)) != SCHEMA_VERSION:
        raise ForwardOddsArchiveError("Forward-odds store schema_version is not recognized")
    if source.get("kind") != SNAPSHOT_KIND:
        raise ForwardOddsArchiveError("Persisted object is not a forward odds snapshot")
    if source.get("capture_mode") != CAPTURE_MODE:
        raise ForwardOddsArchiveError("Persisted forward-odds snapshot is not a manual capture")
    if source.get("append_only") is not True:
        raise ForwardOddsArchiveError("Persisted forward-odds snapshot is not marked append-only")
    snapshot = source.get("snapshot")
    if not isinstance(snapshot, Mapping):
        raise ForwardOddsArchiveError("Persisted forward-odds envelope is missing the snapshot")
    operator_payload = {
        field: snapshot[field]
        for field in OPERATOR_FIELDS + OPTIONAL_OPERATOR_FIELDS
        if field in snapshot
    }
    body = build_snapshot_body(operator_payload)
    if set(snapshot) != set(body):
        raise ForwardOddsArchiveError(
            "Persisted forward-odds snapshot fields do not match the capture contract"
        )
    if body != {key: snapshot[key] for key in body}:
        raise ForwardOddsArchiveError(
            "Persisted forward-odds snapshot does not match its canonical body"
        )
    expected_hash = snapshot_content_hash(body)
    expected_key = logical_duplicate_key(
        body["nhl_game_id"],
        body["book"],
        body["observation_utc"],
    )
    if source.get("snapshot_sha256") != expected_hash:
        raise ForwardOddsArchiveError(
            "Existing forward-odds snapshot was overwritten or edited: "
            "snapshot_sha256 does not match canonical content"
        )
    if source.get("logical_duplicate_key") != expected_key:
        raise ForwardOddsArchiveError(
            "Existing forward-odds snapshot logical duplicate key does not match its body"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": SNAPSHOT_KIND,
        "capture_mode": CAPTURE_MODE,
        "append_only": True,
        "logical_duplicate_key": expected_key,
        "snapshot_sha256": expected_hash,
        "snapshot": body,
    }


@dataclass(frozen=True, slots=True)
class ForwardOddsAppendResult:
    status: str
    written: bool
    snapshot_sha256: str
    logical_duplicate_key: str
    envelope: dict[str, Any]
    store_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "written": self.written,
            "snapshot_sha256": self.snapshot_sha256,
            "logical_duplicate_key": self.logical_duplicate_key,
            "store_path": self.store_path,
            "nhl_game_id": self.envelope["snapshot"]["nhl_game_id"],
            "book": self.envelope["snapshot"]["book"],
            "observation_utc": self.envelope["snapshot"]["observation_utc"],
            "envelope": self.envelope,
        }


class ForwardOddsArchiveStore:
    """JSONL append-only store. No update, replace, or delete API is provided."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else Path(DEFAULT_STORE_RELATIVE_PATH)
        assert_store_path_separate(self.path)

    def _read_envelopes(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        try:
            text = self.path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ForwardOddsArchiveError(
                f"Unreadable forward-odds store {self.path}"
            ) from exc
        envelopes: list[dict[str, Any]] = []
        seen: set[str] = set()
        for line_number, raw in enumerate(text.splitlines(), start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ForwardOddsArchiveError(
                    f"Forward-odds store is not valid JSONL at {self.path}:{line_number}"
                ) from exc
            if not isinstance(payload, Mapping):
                raise ForwardOddsArchiveError(
                    f"Forward-odds store line {line_number} is not a JSON object"
                )
            envelope = _validate_envelope(payload)
            duplicate_key = str(envelope["logical_duplicate_key"])
            if duplicate_key in seen:
                raise ForwardOddsArchiveError(
                    "Existing forward-odds store is corrupt: "
                    f"duplicate logical key {duplicate_key}"
                )
            seen.add(duplicate_key)
            envelopes.append(envelope)
        return envelopes

    def iter_envelopes(self) -> Iterator[dict[str, Any]]:
        yield from self._read_envelopes()

    def get_by_logical_key(self, duplicate_key: str) -> dict[str, Any] | None:
        for envelope in self._read_envelopes():
            if envelope["logical_duplicate_key"] == duplicate_key:
                return envelope
        return None

    def append(self, payload: Mapping[str, Any]) -> ForwardOddsAppendResult:
        """Append one immutable snapshot.

        Exact duplicates return ``existing_identical`` and do not write.
        The same logical key with different content fails closed.
        """

        envelope = build_envelope(payload)
        duplicate_key = str(envelope["logical_duplicate_key"])
        existing = self.get_by_logical_key(duplicate_key)
        if existing is not None:
            if (
                existing["snapshot_sha256"] == envelope["snapshot_sha256"]
                and existing["snapshot"] == envelope["snapshot"]
            ):
                return ForwardOddsAppendResult(
                    status=STATUS_EXISTING_IDENTICAL,
                    written=False,
                    snapshot_sha256=str(existing["snapshot_sha256"]),
                    logical_duplicate_key=duplicate_key,
                    envelope=existing,
                    store_path=str(self.path),
                )
            raise ForwardOddsArchiveError(
                "Conflicting forward-odds snapshot for logical duplicate key "
                f"{duplicate_key}: existing content disagrees; "
                "refusing to mutate the stored snapshot"
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = _canonical_json(envelope) + "\n"
        with self.path.open("ab") as handle:
            handle.write(line.encode("utf-8"))
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
        reread = self.get_by_logical_key(duplicate_key)
        if reread is None or reread["snapshot_sha256"] != envelope["snapshot_sha256"]:
            raise ForwardOddsArchiveError(
                "Append-only forward-odds write did not persist the expected content"
            )
        return ForwardOddsAppendResult(
            status=STATUS_WRITTEN,
            written=True,
            snapshot_sha256=str(envelope["snapshot_sha256"]),
            logical_duplicate_key=duplicate_key,
            envelope=envelope,
            store_path=str(self.path),
        )


def capture_manual_snapshot(
    payload: Mapping[str, Any],
    *,
    store: ForwardOddsArchiveStore | Path | str | None = None,
) -> ForwardOddsAppendResult:
    """Write one operator-supplied snapshot. This is the manual capture entry point."""

    archive = (
        store
        if isinstance(store, ForwardOddsArchiveStore)
        else ForwardOddsArchiveStore(store)
    )
    return archive.append(payload)


@dataclass(frozen=True, slots=True)
class ValidatedForwardOddsSnapshot:
    """One archive line after the writer/validator has accepted it.

    Issue #85 should use this object. It has no model probability, market
    probability, edge, recommendation, stake, or game outcome.
    """

    schema_version: int
    kind: str
    nhl_game_id: str
    home_team: str
    away_team: str
    scheduled_puck_drop_utc: str
    book: str
    home_decimal_odds: str
    away_decimal_odds: str
    observation_utc: str
    capture_method: str
    provenance: str
    snapshot_sha256: str
    logical_duplicate_key: str
    capture_mode: str
    append_only: bool
    home_american_odds: str | None = None
    away_american_odds: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "nhl_game_id": self.nhl_game_id,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "scheduled_puck_drop_utc": self.scheduled_puck_drop_utc,
            "book": self.book,
            "home_decimal_odds": self.home_decimal_odds,
            "away_decimal_odds": self.away_decimal_odds,
            "home_american_odds": self.home_american_odds,
            "away_american_odds": self.away_american_odds,
            "observation_utc": self.observation_utc,
            "capture_method": self.capture_method,
            "capture_mode": self.capture_mode,
            "provenance": self.provenance,
            "snapshot_sha256": self.snapshot_sha256,
            "logical_duplicate_key": self.logical_duplicate_key,
            "append_only": self.append_only,
        }


def validated_snapshot_from_envelope(
    envelope: Mapping[str, Any],
) -> ValidatedForwardOddsSnapshot:
    """Project an already validated envelope into the Issue #85 reader object."""

    checked = _validate_envelope(envelope)
    body = checked["snapshot"]
    return ValidatedForwardOddsSnapshot(
        schema_version=SCHEMA_VERSION,
        kind=SNAPSHOT_KIND,
        nhl_game_id=str(body["nhl_game_id"]),
        home_team=str(body["home_team"]),
        away_team=str(body["away_team"]),
        scheduled_puck_drop_utc=str(body["scheduled_puck_drop_utc"]),
        book=str(body["book"]),
        home_decimal_odds=str(body["home_decimal_odds"]),
        away_decimal_odds=str(body["away_decimal_odds"]),
        observation_utc=str(body["observation_utc"]),
        capture_method=str(body["capture_method"]),
        provenance=str(body["provenance"]),
        snapshot_sha256=str(checked["snapshot_sha256"]),
        logical_duplicate_key=str(checked["logical_duplicate_key"]),
        capture_mode=str(checked["capture_mode"]),
        append_only=True,
        home_american_odds=(
            str(body["home_american_odds"]) if "home_american_odds" in body else None
        ),
        away_american_odds=(
            str(body["away_american_odds"]) if "away_american_odds" in body else None
        ),
    )


def validate_forward_odds_snapshot(
    payload: Mapping[str, Any],
) -> ValidatedForwardOddsSnapshot:
    """Validate one snapshot without writing it."""

    return validated_snapshot_from_envelope(build_envelope(payload))


def write_forward_odds_snapshot(
    payload: Mapping[str, Any],
    *,
    store: ForwardOddsArchiveStore | Path | str | None = None,
) -> ForwardOddsAppendResult:
    """Validate and append one snapshot. Exact duplicates are a no-op."""

    return capture_manual_snapshot(payload, store=store)


def _open_store(
    store: ForwardOddsArchiveStore | Path | str | None,
) -> ForwardOddsArchiveStore:
    if isinstance(store, ForwardOddsArchiveStore):
        return store
    return ForwardOddsArchiveStore(store)


def read_validated_snapshots(
    store: ForwardOddsArchiveStore | Path | str | None = None,
    *,
    nhl_game_id: str | None = None,
) -> tuple[ValidatedForwardOddsSnapshot, ...]:
    """Read the archive and re-validate every line.

    This is the Issue #85 reader. Pass ``nhl_game_id`` to keep one game.
    A missing store returns an empty tuple. A corrupt line fails closed.
    """

    archive = _open_store(store)
    wanted = parse_nhl_game_id(nhl_game_id) if nhl_game_id is not None else None
    rows = [
        validated_snapshot_from_envelope(envelope)
        for envelope in archive.iter_envelopes()
        if wanted is None or envelope["snapshot"]["nhl_game_id"] == wanted
    ]
    rows.sort(
        key=lambda row: (
            row.observation_utc,
            row.nhl_game_id,
            row.book,
            row.snapshot_sha256,
        )
    )
    return tuple(rows)


def decimal_odds_for_market_math(
    snapshot: ValidatedForwardOddsSnapshot,
) -> tuple[float, float]:
    """Return ``(home_decimal_odds, away_decimal_odds)`` as floats.

    Issue #78 market math accepts floats, while the archive stores canonical
    decimal text. Call this on a snapshot the reader already validated. It does
    not reimplement archive rules and it does not read model probabilities.
    """

    if not isinstance(snapshot, ValidatedForwardOddsSnapshot):
        raise ForwardOddsArchiveError(
            "Market-math odds must come from a validated forward-odds snapshot"
        )
    return (float(snapshot.home_decimal_odds), float(snapshot.away_decimal_odds))
