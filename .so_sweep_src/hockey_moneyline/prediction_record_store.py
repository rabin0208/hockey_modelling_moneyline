"""Append-only local store for Issue #55 immutable pregame prediction records.

Durable records live under gitignored ``data/derived/`` so raw provider files,
credentials, odds, staking, and recommendations are never committed. The store
preserves the Issue #52 prediction schema and adds a deterministic content hash.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

from hockey_moneyline.errors import PredictionError
from hockey_moneyline.prospective_artifact import (
    EXPECTED_FITTED_ARTIFACT_SHA256,
    FROZEN_FEATURE_COLUMNS,
)
from hockey_moneyline.prospective_scoring import (
    CURRENT_SEASON_PRIOR_SOURCE_OBSERVED_AT_KEY,
    CURRENT_SEASON_PRIOR_SOURCE_SHA256_KEY,
    FORBIDDEN_MARKET_FIELDS,
    FORBIDDEN_TARGET_OUTCOME_FIELDS,
    PregamePredictionRecord,
)
from hockey_moneyline.schedule_load import RESERVED_FUTURE_SEASON

STORE_SCHEMA_VERSION = 1
STORE_KIND = "immutable_pregame_prediction_record"
EXECUTION_MODE = "manual_explicit"
DEFAULT_STORE_RELATIVE_PATH = (
    "data/derived/prediction_records/pregame_prediction_records.jsonl"
)
REQUIRED_RECORD_FIELDS = (
    "game_id",
    "season",
    "game_type",
    "home_team_id",
    "away_team_id",
    "scheduled_start_time_utc",
    "prediction_timestamp_utc",
    "model_artifact_version",
    "model_artifact_sha256",
    "feature_schema_version",
    "source_versions",
    "feature_columns",
    "feature_values",
    "feature_vector_sha256",
    "model_home_win_probability",
    "probability_kind",
)
FORBIDDEN_STORE_FIELDS = FORBIDDEN_MARKET_FIELDS + FORBIDDEN_TARGET_OUTCOME_FIELDS


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def prediction_record_content_hash(record: PregamePredictionRecord | Mapping[str, Any]) -> str:
    """Deterministic SHA-256 of the Issue #52 prediction-record payload."""

    payload = record.to_dict() if isinstance(record, PregamePredictionRecord) else dict(record)
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


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


def _parse_aware_utc(value: str, *, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PredictionError(f"{label} is not a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise PredictionError(f"{label} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def validate_prediction_record_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed if a durable record is incomplete, drifted, or leaked."""

    missing = [field for field in REQUIRED_RECORD_FIELDS if field not in payload]
    if missing:
        raise PredictionError(
            f"Protected prediction record is missing provenance/hash fields: {missing}"
        )
    leaked = sorted(key for key in _walk_mapping_keys(payload) if key in FORBIDDEN_STORE_FIELDS)
    if leaked:
        raise PredictionError(
            f"Prediction record leaked target outcome or market fields: {leaked}"
        )
    if payload["game_type"] != "regular":
        raise PredictionError(
            f"Frozen prospective target must be regular season, got {payload['game_type']!r}"
        )
    if not payload.get("game_id"):
        raise PredictionError("Protected prediction record is missing game identity")
    if tuple(payload["feature_columns"]) != tuple(FROZEN_FEATURE_COLUMNS):
        raise PredictionError(
            "Feature-order drift: durable record columns are not the frozen 25-column order"
        )
    values = payload["feature_values"]
    if not isinstance(values, list) or len(values) != 25:
        raise PredictionError(
            f"Protected prediction record requires exactly 25 feature values, got {len(values) if isinstance(values, list) else type(values)}"
        )
    if payload["model_artifact_sha256"] != EXPECTED_FITTED_ARTIFACT_SHA256:
        raise PredictionError(
            "Frozen artifact mismatch/tampering: durable record artifact SHA-256 "
            f"{payload['model_artifact_sha256']} != {EXPECTED_FITTED_ARTIFACT_SHA256}"
        )
    if payload["probability_kind"] != "model_home_win":
        raise PredictionError("Prediction record must keep model probability distinct")
    probability = payload["model_home_win_probability"]
    if not isinstance(probability, (int, float)) or not (0.0 < float(probability) < 1.0):
        raise PredictionError("Model home-win probability must be inside (0, 1)")
    start = _parse_aware_utc(
        str(payload["scheduled_start_time_utc"]),
        label="scheduled_start_time_utc",
    )
    observed = _parse_aware_utc(
        str(payload["prediction_timestamp_utc"]),
        label="prediction_timestamp_utc",
    )
    if observed >= start:
        raise PredictionError(
            "Prediction/observation timestamp must occur before scheduled puck drop"
        )
    source_versions = payload["source_versions"]
    if not isinstance(source_versions, Mapping):
        raise PredictionError("Protected prediction record source_versions must be a mapping")
    if int(payload["season"]) >= RESERVED_FUTURE_SEASON:
        sha256 = source_versions.get(CURRENT_SEASON_PRIOR_SOURCE_SHA256_KEY)
        observed_at = source_versions.get(CURRENT_SEASON_PRIOR_SOURCE_OBSERVED_AT_KEY)
        if not isinstance(sha256, str) or len(sha256) != 64:
            raise PredictionError(
                "Protected 2026-27 record is missing current_season_prior_source_sha256"
            )
        if not isinstance(observed_at, str) or not observed_at:
            raise PredictionError(
                "Protected 2026-27 record is missing current_season_prior_source_observed_at_utc"
            )
        _parse_aware_utc(
            observed_at,
            label=CURRENT_SEASON_PRIOR_SOURCE_OBSERVED_AT_KEY,
        )
    if not payload.get("feature_vector_sha256"):
        raise PredictionError("Protected prediction record is missing feature_vector_sha256")
    if not payload.get("feature_schema_version"):
        raise PredictionError("Protected prediction record is missing feature_schema_version")
    return dict(payload)


def build_store_envelope(record: PregamePredictionRecord) -> dict[str, Any]:
    """Wrap an Issue #52 record with the Issue #55 immutable-store envelope."""

    inner = validate_prediction_record_payload(record.to_dict())
    digest = prediction_record_content_hash(inner)
    envelope = {
        "schema_version": STORE_SCHEMA_VERSION,
        "kind": STORE_KIND,
        "execution_mode": EXECUTION_MODE,
        "append_only": True,
        "record_sha256": digest,
        "record": inner,
    }
    leaked = sorted(
        key for key in _walk_mapping_keys(envelope) if key in FORBIDDEN_STORE_FIELDS
    )
    if leaked:
        raise PredictionError(
            f"Prediction record leaked target outcome or market fields: {leaked}"
        )
    return envelope


def _validate_envelope(payload: Mapping[str, Any]) -> dict[str, Any]:
    if int(payload.get("schema_version", 0)) != STORE_SCHEMA_VERSION:
        raise PredictionError("Prediction-record store schema_version is not recognized")
    if payload.get("kind") != STORE_KIND:
        raise PredictionError("Persisted object is not an immutable pregame prediction record")
    if payload.get("execution_mode") != EXECUTION_MODE:
        raise PredictionError("Persisted prediction record is not a manual explicit execution")
    if payload.get("append_only") is not True:
        raise PredictionError("Persisted prediction record is not marked append-only")
    record = payload.get("record")
    if not isinstance(record, Mapping):
        raise PredictionError("Persisted prediction envelope is missing the Issue #52 record")
    inner = validate_prediction_record_payload(record)
    expected = prediction_record_content_hash(inner)
    stored = payload.get("record_sha256")
    if stored != expected:
        raise PredictionError(
            "Existing protected prediction record was overwritten or edited: "
            "record_sha256 does not match canonical content"
        )
    return {
        "schema_version": STORE_SCHEMA_VERSION,
        "kind": STORE_KIND,
        "execution_mode": EXECUTION_MODE,
        "append_only": True,
        "record_sha256": expected,
        "record": inner,
    }


@dataclass(frozen=True, slots=True)
class AppendResult:
    written: bool
    idempotent_duplicate: bool
    envelope: dict[str, Any]
    store_path: str


class PredictionRecordStore:
    """JSONL append-only store. No update, replace, or delete API is provided."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else Path(DEFAULT_STORE_RELATIVE_PATH)

    def _read_envelopes(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        try:
            text = self.path.read_text(encoding="utf-8")
        except OSError as exc:
            raise PredictionError(f"Unreadable prediction-record store {self.path}") from exc
        envelopes: list[dict[str, Any]] = []
        seen: set[str] = set()
        for line_number, raw in enumerate(text.splitlines(), start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PredictionError(
                    f"Prediction-record store is not valid JSONL at {self.path}:{line_number}"
                ) from exc
            if not isinstance(payload, Mapping):
                raise PredictionError(
                    f"Prediction-record store line {line_number} is not a JSON object"
                )
            envelope = _validate_envelope(payload)
            game_id = str(envelope["record"]["game_id"])
            if game_id in seen:
                raise PredictionError(
                    "Existing protected prediction-record store is corrupt: "
                    f"duplicate game_id {game_id}"
                )
            seen.add(game_id)
            envelopes.append(envelope)
        return envelopes

    def iter_envelopes(self) -> Iterator[dict[str, Any]]:
        yield from self._read_envelopes()

    def get_by_game_id(self, game_id: str) -> dict[str, Any] | None:
        for envelope in self._read_envelopes():
            if envelope["record"]["game_id"] == game_id:
                return envelope
        return None

    def append(self, record: PregamePredictionRecord) -> AppendResult:
        """Append one immutable record. Exact duplicates are an explicit no-op."""

        envelope = build_store_envelope(record)
        game_id = str(envelope["record"]["game_id"])
        existing = self.get_by_game_id(game_id)
        if existing is not None:
            if (
                existing["record_sha256"] == envelope["record_sha256"]
                and existing["record"] == envelope["record"]
            ):
                return AppendResult(
                    written=False,
                    idempotent_duplicate=True,
                    envelope=existing,
                    store_path=str(self.path),
                )
            raise PredictionError(
                "Duplicate conflicting prediction record for target "
                f"{game_id}: existing content disagrees with the new record"
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
        reread = self.get_by_game_id(game_id)
        if reread is None or reread["record_sha256"] != envelope["record_sha256"]:
            raise PredictionError(
                "Append-only prediction-record write did not persist the expected content"
            )
        return AppendResult(
            written=True,
            idempotent_duplicate=False,
            envelope=envelope,
            store_path=str(self.path),
        )
