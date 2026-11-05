"""Deterministic pregame audit receipt for an immutable prediction record.

The receipt is a compact identity payload that an operator can post manually
as external timestamp evidence. It does not score, refit, call GitHub, or
replace the local Issue #55 prediction record.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from hockey_moneyline.errors import PredictionError
from hockey_moneyline.prediction_record_store import prediction_record_content_hash
from hockey_moneyline.prospective_scoring import (
    CURRENT_SEASON_PRIOR_SOURCE_SHA256_KEY,
    PregamePredictionRecord,
    feature_vector_sha256,
)

RECEIPT_SCHEMA_VERSION = 1
RECEIPT_KIND = "pregame_audit_receipt"
MODEL_PROBABILITY_KIND = "model_home_win"
RECEIPT_PAYLOAD_FIELDS = (
    "schema_version",
    "kind",
    "canonical_game_id",
    "scheduled_start_time_utc",
    "prediction_timestamp_utc",
    "frozen_artifact_sha256",
    "current_season_source_sha256",
    "feature_vector_sha256",
    "prediction_record_sha256",
    "probability_kind",
)
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and bool(_SHA256_HEX.fullmatch(value))


def _require_sha256(value: object, *, absent_message: str) -> str:
    if not _is_sha256(value):
        raise PredictionError(absent_message)
    return str(value)


def _canonical_utc_timestamp(value: object, *, label: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise PredictionError(f"{label} is missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PredictionError(f"{label} is not a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise PredictionError(f"{label} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _utc_text(value: datetime) -> str:
    return value.isoformat()


def receipt_payload_sha256(payload: Mapping[str, Any]) -> str:
    """SHA-256 of the canonical receipt identity payload.

    ``receipt_sha256`` is excluded. Extra keys are excluded so the hash covers
    only the receipt contract fields.
    """

    try:
        materialized = {field: payload[field] for field in RECEIPT_PAYLOAD_FIELDS}
    except KeyError as exc:
        raise PredictionError(
            f"Receipt payload is missing canonical identity field: {exc.args[0]}"
        ) from exc
    return hashlib.sha256(_canonical_json(materialized).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class PregameAuditReceipt:
    """Compact external receipt. The local prediction record stays authoritative."""

    schema_version: int
    kind: str
    canonical_game_id: str
    scheduled_start_time_utc: str
    prediction_timestamp_utc: str
    frozen_artifact_sha256: str
    current_season_source_sha256: str
    feature_vector_sha256: str
    prediction_record_sha256: str
    probability_kind: str
    receipt_sha256: str

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "canonical_game_id": self.canonical_game_id,
            "scheduled_start_time_utc": self.scheduled_start_time_utc,
            "prediction_timestamp_utc": self.prediction_timestamp_utc,
            "frozen_artifact_sha256": self.frozen_artifact_sha256,
            "current_season_source_sha256": self.current_season_source_sha256,
            "feature_vector_sha256": self.feature_vector_sha256,
            "prediction_record_sha256": self.prediction_record_sha256,
            "probability_kind": self.probability_kind,
        }

    def to_dict(self) -> dict[str, Any]:
        body = self.payload()
        body["receipt_sha256"] = self.receipt_sha256
        return body


def _record_and_claimed_hash(
    record: PregamePredictionRecord | Mapping[str, Any],
    record_sha256: str | None,
) -> tuple[dict[str, Any], object]:
    if isinstance(record, PregamePredictionRecord):
        return record.to_dict(), record_sha256
    if not isinstance(record, Mapping):
        raise PredictionError(
            "Pregame audit receipt requires an immutable prediction record"
        )
    inner = record.get("record")
    if isinstance(inner, Mapping):
        claimed = record.get("record_sha256") if record_sha256 is None else record_sha256
        return dict(inner), claimed
    if isinstance(inner, PregamePredictionRecord):
        claimed = record.get("record_sha256") if record_sha256 is None else record_sha256
        return inner.to_dict(), claimed
    return dict(record), record_sha256


def _require_feature_vector_hash(record: Mapping[str, Any]) -> str:
    stored = record.get("feature_vector_sha256")
    if not _is_sha256(stored):
        raise PredictionError("feature-vector provenance is absent or malformed")
    values = record.get("feature_values")
    if not isinstance(values, (list, tuple)) or len(values) == 0:
        raise PredictionError("feature-vector provenance is absent")
    try:
        expected = feature_vector_sha256(list(values))
    except (TypeError, ValueError) as exc:
        raise PredictionError("feature-vector provenance is absent or malformed") from exc
    if stored != expected:
        raise PredictionError(
            "feature-vector hash is invalid: it does not match the record feature values"
        )
    return str(stored)


def _require_source_sha256(record: Mapping[str, Any]) -> str:
    source_versions = record.get("source_versions")
    if not isinstance(source_versions, Mapping):
        raise PredictionError("current-season source provenance is absent")
    return _require_sha256(
        source_versions.get(CURRENT_SEASON_PRIOR_SOURCE_SHA256_KEY),
        absent_message="current-season source provenance is absent or malformed",
    )


def build_pregame_audit_receipt(
    record: PregamePredictionRecord | Mapping[str, Any],
    *,
    record_sha256: str | None = None,
) -> PregameAuditReceipt:
    """Derive one deterministic receipt from an already-created prediction record.

    Fails closed when the record hash, provenance, pregame timing, probability
    kind, or canonical identity is missing or invalid. Does not score a model
    and does not contact GitHub.
    """

    inner, claimed_hash = _record_and_claimed_hash(record, record_sha256)
    if claimed_hash is None or claimed_hash == "":
        raise PredictionError("prediction-record hash is absent")
    if not _is_sha256(claimed_hash):
        raise PredictionError("prediction-record hash is invalid")
    actual_hash = prediction_record_content_hash(inner)
    if claimed_hash != actual_hash:
        raise PredictionError(
            "prediction-record hash is invalid: it does not match canonical record content"
        )

    game_id = inner.get("game_id")
    if not isinstance(game_id, str) or not game_id.strip():
        raise PredictionError("canonical game id is missing")

    scheduled = _canonical_utc_timestamp(
        inner.get("scheduled_start_time_utc"),
        label="scheduled start time",
    )
    predicted = _canonical_utc_timestamp(
        inner.get("prediction_timestamp_utc"),
        label="prediction timestamp",
    )
    if predicted >= scheduled:
        raise PredictionError(
            "Prediction/observation timestamp must occur before scheduled puck drop"
        )

    artifact_sha256 = _require_sha256(
        inner.get("model_artifact_sha256"),
        absent_message="frozen artifact provenance is absent or malformed",
    )
    source_sha256 = _require_source_sha256(inner)
    vector_sha256 = _require_feature_vector_hash(inner)

    probability_kind = inner.get("probability_kind")
    if probability_kind != MODEL_PROBABILITY_KIND:
        raise PredictionError(
            "probability kind is not the frozen model probability "
            f"{MODEL_PROBABILITY_KIND!r}"
        )

    payload = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "kind": RECEIPT_KIND,
        "canonical_game_id": game_id,
        "scheduled_start_time_utc": _utc_text(scheduled),
        "prediction_timestamp_utc": _utc_text(predicted),
        "frozen_artifact_sha256": artifact_sha256,
        "current_season_source_sha256": source_sha256,
        "feature_vector_sha256": vector_sha256,
        "prediction_record_sha256": actual_hash,
        "probability_kind": MODEL_PROBABILITY_KIND,
    }
    digest = receipt_payload_sha256(payload)
    receipt = PregameAuditReceipt(
        schema_version=RECEIPT_SCHEMA_VERSION,
        kind=RECEIPT_KIND,
        canonical_game_id=payload["canonical_game_id"],
        scheduled_start_time_utc=payload["scheduled_start_time_utc"],
        prediction_timestamp_utc=payload["prediction_timestamp_utc"],
        frozen_artifact_sha256=payload["frozen_artifact_sha256"],
        current_season_source_sha256=payload["current_season_source_sha256"],
        feature_vector_sha256=payload["feature_vector_sha256"],
        prediction_record_sha256=payload["prediction_record_sha256"],
        probability_kind=payload["probability_kind"],
        receipt_sha256=digest,
    )
    if receipt_payload_sha256(receipt.payload()) != receipt.receipt_sha256:
        raise PredictionError("receipt hash does not match canonical payload")
    return receipt
