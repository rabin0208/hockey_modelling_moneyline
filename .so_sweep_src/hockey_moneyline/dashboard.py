"""Read-only local presentation adapter for the Hockey dashboard MVP.

This module loads an NHL schedule snapshot, immutable Issue #55 prediction
records, and Issue #63 receipt evidence. It does not score games, fit or tune
the frozen model, append prediction records, post receipts, or invent
probabilities.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from hockey_moneyline.errors import HockeyMoneylineError, NHLApiError, PredictionError
from hockey_moneyline.identity import nhl_abbrev_for_source
from hockey_moneyline.issue55 import nhl_snapshot_from_dict
from hockey_moneyline.nhl_api import NHLScheduleGame, NHLScheduleSnapshot, parse_schedule_payload
from hockey_moneyline.prediction_record_store import (
    DEFAULT_STORE_RELATIVE_PATH,
    PredictionRecordStore,
)
from hockey_moneyline.pregame_audit_receipt import (
    PregameAuditReceipt,
    build_pregame_audit_receipt,
)
from hockey_moneyline.prospective_artifact import (
    EXPECTED_FITTED_ARTIFACT_SHA256,
    FROZEN_CANDIDATE_FAMILY,
    load_committed_frozen_artifact,
)
from hockey_moneyline.prospective_scoring import CURRENT_SEASON_PRIOR_SOURCE_SHA256_KEY

STATUS_NOT_SCORED = "NOT SCORED"
STATUS_RECORDED = "RECORDED"
STATUS_RECEIPT_READY = "RECEIPT READY"
DASHBOARD_DISCLAIMER = (
    "Model win probabilities are descriptive outputs of the frozen model. "
    "They are not guaranteed outcomes, betting recommendations, or market odds."
)
MARKET_LAYER_ABSENT = (
    "No sportsbook odds, implied probability, edge, or staking layer is loaded."
)
PROSPECTIVE_STATUS = (
    "PRE-OPENING READY — FROZEN MODEL OPERATIONAL FOR PROTECTED MANUAL PREDICTIONS"
)
HISTORICAL_RESULTS_NOTE = (
    "Historical forward-chaining results do not prove superiority over Rabin's "
    "benchmark. 2026-27 is the protected prospective test."
)
# Issue #38 Cycle-2 experimental forward-chaining macro averages. Display
# strings are the published figures. This module does not recompute them.
HISTORICAL_FORWARD_CHAINING_MACROS = {
    "accuracy": "58.6181%",
    "roc_auc": "0.618947",
    "log_loss": "0.672524",
    "brier": "0.239632",
    "ece": "0.051867",
}
DEFAULT_SNAPSHOT_RELATIVE_PATH = "data/local/nhl_schedule_snapshot.json"
DEFAULT_DISPLAY_TIMEZONE = "America/Toronto"
_SHA256_HEX = re.compile(r"^[0-9a-fA-F]{64}$")
_ABSENT_SOURCE_MESSAGE = "current-season source provenance is absent"
_DASHBOARD_MODULE_NAMES = ("dashboard.py", "dashboard_app.py")
_FORBIDDEN_IMPORT_ROOTS = frozenset(
    {
        "github",
        "ghapi",
        "httpx",
        "requests",
        "sklearn",
        "socket",
        "urllib",
        "urllib3",
    }
)
_FORBIDDEN_CALLS = frozenset(
    {
        "emit_prediction_record",
        "execute_historical_dry_run_prediction",
        "execute_manual_pregame_prediction",
        "fit",
        "fit_frozen_prospective_artifact",
        "predict_home_win_probability",
        "score_frozen_snapshot",
        "score_historical_replay_prediction",
        "score_pregame_prediction",
    }
)


class DashboardDataError(HockeyMoneylineError):
    """Local dashboard inputs are missing or malformed."""


@dataclass(frozen=True, slots=True)
class AuditDetail:
    """Full provenance for one recorded game. Short hashes are display-only."""

    frozen_artifact_sha256: str
    prediction_timestamp_utc: str
    source_sha256: str | None
    feature_vector_sha256: str
    prediction_record_sha256: str
    receipt_sha256: str | None
    probability_kind: str
    scheduled_puck_drop_utc: str
    pregame_timing_valid: bool

    @property
    def frozen_artifact_sha256_short(self) -> str:
        return shorten_sha256(self.frozen_artifact_sha256)

    @property
    def source_sha256_short(self) -> str | None:
        if self.source_sha256 is None:
            return None
        return shorten_sha256(self.source_sha256)

    @property
    def feature_vector_sha256_short(self) -> str:
        return shorten_sha256(self.feature_vector_sha256)

    @property
    def prediction_record_sha256_short(self) -> str:
        return shorten_sha256(self.prediction_record_sha256)

    @property
    def receipt_sha256_short(self) -> str | None:
        if self.receipt_sha256 is None:
            return None
        return shorten_sha256(self.receipt_sha256)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frozen_artifact_sha256": self.frozen_artifact_sha256,
            "frozen_artifact_sha256_short": self.frozen_artifact_sha256_short,
            "prediction_timestamp_utc": self.prediction_timestamp_utc,
            "source_sha256": self.source_sha256,
            "source_sha256_short": self.source_sha256_short,
            "feature_vector_sha256": self.feature_vector_sha256,
            "feature_vector_sha256_short": self.feature_vector_sha256_short,
            "prediction_record_sha256": self.prediction_record_sha256,
            "prediction_record_sha256_short": self.prediction_record_sha256_short,
            "receipt_sha256": self.receipt_sha256,
            "receipt_sha256_short": self.receipt_sha256_short,
            "probability_kind": self.probability_kind,
            "scheduled_puck_drop_utc": self.scheduled_puck_drop_utc,
            "pregame_timing_valid": self.pregame_timing_valid,
        }


@dataclass(frozen=True, slots=True)
class GameCard:
    nhl_game_id: str
    away_team: str
    home_team: str
    game_state: str
    game_schedule_state: str
    game_type: str
    season: int
    game_date: str
    scheduled_start_time_utc: str
    scheduled_utc_label: str
    scheduled_local_label: str
    display_timezone: str
    prediction_status: str
    home_model_win_probability: float | None
    away_model_win_probability: float | None
    descriptive_model_output: str | None
    higher_probability_side: str | None
    audit: AuditDetail | None
    receipt_blocker: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "nhl_game_id": self.nhl_game_id,
            "away_team": self.away_team,
            "home_team": self.home_team,
            "game_state": self.game_state,
            "game_schedule_state": self.game_schedule_state,
            "game_type": self.game_type,
            "season": self.season,
            "game_date": self.game_date,
            "scheduled_start_time_utc": self.scheduled_start_time_utc,
            "scheduled_utc_label": self.scheduled_utc_label,
            "scheduled_local_label": self.scheduled_local_label,
            "display_timezone": self.display_timezone,
            "prediction_status": self.prediction_status,
            "home_model_win_probability": self.home_model_win_probability,
            "away_model_win_probability": self.away_model_win_probability,
            "descriptive_model_output": self.descriptive_model_output,
            "higher_probability_side": self.higher_probability_side,
            "audit": None if self.audit is None else self.audit.to_dict(),
            "receipt_blocker": self.receipt_blocker,
            "market_layer": MARKET_LAYER_ABSENT,
        }


@dataclass(frozen=True, slots=True)
class ProtectedRecordRow:
    game_id: str
    away_team_id: str
    home_team_id: str
    game_label: str
    prediction_timestamp_utc: str
    home_model_win_probability: float
    away_model_win_probability: float
    probability_kind: str
    prediction_status: str
    record_sha256: str
    receipt_sha256: str | None
    receipt_blocker: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "away_team_id": self.away_team_id,
            "home_team_id": self.home_team_id,
            "game_label": self.game_label,
            "prediction_timestamp_utc": self.prediction_timestamp_utc,
            "home_model_win_probability": self.home_model_win_probability,
            "away_model_win_probability": self.away_model_win_probability,
            "probability_kind": self.probability_kind,
            "prediction_status": self.prediction_status,
            "record_sha256": self.record_sha256,
            "record_sha256_short": shorten_sha256(self.record_sha256),
            "receipt_sha256": self.receipt_sha256,
            "receipt_sha256_short": (
                None if self.receipt_sha256 is None else shorten_sha256(self.receipt_sha256)
            ),
            "receipt_blocker": self.receipt_blocker,
            "market_layer": MARKET_LAYER_ABSENT,
        }


@dataclass(frozen=True, slots=True)
class ModelStatus:
    model_family: str
    feature_family: str
    feature_column_count: int
    artifact_version: str
    artifact_sha256: str
    prospective_status: str
    historical_accuracy: str
    historical_roc_auc: str
    historical_log_loss: str
    historical_brier: str
    historical_ece: str
    historical_note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_family": self.model_family,
            "feature_family": self.feature_family,
            "feature_column_count": self.feature_column_count,
            "artifact_version": self.artifact_version,
            "artifact_sha256": self.artifact_sha256,
            "artifact_sha256_short": shorten_sha256(self.artifact_sha256),
            "prospective_status": self.prospective_status,
            "historical_accuracy": self.historical_accuracy,
            "historical_roc_auc": self.historical_roc_auc,
            "historical_log_loss": self.historical_log_loss,
            "historical_brier": self.historical_brier,
            "historical_ece": self.historical_ece,
            "historical_note": self.historical_note,
            "market_layer": MARKET_LAYER_ABSENT,
        }


@dataclass(frozen=True, slots=True)
class DashboardView:
    selected_date: date
    selected_date_rule: str
    snapshot_path: str
    snapshot_observed_at_utc: str
    covered_dates: tuple[date, ...]
    store_path: str
    store_present: bool
    games: tuple[GameCard, ...]
    records: tuple[ProtectedRecordRow, ...]
    model_status: ModelStatus
    disclaimer: str
    empty_slate_message: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_date": self.selected_date.isoformat(),
            "selected_date_rule": self.selected_date_rule,
            "snapshot_path": self.snapshot_path,
            "snapshot_observed_at_utc": self.snapshot_observed_at_utc,
            "covered_dates": [day.isoformat() for day in self.covered_dates],
            "store_path": self.store_path,
            "store_present": self.store_present,
            "games": [game.to_dict() for game in self.games],
            "records": [row.to_dict() for row in self.records],
            "model_status": self.model_status.to_dict(),
            "disclaimer": self.disclaimer,
            "empty_slate_message": self.empty_slate_message,
            "market_layer": MARKET_LAYER_ABSENT,
        }


@dataclass(frozen=True, slots=True)
class _PresentedRecord:
    record: dict[str, Any]
    record_sha256: str
    status: str
    home_probability: float
    away_probability: float
    audit: AuditDetail
    receipt_blocker: str | None


def shorten_sha256(value: str) -> str:
    """Shorten a SHA-256 digest for a card while leaving the full value intact."""

    if not isinstance(value, str) or not _SHA256_HEX.fullmatch(value):
        raise DashboardDataError(f"Expected a SHA-256 hex digest, got {value!r}")
    return f"{value[:12]}…{value[-8:]}"


def format_model_probability(probability: float) -> str:
    return f"{probability * 100:.1f}%"


def load_nhl_snapshot(path: Path | str) -> NHLScheduleSnapshot:
    """Load a local NHL snapshot. Missing or malformed files fail visibly."""

    source = Path(path)
    if not source.is_file():
        raise DashboardDataError(
            f"NHL schedule snapshot is missing: {source}. "
            "The dashboard does not fetch a slate or fabricate games."
        )
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise DashboardDataError(f"NHL schedule snapshot is unreadable: {source}") from exc
    except json.JSONDecodeError as exc:
        raise DashboardDataError(
            f"NHL schedule snapshot is not valid JSON: {source}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise DashboardDataError(f"NHL schedule snapshot must be a JSON object: {source}")
    snapshot = _snapshot_from_payload(payload, source)
    if not snapshot.covered_dates:
        raise DashboardDataError(
            "NHL snapshot does not declare covered dates. "
            "Refusing to guess which dates were loaded."
        )
    return snapshot


def build_dashboard_view(
    *,
    snapshot_path: Path | str,
    store_path: Path | str | None = None,
    selected_date: date | None = None,
    display_timezone: str = DEFAULT_DISPLAY_TIMEZONE,
) -> DashboardView:
    """Assemble one read-only dashboard view from local snapshot and records."""

    snapshot_file = Path(snapshot_path)
    store_file = Path(store_path) if store_path is not None else Path(DEFAULT_STORE_RELATIVE_PATH)
    snapshot = load_nhl_snapshot(snapshot_file)
    zone = _display_zone(display_timezone)
    chosen_date, date_rule = _select_date(snapshot, selected_date)
    store = PredictionRecordStore(store_file)
    try:
        envelopes = list(store.iter_envelopes())
    except PredictionError as exc:
        raise DashboardDataError(
            f"Prediction-record store is malformed: {exc}"
        ) from exc
    presented = {
        item.record["game_id"]: item for item in (_present_envelope(envelope) for envelope in envelopes)
    }
    games = tuple(
        _game_card(game, presented.get(game.nhl_game_id), zone, display_timezone)
        for game in snapshot.games
        if game.game_date == chosen_date
    )
    records = tuple(
        _record_row(item)
        for item in sorted(
            presented.values(),
            key=lambda item: (item.audit.prediction_timestamp_utc, item.record["game_id"]),
            reverse=True,
        )
    )
    empty_message = None
    if not games:
        empty_message = (
            f"No games on {chosen_date.isoformat()} in the loaded NHL snapshot. "
            "Nothing was fabricated for this date."
        )
    return DashboardView(
        selected_date=chosen_date,
        selected_date_rule=date_rule,
        snapshot_path=str(snapshot_file),
        snapshot_observed_at_utc=snapshot.observed_at_utc.astimezone(timezone.utc).isoformat(),
        covered_dates=tuple(sorted(snapshot.covered_dates)),
        store_path=str(store_file),
        store_present=store_file.is_file(),
        games=games,
        records=records,
        model_status=load_model_status(),
        disclaimer=DASHBOARD_DISCLAIMER,
        empty_slate_message=empty_message,
    )


def load_model_status() -> ModelStatus:
    """Read committed frozen-artifact metadata. Does not fit or score."""

    try:
        artifact = load_committed_frozen_artifact()
    except PredictionError as exc:
        raise DashboardDataError(f"Frozen artifact is missing or malformed: {exc}") from exc
    if artifact.fitted_artifact_sha256 != EXPECTED_FITTED_ARTIFACT_SHA256:
        raise DashboardDataError(
            "Frozen artifact SHA-256 does not match the committed fingerprint "
            f"{EXPECTED_FITTED_ARTIFACT_SHA256}"
        )
    if len(artifact.feature_columns) != 25:
        raise DashboardDataError(
            "Frozen artifact is not the 25-column Cycle-2 candidate "
            f"(found {len(artifact.feature_columns)} columns)"
        )
    if artifact.feature_family != FROZEN_CANDIDATE_FAMILY:
        raise DashboardDataError("Frozen artifact feature family drifted from the Cycle-2 candidate")
    if "StandardScaler" not in artifact.model_family or "LogisticRegression" not in artifact.model_family:
        raise DashboardDataError(
            f"Frozen model family is not StandardScaler + LogisticRegression: {artifact.model_family}"
        )
    return ModelStatus(
        model_family=artifact.model_family,
        feature_family=artifact.feature_family,
        feature_column_count=len(artifact.feature_columns),
        artifact_version=artifact.artifact_version,
        artifact_sha256=artifact.fitted_artifact_sha256,
        prospective_status=PROSPECTIVE_STATUS,
        historical_accuracy=HISTORICAL_FORWARD_CHAINING_MACROS["accuracy"],
        historical_roc_auc=HISTORICAL_FORWARD_CHAINING_MACROS["roc_auc"],
        historical_log_loss=HISTORICAL_FORWARD_CHAINING_MACROS["log_loss"],
        historical_brier=HISTORICAL_FORWARD_CHAINING_MACROS["brier"],
        historical_ece=HISTORICAL_FORWARD_CHAINING_MACROS["ece"],
        historical_note=HISTORICAL_RESULTS_NOTE,
    )


def render_game_card_text(card: GameCard) -> str:
    """Plain-text card used by tests and as the accessible content contract."""

    lines = [
        f"{card.away_team} at {card.home_team}",
        card.scheduled_local_label,
        card.scheduled_utc_label,
        f"NHL game ID {card.nhl_game_id}",
        f"Game state {card.game_state}",
        f"Schedule state {card.game_schedule_state}",
        card.prediction_status,
    ]
    if card.prediction_status == STATUS_NOT_SCORED:
        if card.home_model_win_probability is not None or card.away_model_win_probability is not None:
            raise DashboardDataError("NOT SCORED game included a model probability")
        if card.audit is not None:
            raise DashboardDataError("NOT SCORED game included audit provenance")
        return "\n".join((*lines, "Model probability is not shown."))
    if card.home_model_win_probability is None or card.away_model_win_probability is None:
        raise DashboardDataError("Recorded game is missing a model probability")
    if card.audit is None or card.descriptive_model_output is None:
        raise DashboardDataError("Recorded game is missing descriptive audit output")
    recorded = (
        f"Home model win probability {format_model_probability(card.home_model_win_probability)}",
        f"Away model win probability {format_model_probability(card.away_model_win_probability)}",
        card.descriptive_model_output,
        "Not a betting recommendation.",
        DASHBOARD_DISCLAIMER,
        "Audit",
        f"Frozen artifact SHA-256 {card.audit.frozen_artifact_sha256}",
        f"Frozen artifact short {card.audit.frozen_artifact_sha256_short}",
        f"Prediction timestamp UTC {card.audit.prediction_timestamp_utc}",
        f"Source SHA-256 {card.audit.source_sha256 or 'not present'}",
        f"Feature-vector SHA-256 {card.audit.feature_vector_sha256}",
        f"Prediction-record SHA-256 {card.audit.prediction_record_sha256}",
        f"Receipt SHA-256 {card.audit.receipt_sha256 or 'not present'}",
        f"Probability kind {card.audit.probability_kind}",
        f"Scheduled puck drop {card.audit.scheduled_puck_drop_utc}",
        f"Pregame timing valid {card.audit.pregame_timing_valid}",
    )
    if card.receipt_blocker:
        recorded = (*recorded, f"Receipt blocker {card.receipt_blocker}")
    return "\n".join((*lines, *recorded))


def render_model_status_text(status: ModelStatus) -> str:
    return "\n".join(
        [
            status.model_family,
            status.feature_family,
            f"{status.feature_column_count} columns",
            status.artifact_version,
            status.artifact_sha256,
            status.prospective_status,
            f"Accuracy {status.historical_accuracy}",
            f"ROC-AUC {status.historical_roc_auc}",
            f"Log loss {status.historical_log_loss}",
            f"Brier {status.historical_brier}",
            f"ECE {status.historical_ece}",
            status.historical_note,
            MARKET_LAYER_ABSENT,
        ]
    )


def assert_dashboard_modules_are_presentation_only() -> None:
    """Fail if the dashboard modules import clients or call scoring/fitting."""

    module_dir = Path(__file__).resolve().parent
    for name in _DASHBOARD_MODULE_NAMES:
        path = module_dir / name
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            modules: tuple[str, ...] = ()
            if isinstance(node, ast.Import):
                modules = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = (node.module,)
            for module_name in modules:
                root = module_name.split(".", 1)[0]
                if root in _FORBIDDEN_IMPORT_ROOTS:
                    raise DashboardDataError(
                        f"{path.name} imports a scoring, network, or GitHub client: {module_name}"
                    )
            called = _called_name(node)
            if called in _FORBIDDEN_CALLS or called == "append":
                raise DashboardDataError(f"{path.name} calls {called}")


def _called_name(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _snapshot_from_payload(payload: Mapping[str, Any], source: Path) -> NHLScheduleSnapshot:
    if _is_rehydrated_snapshot(payload):
        try:
            return nhl_snapshot_from_dict(payload)
        except (PredictionError, KeyError, TypeError, ValueError) as exc:
            raise DashboardDataError(f"NHL snapshot is malformed: {exc}") from exc
    observed = payload.get("observed_at_utc")
    if not isinstance(observed, str) or not observed.strip():
        raise DashboardDataError(
            "NHL schedule payload is missing observed_at_utc. "
            "The dashboard does not invent an observation time."
        )
    try:
        observed_at = datetime.fromisoformat(observed.replace("Z", "+00:00"))
        return parse_schedule_payload(
            payload,
            endpoint_path=str(payload.get("endpoint_path") or source.as_posix()),
            observed_at_utc=observed_at,
        )
    except (NHLApiError, TypeError, ValueError) as exc:
        raise DashboardDataError(f"NHL schedule payload is malformed: {exc}") from exc


def _is_rehydrated_snapshot(payload: Mapping[str, Any]) -> bool:
    games = payload.get("games")
    if isinstance(games, list) and games and isinstance(games[0], Mapping):
        return "nhl_game_id" in games[0]
    return "covered_dates" in payload and "games" in payload and "gameWeek" not in payload


def _display_zone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError as exc:
        raise DashboardDataError(
            f"Display timezone {name} is unavailable. Refusing to invent a local clock."
        ) from exc


def _select_date(
    snapshot: NHLScheduleSnapshot,
    selected_date: date | None,
) -> tuple[date, str]:
    covered = set(snapshot.covered_dates)
    if selected_date is not None:
        if selected_date not in covered:
            listed = ", ".join(day.isoformat() for day in sorted(covered))
            raise DashboardDataError(
                f"Selected date {selected_date.isoformat()} is outside the loaded NHL "
                f"snapshot coverage ({listed}). Refusing to invent a slate."
            )
        return selected_date, "operator_selected"
    today = datetime.now(timezone.utc).date()
    if today in covered:
        return today, "utc_today_in_snapshot_coverage"
    latest = max(covered)
    return latest, "latest_covered_date"


def _present_envelope(envelope: Mapping[str, Any]) -> _PresentedRecord:
    record = envelope.get("record")
    if not isinstance(record, Mapping):
        raise DashboardDataError("Prediction record envelope is missing its record")
    record_body = dict(record)
    home = _require_probability(record_body.get("model_home_win_probability"))
    away = 1.0 - home
    source_state = _source_state(record_body)
    if source_state == "malformed":
        raise DashboardDataError(
            f"Prediction record {record_body.get('game_id')} source SHA-256 is malformed"
        )
    if source_state == "absent":
        audit = _audit_from_record(record_body, envelope, receipt=None)
        status = STATUS_RECORDED
        blocker = _ABSENT_SOURCE_MESSAGE
    else:
        try:
            receipt = build_pregame_audit_receipt(envelope)
        except PredictionError as exc:
            raise DashboardDataError(f"Prediction record is malformed: {exc}") from exc
        audit = _audit_from_record(record_body, envelope, receipt=receipt)
        status = STATUS_RECEIPT_READY
        blocker = None
    if not audit.pregame_timing_valid:
        raise DashboardDataError(
            f"Prediction record {record_body.get('game_id')} is not strictly before puck drop"
        )
    return _PresentedRecord(
        record=record_body,
        record_sha256=str(envelope.get("record_sha256") or ""),
        status=status,
        home_probability=home,
        away_probability=away,
        audit=audit,
        receipt_blocker=blocker,
    )


def _audit_from_record(
    record: Mapping[str, Any],
    envelope: Mapping[str, Any],
    *,
    receipt: PregameAuditReceipt | None,
) -> AuditDetail:
    if receipt is not None:
        predicted = receipt.prediction_timestamp_utc
        scheduled = receipt.scheduled_start_time_utc
        source_sha = receipt.current_season_source_sha256
        vector_sha = receipt.feature_vector_sha256
        record_sha = receipt.prediction_record_sha256
        artifact_sha = receipt.frozen_artifact_sha256
        kind = receipt.probability_kind
        receipt_sha = receipt.receipt_sha256
    else:
        predicted = str(record.get("prediction_timestamp_utc") or "")
        scheduled = str(record.get("scheduled_start_time_utc") or "")
        source_sha = _optional_source_sha(record)
        vector_sha = str(record.get("feature_vector_sha256") or "")
        record_sha = str(envelope.get("record_sha256") or "")
        artifact_sha = str(record.get("model_artifact_sha256") or "")
        kind = str(record.get("probability_kind") or "")
        receipt_sha = None
    predicted_at = _parse_utc(predicted, label="prediction timestamp")
    scheduled_at = _parse_utc(scheduled, label="scheduled puck drop")
    return AuditDetail(
        frozen_artifact_sha256=artifact_sha,
        prediction_timestamp_utc=predicted_at.isoformat(),
        source_sha256=source_sha,
        feature_vector_sha256=vector_sha,
        prediction_record_sha256=record_sha,
        receipt_sha256=receipt_sha,
        probability_kind=kind,
        scheduled_puck_drop_utc=scheduled_at.isoformat(),
        pregame_timing_valid=predicted_at < scheduled_at,
    )


def _source_state(record: Mapping[str, Any]) -> str:
    source_versions = record.get("source_versions")
    if not isinstance(source_versions, Mapping):
        return "absent"
    if CURRENT_SEASON_PRIOR_SOURCE_SHA256_KEY not in source_versions:
        return "absent"
    value = source_versions.get(CURRENT_SEASON_PRIOR_SOURCE_SHA256_KEY)
    if value in (None, ""):
        return "absent"
    if isinstance(value, str) and _SHA256_HEX.fullmatch(value):
        return "present"
    return "malformed"


def _optional_source_sha(record: Mapping[str, Any]) -> str | None:
    state = _source_state(record)
    if state == "malformed":
        raise DashboardDataError("Prediction record source SHA-256 is malformed")
    if state != "present":
        return None
    source_versions = record.get("source_versions")
    if not isinstance(source_versions, Mapping):
        return None
    value = source_versions.get(CURRENT_SEASON_PRIOR_SOURCE_SHA256_KEY)
    return str(value) if isinstance(value, str) else None


def _require_probability(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DashboardDataError("Model home-win probability is missing or malformed")
    probability = float(value)
    if not 0.0 < probability < 1.0:
        raise DashboardDataError("Model home-win probability must be inside (0, 1)")
    return probability


def _parse_utc(value: str, *, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DashboardDataError(f"{label} is not a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise DashboardDataError(f"{label} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _same_instant(recorded: str, scheduled: datetime) -> bool:
    return _parse_utc(recorded, label="scheduled puck drop") == scheduled.astimezone(timezone.utc)


def _require_snapshot_agreement(game: NHLScheduleGame, presented: _PresentedRecord) -> None:
    record = presented.record
    home = nhl_abbrev_for_source(str(record.get("home_team_id") or ""), game.season)
    away = nhl_abbrev_for_source(str(record.get("away_team_id") or ""), game.season)
    scheduled = record.get("scheduled_start_time_utc")
    scheduled_ok = isinstance(scheduled, str) and _same_instant(
        scheduled,
        game.scheduled_start_time_utc,
    )
    reasons = tuple(
        reason
        for reason in (
            None if str(record.get("game_id")) == game.nhl_game_id else "game id",
            None
            if int(record.get("season", -1)) == game.season
            else f"season {record.get('season')} != {game.season}",
            None
            if str(record.get("game_type")) == game.game_type
            else f"game_type {record.get('game_type')!r} != {game.game_type!r}",
            None if home == game.home_team_abbrev else f"home {home!r} != {game.home_team_abbrev!r}",
            None if away == game.away_team_abbrev else f"away {away!r} != {game.away_team_abbrev!r}",
            None if scheduled_ok else "scheduled puck drop",
        )
        if reason is not None
    )
    if reasons:
        raise DashboardDataError(
            f"Prediction record {game.nhl_game_id} disagrees with the NHL snapshot: {list(reasons)}"
        )


def _clock_labels(instant: datetime, zone: ZoneInfo, timezone_name: str) -> tuple[str, str]:
    utc = instant.astimezone(timezone.utc)
    local = instant.astimezone(zone)
    utc_label = utc.strftime("%Y-%m-%d %H:%M UTC")
    local_label = local.strftime("%Y-%m-%d %H:%M ET") + f" ({timezone_name})"
    return utc_label, local_label


def _descriptive_output(
    *,
    home_team: str,
    away_team: str,
    home_probability: float,
    away_probability: float,
) -> tuple[str, str]:
    if home_probability > away_probability:
        return (
            "home",
            f"Descriptive model output: {home_team} has the higher model win probability.",
        )
    if away_probability > home_probability:
        return (
            "away",
            f"Descriptive model output: {away_team} has the higher model win probability.",
        )
    return (
        "even",
        "Descriptive model output: model win probability is even.",
    )


def _game_card(
    game: NHLScheduleGame,
    presented: _PresentedRecord | None,
    zone: ZoneInfo,
    timezone_name: str,
) -> GameCard:
    utc_label, local_label = _clock_labels(
        game.scheduled_start_time_utc,
        zone,
        timezone_name,
    )
    if presented is None:
        return GameCard(
            nhl_game_id=game.nhl_game_id,
            away_team=game.away_team_abbrev,
            home_team=game.home_team_abbrev,
            game_state=game.game_state,
            game_schedule_state=game.game_schedule_state,
            game_type=game.game_type,
            season=game.season,
            game_date=game.game_date.isoformat(),
            scheduled_start_time_utc=game.scheduled_start_time_utc.astimezone(timezone.utc).isoformat(),
            scheduled_utc_label=utc_label,
            scheduled_local_label=local_label,
            display_timezone=timezone_name,
            prediction_status=STATUS_NOT_SCORED,
            home_model_win_probability=None,
            away_model_win_probability=None,
            descriptive_model_output=None,
            higher_probability_side=None,
            audit=None,
            receipt_blocker=None,
        )
    _require_snapshot_agreement(game, presented)
    side, description = _descriptive_output(
        home_team=game.home_team_abbrev,
        away_team=game.away_team_abbrev,
        home_probability=presented.home_probability,
        away_probability=presented.away_probability,
    )
    return GameCard(
        nhl_game_id=game.nhl_game_id,
        away_team=game.away_team_abbrev,
        home_team=game.home_team_abbrev,
        game_state=game.game_state,
        game_schedule_state=game.game_schedule_state,
        game_type=game.game_type,
        season=game.season,
        game_date=game.game_date.isoformat(),
        scheduled_start_time_utc=game.scheduled_start_time_utc.astimezone(timezone.utc).isoformat(),
        scheduled_utc_label=utc_label,
        scheduled_local_label=local_label,
        display_timezone=timezone_name,
        prediction_status=presented.status,
        home_model_win_probability=presented.home_probability,
        away_model_win_probability=presented.away_probability,
        descriptive_model_output=description,
        higher_probability_side=side,
        audit=presented.audit,
        receipt_blocker=presented.receipt_blocker,
    )


def _record_row(presented: _PresentedRecord) -> ProtectedRecordRow:
    return ProtectedRecordRow(
        game_id=str(presented.record["game_id"]),
        away_team_id=str(presented.record["away_team_id"]),
        home_team_id=str(presented.record["home_team_id"]),
        game_label=f"{presented.record['away_team_id']} at {presented.record['home_team_id']}",
        prediction_timestamp_utc=presented.audit.prediction_timestamp_utc,
        home_model_win_probability=presented.home_probability,
        away_model_win_probability=presented.away_probability,
        probability_kind=presented.audit.probability_kind,
        prediction_status=presented.status,
        record_sha256=presented.audit.prediction_record_sha256,
        receipt_sha256=presented.audit.receipt_sha256,
        receipt_blocker=presented.receipt_blocker,
    )
