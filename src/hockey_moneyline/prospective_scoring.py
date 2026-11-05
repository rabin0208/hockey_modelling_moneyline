"""Issue #52 pregame model-probability scoring on the frozen 25-column candidate.

Consumes the merged Issue #45 unscored-target snapshot without changing feature
semantics. Emits auditable model-probability records only. No odds, market,
ROI, staking, or recommendation fields are produced.

Protected 2026-27 scoring still requires current-season prior-state source
identity. Issue #54 (`hockey_moneyline.current_season_prior`) is the acquisition
path that supplies those keys; this module does not change scoring semantics.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from hockey_moneyline.advanced_stats import AdvancedTeamGame
from hockey_moneyline.errors import FeatureError, PredictionError
from hockey_moneyline.provenance import MIN_PRIOR_GAMES
from hockey_moneyline.prospective_artifact import (
    FEATURE_SCHEMA_VERSION,
    FROZEN_FEATURE_COLUMNS,
    FrozenFittedArtifact,
    assert_frozen_prospective_contract,
    predict_home_win_probability,
    resolve_scoring_artifact,
)
from hockey_moneyline.schedule_load import (
    RESERVED_FUTURE_SEASON,
    ScheduleHistoryGame,
)
from hockey_moneyline.schema import NormalizedGame
from hockey_moneyline.unscored_target import (
    FrozenFeatureSnapshot,
    UnscoredTarget,
    build_historical_replay_snapshot,
    build_unscored_frozen_feature_snapshot,
)

FORBIDDEN_MARKET_FIELDS = (
    "market_probability",
    "implied_probability",
    "fair_probability",
    "edge",
    "roi",
    "stake",
    "recommendation",
    "odds",
)
FORBIDDEN_TARGET_OUTCOME_FIELDS = (
    "home_score",
    "away_score",
    "home_win",
    "xGoalsFor",
    "xGoalsAgainst",
)
CURRENT_SEASON_PRIOR_SOURCE_SHA256_KEY = "current_season_prior_source_sha256"
CURRENT_SEASON_PRIOR_SOURCE_OBSERVED_AT_KEY = (
    "current_season_prior_source_observed_at_utc"
)
CURRENT_SEASON_PRIOR_SOURCE_IDENTITY_KEYS = (
    CURRENT_SEASON_PRIOR_SOURCE_SHA256_KEY,
    CURRENT_SEASON_PRIOR_SOURCE_OBSERVED_AT_KEY,
)
_SHA256_HEX = re.compile(r"^[0-9a-fA-F]{64}$")


def _aware_utc(value: datetime, *, label: str) -> datetime:
    if value.tzinfo is None:
        raise PredictionError(f"{label} must be timezone-aware")
    return value.astimezone(timezone.utc)


def require_pregame_timestamps(target: UnscoredTarget) -> tuple[datetime, datetime]:
    """Require a scheduled start and a strictly earlier observation."""

    if target.scheduled_start_time_utc is None or target.observed_at_utc is None:
        raise PredictionError(
            "Pregame scoring requires scheduled_start_time_utc and observed_at_utc"
        )
    start = _aware_utc(target.scheduled_start_time_utc, label="scheduled_start_time_utc")
    observed = _aware_utc(target.observed_at_utc, label="observed_at_utc")
    if observed >= start:
        raise PredictionError(
            "Prediction/observation timestamp must occur before scheduled puck drop"
        )
    return start, observed


def _reject_preseason_history(completed_history: Sequence[NormalizedGame]) -> None:
    bad = [game.game_id for game in completed_history if game.game_type != "regular"]
    if bad:
        raise PredictionError(
            "Pregame scoring rejects preseason/non-regular completed history; "
            f"ids={bad[:10]}"
        )


def _is_sha256_hex(value: object) -> bool:
    return isinstance(value, str) and bool(_SHA256_HEX.fullmatch(value))


def _is_aware_iso_timestamp(value: object) -> bool:
    if isinstance(value, datetime):
        return value.tzinfo is not None
    if not isinstance(value, str) or not value:
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def require_current_season_prior_source_identity(
    source_versions: Mapping[str, Any] | None,
) -> None:
    """Require durable current-season prior-state content identity + timestamp."""

    extra = dict(source_versions or {})
    sha256 = extra.get(CURRENT_SEASON_PRIOR_SOURCE_SHA256_KEY)
    observed_at = extra.get(CURRENT_SEASON_PRIOR_SOURCE_OBSERVED_AT_KEY)
    if _is_sha256_hex(sha256) and _is_aware_iso_timestamp(observed_at):
        return
    raise PredictionError(
        "NOT READY — ONE MATERIAL PROSPECTIVE-SCORING DEPENDENCY: "
        "current-season prior-state source identity is absent; "
        "protected 2026-27 scoring requires a durable content/version/hash "
        f"({CURRENT_SEASON_PRIOR_SOURCE_SHA256_KEY}) plus a timezone-aware "
        "observation/retrieval timestamp "
        f"({CURRENT_SEASON_PRIOR_SOURCE_OBSERVED_AT_KEY}) sufficient to "
        "reproduce the prior-state snapshot"
    )


def require_current_season_prior_source(
    target: UnscoredTarget,
    *,
    current_season_prior_source_available: bool | None,
    source_versions: Mapping[str, Any] | None = None,
) -> None:
    """Fail closed when 2026-27 prior history or its source identity is missing."""

    if target.season < RESERVED_FUTURE_SEASON:
        return
    if current_season_prior_source_available is not True:
        raise PredictionError(
            "NOT READY — ONE MATERIAL PROSPECTIVE-SCORING DEPENDENCY: "
            "current-season prior completed history is unavailable under the "
            "approved MoneyPuck/NHL contracts; refuse to fabricate, impute, or "
            "silently freeze stale current-season state"
        )
    require_current_season_prior_source_identity(source_versions)


def feature_vector_sha256(values: Sequence[float]) -> str:
    payload = json.dumps(list(values), separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def score_frozen_snapshot(
    snapshot: FrozenFeatureSnapshot,
    artifact: FrozenFittedArtifact | None = None,
) -> float:
    """Score an Issue #45 snapshot without mutating its 25 values."""

    artifact = resolve_scoring_artifact(artifact)
    assert_frozen_prospective_contract()
    if snapshot.feature_columns != tuple(FROZEN_FEATURE_COLUMNS):
        raise PredictionError(
            "Issue #45 snapshot feature columns drifted from the frozen 25-column order"
        )
    if artifact.feature_columns != tuple(FROZEN_FEATURE_COLUMNS):
        raise PredictionError("Artifact feature columns drifted from the frozen order")
    if snapshot.feature_columns != artifact.feature_columns:
        raise PredictionError("Snapshot and artifact feature columns do not match")
    values = tuple(float(value) for value in snapshot.feature_values)
    if values != snapshot.feature_values:
        raise PredictionError("Scoring must not change Issue #45 snapshot values")
    return predict_home_win_probability(artifact, values)


@dataclass(frozen=True, slots=True)
class PregamePredictionRecord:
    """Auditable model-probability record. Market fields are intentionally absent."""

    game_id: str
    season: int
    game_type: str
    home_team_id: str
    away_team_id: str
    scheduled_start_time_utc: str
    prediction_timestamp_utc: str
    model_artifact_version: str
    model_artifact_sha256: str
    feature_schema_version: str
    source_versions: dict[str, Any]
    feature_columns: tuple[str, ...]
    feature_values: tuple[float, ...]
    feature_vector_sha256: str
    model_home_win_probability: float
    probability_kind: str = "model_home_win"

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "game_id": self.game_id,
            "season": self.season,
            "game_type": self.game_type,
            "home_team_id": self.home_team_id,
            "away_team_id": self.away_team_id,
            "scheduled_start_time_utc": self.scheduled_start_time_utc,
            "prediction_timestamp_utc": self.prediction_timestamp_utc,
            "model_artifact_version": self.model_artifact_version,
            "model_artifact_sha256": self.model_artifact_sha256,
            "feature_schema_version": self.feature_schema_version,
            "source_versions": dict(self.source_versions),
            "feature_columns": list(self.feature_columns),
            "feature_values": list(self.feature_values),
            "feature_vector_sha256": self.feature_vector_sha256,
            "model_home_win_probability": self.model_home_win_probability,
            "probability_kind": self.probability_kind,
        }
        leaked = [field for field in FORBIDDEN_MARKET_FIELDS if field in payload]
        if leaked:
            raise PredictionError(f"Prediction record leaked market fields: {leaked}")
        leaked_outcomes = [
            field for field in FORBIDDEN_TARGET_OUTCOME_FIELDS if field in payload
        ]
        if leaked_outcomes:
            raise PredictionError(
                f"Prediction record leaked target outcomes: {leaked_outcomes}"
            )
        return payload


def _jsonable_source_versions(extra: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in extra.items():
        if isinstance(value, datetime):
            payload[key] = _aware_utc(value, label=str(key)).isoformat()
        else:
            payload[key] = value
    return payload


def _source_versions(
    artifact: FrozenFittedArtifact,
    target: UnscoredTarget,
    extra: Mapping[str, Any] | None,
) -> dict[str, Any]:
    payload = {
        "historical_cache_sha256": artifact.historical_cache_sha256,
        "code_freeze_commit": artifact.code_freeze_commit,
        "feature_schema_version": artifact.feature_schema_version,
        "target_source": target.source,
        "snapshot_primitive": "issue45_unscored_frozen_feature_snapshot",
    }
    if extra:
        payload.update(_jsonable_source_versions(extra))
    if target.season >= RESERVED_FUTURE_SEASON:
        require_current_season_prior_source_identity(payload)
    return payload


def emit_prediction_record(
    snapshot: FrozenFeatureSnapshot,
    artifact: FrozenFittedArtifact | None = None,
    *,
    scheduled_start_time_utc: datetime,
    prediction_timestamp_utc: datetime,
    source_versions: Mapping[str, Any] | None = None,
) -> PregamePredictionRecord:
    """Build an auditable record from an already-built Issue #45 snapshot."""

    artifact = resolve_scoring_artifact(artifact)
    start = _aware_utc(scheduled_start_time_utc, label="scheduled_start_time_utc")
    observed = _aware_utc(prediction_timestamp_utc, label="prediction_timestamp_utc")
    if observed >= start:
        raise PredictionError(
            "Prediction/observation timestamp must occur before scheduled puck drop"
        )
    target = snapshot.target
    if target.game_type != "regular":
        raise PredictionError(
            f"Frozen prospective target must be regular season, got {target.game_type!r}"
        )
    probability = score_frozen_snapshot(snapshot, artifact)
    values = tuple(snapshot.feature_values)
    return PregamePredictionRecord(
        game_id=target.game_id,
        season=target.season,
        game_type=target.game_type,
        home_team_id=target.home_team_id,
        away_team_id=target.away_team_id,
        scheduled_start_time_utc=start.isoformat(),
        prediction_timestamp_utc=observed.isoformat(),
        model_artifact_version=artifact.artifact_version,
        model_artifact_sha256=artifact.fitted_artifact_sha256,
        feature_schema_version=artifact.feature_schema_version or FEATURE_SCHEMA_VERSION,
        source_versions=_source_versions(artifact, target, source_versions),
        feature_columns=tuple(snapshot.feature_columns),
        feature_values=values,
        feature_vector_sha256=feature_vector_sha256(values),
        model_home_win_probability=probability,
    )


def score_pregame_prediction(
    target: UnscoredTarget,
    completed_history: Sequence[NormalizedGame],
    advanced_records: Mapping[tuple[str, str], AdvancedTeamGame],
    schedule_history: Sequence[ScheduleHistoryGame],
    artifact: FrozenFittedArtifact | None = None,
    *,
    current_season_prior_source_available: bool | None = None,
    source_versions: Mapping[str, Any] | None = None,
    min_prior_games: int = MIN_PRIOR_GAMES,
) -> PregamePredictionRecord:
    """Build the Issue #45 snapshot and emit one auditable pregame record."""

    artifact = resolve_scoring_artifact(artifact)
    assert_frozen_prospective_contract()
    if target.game_type != "regular":
        raise PredictionError(
            f"Frozen prospective target must be regular season, got {target.game_type!r}"
        )
    start, observed = require_pregame_timestamps(target)
    require_current_season_prior_source(
        target,
        current_season_prior_source_available=current_season_prior_source_available,
        source_versions=source_versions,
    )
    _reject_preseason_history(completed_history)
    try:
        snapshot = build_unscored_frozen_feature_snapshot(
            target,
            completed_history,
            advanced_records,
            schedule_history,
            min_prior_games=min_prior_games,
        )
    except FeatureError as exc:
        raise PredictionError(str(exc)) from exc
    return emit_prediction_record(
        snapshot,
        artifact,
        scheduled_start_time_utc=start,
        prediction_timestamp_utc=observed,
        source_versions=source_versions,
    )


def score_historical_replay_prediction(
    target_game: NormalizedGame,
    all_completed_games: Sequence[NormalizedGame],
    advanced_records: Mapping[tuple[str, str], AdvancedTeamGame],
    schedule_history: Sequence[ScheduleHistoryGame],
    artifact: FrozenFittedArtifact | None = None,
    *,
    scheduled_start_time_utc: datetime,
    prediction_timestamp_utc: datetime,
    source_versions: Mapping[str, Any] | None = None,
    min_prior_games: int = MIN_PRIOR_GAMES,
) -> tuple[FrozenFeatureSnapshot, PregamePredictionRecord]:
    """Score a historical target through the Issue #45 snapshot without using its outcome."""

    artifact = resolve_scoring_artifact(artifact)
    assert_frozen_prospective_contract()
    _reject_preseason_history(all_completed_games)
    if target_game.season >= RESERVED_FUTURE_SEASON:
        raise PredictionError(
            "Historical replay dry-run cannot select a 2026-27 target"
        )
    if any(game.season >= RESERVED_FUTURE_SEASON for game in all_completed_games):
        raise PredictionError(
            "Historical replay dry-run cannot consume 2026-27 completed outcomes"
        )
    try:
        snapshot = build_historical_replay_snapshot(
            target_game,
            all_completed_games,
            advanced_records,
            schedule_history,
            min_prior_games=min_prior_games,
        )
    except FeatureError as exc:
        raise PredictionError(str(exc)) from exc
    record = emit_prediction_record(
        snapshot,
        artifact,
        scheduled_start_time_utc=scheduled_start_time_utc,
        prediction_timestamp_utc=prediction_timestamp_utc,
        source_versions=source_versions,
    )
    if record.feature_values != snapshot.feature_values:
        raise PredictionError("Prospective scorer changed the Issue #45 snapshot values")
    return snapshot, record
