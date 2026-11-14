"""Issue #52 frozen prospective fitted artifact.

Fits the already-frozen 25-column logistic candidate exactly once on eligible
development history through consumed 2025-26. Fitted parameters are
fingerprinted before any scored 2026-27 prediction. This module does not
change frozen feature or model semantics.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import sklearn
from sklearn.pipeline import Pipeline

from hockey_moneyline.advanced_stats import (
    AdvancedTeamGame,
    build_pregame_xgoal_share_features,
)
from hockey_moneyline.errors import EvaluationError, PredictionError
from hockey_moneyline.evaluate import (
    LOGISTIC_MAX_ITER,
    LOGISTIC_MODEL_DESCRIPTION,
    LOGISTIC_RANDOM_STATE,
    LOGISTIC_SOLVER,
    _new_logistic_model,
    shared_logistic_model_record,
)
from hockey_moneyline.features import build_features, feature_matrix
from hockey_moneyline.provenance import MIN_PRIOR_GAMES
from hockey_moneyline.schedule_load import (
    CYCLE2_DEVELOPMENT_SEASONS,
    RESERVED_FUTURE_SEASON,
    ScheduleHistoryGame,
    build_prior_schedule_load_features,
)
from hockey_moneyline.schedule_load_ablation import (
    BASELINE_V1_PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE_PLUS_PRIOR_SCHEDULE_LOAD_COMPACT,
    KNOWN_CYCLE2_MONEYPUCK_SHA256,
    PLUS_XGOAL_PLUS_PRIOR_SCHEDULE_LOAD_COMPACT_FEATURE_COLUMNS,
    assert_prior_schedule_load_compact_contract,
    attach_pregame_schedule_load_compact_features,
    schedule_load_compact_features_valid,
)
from hockey_moneyline.schema import NormalizedGame
from hockey_moneyline.unscored_target import FROZEN_ISSUE40_FEATURE_COLUMNS
from hockey_moneyline.xgoal_share import (
    attach_pregame_xgoal_share_features,
    xgoal_features_valid,
)

FROZEN_CANDIDATE_FAMILY = (
    BASELINE_V1_PLUS_ELO_PLUS_PRIOR_XGOAL_SHARE_PLUS_PRIOR_SCHEDULE_LOAD_COMPACT
)
FROZEN_FEATURE_COLUMNS = FROZEN_ISSUE40_FEATURE_COLUMNS
FROZEN_CANDIDATE_MODEL_DEFINITION_COMMIT = (
    "946a5ad34788cb7016da27a491f85332f2346888"
)
FROZEN_HISTORICAL_CACHE_SHA256 = KNOWN_CYCLE2_MONEYPUCK_SHA256
FROZEN_TRAINING_SEASONS = CYCLE2_DEVELOPMENT_SEASONS
ARTIFACT_SCHEMA_VERSION = 1
ARTIFACT_KIND = "frozen_prospective_fitted_artifact"
ARTIFACT_VERSION = "frozen_prospective_v1"
FEATURE_SCHEMA_VERSION = (
    f"{FROZEN_CANDIDATE_FAMILY}@{FROZEN_CANDIDATE_MODEL_DEFINITION_COMMIT}"
)
SYNTHETIC_CACHE_IDENTITY = "synthetic-development-fixture"
COMMITTED_FROZEN_ARTIFACT_RELATIVE_PATH = (
    "src/hockey_moneyline/artifacts/frozen_prospective_fitted_artifact.json"
)
EXPECTED_FITTED_ARTIFACT_SHA256 = (
    "763ef69cce4e414b926546fee09e7a66ff54eab41d695f499f01795f150238df"
)
EXPECTED_TRAINING_ROW_IDENTITY_SHA256 = (
    "f1847cc7808bc081790755b8d10034bc300f058956095a7842f0719f0b8c2719"
)
EXPECTED_FROZEN_TRAINING_ROW_COUNT = 6469


def assert_frozen_prospective_contract() -> None:
    """Guard the Issue #40/#52 frozen candidate, model, and 25-column order."""

    assert_prior_schedule_load_compact_contract()
    if FROZEN_CANDIDATE_FAMILY != (
        "baseline_v1_plus_elo_plus_prior_xgoal_share_plus_prior_schedule_load_compact"
    ):
        raise RuntimeError("frozen prospective candidate family was altered")
    if tuple(FROZEN_FEATURE_COLUMNS) != tuple(
        PLUS_XGOAL_PLUS_PRIOR_SCHEDULE_LOAD_COMPACT_FEATURE_COLUMNS
    ):
        raise RuntimeError("frozen prospective 25-column order was altered")
    if len(FROZEN_FEATURE_COLUMNS) != 25:
        raise RuntimeError("frozen prospective family must remain exactly 25 columns")
    if FROZEN_FEATURE_COLUMNS != (
        "home_prior_games",
        "away_prior_games",
        "home_win_rate",
        "away_win_rate",
        "home_goals_for_per_game",
        "away_goals_for_per_game",
        "home_goals_against_per_game",
        "away_goals_against_per_game",
        "home_goal_diff_per_game",
        "away_goal_diff_per_game",
        "home_home_win_rate",
        "away_away_win_rate",
        "home_season_games",
        "away_season_games",
        "home_season_win_rate",
        "away_season_win_rate",
        "home_recent_win_rate",
        "away_recent_win_rate",
        "home_elo_minus_away_elo",
        "home_prior_xgoal_share",
        "away_prior_xgoal_share",
        "home_back_to_back",
        "away_back_to_back",
        "home_games_prior_7d",
        "away_games_prior_7d",
    ):
        raise RuntimeError("frozen prospective feature order does not match Issue #40")
    if LOGISTIC_SOLVER != "lbfgs":
        raise RuntimeError("frozen logistic solver was altered")
    if LOGISTIC_MAX_ITER != 1000:
        raise RuntimeError("frozen logistic max_iter was altered")
    if LOGISTIC_RANDOM_STATE != 42:
        raise RuntimeError("frozen logistic random_state was altered")
    if LOGISTIC_MODEL_DESCRIPTION != (
        "StandardScaler + LogisticRegression(solver=lbfgs, max_iter=1000, random_state=42)"
    ):
        raise RuntimeError("frozen logistic description was altered")
    if FROZEN_CANDIDATE_MODEL_DEFINITION_COMMIT != (
        "946a5ad34788cb7016da27a491f85332f2346888"
    ):
        raise RuntimeError("frozen candidate/model-definition commit was altered")
    if FROZEN_HISTORICAL_CACHE_SHA256 != (
        "9eb473051e1d2c827990d802ea7d996684f48d3377c8da7e34db051adfc3bcab"
    ):
        raise RuntimeError("frozen historical MoneyPuck cache SHA-256 was altered")
    if FROZEN_TRAINING_SEASONS != (20212022, 20222023, 20232024, 20242025, 20252026):
        raise RuntimeError("frozen training seasons must remain 2021-22 through 2025-26")
    if RESERVED_FUTURE_SEASON != 20262027:
        raise RuntimeError("reserved prospective season must remain 2026-27")
    if MIN_PRIOR_GAMES != 5:
        raise RuntimeError("frozen min-prior-games must remain 5")
    if EXPECTED_FITTED_ARTIFACT_SHA256 != (
        "763ef69cce4e414b926546fee09e7a66ff54eab41d695f499f01795f150238df"
    ):
        raise RuntimeError("expected fitted artifact SHA-256 was altered")
    if EXPECTED_TRAINING_ROW_IDENTITY_SHA256 != (
        "f1847cc7808bc081790755b8d10034bc300f058956095a7842f0719f0b8c2719"
    ):
        raise RuntimeError("expected training-row identity SHA-256 was altered")
    if EXPECTED_FROZEN_TRAINING_ROW_COUNT != 6469:
        raise RuntimeError("expected frozen training-row count was altered")


def _reject_reserved_future_games(games: Sequence[NormalizedGame], *, context: str) -> None:
    future = [game.game_id for game in games if game.season >= RESERVED_FUTURE_SEASON]
    if future:
        raise EvaluationError(
            f"{context} cannot use 2026-27 games; ids={future[:10]}"
        )


def _reject_non_regular_games(games: Sequence[NormalizedGame], *, context: str) -> None:
    bad = [
        game.game_id for game in games if game.game_type != "regular"
    ]
    if bad:
        raise EvaluationError(
            f"{context} rejects preseason/non-regular games; ids={bad[:10]}"
        )


def build_frozen_training_rows(
    games: Sequence[NormalizedGame],
    records: Mapping[tuple[str, str], AdvancedTeamGame],
    schedule_history: Sequence[ScheduleHistoryGame],
) -> list[dict[str, Any]]:
    """Build training-time frozen 25-column rows. Labels are not used as features."""

    assert_frozen_prospective_contract()
    _reject_reserved_future_games(games, context="Frozen prospective fitting")
    _reject_non_regular_games(games, context="Frozen prospective fitting")
    if any(item.season >= RESERVED_FUTURE_SEASON for item in schedule_history):
        raise EvaluationError(
            "Frozen prospective fitting cannot use 2026-27 schedule history"
        )
    rows = build_features(games)
    rows = attach_pregame_xgoal_share_features(
        rows,
        build_pregame_xgoal_share_features(games, records),
    )
    rows = attach_pregame_schedule_load_compact_features(
        rows,
        build_prior_schedule_load_features(schedule_history, games),
    )
    return rows


def eligible_frozen_training_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    min_prior_games: int = MIN_PRIOR_GAMES,
) -> list[dict[str, Any]]:
    """Return eligible frozen-candidate development rows in existing row order."""

    assert_frozen_prospective_contract()
    eligible: list[dict[str, Any]] = []
    for row in rows:
        if int(row["season"]) >= RESERVED_FUTURE_SEASON:
            raise EvaluationError(
                "Frozen prospective fitting received a reserved 2026-27 training row"
            )
        if int(row["season"]) not in FROZEN_TRAINING_SEASONS:
            raise EvaluationError(
                f"Frozen prospective fitting received non-development season {row['season']}"
            )
        if (
            float(row["home_prior_games"]) >= min_prior_games
            and float(row["away_prior_games"]) >= min_prior_games
            and xgoal_features_valid(row)
            and schedule_load_compact_features_valid(row)
        ):
            eligible.append(dict(row))
    if not eligible:
        raise EvaluationError("Frozen prospective fitting has no eligible training rows")
    return eligible


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True, allow_nan=False)


def _sha256_text(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _row_identity_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    joined = ",".join(str(row["game_id"]) for row in rows)
    return _sha256_text(joined)


def _float_list(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.asarray(values, dtype=float).ravel()]


def _reconstruct_pipeline(
    scaler_mean: Sequence[float],
    scaler_scale: Sequence[float],
    classifier_coef: Sequence[float],
    classifier_intercept: Sequence[float],
) -> Pipeline:
    model = _new_logistic_model()
    scaler = model.named_steps["scaler"]
    classifier = model.named_steps["classifier"]
    scaler.mean_ = np.asarray(scaler_mean, dtype=float)
    scaler.scale_ = np.asarray(scaler_scale, dtype=float)
    scaler.var_ = np.square(scaler.scale_)
    scaler.n_features_in_ = len(FROZEN_FEATURE_COLUMNS)
    scaler.n_samples_seen_ = 1
    classifier.coef_ = np.asarray(classifier_coef, dtype=float).reshape(1, -1)
    classifier.intercept_ = np.asarray(classifier_intercept, dtype=float).reshape(-1)
    classifier.classes_ = np.array([0, 1])
    classifier.n_features_in_ = len(FROZEN_FEATURE_COLUMNS)
    return model


@dataclass(frozen=True, slots=True)
class FrozenFittedArtifact:
    """Auditable frozen fitted parameters for prospective model-probability scoring."""

    model_family: str
    model_config: dict[str, Any]
    feature_family: str
    feature_columns: tuple[str, ...]
    feature_schema_version: str
    training_seasons: tuple[int, ...]
    training_window_start: str
    training_window_end: str
    training_row_count: int
    training_row_ids: tuple[str, ...]
    training_row_identity_sha256: str
    code_freeze_commit: str
    historical_cache_sha256: str
    sklearn_version: str
    scaler_mean: tuple[float, ...]
    scaler_scale: tuple[float, ...]
    classifier_coef: tuple[float, ...]
    classifier_intercept: tuple[float, ...]
    fitted_artifact_sha256: str
    artifact_version: str
    created_at_utc: str

    def pipeline(self) -> Pipeline:
        return _reconstruct_pipeline(
            self.scaler_mean,
            self.scaler_scale,
            self.classifier_coef,
            self.classifier_intercept,
        )

    def fingerprint_payload(self) -> dict[str, Any]:
        return {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "kind": ARTIFACT_KIND,
            "model_family": self.model_family,
            "model_config": self.model_config,
            "feature_family": self.feature_family,
            "feature_columns": list(self.feature_columns),
            "feature_schema_version": self.feature_schema_version,
            "training_seasons": list(self.training_seasons),
            "training_window_start": self.training_window_start,
            "training_window_end": self.training_window_end,
            "training_row_count": self.training_row_count,
            "training_row_identity_sha256": self.training_row_identity_sha256,
            "code_freeze_commit": self.code_freeze_commit,
            "historical_cache_sha256": self.historical_cache_sha256,
            "sklearn_version": self.sklearn_version,
            "scaler_mean": list(self.scaler_mean),
            "scaler_scale": list(self.scaler_scale),
            "classifier_coef": list(self.classifier_coef),
            "classifier_intercept": list(self.classifier_intercept),
            "artifact_version": self.artifact_version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.fingerprint_payload()
        payload["training_row_ids"] = list(self.training_row_ids)
        payload["fitted_artifact_sha256"] = self.fitted_artifact_sha256
        payload["created_at_utc"] = self.created_at_utc
        return payload


def _artifact_hash(payload: Mapping[str, Any]) -> str:
    return _sha256_text(_canonical_json(payload))


def fit_frozen_prospective_artifact(
    games: Sequence[NormalizedGame],
    records: Mapping[tuple[str, str], AdvancedTeamGame],
    schedule_history: Sequence[ScheduleHistoryGame],
    *,
    historical_cache_sha256: str,
    created_at_utc: datetime | None = None,
    min_prior_games: int = MIN_PRIOR_GAMES,
) -> FrozenFittedArtifact:
    """Fit the frozen candidate once on eligible 2021-22 through 2025-26 rows."""

    assert_frozen_prospective_contract()
    if historical_cache_sha256 not in {
        FROZEN_HISTORICAL_CACHE_SHA256,
        SYNTHETIC_CACHE_IDENTITY,
    }:
        raise EvaluationError(
            "Frozen prospective fitting requires the frozen MoneyPuck cache "
            f"SHA-256 {FROZEN_HISTORICAL_CACHE_SHA256} or an explicit synthetic "
            "test identity"
        )
    seasons = sorted({game.season for game in games})
    if historical_cache_sha256 == FROZEN_HISTORICAL_CACHE_SHA256:
        if tuple(seasons) != tuple(FROZEN_TRAINING_SEASONS):
            raise EvaluationError(
                "Frozen prospective fitting on the approved cache requires exactly "
                "development seasons 2021-22 through 2025-26"
            )
    elif any(season >= RESERVED_FUTURE_SEASON for season in seasons):
        raise EvaluationError(
            "Frozen prospective fitting cannot include reserved 2026-27 seasons"
        )
    elif any(season not in FROZEN_TRAINING_SEASONS for season in seasons):
        raise EvaluationError(
            "Frozen prospective fitting received a season outside the development window"
        )

    rows = build_frozen_training_rows(games, records, schedule_history)
    eligible = eligible_frozen_training_rows(rows, min_prior_games=min_prior_games)
    train_y = np.array([int(row["home_win"]) for row in eligible], dtype=int)
    if not 0.0 < float(train_y.mean()) < 1.0:
        raise EvaluationError(
            "Frozen prospective training home-win rate must be inside (0, 1)"
        )
    train_x = np.array(
        feature_matrix(eligible, columns=FROZEN_FEATURE_COLUMNS),
        dtype=float,
    )
    if train_x.shape != (len(eligible), len(FROZEN_FEATURE_COLUMNS)):
        raise EvaluationError(
            "Frozen prospective train matrix does not match the 25-column contract"
        )
    model = _new_logistic_model()
    model.fit(train_x, train_y)
    scaler = model.named_steps["scaler"]
    classifier = model.named_steps["classifier"]
    created = created_at_utc or datetime.now(timezone.utc)
    if created.tzinfo is None:
        raise EvaluationError("Artifact creation timestamp must be timezone-aware")
    created_iso = created.astimezone(timezone.utc).isoformat()
    training_ids = tuple(str(row["game_id"]) for row in eligible)
    artifact = FrozenFittedArtifact(
        model_family=LOGISTIC_MODEL_DESCRIPTION,
        model_config=shared_logistic_model_record(),
        feature_family=FROZEN_CANDIDATE_FAMILY,
        feature_columns=tuple(FROZEN_FEATURE_COLUMNS),
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        training_seasons=tuple(sorted({int(row["season"]) for row in eligible})),
        training_window_start=min(str(row["game_date"]) for row in eligible),
        training_window_end=max(str(row["game_date"]) for row in eligible),
        training_row_count=len(eligible),
        training_row_ids=training_ids,
        training_row_identity_sha256=_row_identity_sha256(eligible),
        code_freeze_commit=FROZEN_CANDIDATE_MODEL_DEFINITION_COMMIT,
        historical_cache_sha256=historical_cache_sha256,
        sklearn_version=sklearn.__version__,
        scaler_mean=tuple(_float_list(scaler.mean_)),
        scaler_scale=tuple(_float_list(scaler.scale_)),
        classifier_coef=tuple(_float_list(classifier.coef_)),
        classifier_intercept=tuple(_float_list(classifier.intercept_)),
        fitted_artifact_sha256="",
        artifact_version=ARTIFACT_VERSION,
        created_at_utc=created_iso,
    )
    digest = _artifact_hash(artifact.fingerprint_payload())
    return replace(artifact, fitted_artifact_sha256=digest)


def artifact_from_dict(payload: Mapping[str, Any]) -> FrozenFittedArtifact:
    """Rehydrate a persisted frozen artifact and re-verify its fingerprint."""

    assert_frozen_prospective_contract()
    if int(payload.get("schema_version", 0)) != ARTIFACT_SCHEMA_VERSION:
        raise PredictionError("Frozen artifact schema_version is not recognized")
    if payload.get("kind") != ARTIFACT_KIND:
        raise PredictionError("Persisted object is not a frozen prospective artifact")
    if tuple(payload.get("feature_columns") or ()) != tuple(FROZEN_FEATURE_COLUMNS):
        raise PredictionError("Persisted artifact feature columns drifted from the freeze")
    artifact = FrozenFittedArtifact(
        model_family=str(payload["model_family"]),
        model_config=dict(payload["model_config"]),
        feature_family=str(payload["feature_family"]),
        feature_columns=tuple(payload["feature_columns"]),
        feature_schema_version=str(payload["feature_schema_version"]),
        training_seasons=tuple(int(season) for season in payload["training_seasons"]),
        training_window_start=str(payload["training_window_start"]),
        training_window_end=str(payload["training_window_end"]),
        training_row_count=int(payload["training_row_count"]),
        training_row_ids=tuple(str(item) for item in payload["training_row_ids"]),
        training_row_identity_sha256=str(payload["training_row_identity_sha256"]),
        code_freeze_commit=str(payload["code_freeze_commit"]),
        historical_cache_sha256=str(payload["historical_cache_sha256"]),
        sklearn_version=str(payload["sklearn_version"]),
        scaler_mean=tuple(float(value) for value in payload["scaler_mean"]),
        scaler_scale=tuple(float(value) for value in payload["scaler_scale"]),
        classifier_coef=tuple(float(value) for value in payload["classifier_coef"]),
        classifier_intercept=tuple(
            float(value) for value in payload["classifier_intercept"]
        ),
        fitted_artifact_sha256=str(payload["fitted_artifact_sha256"]),
        artifact_version=str(payload["artifact_version"]),
        created_at_utc=str(payload["created_at_utc"]),
    )
    expected = _artifact_hash(artifact.fingerprint_payload())
    if artifact.fitted_artifact_sha256 != expected:
        raise PredictionError("Frozen artifact fingerprint does not match persisted parameters")
    if artifact.training_row_count != len(artifact.training_row_ids):
        raise PredictionError("Frozen artifact training-row identity count disagrees")
    if artifact.training_row_identity_sha256 != _sha256_text(
        ",".join(artifact.training_row_ids)
    ):
        raise PredictionError("Frozen artifact training-row identity hash disagrees")
    return artifact


def committed_frozen_artifact_path() -> Path:
    """Return the tracked package path of the durable frozen parameter artifact."""

    return (
        Path(__file__).resolve().parent
        / "artifacts"
        / "frozen_prospective_fitted_artifact.json"
    )


def write_frozen_artifact(path: Path | str, artifact: FrozenFittedArtifact) -> Path:
    """Persist fitted parameters + fingerprint metadata without raw MoneyPuck rows."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(artifact.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return destination


def load_frozen_artifact_from_path(path: Path | str) -> FrozenFittedArtifact:
    """Load a persisted frozen artifact and re-verify its fingerprint."""

    source = Path(path)
    if not source.is_file():
        raise PredictionError(f"Frozen artifact file is missing: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PredictionError(f"Frozen artifact file is not valid JSON: {source}") from exc
    if not isinstance(payload, dict):
        raise PredictionError(f"Frozen artifact file is not a JSON object: {source}")
    return artifact_from_dict(payload)


def assert_canonical_frozen_artifact(artifact: FrozenFittedArtifact) -> FrozenFittedArtifact:
    """Require the committed prospective freeze identity, not a synthetic refit."""

    assert_frozen_prospective_contract()
    expected = _artifact_hash(artifact.fingerprint_payload())
    if artifact.fitted_artifact_sha256 != expected:
        raise PredictionError("Frozen artifact fingerprint does not match persisted parameters")
    if artifact.historical_cache_sha256 != FROZEN_HISTORICAL_CACHE_SHA256:
        raise PredictionError(
            "Committed frozen artifact is not bound to the frozen MoneyPuck cache identity"
        )
    if artifact.code_freeze_commit != FROZEN_CANDIDATE_MODEL_DEFINITION_COMMIT:
        raise PredictionError("Committed frozen artifact freeze commit drifted")
    if artifact.feature_family != FROZEN_CANDIDATE_FAMILY:
        raise PredictionError("Committed frozen artifact feature family drifted")
    if artifact.model_family != LOGISTIC_MODEL_DESCRIPTION:
        raise PredictionError("Committed frozen artifact model family drifted")
    if tuple(artifact.feature_columns) != tuple(FROZEN_FEATURE_COLUMNS):
        raise PredictionError("Committed frozen artifact columns drifted from the freeze")
    if artifact.training_row_count != EXPECTED_FROZEN_TRAINING_ROW_COUNT:
        raise PredictionError(
            "Committed frozen artifact training-row count is not the frozen 6,469-row identity"
        )
    if artifact.training_row_identity_sha256 != EXPECTED_TRAINING_ROW_IDENTITY_SHA256:
        raise PredictionError(
            "Committed frozen artifact training-row identity SHA-256 does not match freeze"
        )
    if artifact.fitted_artifact_sha256 != EXPECTED_FITTED_ARTIFACT_SHA256:
        raise PredictionError(
            "Committed frozen artifact SHA-256 does not match the frozen fitted fingerprint"
        )
    return artifact


def load_committed_frozen_artifact() -> FrozenFittedArtifact:
    """Load the tracked frozen parameters and verify integrity without refitting."""

    return assert_canonical_frozen_artifact(
        load_frozen_artifact_from_path(committed_frozen_artifact_path())
    )


def resolve_scoring_artifact(
    artifact: FrozenFittedArtifact | None,
) -> FrozenFittedArtifact:
    """Use an explicit artifact, or the committed freeze, after integrity checks."""

    if artifact is None:
        return load_committed_frozen_artifact()
    loaded = artifact_from_dict(artifact.to_dict())
    if loaded.historical_cache_sha256 == FROZEN_HISTORICAL_CACHE_SHA256:
        return assert_canonical_frozen_artifact(loaded)
    return loaded


def compare_refit_to_committed_artifact(
    committed: FrozenFittedArtifact,
    refit: FrozenFittedArtifact,
) -> list[str]:
    """Return blockers if a reproducibility refit disagrees with the committed freeze."""

    blockers: list[str] = []
    try:
        assert_canonical_frozen_artifact(committed)
    except PredictionError as exc:
        blockers.append(str(exc))
        return blockers
    if refit.fitted_artifact_sha256 != committed.fitted_artifact_sha256:
        blockers.append(
            "reproducibility refit artifact SHA-256 disagrees with the committed freeze"
        )
    if refit.training_row_identity_sha256 != committed.training_row_identity_sha256:
        blockers.append(
            "reproducibility refit training-row identity disagrees with the committed freeze"
        )
    if refit.scaler_mean != committed.scaler_mean:
        blockers.append("reproducibility refit scaler mean disagrees with the committed freeze")
    if refit.scaler_scale != committed.scaler_scale:
        blockers.append("reproducibility refit scaler scale disagrees with the committed freeze")
    if refit.classifier_coef != committed.classifier_coef:
        blockers.append(
            "reproducibility refit classifier coefficients disagree with the committed freeze"
        )
    if refit.classifier_intercept != committed.classifier_intercept:
        blockers.append(
            "reproducibility refit classifier intercept disagrees with the committed freeze"
        )
    return blockers


def predict_home_win_probability(
    artifact: FrozenFittedArtifact,
    feature_values: Sequence[float],
) -> float:
    """Return the model home-win probability for one frozen 25-vector."""

    assert_frozen_prospective_contract()
    if artifact.feature_columns != tuple(FROZEN_FEATURE_COLUMNS):
        raise PredictionError("Artifact feature columns are not the frozen 25-column order")
    if len(feature_values) != 25:
        raise PredictionError(
            f"Prospective scoring requires exactly 25 feature values, got {len(feature_values)}"
        )
    values = []
    for index, value in enumerate(feature_values):
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise PredictionError(
                f"Frozen feature at index {index} is not numeric"
            ) from exc
        if not np.isfinite(number):
            raise PredictionError(f"Frozen feature at index {index} is not finite")
        values.append(number)
    matrix = np.array([values], dtype=float)
    probability = float(artifact.pipeline().predict_proba(matrix)[0, 1])
    clipped = float(np.clip(probability, 1e-15, 1.0 - 1e-15))
    if not (0.0 < clipped < 1.0):
        raise PredictionError("Model home-win probability must be inside (0, 1)")
    return clipped
