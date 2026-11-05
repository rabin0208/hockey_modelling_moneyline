"""Issue #92 read-only hosted frozen-model scoring.

Consumes only validated ``HostedRefreshBundle.scoring_inputs()`` from Issue #91
and reuses the existing Issue #45/#52 frozen scoring semantics. Results are
in-memory current web calculations for presentation. This module does not write
protected prediction records, post receipts, read odds, size stakes, or fit a
model.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hockey_moneyline.errors import HockeyMoneylineError, PredictionError
from hockey_moneyline.hosted_refresh import (
    BLOCKED as REFRESH_BLOCKED,
    FUT_GAME_STATE,
    READY as REFRESH_READY,
    HostedRefreshBundle,
    HostedScoringInput,
)
from hockey_moneyline.issue55 import merge_frozen_history_with_current_season_package
from hockey_moneyline.prospective_artifact import (
    EXPECTED_FITTED_ARTIFACT_SHA256,
    FROZEN_FEATURE_COLUMNS,
    load_committed_frozen_artifact,
)
from hockey_moneyline.prospective_scoring import (
    CURRENT_SEASON_PRIOR_SOURCE_SHA256_KEY,
    feature_vector_sha256,
    score_pregame_prediction,
)

HOSTED_SCORING_SCHEMA_VERSION = 1
HOSTED_SCORING_KIND = "hosted_frozen_scoring_bundle"
CALCULATION_KIND = "current_hosted_web_calculation"
SCORED = "SCORED"
NOT_SCORED = "NOT_SCORED"
SCORING_COMPLETE = "COMPLETE"
SCORING_PARTIAL = "PARTIAL"
SCORING_NO_ELIGIBLE = "NO_ELIGIBLE_TARGETS"
SCORING_BLOCKED = "BLOCKED"

_FORBIDDEN_IMPORT_PREFIXES = (
    "hockey_moneyline.prediction_record_store",
    "hockey_moneyline.pregame_audit_receipt",
    "hockey_moneyline.issue63",
    "hockey_moneyline.issue68",
    "hockey_moneyline.forward_odds_archive",
    "hockey_moneyline.market_math",
    "hockey_moneyline.recommendation_contract",
    "hockey_moneyline.odds_gap_assist",
    "streamlit",
)
_FORBIDDEN_CALL_NAMES = frozenset(
    {
        "fit",
        "fit_transform",
        "partial_fit",
        "write_prediction_record",
        "append_prediction_record",
        "write_forward_odds_snapshot",
        "build_pregame_audit_receipt",
        "post_receipt",
        "write_text",
        "write_bytes",
        "unlink",
    }
)


class HostedScoringError(PredictionError):
    """Hosted scoring contract failed closed before a trustworthy score existed."""


def _iso_utc(value: datetime, *, label: str) -> str:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise HostedScoringError(f"{label} must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class HostedGameScore:
    game_id: str
    season: int
    game_type: str
    home_team_id: str
    away_team_id: str
    scheduled_start_time_utc: str
    observed_at_utc: str
    status: str
    reason_code: str | None
    reason: str | None
    model_home_win_probability: float | None
    model_away_win_probability: float | None
    probability_kind: str | None
    model_artifact_version: str | None
    model_artifact_sha256: str | None
    feature_schema_version: str | None
    feature_columns: tuple[str, ...]
    feature_vector_sha256: str | None
    current_season_prior_source_sha256: str | None
    moneypuck_source_sha256: str | None
    moneypuck_source_id: str | None
    moneypuck_source_file_url: str | None
    calculation_kind: str = CALCULATION_KIND
    durable_prospective_evidence: bool = False
    protected_prediction_record_written: bool = False
    github_receipt_posted: bool = False

    def __post_init__(self) -> None:
        if self.status not in {SCORED, NOT_SCORED}:
            raise HostedScoringError(f"unsupported hosted game status {self.status!r}")
        if self.status == SCORED:
            if self.reason_code is not None or self.reason is not None:
                raise HostedScoringError("scored hosted game cannot carry a failure reason")
            if self.model_home_win_probability is None or self.model_away_win_probability is None:
                raise HostedScoringError("scored hosted game requires both model probabilities")
            if not (0.0 < self.model_home_win_probability < 1.0):
                raise HostedScoringError("home model probability must be inside (0, 1)")
            if not (0.0 < self.model_away_win_probability < 1.0):
                raise HostedScoringError("away model probability must be inside (0, 1)")
            if abs(
                self.model_home_win_probability + self.model_away_win_probability - 1.0
            ) > 1e-12:
                raise HostedScoringError("hosted home/away probabilities must sum to 1")
            if self.probability_kind != "model_home_win":
                raise HostedScoringError("hosted scoring probability_kind drifted")
            if self.model_artifact_sha256 != EXPECTED_FITTED_ARTIFACT_SHA256:
                raise HostedScoringError("hosted scoring artifact SHA-256 drifted")
            if self.feature_columns != tuple(FROZEN_FEATURE_COLUMNS):
                raise HostedScoringError("hosted scoring feature order drifted")
            if len(self.feature_columns) != 25:
                raise HostedScoringError("hosted scoring must expose exactly 25 feature columns")
            if not self.feature_vector_sha256:
                raise HostedScoringError("scored hosted game requires feature-vector SHA-256")
            if not self.current_season_prior_source_sha256:
                raise HostedScoringError("scored hosted game requires prior-source SHA-256")
            if not self.moneypuck_source_sha256:
                raise HostedScoringError("scored hosted game requires raw MoneyPuck SHA-256")
        else:
            if not self.reason_code or not self.reason:
                raise HostedScoringError("not-scored hosted game requires an explicit reason")
            if self.model_home_win_probability is not None or self.model_away_win_probability is not None:
                raise HostedScoringError("not-scored hosted game cannot expose a probability")
            if self.feature_vector_sha256 is not None:
                raise HostedScoringError("not-scored hosted game cannot expose a feature hash")

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "season": self.season,
            "game_type": self.game_type,
            "home_team_id": self.home_team_id,
            "away_team_id": self.away_team_id,
            "scheduled_start_time_utc": self.scheduled_start_time_utc,
            "observed_at_utc": self.observed_at_utc,
            "status": self.status,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "model_home_win_probability": self.model_home_win_probability,
            "model_away_win_probability": self.model_away_win_probability,
            "probability_kind": self.probability_kind,
            "model_artifact_version": self.model_artifact_version,
            "model_artifact_sha256": self.model_artifact_sha256,
            "feature_schema_version": self.feature_schema_version,
            "feature_columns": list(self.feature_columns),
            "feature_vector_sha256": self.feature_vector_sha256,
            "current_season_prior_source_sha256": self.current_season_prior_source_sha256,
            "moneypuck_source_sha256": self.moneypuck_source_sha256,
            "moneypuck_source_id": self.moneypuck_source_id,
            "moneypuck_source_file_url": self.moneypuck_source_file_url,
            "calculation_kind": self.calculation_kind,
            "durable_prospective_evidence": self.durable_prospective_evidence,
            "protected_prediction_record_written": self.protected_prediction_record_written,
            "github_receipt_posted": self.github_receipt_posted,
        }


@dataclass(frozen=True, slots=True)
class HostedScoringBundle:
    schema_version: int
    kind: str
    refresh_readiness: str
    scoring_state: str
    blocker_code: str | None
    blocker_message: str | None
    slate_date: str
    observed_at_utc: str
    nhl_endpoint_path: str | None
    nhl_observed_at_utc: str | None
    moneypuck_source_sha256: str | None
    moneypuck_source_id: str | None
    moneypuck_source_file_url: str | None
    moneypuck_retrieved_at_utc: str | None
    games: tuple[HostedGameScore, ...]
    calculation_kind: str = CALCULATION_KIND
    durable_prospective_evidence: bool = False
    protected_prediction_record_written: bool = False
    github_receipt_posted: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != HOSTED_SCORING_SCHEMA_VERSION:
            raise HostedScoringError("unsupported hosted scoring schema version")
        if self.kind != HOSTED_SCORING_KIND:
            raise HostedScoringError("unsupported hosted scoring kind")
        if self.refresh_readiness not in {REFRESH_READY, REFRESH_BLOCKED}:
            raise HostedScoringError("unsupported source refresh readiness")
        if self.scoring_state not in {
            SCORING_COMPLETE,
            SCORING_PARTIAL,
            SCORING_NO_ELIGIBLE,
            SCORING_BLOCKED,
        }:
            raise HostedScoringError("unsupported hosted scoring state")
        if self.scoring_state == SCORING_BLOCKED:
            if not self.blocker_code or not self.blocker_message:
                raise HostedScoringError("blocked hosted scoring requires a blocker")
            if any(game.status == SCORED for game in self.games):
                raise HostedScoringError("blocked hosted scoring cannot contain scored games")
        elif self.blocker_code is not None or self.blocker_message is not None:
            raise HostedScoringError("non-blocked hosted scoring cannot carry a blocker")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "refresh_readiness": self.refresh_readiness,
            "scoring_state": self.scoring_state,
            "blocker_code": self.blocker_code,
            "blocker_message": self.blocker_message,
            "slate_date": self.slate_date,
            "observed_at_utc": self.observed_at_utc,
            "nhl_endpoint_path": self.nhl_endpoint_path,
            "nhl_observed_at_utc": self.nhl_observed_at_utc,
            "moneypuck_source_sha256": self.moneypuck_source_sha256,
            "moneypuck_source_id": self.moneypuck_source_id,
            "moneypuck_source_file_url": self.moneypuck_source_file_url,
            "moneypuck_retrieved_at_utc": self.moneypuck_retrieved_at_utc,
            "calculation_kind": self.calculation_kind,
            "durable_prospective_evidence": self.durable_prospective_evidence,
            "protected_prediction_record_written": self.protected_prediction_record_written,
            "github_receipt_posted": self.github_receipt_posted,
            "games": [game.to_dict() for game in self.games],
        }


def _not_scored(
    target: Any,
    *,
    code: str,
    reason: str,
    raw_moneypuck_sha256: str | None,
    moneypuck_source_id: str | None,
    moneypuck_source_file_url: str | None,
) -> HostedGameScore:
    return HostedGameScore(
        game_id=str(target.game_id),
        season=int(target.season),
        game_type=str(target.game_type),
        home_team_id=str(target.home_team_id),
        away_team_id=str(target.away_team_id),
        scheduled_start_time_utc=_iso_utc(
            target.scheduled_start_time_utc,
            label="scheduled_start_time_utc",
        ),
        observed_at_utc=_iso_utc(target.observed_at_utc, label="observed_at_utc"),
        status=NOT_SCORED,
        reason_code=code,
        reason=reason,
        model_home_win_probability=None,
        model_away_win_probability=None,
        probability_kind=None,
        model_artifact_version=None,
        model_artifact_sha256=None,
        feature_schema_version=None,
        feature_columns=(),
        feature_vector_sha256=None,
        current_season_prior_source_sha256=None,
        moneypuck_source_sha256=raw_moneypuck_sha256,
        moneypuck_source_id=moneypuck_source_id,
        moneypuck_source_file_url=moneypuck_source_file_url,
    )


def _bundle_source_metadata(
    bundle: HostedRefreshBundle,
) -> tuple[str | None, str | None, str | None, str | None, str | None, str | None]:
    nhl_endpoint = None
    nhl_observed = None
    if bundle.nhl_coverage is not None:
        nhl_endpoint = bundle.nhl_coverage.endpoint_path
        nhl_observed = _iso_utc(
            bundle.nhl_coverage.observed_at_utc,
            label="nhl_coverage.observed_at_utc",
        )
    raw_sha = None
    source_id = None
    file_url = None
    retrieved = None
    if bundle.moneypuck_provenance is not None:
        raw_sha = bundle.moneypuck_provenance.sha256
        source_id = bundle.moneypuck_provenance.source_id
        file_url = bundle.moneypuck_provenance.source_file_url
        retrieved = _iso_utc(
            bundle.moneypuck_provenance.retrieved_at_utc,
            label="moneypuck_provenance.retrieved_at_utc",
        )
    return nhl_endpoint, nhl_observed, raw_sha, source_id, file_url, retrieved


def _blocked_from_refresh(bundle: HostedRefreshBundle) -> HostedScoringBundle:
    nhl_endpoint, nhl_observed, raw_sha, source_id, file_url, retrieved = (
        _bundle_source_metadata(bundle)
    )
    blocker_code = bundle.blocker.code if bundle.blocker is not None else "refresh_not_ready"
    blocker_message = (
        bundle.blocker.message
        if bundle.blocker is not None
        else "Hosted refresh is not READY. No model probability was generated."
    )
    games = tuple(
        _not_scored(
            target,
            code=blocker_code,
            reason=blocker_message,
            raw_moneypuck_sha256=raw_sha,
            moneypuck_source_id=source_id,
            moneypuck_source_file_url=file_url,
        )
        for target in bundle.slate_targets
    )
    return HostedScoringBundle(
        schema_version=HOSTED_SCORING_SCHEMA_VERSION,
        kind=HOSTED_SCORING_KIND,
        refresh_readiness=bundle.readiness,
        scoring_state=SCORING_BLOCKED,
        blocker_code=blocker_code,
        blocker_message=blocker_message,
        slate_date=bundle.slate_date.isoformat(),
        observed_at_utc=_iso_utc(bundle.observed_at_utc, label="observed_at_utc"),
        nhl_endpoint_path=nhl_endpoint,
        nhl_observed_at_utc=nhl_observed,
        moneypuck_source_sha256=raw_sha,
        moneypuck_source_id=source_id,
        moneypuck_source_file_url=file_url,
        moneypuck_retrieved_at_utc=retrieved,
        games=games,
    )


def _validate_ready_sources(bundle: HostedRefreshBundle) -> tuple[str, str, str, str, str, str]:
    if bundle.nhl_coverage is None:
        raise HostedScoringError("READY refresh is missing NHL coverage metadata")
    if bundle.moneypuck_provenance is None:
        raise HostedScoringError("READY refresh is missing MoneyPuck provenance")
    observed = _iso_utc(bundle.observed_at_utc, label="observed_at_utc")
    nhl_observed = _iso_utc(
        bundle.nhl_coverage.observed_at_utc,
        label="nhl_coverage.observed_at_utc",
    )
    retrieved = _iso_utc(
        bundle.moneypuck_provenance.retrieved_at_utc,
        label="moneypuck_provenance.retrieved_at_utc",
    )
    if nhl_observed != observed or retrieved != observed:
        raise HostedScoringError(
            "READY refresh source observation metadata disagrees with bundle observation"
        )
    if not bundle.moneypuck_provenance.sha256:
        raise HostedScoringError("READY refresh is missing raw MoneyPuck SHA-256")
    return (
        bundle.nhl_coverage.endpoint_path,
        nhl_observed,
        bundle.moneypuck_provenance.sha256,
        bundle.moneypuck_provenance.source_id,
        bundle.moneypuck_provenance.source_file_url,
        retrieved,
    )


def _cross_check_input(
    item: HostedScoringInput,
    target: Any,
    bundle: HostedRefreshBundle,
) -> None:
    if target.game_state != FUT_GAME_STATE:
        raise HostedScoringError(
            f"Target {target.game_id} is no longer FUT; hosted scoring is rejected"
        )
    if not target.scoring_eligible:
        raise HostedScoringError(
            target.ineligibility_reason
            or f"Target {target.game_id} is not eligible for hosted scoring"
        )
    if item.prior_state is not target.prior_state:
        raise HostedScoringError("scoring input prior package is not the canonical bundle package")
    if item.target.game_id != target.game_id:
        raise HostedScoringError("scoring input game identity disagrees with slate target")
    if item.target.home_team_id != target.home_team_id or item.target.away_team_id != target.away_team_id:
        raise HostedScoringError("scoring input team identity disagrees with slate target")
    if item.target.scheduled_start_time_utc != target.scheduled_start_time_utc:
        raise HostedScoringError("scoring input scheduled start disagrees with slate target")
    if item.target.observed_at_utc != target.observed_at_utc:
        raise HostedScoringError("scoring input observation disagrees with slate target")
    if item.target.observed_at_utc != bundle.observed_at_utc:
        raise HostedScoringError("scoring input observation disagrees with refresh bundle")
    if item.target.observed_at_utc is None or item.target.scheduled_start_time_utc is None:
        raise HostedScoringError("hosted scoring requires complete target timing")
    if item.target.observed_at_utc >= item.target.scheduled_start_time_utc:
        raise HostedScoringError("hosted scoring observation must be strictly pre-puck")
    if item.prior_state.cutoff_game_id != item.target.game_id:
        raise HostedScoringError("prior package cutoff game disagrees with scoring target")
    if item.prior_state.observed_at_utc != bundle.observed_at_utc:
        raise HostedScoringError("prior package observation disagrees with refresh bundle")
    if not item.prior_state.source_sha256:
        raise HostedScoringError("prior package is missing source SHA-256")


def _score_one(
    item: HostedScoringInput,
    *,
    frozen_completed_history: Any,
    frozen_advanced_records: Any,
    frozen_schedule_history: Any,
    raw_moneypuck_sha256: str,
    moneypuck_source_id: str,
    moneypuck_source_file_url: str,
    artifact: Any,
) -> HostedGameScore:
    prior = item.prior_state
    merged_games, merged_records, merged_schedule = (
        merge_frozen_history_with_current_season_package(
            frozen_completed_history,
            frozen_advanced_records,
            frozen_schedule_history,
            prior,
        )
    )
    record = score_pregame_prediction(
        item.target,
        merged_games,
        merged_records,
        merged_schedule,
        artifact=artifact,
        **item.issue52_scoring_kwargs(),
    )
    if record.model_artifact_sha256 != EXPECTED_FITTED_ARTIFACT_SHA256:
        raise HostedScoringError("in-memory scorer returned an unexpected artifact SHA-256")
    if record.feature_columns != tuple(FROZEN_FEATURE_COLUMNS):
        raise HostedScoringError("in-memory scorer returned a non-frozen feature order")
    if record.feature_vector_sha256 != feature_vector_sha256(record.feature_values):
        raise HostedScoringError("in-memory scorer returned a non-reproducible feature hash")
    if (
        record.source_versions.get(CURRENT_SEASON_PRIOR_SOURCE_SHA256_KEY)
        != prior.source_sha256
    ):
        raise HostedScoringError("in-memory scorer source hash disagrees with prior package")
    home_probability = float(record.model_home_win_probability)
    return HostedGameScore(
        game_id=record.game_id,
        season=record.season,
        game_type=record.game_type,
        home_team_id=record.home_team_id,
        away_team_id=record.away_team_id,
        scheduled_start_time_utc=record.scheduled_start_time_utc,
        observed_at_utc=record.prediction_timestamp_utc,
        status=SCORED,
        reason_code=None,
        reason=None,
        model_home_win_probability=home_probability,
        model_away_win_probability=1.0 - home_probability,
        probability_kind=record.probability_kind,
        model_artifact_version=record.model_artifact_version,
        model_artifact_sha256=record.model_artifact_sha256,
        feature_schema_version=record.feature_schema_version,
        feature_columns=tuple(record.feature_columns),
        feature_vector_sha256=record.feature_vector_sha256,
        current_season_prior_source_sha256=prior.source_sha256,
        moneypuck_source_sha256=raw_moneypuck_sha256,
        moneypuck_source_id=moneypuck_source_id,
        moneypuck_source_file_url=moneypuck_source_file_url,
    )


def score_hosted_refresh_bundle(
    bundle: HostedRefreshBundle,
    *,
    frozen_completed_history: Any | None = None,
    frozen_advanced_records: Any | None = None,
    frozen_schedule_history: Any | None = None,
) -> HostedScoringBundle:
    """Score eligible READY targets in memory with protected feature semantics.

    Frozen historical material is read-only input supplied by the hosted
    composition layer. This function never loads or writes the protected store,
    never builds receipts, and never persists the temporary history material.
    """

    if not isinstance(bundle, HostedRefreshBundle):
        raise HostedScoringError("hosted scoring requires an Issue #91 HostedRefreshBundle")
    if bundle.readiness != REFRESH_READY or bundle.blocker is not None:
        return _blocked_from_refresh(bundle)

    try:
        (
            nhl_endpoint,
            nhl_observed,
            raw_sha,
            source_id,
            file_url,
            retrieved,
        ) = _validate_ready_sources(bundle)
        scoring_inputs = bundle.scoring_inputs()
    except HockeyMoneylineError as exc:
        games = tuple(
            _not_scored(
                target,
                code="ready_bundle_validation_failed",
                reason=str(exc),
                raw_moneypuck_sha256=(
                    bundle.moneypuck_provenance.sha256
                    if bundle.moneypuck_provenance is not None
                    else None
                ),
                moneypuck_source_id=(
                    bundle.moneypuck_provenance.source_id
                    if bundle.moneypuck_provenance is not None
                    else None
                ),
                moneypuck_source_file_url=(
                    bundle.moneypuck_provenance.source_file_url
                    if bundle.moneypuck_provenance is not None
                    else None
                ),
            )
            for target in bundle.slate_targets
        )
        return HostedScoringBundle(
            schema_version=HOSTED_SCORING_SCHEMA_VERSION,
            kind=HOSTED_SCORING_KIND,
            refresh_readiness=bundle.readiness,
            scoring_state=SCORING_BLOCKED,
            blocker_code="ready_bundle_validation_failed",
            blocker_message=str(exc),
            slate_date=bundle.slate_date.isoformat(),
            observed_at_utc=_iso_utc(bundle.observed_at_utc, label="observed_at_utc"),
            nhl_endpoint_path=(
                bundle.nhl_coverage.endpoint_path if bundle.nhl_coverage is not None else None
            ),
            nhl_observed_at_utc=(
                _iso_utc(
                    bundle.nhl_coverage.observed_at_utc,
                    label="nhl_coverage.observed_at_utc",
                )
                if bundle.nhl_coverage is not None
                else None
            ),
            moneypuck_source_sha256=(
                bundle.moneypuck_provenance.sha256
                if bundle.moneypuck_provenance is not None
                else None
            ),
            moneypuck_source_id=(
                bundle.moneypuck_provenance.source_id
                if bundle.moneypuck_provenance is not None
                else None
            ),
            moneypuck_source_file_url=(
                bundle.moneypuck_provenance.source_file_url
                if bundle.moneypuck_provenance is not None
                else None
            ),
            moneypuck_retrieved_at_utc=(
                _iso_utc(
                    bundle.moneypuck_provenance.retrieved_at_utc,
                    label="moneypuck_provenance.retrieved_at_utc",
                )
                if bundle.moneypuck_provenance is not None
                else None
            ),
            games=games,
        )

    if scoring_inputs and (
        frozen_completed_history is None
        or frozen_advanced_records is None
        or frozen_schedule_history is None
    ):
        reason = (
            "Frozen historical feature history is unavailable for hosted scoring; "
            "refuse current-season-only feature semantics."
        )
        games = tuple(
            _not_scored(
                target,
                code="frozen_history_unavailable",
                reason=reason,
                raw_moneypuck_sha256=raw_sha,
                moneypuck_source_id=source_id,
                moneypuck_source_file_url=file_url,
            )
            for target in bundle.slate_targets
        )
        return HostedScoringBundle(
            schema_version=HOSTED_SCORING_SCHEMA_VERSION,
            kind=HOSTED_SCORING_KIND,
            refresh_readiness=bundle.readiness,
            scoring_state=SCORING_BLOCKED,
            blocker_code="frozen_history_unavailable",
            blocker_message=reason,
            slate_date=bundle.slate_date.isoformat(),
            observed_at_utc=_iso_utc(bundle.observed_at_utc, label="observed_at_utc"),
            nhl_endpoint_path=nhl_endpoint,
            nhl_observed_at_utc=nhl_observed,
            moneypuck_source_sha256=raw_sha,
            moneypuck_source_id=source_id,
            moneypuck_source_file_url=file_url,
            moneypuck_retrieved_at_utc=retrieved,
            games=games,
        )

    artifact = load_committed_frozen_artifact()
    if artifact.fitted_artifact_sha256 != EXPECTED_FITTED_ARTIFACT_SHA256:
        reason = "committed frozen artifact SHA-256 does not match the protected freeze"
        games = tuple(
            _not_scored(
                target,
                code="artifact_sha_mismatch",
                reason=reason,
                raw_moneypuck_sha256=raw_sha,
                moneypuck_source_id=source_id,
                moneypuck_source_file_url=file_url,
            )
            for target in bundle.slate_targets
        )
        return HostedScoringBundle(
            schema_version=HOSTED_SCORING_SCHEMA_VERSION,
            kind=HOSTED_SCORING_KIND,
            refresh_readiness=bundle.readiness,
            scoring_state=SCORING_BLOCKED,
            blocker_code="artifact_sha_mismatch",
            blocker_message=reason,
            slate_date=bundle.slate_date.isoformat(),
            observed_at_utc=_iso_utc(bundle.observed_at_utc, label="observed_at_utc"),
            nhl_endpoint_path=nhl_endpoint,
            nhl_observed_at_utc=nhl_observed,
            moneypuck_source_sha256=raw_sha,
            moneypuck_source_id=source_id,
            moneypuck_source_file_url=file_url,
            moneypuck_retrieved_at_utc=retrieved,
            games=games,
        )
    if artifact.feature_columns != tuple(FROZEN_FEATURE_COLUMNS) or len(artifact.feature_columns) != 25:
        reason = "committed frozen artifact feature order does not match the 25-column freeze"
        games = tuple(
            _not_scored(
                target,
                code="artifact_feature_order_mismatch",
                reason=reason,
                raw_moneypuck_sha256=raw_sha,
                moneypuck_source_id=source_id,
                moneypuck_source_file_url=file_url,
            )
            for target in bundle.slate_targets
        )
        return HostedScoringBundle(
            schema_version=HOSTED_SCORING_SCHEMA_VERSION,
            kind=HOSTED_SCORING_KIND,
            refresh_readiness=bundle.readiness,
            scoring_state=SCORING_BLOCKED,
            blocker_code="artifact_feature_order_mismatch",
            blocker_message=reason,
            slate_date=bundle.slate_date.isoformat(),
            observed_at_utc=_iso_utc(bundle.observed_at_utc, label="observed_at_utc"),
            nhl_endpoint_path=nhl_endpoint,
            nhl_observed_at_utc=nhl_observed,
            moneypuck_source_sha256=raw_sha,
            moneypuck_source_id=source_id,
            moneypuck_source_file_url=file_url,
            moneypuck_retrieved_at_utc=retrieved,
            games=games,
        )

    by_target_id = {target.game_id: target for target in bundle.slate_targets}
    if len(by_target_id) != len(bundle.slate_targets):
        raise HostedScoringError("hosted refresh contains duplicate slate target ids")
    by_input_id: dict[str, HostedScoringInput] = {}
    for item in scoring_inputs:
        if item.target.game_id in by_input_id:
            raise HostedScoringError("hosted refresh returned duplicate scoring inputs")
        target = by_target_id.get(item.target.game_id)
        if target is None:
            raise HostedScoringError("scoring input is not present in canonical slate targets")
        _cross_check_input(item, target, bundle)
        by_input_id[item.target.game_id] = item

    results: list[HostedGameScore] = []
    for target in bundle.slate_targets:
        item = by_input_id.get(target.game_id)
        if item is None:
            if target.game_state != FUT_GAME_STATE:
                code = "target_not_fut"
                reason = (
                    f"Target {target.game_id} is no longer FUT; NHL game_state is "
                    f"{target.game_state!r}. Hosted pregame scoring is rejected."
                )
            elif target.observed_at_utc >= target.scheduled_start_time_utc:
                code = "observation_at_or_after_puck_drop"
                reason = (
                    f"Target {target.game_id} observation is at or after scheduled puck drop. "
                    "Hosted pregame scoring is rejected."
                )
            else:
                code = "target_not_eligible"
                reason = target.ineligibility_reason or (
                    f"Target {target.game_id} was not returned by validated scoring_inputs()."
                )
            results.append(
                _not_scored(
                    target,
                    code=code,
                    reason=reason,
                    raw_moneypuck_sha256=raw_sha,
                    moneypuck_source_id=source_id,
                    moneypuck_source_file_url=file_url,
                )
            )
            continue
        try:
            results.append(
                _score_one(
                    item,
                    frozen_completed_history=frozen_completed_history,
                    frozen_advanced_records=frozen_advanced_records,
                    frozen_schedule_history=frozen_schedule_history,
                    raw_moneypuck_sha256=raw_sha,
                    moneypuck_source_id=source_id,
                    moneypuck_source_file_url=file_url,
                    artifact=artifact,
                )
            )
        except (HockeyMoneylineError, ValueError, TypeError) as exc:
            results.append(
                _not_scored(
                    target,
                    code="scoring_failure",
                    reason=str(exc),
                    raw_moneypuck_sha256=raw_sha,
                    moneypuck_source_id=source_id,
                    moneypuck_source_file_url=file_url,
                )
            )

    scored_count = sum(game.status == SCORED for game in results)
    if not scoring_inputs:
        scoring_state = SCORING_NO_ELIGIBLE
    elif scored_count == len(scoring_inputs):
        scoring_state = (
            SCORING_COMPLETE
            if all(game.status == SCORED for game in results)
            else SCORING_PARTIAL
        )
    else:
        scoring_state = SCORING_PARTIAL

    return HostedScoringBundle(
        schema_version=HOSTED_SCORING_SCHEMA_VERSION,
        kind=HOSTED_SCORING_KIND,
        refresh_readiness=bundle.readiness,
        scoring_state=scoring_state,
        blocker_code=None,
        blocker_message=None,
        slate_date=bundle.slate_date.isoformat(),
        observed_at_utc=_iso_utc(bundle.observed_at_utc, label="observed_at_utc"),
        nhl_endpoint_path=nhl_endpoint,
        nhl_observed_at_utc=nhl_observed,
        moneypuck_source_sha256=raw_sha,
        moneypuck_source_id=source_id,
        moneypuck_source_file_url=file_url,
        moneypuck_retrieved_at_utc=retrieved,
        games=tuple(results),
    )


def assert_hosted_scoring_is_read_only() -> None:
    """Static guard against persistence, fitting, market, and UI dependencies."""

    source_path = Path(__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    forbidden_imports: list[str] = []
    forbidden_calls: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module or ""]
        else:
            modules = []
        for module in modules:
            if any(
                module == prefix or module.startswith(prefix + ".")
                for prefix in _FORBIDDEN_IMPORT_PREFIXES
            ):
                forbidden_imports.append(module)

        if isinstance(node, ast.Call):
            name = None
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            if name in _FORBIDDEN_CALL_NAMES:
                forbidden_calls.append(str(name))

    if forbidden_imports:
        raise HostedScoringError(
            f"hosted scoring imported forbidden runtime dependencies: {sorted(set(forbidden_imports))}"
        )
    if forbidden_calls:
        raise HostedScoringError(
            f"hosted scoring contains forbidden mutation/fit calls: {sorted(set(forbidden_calls))}"
        )
