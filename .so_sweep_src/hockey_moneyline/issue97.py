"""Issue #97 one-command daily protected prediction workflow.

Operator usage:

    python -m hockey_moneyline.issue97 2026-10-03

The command captures one UTC observation, refreshes NHL + current MoneyPuck
through the merged Issue #91 contract, verifies/reconstructs the frozen
historical cache, scores only validated FUT/pre-puck targets through Issue #55,
writes the existing immutable protected store, and generates Issue #63 receipt
JSON.

It never posts receipts externally. Receipt posting remains a separate
designated-operator/03 action.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import sys
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.request import urlopen

from hockey_moneyline.acquire import DEFAULT_CACHE_DIR, cache_path
from hockey_moneyline.errors import HockeyMoneylineError, PredictionError
from hockey_moneyline.hosted_refresh import (
    READY as REFRESH_READY,
    HostedRefreshBundle,
    refresh_hosted_slate,
    require_hosted_pregame_scoring_eligibility,
)
from hockey_moneyline.issue55 import (
    execute_manual_pregame_prediction,
    load_frozen_development_history,
    require_committed_execution_artifact,
)
from hockey_moneyline.prediction_record_store import (
    DEFAULT_STORE_RELATIVE_PATH,
    PredictionRecordStore,
)
from hockey_moneyline.pregame_audit_receipt import build_pregame_audit_receipt
from hockey_moneyline.prospective_artifact import (
    EXPECTED_FITTED_ARTIFACT_SHA256,
    FROZEN_FEATURE_COLUMNS,
)
from hockey_moneyline.prospective_scoring import (
    CURRENT_SEASON_PRIOR_SOURCE_SHA256_KEY,
)
from hockey_moneyline.schedule_load_ablation import (
    KNOWN_CYCLE2_MONEYPUCK_SHA256,
    sha256_path,
)

FROZEN_HISTORICAL_CACHE_SIZE = 126_483_024
FROZEN_HISTORICAL_CACHE_SHA256 = KNOWN_CYCLE2_MONEYPUCK_SHA256
CURRENT_SEASON_MONEYPUCK_VALUE = "2026"
READY = "READY FOR DESIGNATED-OPERATOR RECEIPT POSTING"
BLOCKED = "BLOCKED — ONE MATERIAL DAILY PROTECTED-PREDICTION DEPENDENCY"


@dataclass(frozen=True, slots=True)
class FrozenCacheStatus:
    path: str
    sha256: str
    byte_count: int
    reconstructed: bool
    removed_current_season_rows: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "byte_count": self.byte_count,
            "reconstructed": self.reconstructed,
            "removed_current_season_rows": self.removed_current_season_rows,
        }


@dataclass(frozen=True, slots=True)
class DailyProtectedGameResult:
    game_id: str
    away_team_id: str
    home_team_id: str
    scheduled_start_time_utc: str
    status: str
    model_home_win_probability: float
    model_away_win_probability: float
    model_artifact_sha256: str
    current_season_source_sha256: str
    current_moneypuck_sha256: str
    frozen_historical_cache_sha256: str
    feature_vector_sha256: str
    prediction_record_sha256: str
    receipt_sha256: str
    receipt: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "away_team_id": self.away_team_id,
            "home_team_id": self.home_team_id,
            "scheduled_start_time_utc": self.scheduled_start_time_utc,
            "status": self.status,
            "model_home_win_probability": self.model_home_win_probability,
            "model_away_win_probability": self.model_away_win_probability,
            "model_artifact_sha256": self.model_artifact_sha256,
            "current_season_source_sha256": self.current_season_source_sha256,
            "current_moneypuck_sha256": self.current_moneypuck_sha256,
            "frozen_historical_cache_sha256": self.frozen_historical_cache_sha256,
            "feature_vector_sha256": self.feature_vector_sha256,
            "prediction_record_sha256": self.prediction_record_sha256,
            "receipt_sha256": self.receipt_sha256,
            "receipt": dict(self.receipt),
        }


@dataclass(frozen=True, slots=True)
class DailyProtectedSlateResult:
    slate_date: str
    observed_at_utc: str
    current_moneypuck_sha256: str
    frozen_cache: FrozenCacheStatus
    artifact_sha256: str
    store_path: str
    target_count: int
    written_count: int
    already_protected_count: int
    games: tuple[DailyProtectedGameResult, ...]
    external_receipts_posted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": READY,
            "slate_date": self.slate_date,
            "observed_at_utc": self.observed_at_utc,
            "current_moneypuck_sha256": self.current_moneypuck_sha256,
            "frozen_cache": self.frozen_cache.to_dict(),
            "artifact_sha256": self.artifact_sha256,
            "store_path": self.store_path,
            "target_count": self.target_count,
            "written_count": self.written_count,
            "already_protected_count": self.already_protected_count,
            "external_receipts_posted": self.external_receipts_posted,
            "games": [game.to_dict() for game in self.games],
        }


class _CapturingResponse:
    def __init__(self, response: Any, sink: bytearray) -> None:
        self._response = response
        self._active = response
        self._sink = sink

    @property
    def status(self) -> int:
        return int(getattr(self._active, "status", getattr(self._response, "status", 200)))

    def __enter__(self) -> "_CapturingResponse":
        entered = self._response.__enter__()
        self._active = entered
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        return self._response.__exit__(exc_type, exc, tb)

    def read(self, size: int = -1) -> bytes:
        chunk = self._active.read(size)
        if chunk:
            if not isinstance(chunk, (bytes, bytearray)):
                raise PredictionError("MoneyPuck source response returned non-byte content")
            self._sink.extend(bytes(chunk))
        return chunk


class _CapturingMoneyPuckOpener:
    """Capture exact bytes consumed by Issue #91 without a second source fetch."""

    def __init__(self, opener: Callable[..., Any]) -> None:
        self._opener = opener
        self._payload = bytearray()
        self._called = False

    @property
    def payload(self) -> bytes:
        return bytes(self._payload)

    def __call__(self, request: Any, timeout: int = 120) -> _CapturingResponse:
        if self._called:
            raise PredictionError(
                "Daily protected workflow attempted more than one MoneyPuck source fetch"
            )
        self._called = True
        response = self._opener(request, timeout=timeout)
        return _CapturingResponse(response, self._payload)


def _aware_utc(value: datetime, *, label: str) -> datetime:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise PredictionError(f"{label} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _csv_row_from_raw_line(raw: bytes, *, line_number: int, header: bool = False) -> list[str]:
    try:
        text = raw.decode("utf-8-sig" if header else "utf-8").rstrip("\r\n")
    except UnicodeDecodeError as exc:
        raise PredictionError(
            f"MoneyPuck CSV is not valid UTF-8 at line {line_number}"
        ) from exc
    try:
        return next(csv.reader([text]))
    except (csv.Error, StopIteration) as exc:
        raise PredictionError(
            f"MoneyPuck CSV could not be parsed at line {line_number}"
        ) from exc


def reconstruct_frozen_historical_bytes(
    current_payload: bytes,
    *,
    expected_size: int = FROZEN_HISTORICAL_CACHE_SIZE,
    expected_sha256: str = FROZEN_HISTORICAL_CACHE_SHA256,
) -> tuple[bytes, int]:
    """Reproduce the Issue #83 freeze by removing raw season=2026 rows."""

    if not current_payload:
        raise PredictionError("Cannot reconstruct frozen cache from an empty MoneyPuck payload")
    lines = current_payload.splitlines(keepends=True)
    if not lines:
        raise PredictionError("Cannot reconstruct frozen cache without a CSV header")
    header = _csv_row_from_raw_line(lines[0], line_number=1, header=True)
    if "season" not in header:
        raise PredictionError("MoneyPuck CSV is missing the season column")
    season_index = header.index("season")

    kept: list[bytes] = [lines[0]]
    removed = 0
    for line_number, raw in enumerate(lines[1:], start=2):
        if not raw.strip():
            kept.append(raw)
            continue
        row = _csv_row_from_raw_line(raw, line_number=line_number)
        if season_index >= len(row):
            raise PredictionError(
                f"MoneyPuck CSV row {line_number} is missing the season value"
            )
        if row[season_index].strip() == CURRENT_SEASON_MONEYPUCK_VALUE:
            removed += 1
        else:
            kept.append(raw)

    if removed == 0:
        raise PredictionError(
            "Frozen-cache reconstruction removed zero season=2026 rows; "
            "refuse to treat the published file as the historical freeze"
        )
    rebuilt = b"".join(kept)
    digest = _sha256_bytes(rebuilt)
    if len(rebuilt) != expected_size or digest != expected_sha256:
        raise PredictionError(
            "Frozen MoneyPuck reconstruction identity mismatch: "
            f"size={len(rebuilt)} sha256={digest}; expected "
            f"size={expected_size} sha256={expected_sha256}"
        )
    return rebuilt, removed


def ensure_frozen_historical_cache(
    cache_dir: Path,
    current_payload: bytes,
) -> FrozenCacheStatus:
    """Verify the existing freeze, or reconstruct it exactly when absent."""

    path = cache_path(cache_dir)
    if path.exists():
        if not path.is_file() or path.stat().st_size == 0:
            raise PredictionError(f"Frozen MoneyPuck cache is unusable at {path}")
        size = path.stat().st_size
        digest = sha256_path(path)
        if size != FROZEN_HISTORICAL_CACHE_SIZE or digest != FROZEN_HISTORICAL_CACHE_SHA256:
            raise PredictionError(
                "Existing frozen MoneyPuck cache identity mismatch; refuse to overwrite it: "
                f"size={size} sha256={digest}"
            )
        return FrozenCacheStatus(
            path=str(path),
            sha256=digest,
            byte_count=size,
            reconstructed=False,
            removed_current_season_rows=0,
        )

    rebuilt, removed = reconstruct_frozen_historical_bytes(
        current_payload,
        expected_size=FROZEN_HISTORICAL_CACHE_SIZE,
        expected_sha256=FROZEN_HISTORICAL_CACHE_SHA256,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    try:
        partial.write_bytes(rebuilt)
        partial.replace(path)
    except OSError as exc:
        try:
            partial.unlink(missing_ok=True)
        except OSError:
            pass
        raise PredictionError(f"Could not write reconstructed frozen cache at {path}") from exc

    size = path.stat().st_size
    digest = sha256_path(path)
    if size != FROZEN_HISTORICAL_CACHE_SIZE or digest != FROZEN_HISTORICAL_CACHE_SHA256:
        raise PredictionError(
            "Reconstructed frozen MoneyPuck cache failed post-write identity verification"
        )
    return FrozenCacheStatus(
        path=str(path),
        sha256=digest,
        byte_count=size,
        reconstructed=True,
        removed_current_season_rows=removed,
    )


def require_concrete_cloud_agent_run_id(run_id: str | None) -> str:
    """Gate delegated-active reporting on a concrete Cursor/Cloud Agent id."""

    value = str(run_id or "").strip()
    if not value.startswith("bc-"):
        raise PredictionError(
            "Delegated execution is not active without a concrete Cursor/Cloud Agent run id"
        )
    try:
        uuid.UUID(value[3:])
    except ValueError as exc:
        raise PredictionError(
            "Delegated execution run id is not a concrete Cursor/Cloud Agent UUID"
        ) from exc
    return value


def _expected_start_iso(target: Any) -> str:
    return _aware_utc(
        target.scheduled_start_time_utc,
        label="scheduled_start_time_utc",
    ).isoformat()


def _existing_record_conflicts(
    envelope: Mapping[str, Any],
    scoring_input: Any,
) -> list[str]:
    record = envelope["record"]
    target = scoring_input.target
    source_versions = record.get("source_versions") or {}
    expected = {
        "game_id": target.game_id,
        "season": target.season,
        "game_type": target.game_type,
        "home_team_id": target.home_team_id,
        "away_team_id": target.away_team_id,
        "scheduled_start_time_utc": _expected_start_iso(target),
        "model_artifact_sha256": EXPECTED_FITTED_ARTIFACT_SHA256,
        "probability_kind": "model_home_win",
    }
    conflicts = [
        f"{key}: existing={record.get(key)!r} expected={value!r}"
        for key, value in expected.items()
        if record.get(key) != value
    ]
    if tuple(record.get("feature_columns") or ()) != tuple(FROZEN_FEATURE_COLUMNS):
        conflicts.append("feature_columns disagree with the frozen 25-column order")
    if source_versions.get("historical_cache_sha256") != FROZEN_HISTORICAL_CACHE_SHA256:
        conflicts.append("historical_cache_sha256 disagrees with the frozen cache")
    if (
        source_versions.get(CURRENT_SEASON_PRIOR_SOURCE_SHA256_KEY)
        != scoring_input.prior_state.source_sha256
    ):
        conflicts.append("current-season prior source SHA-256 disagrees")
    try:
        prediction_time = datetime.fromisoformat(
            str(record.get("prediction_timestamp_utc", "")).replace("Z", "+00:00")
        )
        if _aware_utc(prediction_time, label="prediction_timestamp_utc") >= _aware_utc(
            target.scheduled_start_time_utc,
            label="scheduled_start_time_utc",
        ):
            conflicts.append("existing prediction timestamp is not pre-puck")
    except (TypeError, ValueError, PredictionError):
        conflicts.append("existing prediction timestamp is invalid")
    return conflicts


def _game_result_from_existing(
    envelope: Mapping[str, Any],
    scoring_input: Any,
    *,
    current_moneypuck_sha256: str,
    status: str,
) -> DailyProtectedGameResult:
    receipt = build_pregame_audit_receipt(envelope)
    record = envelope["record"]
    home_probability = float(record["model_home_win_probability"])
    return DailyProtectedGameResult(
        game_id=str(record["game_id"]),
        away_team_id=str(record["away_team_id"]),
        home_team_id=str(record["home_team_id"]),
        scheduled_start_time_utc=str(record["scheduled_start_time_utc"]),
        status=status,
        model_home_win_probability=home_probability,
        model_away_win_probability=1.0 - home_probability,
        model_artifact_sha256=str(record["model_artifact_sha256"]),
        current_season_source_sha256=scoring_input.prior_state.source_sha256,
        current_moneypuck_sha256=current_moneypuck_sha256,
        frozen_historical_cache_sha256=FROZEN_HISTORICAL_CACHE_SHA256,
        feature_vector_sha256=str(record["feature_vector_sha256"]),
        prediction_record_sha256=str(envelope["record_sha256"]),
        receipt_sha256=receipt.receipt_sha256,
        receipt=receipt.to_dict(),
    )


def _validate_ready_refresh(
    bundle: HostedRefreshBundle,
    observed: datetime,
) -> tuple[Any, ...]:
    if bundle.readiness != REFRESH_READY or bundle.blocker is not None:
        detail = bundle.blocker.detail if bundle.blocker is not None else "refresh not READY"
        code = bundle.blocker.code if bundle.blocker is not None else "refresh_not_ready"
        raise PredictionError(f"Source refresh blocked ({code}): {detail}")
    if _aware_utc(bundle.observed_at_utc, label="bundle.observed_at_utc") != observed:
        raise PredictionError("Source refresh observation disagrees with the command observation")
    if bundle.moneypuck_provenance is None:
        raise PredictionError("READY source refresh is missing MoneyPuck provenance")

    for target in bundle.slate_targets:
        try:
            require_hosted_pregame_scoring_eligibility(target)
        except HockeyMoneylineError as exc:
            raise PredictionError(
                f"Slate target {target.game_id} rejected before scoring: {exc}"
            ) from exc

    scoring_inputs = bundle.scoring_inputs()
    if len(scoring_inputs) != len(bundle.slate_targets):
        raise PredictionError(
            "READY slate target inventory and validated scoring inputs disagree"
        )
    return scoring_inputs


def run_daily_protected_predictions(
    slate_date: date | str,
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    store_path: Path | str = DEFAULT_STORE_RELATIVE_PATH,
    observed_at_utc: datetime | None = None,
    nhl_opener: Callable[..., Any] = urlopen,
    moneypuck_opener: Callable[..., Any] = urlopen,
    _refresh_fn: Callable[..., HostedRefreshBundle] = refresh_hosted_slate,
    _history_loader: Callable[..., Any] = load_frozen_development_history,
) -> DailyProtectedSlateResult:
    """Execute one bounded daily protected slate from one source observation."""

    observed = _aware_utc(
        observed_at_utc or datetime.now(timezone.utc),
        label="observed_at_utc",
    )
    capture = _CapturingMoneyPuckOpener(moneypuck_opener)
    bundle = _refresh_fn(
        slate_date,
        observed,
        nhl_opener=nhl_opener,
        moneypuck_opener=capture,
    )
    scoring_inputs = _validate_ready_refresh(bundle, observed)

    payload = capture.payload
    provenance = bundle.moneypuck_provenance
    if provenance is None:
        raise PredictionError("READY source refresh is missing MoneyPuck provenance")
    if not payload:
        raise PredictionError(
            "Issue #91 refresh did not expose the exact MoneyPuck bytes consumed by this command"
        )
    raw_sha = _sha256_bytes(payload)
    if raw_sha != provenance.sha256 or len(payload) != provenance.byte_count:
        raise PredictionError(
            "Captured MoneyPuck bytes disagree with Issue #91 provenance identity"
        )

    frozen_cache = ensure_frozen_historical_cache(Path(cache_dir), payload)
    artifact = require_committed_execution_artifact()
    if artifact.fitted_artifact_sha256 != EXPECTED_FITTED_ARTIFACT_SHA256:
        raise PredictionError("Committed frozen artifact identity changed")
    if tuple(artifact.feature_columns) != tuple(FROZEN_FEATURE_COLUMNS):
        raise PredictionError("Committed frozen artifact feature order changed")

    frozen_games, frozen_records, frozen_schedule = _history_loader(Path(cache_dir))
    store = PredictionRecordStore(store_path)

    existing_by_game: dict[str, Mapping[str, Any]] = {}
    for item in scoring_inputs:
        existing = store.get_by_game_id(item.target.game_id)
        if existing is None:
            continue
        conflicts = _existing_record_conflicts(existing, item)
        if conflicts:
            raise PredictionError(
                "Conflicting duplicate protected prediction for "
                f"{item.target.game_id}: {conflicts[0]}"
            )
        existing_by_game[item.target.game_id] = existing

    results: list[DailyProtectedGameResult] = []
    written_count = 0
    already_count = 0
    for item in scoring_inputs:
        existing = existing_by_game.get(item.target.game_id)
        if existing is not None:
            results.append(
                _game_result_from_existing(
                    existing,
                    item,
                    current_moneypuck_sha256=raw_sha,
                    status="already_protected_no_rescore",
                )
            )
            already_count += 1
            continue

        appeared = store.get_by_game_id(item.target.game_id)
        if appeared is not None:
            conflicts = _existing_record_conflicts(appeared, item)
            if conflicts:
                raise PredictionError(
                    "Conflicting duplicate protected prediction for "
                    f"{item.target.game_id}: {conflicts[0]}"
                )
            results.append(
                _game_result_from_existing(
                    appeared,
                    item,
                    current_moneypuck_sha256=raw_sha,
                    status="already_protected_no_rescore",
                )
            )
            already_count += 1
            continue

        execution = execute_manual_pregame_prediction(
            item.target,
            item.prior_state,
            frozen_completed_history=frozen_games,
            frozen_advanced_records=frozen_records,
            frozen_schedule_history=frozen_schedule,
            store=store,
        )
        if not execution.written:
            raise PredictionError(
                f"Unexpected non-write after no-rescore preflight for {item.target.game_id}"
            )
        receipt = build_pregame_audit_receipt(execution.envelope)
        home_probability = float(execution.record.model_home_win_probability)
        results.append(
            DailyProtectedGameResult(
                game_id=execution.record.game_id,
                away_team_id=execution.record.away_team_id,
                home_team_id=execution.record.home_team_id,
                scheduled_start_time_utc=execution.record.scheduled_start_time_utc,
                status="written",
                model_home_win_probability=home_probability,
                model_away_win_probability=1.0 - home_probability,
                model_artifact_sha256=execution.record.model_artifact_sha256,
                current_season_source_sha256=item.prior_state.source_sha256,
                current_moneypuck_sha256=raw_sha,
                frozen_historical_cache_sha256=frozen_cache.sha256,
                feature_vector_sha256=execution.record.feature_vector_sha256,
                prediction_record_sha256=execution.record_sha256,
                receipt_sha256=receipt.receipt_sha256,
                receipt=receipt.to_dict(),
            )
        )
        written_count += 1

    return DailyProtectedSlateResult(
        slate_date=bundle.slate_date.isoformat(),
        observed_at_utc=observed.isoformat(),
        current_moneypuck_sha256=raw_sha,
        frozen_cache=frozen_cache,
        artifact_sha256=artifact.fitted_artifact_sha256,
        store_path=str(store.path),
        target_count=len(scoring_inputs),
        written_count=written_count,
        already_protected_count=already_count,
        games=tuple(results),
    )


def assert_issue97_has_no_external_posting_path() -> None:
    """Static guard: provider reads are allowed; external posting is not."""

    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    forbidden_import_prefixes = ("github", "requests", "httpx")
    forbidden_calls = {
        "add_comment_to_issue",
        "create_issue_comment",
        "post_receipt",
        "post_comment",
        "requests.post",
    }
    found_imports: list[str] = []
    found_calls: list[str] = []

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
                for prefix in forbidden_import_prefixes
            ):
                found_imports.append(module)

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called = node.func.id
            elif isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name):
                    called = f"{node.func.value.id}.{node.func.attr}"
                else:
                    called = node.func.attr
            else:
                called = ""
            if called in forbidden_calls:
                found_calls.append(called)

    if found_imports or found_calls:
        raise PredictionError(
            "Issue #97 contains an external posting path: "
            f"imports={sorted(set(found_imports))} calls={sorted(set(found_calls))}"
        )


def format_slate_table(result: DailyProtectedSlateResult) -> str:
    header = (
        "GAME         MATCHUP       HOME_P    AWAY_P    STATUS\n"
        "-----------  ------------  --------  --------  ----------------------------"
    )
    rows = [header]
    for game in result.games:
        matchup = f"{game.away_team_id}@{game.home_team_id}"
        rows.append(
            f"{game.game_id:<11}  {matchup:<12}  "
            f"{game.model_home_win_probability:>8.4%}  "
            f"{game.model_away_win_probability:>8.4%}  {game.status}"
        )
    return "\n".join(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one manual protected NHL prediction slate from a single observation."
    )
    parser.add_argument("slate_date", help="NHL slate date, YYYY-MM-DD")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Frozen historical MoneyPuck cache directory.",
    )
    parser.add_argument(
        "--store",
        type=Path,
        default=Path(DEFAULT_STORE_RELATIVE_PATH),
        help="Append-only protected prediction-record JSONL path.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    assert_issue97_has_no_external_posting_path()
    try:
        result = run_daily_protected_predictions(
            args.slate_date,
            cache_dir=args.cache_dir,
            store_path=args.store,
        )
    except HockeyMoneylineError as exc:
        print(
            json.dumps(
                {
                    "outcome": BLOCKED,
                    "slate_date": args.slate_date,
                    "error": str(exc),
                    "external_receipts_posted": False,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        print(BLOCKED, file=sys.stderr)
        return 1

    print(format_slate_table(result))
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    print(READY)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
