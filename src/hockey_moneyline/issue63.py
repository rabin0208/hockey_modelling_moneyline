"""Issue #63 deterministic pregame audit-receipt proof.

Usage:
    python -m hockey_moneyline.issue63
    python -m hockey_moneyline.issue63 receipt --store PATH --game-id GAME_ID
    python -m hockey_moneyline.issue63 receipt --record-json PATH

The proof uses one historical/synthetic prediction record. It does not emit a
real protected 2026-27 prediction, call GitHub, or post a receipt.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from hockey_moneyline.errors import HockeyMoneylineError, PredictionError
from hockey_moneyline.prediction_record_store import (
    PredictionRecordStore,
    prediction_record_content_hash,
)
from hockey_moneyline.pregame_audit_receipt import (
    MODEL_PROBABILITY_KIND,
    RECEIPT_PAYLOAD_FIELDS,
    build_pregame_audit_receipt,
)
from hockey_moneyline.prospective_artifact import (
    EXPECTED_FITTED_ARTIFACT_SHA256,
    FROZEN_FEATURE_COLUMNS,
    load_committed_frozen_artifact,
)
from hockey_moneyline.prospective_scoring import (
    CURRENT_SEASON_PRIOR_SOURCE_SHA256_KEY,
    FORBIDDEN_MARKET_FIELDS,
    feature_vector_sha256,
)

DEFAULT_OUTPUT = Path("data/derived/issue63_pregame_audit_receipt_report.json")
READY = "READY FOR TIMESTAMPED PROSPECTIVE RECEIPTS"
NOT_READY = "NOT READY — ONE MATERIAL AUDIT-RECEIPT DEPENDENCY"
_SYNTHETIC_GAME_ID = "synthetic-hist-2022020999"
_SYNTHETIC_SOURCE_SHA256 = "ab" * 32
_ALT_ARTIFACT_SHA256 = "11" * 32
_ALT_SOURCE_SHA256 = "22" * 32
_FORBIDDEN_IMPORT_ROOTS = frozenset(
    {
        "aiohttp",
        "ftplib",
        "github",
        "ghapi",
        "http",
        "httpx",
        "requests",
        "smtplib",
        "socket",
        "ssl",
        "urllib",
        "urllib3",
    }
)
_PRODUCT_MODULE_NAMES = (
    "pregame_audit_receipt.py",
    "issue63.py",
)


def synthetic_protected_prediction_record() -> dict[str, Any]:
    """Historical/synthetic record used only to exercise the receipt contract.

    This is not a protected 2026-27 prediction and it is not produced by scoring.
    """

    values = [float(index) / 100.0 for index in range(1, 26)]
    return {
        "game_id": _SYNTHETIC_GAME_ID,
        "season": 20222023,
        "game_type": "regular",
        "home_team_id": "BOS",
        "away_team_id": "BUF",
        "scheduled_start_time_utc": "2023-03-01T23:00:00+00:00",
        "prediction_timestamp_utc": "2023-03-01T12:00:00+00:00",
        "model_artifact_version": "frozen-prospective-fixture",
        "model_artifact_sha256": EXPECTED_FITTED_ARTIFACT_SHA256,
        "feature_schema_version": "synthetic-audit-receipt-fixture",
        "source_versions": {
            CURRENT_SEASON_PRIOR_SOURCE_SHA256_KEY: _SYNTHETIC_SOURCE_SHA256,
            "historical_replay_dry_run": True,
            "issue63_synthetic_receipt_fixture": True,
        },
        "feature_columns": list(FROZEN_FEATURE_COLUMNS),
        "feature_values": values,
        "feature_vector_sha256": feature_vector_sha256(values),
        "model_home_win_probability": 0.51,
        "probability_kind": MODEL_PROBABILITY_KIND,
    }


def synthetic_prediction_envelope(
    record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Envelope shape accepted by receipt generation. Not a store write."""

    inner = dict(synthetic_protected_prediction_record() if record is None else record)
    if "source_versions" in inner:
        inner["source_versions"] = dict(inner["source_versions"])
    return {
        "record": inner,
        "record_sha256": prediction_record_content_hash(inner),
    }


def _product_module_paths() -> tuple[Path, Path]:
    module_dir = Path(__file__).resolve().parent
    return tuple(module_dir / name for name in _PRODUCT_MODULE_NAMES)


def assert_receipt_product_surface_is_local() -> None:
    """Fail closed if the receipt modules import a network or GitHub client."""

    for path in _product_module_paths():
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
            for module_name in modules:
                root = module_name.split(".", 1)[0]
                if root in _FORBIDDEN_IMPORT_ROOTS:
                    raise PredictionError(
                        f"{path.name} imports network/GitHub client {module_name}"
                    )


def _replace(record: dict[str, Any], **changes: Any) -> dict[str, Any]:
    updated = dict(record)
    updated["source_versions"] = dict(record["source_versions"])
    updated["feature_values"] = list(record["feature_values"])
    updated["feature_columns"] = list(record["feature_columns"])
    for key, value in changes.items():
        if key == "source_versions":
            updated["source_versions"] = dict(value)
        else:
            updated[key] = value
    return updated


def _fail_closed_cases(baseline: Mapping[str, Any]) -> list[dict[str, Any]]:
    record = dict(baseline["record"])
    cases: list[tuple[str, dict[str, Any] | None, str | None]] = []

    missing_hash = {"record": record}
    cases.append(("missing_prediction_record_hash", missing_hash, None))

    invalid_hash = {"record": record, "record_sha256": "0" * 64}
    cases.append(("invalid_prediction_record_hash", invalid_hash, None))

    malformed_hash = {"record": record, "record_sha256": "not-a-hash"}
    cases.append(("malformed_prediction_record_hash", malformed_hash, None))

    missing_artifact = _replace(record)
    missing_artifact.pop("model_artifact_sha256")
    cases.append(("missing_artifact_provenance", synthetic_prediction_envelope(missing_artifact), None))

    malformed_artifact = _replace(record, model_artifact_sha256="abcd")
    cases.append(
        ("malformed_artifact_provenance", synthetic_prediction_envelope(malformed_artifact), None)
    )

    missing_source = _replace(record)
    missing_source["source_versions"].pop(CURRENT_SEASON_PRIOR_SOURCE_SHA256_KEY)
    cases.append(("missing_source_provenance", synthetic_prediction_envelope(missing_source), None))

    malformed_source = _replace(record)
    malformed_source["source_versions"][CURRENT_SEASON_PRIOR_SOURCE_SHA256_KEY] = "zz" * 32
    cases.append(
        ("malformed_source_provenance", synthetic_prediction_envelope(malformed_source), None)
    )

    missing_vector = _replace(record)
    missing_vector.pop("feature_vector_sha256")
    cases.append(("missing_vector_provenance", synthetic_prediction_envelope(missing_vector), None))

    mismatched_vector = _replace(record, feature_vector_sha256="33" * 32)
    cases.append(
        (
            "invalid_vector_provenance",
            synthetic_prediction_envelope(mismatched_vector),
            None,
        )
    )

    at_drop = _replace(
        record,
        prediction_timestamp_utc=record["scheduled_start_time_utc"],
    )
    cases.append(("prediction_at_puck_drop", synthetic_prediction_envelope(at_drop), None))

    after_drop = _replace(
        record,
        prediction_timestamp_utc="2023-03-02T00:30:00+00:00",
    )
    cases.append(("prediction_after_puck_drop", synthetic_prediction_envelope(after_drop), None))

    wrong_kind = _replace(record, probability_kind="market_home_win")
    cases.append(("probability_kind_not_model", synthetic_prediction_envelope(wrong_kind), None))

    missing_kind = _replace(record)
    missing_kind.pop("probability_kind")
    cases.append(("missing_probability_kind", synthetic_prediction_envelope(missing_kind), None))

    missing_game = _replace(record, game_id="")
    cases.append(("missing_canonical_game_id", synthetic_prediction_envelope(missing_game), None))

    missing_start = _replace(record)
    missing_start.pop("scheduled_start_time_utc")
    cases.append(
        ("missing_scheduled_start", synthetic_prediction_envelope(missing_start), None)
    )

    results: list[dict[str, Any]] = []
    for name, payload, _unused in cases:
        try:
            build_pregame_audit_receipt(payload)
        except PredictionError as exc:
            results.append({"case": name, "failed_closed": True, "message": str(exc)})
        else:
            results.append({"case": name, "failed_closed": False, "message": None})
    return results


def build_issue63_receipt_report() -> dict[str, Any]:
    blockers: list[str] = []
    try:
        assert_receipt_product_surface_is_local()
    except PredictionError as exc:
        blockers.append(str(exc))

    artifact = load_committed_frozen_artifact()
    if artifact.fitted_artifact_sha256 != EXPECTED_FITTED_ARTIFACT_SHA256:
        blockers.append("committed frozen artifact hash drifted")
    if artifact.fitted_artifact_sha256 != (
        "1739c7edf14ecd23ab3e94b54ef9194a2d868026ceb320663a85fb9a292aac28"
    ):
        blockers.append("frozen fitted artifact SHA-256 changed")

    envelope = synthetic_prediction_envelope()
    first = build_pregame_audit_receipt(envelope)
    second = build_pregame_audit_receipt(envelope)
    if first.to_dict() != second.to_dict() or first.receipt_sha256 != second.receipt_sha256:
        blockers.append("identical synthetic records did not produce identical receipts")

    record = envelope["record"]
    changed_record = _replace(record, home_team_id="CAR")
    changed_record_receipt = build_pregame_audit_receipt(
        synthetic_prediction_envelope(changed_record)
    )
    if changed_record_receipt.prediction_record_sha256 == first.prediction_record_sha256:
        blockers.append("changed record content did not change the prediction-record hash")
    if changed_record_receipt.receipt_sha256 == first.receipt_sha256:
        blockers.append("changed prediction-record hash did not change the receipt hash")

    changed_artifact = _replace(record, model_artifact_sha256=_ALT_ARTIFACT_SHA256)
    changed_source = _replace(record)
    changed_source["source_versions"][CURRENT_SEASON_PRIOR_SOURCE_SHA256_KEY] = (
        _ALT_SOURCE_SHA256
    )
    changed_values = _replace(record)
    changed_values["feature_values"][0] = changed_values["feature_values"][0] + 1.0
    changed_values["feature_vector_sha256"] = feature_vector_sha256(
        changed_values["feature_values"]
    )
    artifact_receipt = build_pregame_audit_receipt(
        synthetic_prediction_envelope(changed_artifact)
    )
    source_receipt = build_pregame_audit_receipt(
        synthetic_prediction_envelope(changed_source)
    )
    vector_receipt = build_pregame_audit_receipt(
        synthetic_prediction_envelope(changed_values)
    )
    if artifact_receipt.receipt_sha256 == first.receipt_sha256:
        blockers.append("changed artifact identity did not change the receipt hash")
    if source_receipt.receipt_sha256 == first.receipt_sha256:
        blockers.append("changed source identity did not change the receipt hash")
    if vector_receipt.receipt_sha256 == first.receipt_sha256:
        blockers.append("changed feature-vector identity did not change the receipt hash")
    if artifact_receipt.frozen_artifact_sha256 == first.frozen_artifact_sha256:
        blockers.append("artifact identity was not copied into the receipt")
    if source_receipt.current_season_source_sha256 == first.current_season_source_sha256:
        blockers.append("source identity was not copied into the receipt")
    if vector_receipt.feature_vector_sha256 == first.feature_vector_sha256:
        blockers.append("feature-vector identity was not copied into the receipt")

    leaked = [
        field
        for field in (*FORBIDDEN_MARKET_FIELDS, "feature_values", "source_versions")
        if field in first.payload() or field in first.to_dict()
    ]
    if leaked:
        blockers.append(f"receipt included non-receipt fields: {leaked}")
    if tuple(first.payload()) != RECEIPT_PAYLOAD_FIELDS:
        blockers.append("receipt payload fields drifted from the receipt contract")
    if first.probability_kind != MODEL_PROBABILITY_KIND:
        blockers.append("receipt probability kind drifted from model_home_win")

    fail_closed = _fail_closed_cases(envelope)
    if any(not case["failed_closed"] for case in fail_closed):
        names = [case["case"] for case in fail_closed if not case["failed_closed"]]
        blockers.append(f"fail-closed cases did not raise: {names}")

    outcome = READY if not blockers else NOT_READY
    return {
        "issue": 63,
        "outcome": outcome,
        "blockers": blockers,
        "frozen_contract_unchanged": {
            "model_parameters_changed": False,
            "feature_protocol_changed": False,
            "prediction_record_contents_changed": False,
            "fitted_artifact_sha256": artifact.fitted_artifact_sha256,
            "expected_fitted_artifact_sha256": EXPECTED_FITTED_ARTIFACT_SHA256,
            "matches_expected_fitted_artifact_sha256": (
                artifact.fitted_artifact_sha256 == EXPECTED_FITTED_ARTIFACT_SHA256
            ),
            "feature_count": len(FROZEN_FEATURE_COLUMNS),
            "2026_27_prediction_emitted": False,
            "refit_during_receipt_generation": False,
        },
        "receipt_surface": {
            "manual_generation_only": True,
            "github_api_in_product_code": False,
            "automatic_posting": False,
            "recurring_schedule": False,
            "daemon": False,
            "webhook": False,
            "production_deployed": False,
            "odds_staking_recommendations_required": False,
            "raw_provider_payload_included": False,
            "external_timestamp_is_evidence_only": True,
            "local_record_remains_full_audit_record": True,
        },
        "synthetic_receipt": first.to_dict()
        | {
            "deterministic_identical_input": first.receipt_sha256 == second.receipt_sha256,
            "changed_record_hash_changed_receipt_hash": (
                changed_record_receipt.receipt_sha256 != first.receipt_sha256
            ),
            "changed_artifact_identity_changed_receipt_hash": (
                artifact_receipt.receipt_sha256 != first.receipt_sha256
            ),
            "changed_source_identity_changed_receipt_hash": (
                source_receipt.receipt_sha256 != first.receipt_sha256
            ),
            "changed_vector_identity_changed_receipt_hash": (
                vector_receipt.receipt_sha256 != first.receipt_sha256
            ),
        },
        "fail_closed": fail_closed,
        "remaining_dependency": (
            None
            if not blockers
            else {
                "owner": "03 — Prediction, Odds & Recommendation Engine",
                "dependency": blockers[0],
                "unblock_signal": (
                    "Receipt generation must be deterministic, fail closed on missing "
                    "provenance or late timestamps, and stay local to the prediction record"
                ),
            }
        ),
        "notes": [
            "Issue #63 adds a receipt contract only. It does not change model, artifact, features, or prediction-record contents.",
            "Validation used one historical/synthetic prediction record. No real protected 2026-27 prediction was emitted.",
            "The external GitHub comment timestamp is operator evidence. Product code does not call GitHub.",
            "The local immutable prediction record remains the full audit record.",
        ],
    }


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PredictionError(f"Unreadable prediction-record JSON {path}") from exc
    stripped = text.strip()
    if not stripped:
        raise PredictionError(f"Prediction-record JSON is empty: {path}")
    try:
        if stripped.startswith("{"):
            payload = json.loads(stripped)
        else:
            raise json.JSONDecodeError("expected object or JSONL", stripped, 0)
    except json.JSONDecodeError:
        envelopes = []
        for line_number, raw in enumerate(text.splitlines(), start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PredictionError(
                    f"Prediction-record JSONL is invalid at {path}:{line_number}"
                ) from exc
            if not isinstance(item, dict):
                raise PredictionError(
                    f"Prediction-record JSONL line {line_number} is not an object"
                )
            envelopes.append(item)
        if len(envelopes) != 1:
            raise PredictionError(
                "JSONL receipt input must contain exactly one envelope; "
                "use --store and --game-id for a multi-record store"
            )
        payload = envelopes[0]
    if not isinstance(payload, dict):
        raise PredictionError("Prediction-record JSON must be an object")
    return payload


def receipt_from_store(store: Path, game_id: str) -> dict[str, Any]:
    envelope = PredictionRecordStore(store).get_by_game_id(game_id)
    if envelope is None:
        raise PredictionError(f"No immutable prediction record for game_id {game_id}")
    return build_pregame_audit_receipt(envelope).to_dict()


def receipt_from_json(path: Path) -> dict[str, Any]:
    return build_pregame_audit_receipt(_load_json_object(path)).to_dict()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Issue #63 deterministic pregame audit receipt"
    )
    sub = parser.add_subparsers(dest="command")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)

    receipt = sub.add_parser(
        "receipt",
        help="Print a deterministic receipt for one local immutable prediction record",
    )
    receipt.add_argument("--store", type=Path, default=None)
    receipt.add_argument("--game-id", default=None)
    receipt.add_argument("--record-json", type=Path, default=None)
    receipt.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "receipt":
            if args.record_json is not None:
                payload = receipt_from_json(args.record_json)
            elif args.store is not None and args.game_id:
                payload = receipt_from_store(args.store, args.game_id)
            else:
                raise PredictionError(
                    "Receipt generation requires --record-json or both --store and --game-id"
                )
            rendered = json.dumps(payload, indent=2, sort_keys=True)
            print(rendered)
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(rendered + "\n", encoding="utf-8")
                print(f"Wrote {args.output}")
            return 0
        report = build_issue63_receipt_report()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        print(f"Wrote {args.output}")
        return 0 if report["outcome"] == READY else 1
    except HockeyMoneylineError as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
