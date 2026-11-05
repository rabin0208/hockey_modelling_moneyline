"""Issue #84 manual immutable forward odds archive.

Usage:
    python -m hockey_moneyline.issue84
    python -m hockey_moneyline.issue84 capture \\
        --nhl-game-id ... --home-team ... --away-team ... \\
        --scheduled-puck-drop-utc ... --book ... \\
        --home-decimal-odds ... --away-decimal-odds ... \\
        --observation-utc ... --capture-method ... --provenance ...

This entry point only writes a snapshot when the operator supplies every field.
It does not log into a sportsbook, scrape, schedule itself, or place a wager.
Issue #83's six screenshot observations are preserved as noncanonical evidence
and are not seeded into the canonical store.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from hockey_moneyline.errors import ForwardOddsArchiveError, HockeyMoneylineError
from hockey_moneyline.forward_odds_archive import (
    DEFAULT_STORE_RELATIVE_PATH,
    FORBIDDEN_SNAPSHOT_FIELDS,
    PREDICTION_RECORD_FILENAME,
    SNAPSHOT_BODY_FIELDS,
    STATUS_EXISTING_IDENTICAL,
    STATUS_WRITTEN,
    ForwardOddsArchiveStore,
    assert_store_path_separate,
    build_envelope,
    capture_manual_snapshot,
)

READY = "READY FOR FORWARD ODDS ARCHIVE REVIEW"
BLOCKED = "BLOCKED — ONE MATERIAL FORWARD-ARCHIVE DEPENDENCY"
TREATMENT_NONCANONICAL = "noncanonical"
ISSUE83_OPERATOR_ATTESTATION_COMMENT = "5915092107"
ISSUE83_FILE_METADATA_COMMENT = "5915172121"
DEFAULT_OUTPUT = Path("data/derived/issue84_forward_odds_archive_report.json")
DEFAULT_PROOF_STORE = Path("data/derived/odds_archive/forward/issue84_proof_snapshots.jsonl")
FROZEN_FITTED_ARTIFACT_SHA256 = (
    "1739c7edf14ecd23ab3e94b54ef9194a2d868026ceb320663a85fb9a292aac28"
)

# Screenshot files do not encode a UTC offset. Comment 5915172121 records the
# naive capture clocks below. Comment 5915092107 separately says the screenshots
# were contemporaneous around 09:07 PDT. Those facts are not combined into an
# observation_utc here.
_ISSUE83_GAMES = {
    "2026020006": ("PIT", "PHI", "2026-09-30T23:30:00Z"),
    "2026020008": ("NYI", "TOR", "2026-09-30T23:30:00Z"),
    "2026020007": ("LAK", "COL", "2026-10-01T02:00:00Z"),
}
_ISSUE83_PRICES = (
    ("Bet365", "2026-09-30 09:07:43", "2026020006", "2.20", "1.71"),
    ("Bet365", "2026-09-30 09:07:43", "2026020008", "2.00", "1.83"),
    ("Bet365", "2026-09-30 09:07:43", "2026020007", "2.55", "1.54"),
    ("PlayNow", "2026-09-30 09:07:20", "2026020006", "2.20", "1.71"),
    ("PlayNow", "2026-09-30 09:07:20", "2026020008", "2.05", "1.80"),
    ("PlayNow", "2026-09-30 09:07:20", "2026020007", "2.60", "1.52"),
)

NONCANONICAL_REASON = (
    "Issue #83's six Bet365/PlayNow observations stay noncanonical. "
    "Exact local wall-clock seconds came from screenshot capture metadata "
    "(Bet365 2026-09-30 09:07:43; PlayNow 2026-09-30 09:07:20; "
    f"Issue #83 comment {ISSUE83_FILE_METADATA_COMMENT}). "
    "Those screenshot files contain only naive capture metadata, and the UTC "
    "offset did not come from screenshot file metadata. "
    "The PDT/timezone basis came from the separate Issue #83 operator attestation "
    "that the screenshots were contemporaneous around 09:07 PDT "
    f"(comment {ISSUE83_OPERATOR_ATTESTATION_COMMENT}). "
    "Writing a timezone-aware observation_utc would combine those sources into a "
    "second-precision UTC instant and overstate timing certainty. "
    "The canonical forward-odds contract requires a timezone-aware observation "
    "strictly before puck drop, so these six observations are preserved as "
    "noncanonical evidence and are not seeded into "
    "data/derived/odds_archive/forward/. "
    "No screenshot file hash is available, and none is invented. "
    "Issue #83 model probabilities are not included."
)


def issue83_first_evidence() -> dict[str, Any]:
    """Auditable decision for the six Issue #83 observations. Nothing is seeded."""

    observations: list[dict[str, Any]] = []
    for book, naive_clock, game_id, away_odds, home_odds in _ISSUE83_PRICES:
        away_team, home_team, puck_drop = _ISSUE83_GAMES[game_id]
        observations.append(
            {
                "book": book,
                "nhl_game_id": game_id,
                "away_team": away_team,
                "home_team": home_team,
                "away_decimal_odds_text": away_odds,
                "home_decimal_odds_text": home_odds,
                "scheduled_puck_drop_utc": puck_drop,
                "naive_screenshot_capture_metadata": naive_clock,
                "canonical_snapshot": False,
                "observation_utc_asserted": False,
            }
        )
    if len(observations) != 6:
        raise ForwardOddsArchiveError("Issue #83 first evidence must contain six observations")
    return {
        "treatment": TREATMENT_NONCANONICAL,
        "canonical_store_seeded": False,
        "observation_count": 6,
        "reason": NONCANONICAL_REASON,
        "issue_83_comments": {
            "operator_attestation": ISSUE83_OPERATOR_ATTESTATION_COMMENT,
            "screenshot_file_metadata": ISSUE83_FILE_METADATA_COMMENT,
        },
        "timing": {
            "local_wall_clock_seconds_source": "screenshot capture metadata",
            "timezone_basis_source": (
                "Issue #83 operator attestation comment "
                f"{ISSUE83_OPERATOR_ATTESTATION_COMMENT} around 09:07 PDT"
            ),
            "utc_offset_from_screenshot_file_metadata": False,
            "screenshot_file_hashes_available": False,
            "screenshot_hashes_invented": False,
        },
        "observations": observations,
    }


def seed_issue83_observations(store: ForwardOddsArchiveStore | None = None) -> None:
    """Fail closed. Do not write the six observations into the canonical store."""

    if store is not None:
        assert_store_path_separate(store.path)
    raise ForwardOddsArchiveError(NONCANONICAL_REASON)


def _proof_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "nhl_game_id": "2099020001",
        "home_team": "PHI",
        "away_team": "PIT",
        "scheduled_puck_drop_utc": "2099-10-15T23:30:00Z",
        "book": "ExampleBook",
        "home_decimal_odds": "1.91",
        "away_decimal_odds": "1.95",
        "observation_utc": "2099-10-15T18:00:00Z",
        "capture_method": "manual_operator_entry",
        "provenance": "Issue #84 contract proof. Synthetic id 2099020001 is not a live capture.",
    }
    payload.update(overrides)
    return payload


def _expect_failure(payload: Mapping[str, Any], store: ForwardOddsArchiveStore, token: str) -> None:
    try:
        store.append(dict(payload))
    except ForwardOddsArchiveError as exc:
        if token not in str(exc):
            raise ForwardOddsArchiveError(
                f"Fail-closed error did not mention {token!r}: {exc}"
            ) from exc
        return
    raise ForwardOddsArchiveError(f"Expected fail-closed for {token}")


def _run_proof(store_path: Path) -> dict[str, Any]:
    from hockey_moneyline.prospective_artifact import (
        EXPECTED_FITTED_ARTIFACT_SHA256,
        load_committed_frozen_artifact,
    )

    if Path(store_path) == Path(DEFAULT_STORE_RELATIVE_PATH):
        raise ForwardOddsArchiveError(
            "Contract proof must not use the canonical forward-odds archive path"
        )
    if store_path.exists():
        store_path.unlink()

    checks: list[str] = []
    artifact = load_committed_frozen_artifact()
    if artifact.fitted_artifact_sha256 != EXPECTED_FITTED_ARTIFACT_SHA256:
        raise ForwardOddsArchiveError("Frozen fitted artifact SHA-256 changed")
    if artifact.fitted_artifact_sha256 != FROZEN_FITTED_ARTIFACT_SHA256:
        raise ForwardOddsArchiveError("Frozen fitted artifact SHA-256 changed")
    checks.append("frozen_fitted_artifact_unchanged")

    assert_store_path_separate(Path(DEFAULT_STORE_RELATIVE_PATH))
    if PREDICTION_RECORD_FILENAME in DEFAULT_STORE_RELATIVE_PATH:
        raise ForwardOddsArchiveError("Default odds archive path collides with prediction records")
    try:
        assert_store_path_separate(
            Path("data/derived/prediction_records") / PREDICTION_RECORD_FILENAME
        )
    except ForwardOddsArchiveError:
        checks.append("store_path_separate_from_prediction_records")
    else:
        raise ForwardOddsArchiveError(
            "Prediction-record path was accepted as a forward-odds store"
        )

    store = ForwardOddsArchiveStore(store_path)
    first = store.append(_proof_payload())
    if first.status != STATUS_WRITTEN or not first.written:
        raise ForwardOddsArchiveError("Valid two-sided pregame snapshot did not write")
    checks.append("valid_two_sided_pregame_write")

    second = store.append(
        _proof_payload(
            home_decimal_odds="1.9100",
            observation_utc="2099-10-15T18:00:00+00:00",
        )
    )
    if second.status != STATUS_EXISTING_IDENTICAL or second.written:
        raise ForwardOddsArchiveError("Exact duplicate was not an idempotent no-op")
    if sum(1 for _ in store.iter_envelopes()) != 1:
        raise ForwardOddsArchiveError("Idempotent duplicate appended another line")
    checks.append("exact_duplicate_idempotent")

    before = store.path.read_bytes()
    _expect_failure(_proof_payload(home_decimal_odds="1.50"), store, "Conflicting")
    if store.path.read_bytes() != before:
        raise ForwardOddsArchiveError("Conflicting duplicate mutated the store")
    checks.append("conflicting_duplicate_fails_closed")

    _expect_failure(
        {key: value for key, value in _proof_payload().items() if key != "away_decimal_odds"},
        store,
        "missing one side",
    )
    checks.append("missing_one_side_fails_closed")

    for bad in ("nan", "inf", "1", "1.0", "0.5", "nope"):
        _expect_failure(_proof_payload(away_decimal_odds=bad), store, "decimal odds")
    checks.append("malformed_odds_fail_closed")

    _expect_failure(
        _proof_payload(observation_utc="2099-10-15 18:00:00"),
        store,
        "timezone-aware",
    )
    _expect_failure(
        _proof_payload(scheduled_puck_drop_utc="2099-10-15T23:30:00"),
        store,
        "timezone-aware",
    )
    checks.append("naive_timestamps_fail_closed")

    _expect_failure(
        _proof_payload(observation_utc="2099-10-15T23:30:00Z"),
        store,
        "strictly before",
    )
    _expect_failure(
        _proof_payload(observation_utc="2099-10-15T23:30:01Z"),
        store,
        "strictly before",
    )
    checks.append("observation_not_before_puck_drop_fails_closed")

    _expect_failure(
        {key: value for key, value in _proof_payload().items() if key != "provenance"},
        store,
        "provenance",
    )
    _expect_failure(
        {key: value for key, value in _proof_payload().items() if key != "book"},
        store,
        "book/source",
    )
    _expect_failure(
        {key: value for key, value in _proof_payload().items() if key != "home_team"},
        store,
        "home_team",
    )
    _expect_failure(
        {key: value for key, value in _proof_payload().items() if key != "nhl_game_id"},
        store,
        "NHL game ID",
    )
    checks.append("missing_provenance_book_team_game_fails_closed")

    envelope = build_envelope(_proof_payload())
    keys = set(envelope) | set(envelope["snapshot"])
    if keys & FORBIDDEN_SNAPSHOT_FIELDS:
        raise ForwardOddsArchiveError(f"Snapshot serialization leaked forbidden fields: {keys}")
    if set(envelope["snapshot"]) != set(SNAPSHOT_BODY_FIELDS):
        raise ForwardOddsArchiveError("Snapshot body is not the closed capture contract")
    checks.append("serialization_excludes_model_outcome_recommendation_stake")

    evidence = issue83_first_evidence()
    if evidence["treatment"] != TREATMENT_NONCANONICAL or evidence["canonical_store_seeded"]:
        raise ForwardOddsArchiveError("Issue #83 first evidence was not kept noncanonical")
    if evidence["observation_count"] != 6 or len(evidence["observations"]) != 6:
        raise ForwardOddsArchiveError("Issue #83 first evidence is not the six observations")
    proof_lines_before = store.path.read_bytes()
    try:
        seed_issue83_observations(store)
    except ForwardOddsArchiveError as exc:
        if "noncanonical" not in str(exc):
            raise
    else:
        raise ForwardOddsArchiveError("Issue #83 canonical seed did not fail closed")
    if store.path.read_bytes() != proof_lines_before:
        raise ForwardOddsArchiveError("Issue #83 refusal wrote into the canonical store")
    checks.append("issue83_six_observations_noncanonical")

    old = store.append(
        _proof_payload(
            nhl_game_id="2099020002",
            observation_utc="2099-09-01T00:00:00Z",
            scheduled_puck_drop_utc="2099-10-15T23:30:00Z",
        )
    )
    if old.status != STATUS_WRITTEN:
        raise ForwardOddsArchiveError("A quote-age cutoff rejected an early pregame snapshot")
    checks.append("no_quote_age_cutoff")

    return {
        "checks_passed": checks,
        "proof_store_path": str(store.path),
        "proof_snapshot_count": sum(1 for _ in store.iter_envelopes()),
        "frozen_fitted_artifact_sha256": artifact.fitted_artifact_sha256,
        "default_store_relative_path": DEFAULT_STORE_RELATIVE_PATH,
        "prediction_record_filename": PREDICTION_RECORD_FILENAME,
        "issue83_first_evidence": evidence,
    }


def build_issue84_report(store_path: Path | None = None) -> dict[str, Any]:
    """Run the local contract proof. Does not seed Issue #83 into the canonical store."""

    path = store_path if store_path is not None else DEFAULT_PROOF_STORE
    try:
        evidence = _run_proof(path)
    except (HockeyMoneylineError, OSError, RuntimeError) as exc:
        return {
            "outcome": BLOCKED,
            "error": str(exc),
            "issue83_first_evidence": issue83_first_evidence(),
            "default_store_relative_path": DEFAULT_STORE_RELATIVE_PATH,
        }
    return {
        "outcome": READY,
        "owner": "01 — Data, Features & Validation",
        "manual_capture_only": True,
        "quote_age_cutoff_authorized": False,
        "moneypuck_oddshistory_adopted": False,
        "moneypuck_inquiry_sent": False,
        **evidence,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Issue #84 manual forward odds archive. "
            "Operator-supplied values only. No scraper, schedule, or wager placement."
        )
    )
    sub = parser.add_subparsers(dest="command")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--proof-store", type=Path, default=DEFAULT_PROOF_STORE)

    capture = sub.add_parser(
        "capture",
        help="Write one immutable snapshot from explicit operator-supplied values",
    )
    capture.add_argument("--nhl-game-id", required=True)
    capture.add_argument("--home-team", required=True)
    capture.add_argument("--away-team", required=True)
    capture.add_argument("--scheduled-puck-drop-utc", required=True)
    capture.add_argument("--book", required=True, help="Sportsbook or other odds source name")
    capture.add_argument("--home-decimal-odds", required=True)
    capture.add_argument("--away-decimal-odds", required=True)
    capture.add_argument("--observation-utc", required=True)
    capture.add_argument("--capture-method", required=True)
    capture.add_argument("--provenance", required=True)
    capture.add_argument("--home-american-odds", default=None)
    capture.add_argument("--away-american-odds", default=None)
    capture.add_argument("--store", type=Path, default=Path(DEFAULT_STORE_RELATIVE_PATH))
    return parser


def _capture_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "nhl_game_id": args.nhl_game_id,
        "home_team": args.home_team,
        "away_team": args.away_team,
        "scheduled_puck_drop_utc": args.scheduled_puck_drop_utc,
        "book": args.book,
        "home_decimal_odds": args.home_decimal_odds,
        "away_decimal_odds": args.away_decimal_odds,
        "observation_utc": args.observation_utc,
        "capture_method": args.capture_method,
        "provenance": args.provenance,
    }
    if args.home_american_odds is not None or args.away_american_odds is not None:
        payload["home_american_odds"] = args.home_american_odds
        payload["away_american_odds"] = args.away_american_odds


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "capture":
            result = capture_manual_snapshot(_capture_payload(args), store=args.store)
            print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
            return 0
        report = build_issue84_report(store_path=args.proof_store)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        print(f"Wrote {args.output}")
        print(report["outcome"])
        return 0 if report["outcome"] == READY else 1
    except HockeyMoneylineError as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
