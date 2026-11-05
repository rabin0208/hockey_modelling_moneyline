"""Issue #87 forward odds archive.

The store remains the Issue #84 archive at
``data/derived/odds_archive/forward/forward_odds_snapshots.jsonl``.

Issue #85 should import the reader surface from
``hockey_moneyline.forward_odds_archive``:

- ``read_validated_snapshots``
- ``validate_forward_odds_snapshot``
- ``write_forward_odds_snapshot``
- ``ValidatedForwardOddsSnapshot``
- ``decimal_odds_for_market_math``

Usage:
    python -m hockey_moneyline.issue87
    python -m hockey_moneyline.issue87 read [--nhl-game-id ID] [--store PATH]
    python -m hockey_moneyline.issue87 capture ...
    python -m hockey_moneyline.issue87 odds-gap --csv PATH --schedule PATH [--confirm]
    python -m hockey_moneyline.issue87 odds-gap --fetch --schedule PATH [--confirm]

``odds-gap --fetch`` performs one manual GET. It does not retry, paginate, or
send a credential. Nothing is written unless ``--confirm`` is present.
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
    ISSUE_85_READER_SURFACE,
    read_validated_snapshots,
    validate_forward_odds_snapshot,
    write_forward_odds_snapshot,
)
from hockey_moneyline.issue84 import (
    BLOCKED,
    FROZEN_FITTED_ARTIFACT_SHA256,
    READY,
    issue83_first_evidence,
    seed_issue83_observations,
)
from hockey_moneyline.odds_gap_assist import (
    commit_odds_gap_proposal,
    fetch_odds_gap_csv,
    prepare_odds_gap_csv_file,
    prepare_odds_gap_snapshots,
)

DEFAULT_OUTPUT = Path("data/derived/issue87_forward_odds_archive_report.json")
DEFAULT_PROOF_STORE = Path("data/derived/odds_archive/forward/issue87_proof_snapshots.jsonl")

_PROOF_CSV = """snapshot_ts,sport,away,home,commence_time,market,side,line,book,american_odds,devig_fair_prob,best_price_flag
2099-10-15T16:00:00+00:00,icehockey_nhl,Pittsburgh Penguins,Philadelphia Flyers,2099-10-15T23:30:00Z,ml,home,,examplebook,-110,0.523810,0
2099-10-15T16:00:00+00:00,icehockey_nhl,Pittsburgh Penguins,Philadelphia Flyers,2099-10-15T23:30:00Z,ml,away,,examplebook,-110,0.476190,1
2099-10-15T18:00:00+00:00,icehockey_nhl,Pittsburgh Penguins,Philadelphia Flyers,2099-10-15T23:30:00Z,ml,home,,examplebook,-120,0.545455,0
2099-10-15T18:00:00+00:00,icehockey_nhl,Pittsburgh Penguins,Philadelphia Flyers,2099-10-15T23:30:00Z,ml,away,,examplebook,+100,0.454545,1
"""
_PROOF_SCHEDULE = (
    {
        "nhl_game_id": "2099020001",
        "home_team": "PHI",
        "away_team": "PIT",
        "scheduled_puck_drop_utc": "2099-10-15T23:30:00Z",
    },
)


def _proof_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "nhl_game_id": "2099020001",
        "home_team": "PHI",
        "away_team": "PIT",
        "scheduled_puck_drop_utc": "2099-10-15T23:30:00Z",
        "book": "ExampleBook",
        "home_decimal_odds": "1.91",
        "away_decimal_odds": "1.91",
        "home_american_odds": "-110",
        "away_american_odds": "-110",
        "observation_utc": "2099-10-15T18:00:00Z",
        "capture_method": "manual_operator_entry",
        "provenance": "Issue #87 contract proof. Synthetic id 2099020001 is not a live capture.",
    }
    payload.update(overrides)
    return payload


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

    view = validate_forward_odds_snapshot(_proof_payload())
    if view.nhl_game_id != "2099020001" or view.home_team != "PHI" or view.away_team != "PIT":
        raise ForwardOddsArchiveError("NHL identity was not preserved")
    again = validate_forward_odds_snapshot(_proof_payload())
    if view.snapshot_sha256 != again.snapshot_sha256:
        raise ForwardOddsArchiveError("Snapshot hash was not deterministic")
    field_names = set(view.to_dict())
    if field_names & FORBIDDEN_SNAPSHOT_FIELDS:
        raise ForwardOddsArchiveError(
            f"Reader surface leaked forbidden fields: {sorted(field_names & FORBIDDEN_SNAPSHOT_FIELDS)}"
        )
    checks.append("reader_preserves_identity_and_hash")

    written = write_forward_odds_snapshot(_proof_payload(), store=store_path)
    if not written.written:
        raise ForwardOddsArchiveError("Valid two-sided snapshot did not write")
    duplicate = write_forward_odds_snapshot(_proof_payload(), store=store_path)
    if duplicate.written or duplicate.snapshot_sha256 != written.snapshot_sha256:
        raise ForwardOddsArchiveError("Exact duplicate was not an idempotent no-op")
    rows = read_validated_snapshots(store_path, nhl_game_id="2099020001")
    if len(rows) != 1 or rows[0].nhl_game_id != "2099020001":
        raise ForwardOddsArchiveError("Issue #85 reader did not return the stored NHL identity")
    if rows[0].home_american_odds != "-110" or rows[0].away_american_odds != "-110":
        raise ForwardOddsArchiveError("Optional American prices were not preserved")
    checks.append("issue85_reader_round_trip")

    proposal = prepare_odds_gap_snapshots(_PROOF_CSV, _PROOF_SCHEDULE, book="examplebook")
    if proposal.to_dict()["moneypuck_oddshistory_adopted"] or proposal.to_dict()["credentials_used"]:
        raise ForwardOddsArchiveError("Odds Gap proposal adopted a forbidden source")
    if len(proposal.ready) != 1:
        raise ForwardOddsArchiveError(
            "Bounded Odds Gap assist did not keep the latest two-sided snapshot"
        )
    ready = proposal.ready[0]
    if ready["nhl_game_id"] != "2099020001" or "devig_fair_prob" in ready:
        raise ForwardOddsArchiveError("Odds Gap assist invented an id or stored a fair probability")
    if "0.523810" in json.dumps(ready) or "0.476190" in json.dumps(ready):
        raise ForwardOddsArchiveError("Odds Gap devig probability was copied into the snapshot")
    if commit_odds_gap_proposal(proposal, store=store_path, confirm=False):
        raise ForwardOddsArchiveError("Odds Gap assist wrote without confirm")
    confirmed = commit_odds_gap_proposal(proposal, store=store_path, confirm=True)
    if len(confirmed) != 1 or not confirmed[0]["written"]:
        raise ForwardOddsArchiveError("Confirmed Odds Gap snapshot did not write")
    checks.append("odds_gap_bounded_assist")

    evidence = issue83_first_evidence()
    if evidence["canonical_store_seeded"] or evidence["observation_count"] != 6:
        raise ForwardOddsArchiveError("Issue #83 evidence was not kept noncanonical")
    before = store_path.read_bytes()
    try:
        seed_issue83_observations()
    except ForwardOddsArchiveError as exc:
        if "noncanonical" not in str(exc):
            raise
    else:
        raise ForwardOddsArchiveError("Issue #83 canonical seed did not fail closed")
    if store_path.read_bytes() != before:
        raise ForwardOddsArchiveError("Issue #83 refusal wrote into the proof store")
    checks.append("issue83_six_observations_noncanonical")

    return {
        "checks_passed": checks,
        "issue_85_reader_surface": list(ISSUE_85_READER_SURFACE),
        "default_store_relative_path": DEFAULT_STORE_RELATIVE_PATH,
        "frozen_fitted_artifact_sha256": artifact.fitted_artifact_sha256,
        "proof_store_path": str(store_path),
        "issue83_first_evidence": evidence,
        "odds_gap_ready_count": len(proposal.ready),
    }


def build_issue87_report(store_path: Path | None = None) -> dict[str, Any]:
    """Run the local contract proof. No network and no canonical-store seed."""

    path = store_path if store_path is not None else DEFAULT_PROOF_STORE
    try:
        evidence = _run_proof(path)
    except (HockeyMoneylineError, OSError, RuntimeError) as exc:
        return {
            "outcome": BLOCKED,
            "error": str(exc),
            "issue_85_reader_surface": list(ISSUE_85_READER_SURFACE),
            "default_store_relative_path": DEFAULT_STORE_RELATIVE_PATH,
            "issue83_first_evidence": issue83_first_evidence(),
        }
    return {
        "outcome": READY,
        "owner": "01 — Data, Features & Validation",
        "canonical_issue": 87,
        "historical_issue": 84,
        "manual_capture": True,
        "odds_gap_assist": "manual_bounded_free_tier",
        "moneypuck_oddshistory_adopted": False,
        **evidence,
    }


def _capture_payload(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
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
    return payload


def _load_schedule(path: Path) -> list[Mapping[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ForwardOddsArchiveError(f"Unreadable NHL schedule identity file {path}") from exc
    if not isinstance(payload, list):
        raise ForwardOddsArchiveError(
            "NHL schedule identity file must be a JSON list; refusing to invent NHL ids"
        )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Issue #87 forward odds archive. "
            "Manual capture, validated reader, and one optional bounded Odds Gap assist."
        )
    )
    sub = parser.add_subparsers(dest="command")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--proof-store", type=Path, default=DEFAULT_PROOF_STORE)

    reader = sub.add_parser("read", help="Print validated snapshots for Issue #85")
    reader.add_argument("--store", type=Path, default=Path(DEFAULT_STORE_RELATIVE_PATH))
    reader.add_argument("--nhl-game-id", default=None)

    capture = sub.add_parser("capture", help="Write one operator-supplied snapshot")
    capture.add_argument("--nhl-game-id", required=True)
    capture.add_argument("--home-team", required=True)
    capture.add_argument("--away-team", required=True)
    capture.add_argument("--scheduled-puck-drop-utc", required=True)
    capture.add_argument("--book", required=True)
    capture.add_argument("--home-decimal-odds", required=True)
    capture.add_argument("--away-decimal-odds", required=True)
    capture.add_argument("--observation-utc", required=True)
    capture.add_argument("--capture-method", required=True)
    capture.add_argument("--provenance", required=True)
    capture.add_argument("--home-american-odds", default=None)
    capture.add_argument("--away-american-odds", default=None)
    capture.add_argument("--store", type=Path, default=Path(DEFAULT_STORE_RELATIVE_PATH))

    assist = sub.add_parser(
        "odds-gap",
        help="Prepare snapshots from one free Odds Gap CSV. Write only with --confirm",
    )
    source = assist.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", type=Path)
    source.add_argument("--fetch", action="store_true")
    assist.add_argument("--schedule", type=Path, required=True)
    assist.add_argument("--book", default=None)
    assist.add_argument("--all-pregame", action="store_true")
    assist.add_argument("--confirm", action="store_true")
    assist.add_argument("--store", type=Path, default=Path(DEFAULT_STORE_RELATIVE_PATH))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "read":
            rows = read_validated_snapshots(args.store, nhl_game_id=args.nhl_game_id)
            print(json.dumps([row.to_dict() for row in rows], indent=2, sort_keys=True))
            return 0
        if args.command == "capture":
            result = write_forward_odds_snapshot(_capture_payload(args), store=args.store)
            print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
            return 0
        if args.command == "odds-gap":
            schedule = _load_schedule(args.schedule)
            fetch = None
            if args.fetch:
                fetch = fetch_odds_gap_csv()
                proposal = prepare_odds_gap_snapshots(
                    fetch.text,
                    schedule,
                    book=args.book,
                    all_pregame_observations=args.all_pregame,
                    fetch=fetch,
                )
            else:
                proposal = prepare_odds_gap_csv_file(
                    args.csv,
                    schedule,
                    book=args.book,
                    all_pregame_observations=args.all_pregame,
                )
            report = proposal.to_dict()
            report["written"] = commit_odds_gap_proposal(
                proposal,
                store=args.store,
                confirm=args.confirm,
            )
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        report = build_issue87_report(store_path=args.proof_store)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
        print(f"Wrote {args.output}")
        print(report["outcome"])
        return 0 if report["outcome"] == READY else 1
    except HockeyMoneylineError as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
