"""Issue #35 Project HQ lifecycle sender/correlation helpers.

This module is stdlib-only so GitHub Actions can parse and emit the v1
envelope without installing modeling dependencies. It does not activate a
worker route, webhook, wake, or schedule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Sequence

from hockey_moneyline.errors import ProjectHqError

CONTROL_PLANE_REPOSITORY = "EduardoRasanmar/project-hq-automation"
LIFECYCLE_DISPATCH_EVENT = "project-hq-lifecycle"
SOURCE_EVENT_TYPE_V1 = "issue_comment"
DISPATCH_ID_PREFIX = "disp-v1-"

ALLOWED_LIFECYCLES = frozenset(
    {
        "specifying",
        "ready",
        "started",
        "review",
        "changes_requested",
        "blocked",
        "done",
    }
)
CORRELATED_DISPATCH_LIFECYCLES = frozenset({"started", "review", "blocked", "done"})
WORKER_SLUGS = {
    "eduardo": "Eduardo",
    "cursor": "Cursor",
    "grok-bot": "Grok Bot",
    "codex": "Codex",
}
PRIORITY_SLUGS = {
    "high": "High",
    "normal": "Normal",
    "low": "Low",
}
DISPATCH_TOKEN_RE = re.compile(r"^dispatch=(disp-v1-[0-9a-f]{32})$")
DISPATCH_ID_RE = re.compile(r"^disp-v1-[0-9a-f]{32}$")
NUMERIC_RE = re.compile(r"^[0-9]+$")

CLIENT_PAYLOAD_FIELDS = (
    "repository",
    "issue_number",
    "issue_node_id",
    "lifecycle_event",
    "worker",
    "priority",
    "source_event_type",
    "source_event_id",
    "dispatch_id",
)


@dataclass(frozen=True)
class ParsedProjectHqCommand:
    lifecycle: str
    worker: str = ""
    priority: str = ""
    dispatch_id: str = ""


def _single_line_command(raw: str) -> str:
    command = raw.replace("\r", "")
    if command.endswith("\n"):
        command = command[:-1]
    if "\n" in command:
        raise ProjectHqError("Project HQ lifecycle commands must be a single line.")
    return command


def parse_project_hq_command(raw: str) -> ParsedProjectHqCommand:
    command = _single_line_command(raw)
    tokens = command.split()
    if len(tokens) < 2 or tokens[0] != "/project-hq":
        raise ProjectHqError("Invalid Project HQ lifecycle command format.")
    if len(tokens) > 5:
        raise ProjectHqError("Invalid Project HQ lifecycle command format.")

    lifecycle = tokens[1]
    if lifecycle not in ALLOWED_LIFECYCLES:
        raise ProjectHqError(f"Unsupported lifecycle event '{lifecycle}'.")

    worker_slug = tokens[2] if len(tokens) >= 3 else ""
    priority_slug = tokens[3] if len(tokens) >= 4 else ""
    fifth = tokens[4] if len(tokens) == 5 else ""

    worker_value = ""
    if worker_slug:
        if worker_slug not in WORKER_SLUGS:
            raise ProjectHqError(f"Unsupported worker '{worker_slug}'.")
        worker_value = WORKER_SLUGS[worker_slug]

    priority_value = ""
    if priority_slug:
        if priority_slug not in PRIORITY_SLUGS:
            raise ProjectHqError(f"Unsupported priority '{priority_slug}'.")
        priority_value = PRIORITY_SLUGS[priority_slug]

    dispatch_id_value = ""
    if fifth:
        if lifecycle == "ready":
            raise ProjectHqError(
                "Ready lifecycle commands must not accept a caller-supplied dispatch id."
            )
        if lifecycle not in CORRELATED_DISPATCH_LIFECYCLES:
            raise ProjectHqError(
                "Dispatch correlation is only accepted for started, review, blocked, and done lifecycle commands."
            )
        match = DISPATCH_TOKEN_RE.fullmatch(fifth)
        if match is None:
            raise ProjectHqError("Lifecycle dispatch token is malformed.")
        dispatch_id_value = match.group(1)

    if lifecycle == "ready" and not worker_value:
        raise ProjectHqError("The ready lifecycle command requires a worker.")
    if lifecycle == "ready" and not priority_value:
        priority_value = "Normal"
    if lifecycle in CORRELATED_DISPATCH_LIFECYCLES and dispatch_id_value and not worker_value:
        raise ProjectHqError(f"A correlated {lifecycle} lifecycle command requires a worker.")

    return ParsedProjectHqCommand(
        lifecycle=lifecycle,
        worker=worker_value,
        priority=priority_value,
        dispatch_id=dispatch_id_value,
    )


def dispatch_id_looks_valid(dispatch_id: str) -> bool:
    return bool(DISPATCH_ID_RE.fullmatch(dispatch_id))


def compute_dispatch_id(
    repository: str,
    issue_number: str,
    source_event_type: str,
    source_event_id: str,
) -> str:
    if not repository:
        raise ProjectHqError("Repository is required to compute dispatch_id.")
    if not NUMERIC_RE.fullmatch(str(issue_number)):
        raise ProjectHqError("Issue number must be numeric to compute dispatch_id.")
    if source_event_type != SOURCE_EVENT_TYPE_V1:
        raise ProjectHqError(f"Unsupported source_event_type '{source_event_type}'.")
    if not NUMERIC_RE.fullmatch(str(source_event_id)):
        raise ProjectHqError("source_event_id must be a numeric GitHub comment id.")

    seed = f"{repository}|{issue_number}|{source_event_type}|{source_event_id}"
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return f"{DISPATCH_ID_PREFIX}{digest[:32]}"


def build_lifecycle_envelope(
    *,
    repository: str,
    issue_number: str,
    issue_node_id: str,
    lifecycle_event: str,
    worker: str = "",
    priority: str = "",
    parsed_dispatch_id: str = "",
    comment_id: str = "",
) -> dict[str, object]:
    if not repository:
        raise ProjectHqError("Repository is required.")
    if not NUMERIC_RE.fullmatch(str(issue_number)):
        raise ProjectHqError("Issue number must be numeric.")
    if not issue_node_id:
        raise ProjectHqError("Issue node ID is unavailable; refusing lifecycle dispatch.")
    if lifecycle_event not in ALLOWED_LIFECYCLES:
        raise ProjectHqError(f"Unsupported lifecycle event '{lifecycle_event}'.")
    if worker and worker not in WORKER_SLUGS.values():
        raise ProjectHqError(f"Unsupported worker '{worker}'.")
    if priority and priority not in PRIORITY_SLUGS.values():
        raise ProjectHqError(f"Unsupported priority '{priority}'.")

    source_event_type = ""
    source_event_id = ""
    dispatch_id = parsed_dispatch_id

    if lifecycle_event == "ready":
        if parsed_dispatch_id:
            raise ProjectHqError(
                "Ready lifecycle commands must not accept a caller-supplied dispatch id."
            )
        source_event_type = SOURCE_EVENT_TYPE_V1
        source_event_id = str(comment_id)
        dispatch_id = compute_dispatch_id(
            repository,
            str(issue_number),
            source_event_type,
            source_event_id,
        )
    elif lifecycle_event in CORRELATED_DISPATCH_LIFECYCLES and parsed_dispatch_id:
        if not dispatch_id_looks_valid(parsed_dispatch_id):
            raise ProjectHqError("Lifecycle dispatch token is malformed.")
        source_event_type = SOURCE_EVENT_TYPE_V1
        source_event_id = str(comment_id)
        if not NUMERIC_RE.fullmatch(source_event_id):
            raise ProjectHqError("source_event_id must be a numeric GitHub comment id.")
    elif parsed_dispatch_id:
        raise ProjectHqError(
            "Dispatch correlation is only accepted for started, review, blocked, and done lifecycle commands."
        )

    payload = {
        "repository": repository,
        "issue_number": str(issue_number),
        "issue_node_id": issue_node_id,
        "lifecycle_event": lifecycle_event,
        "worker": worker,
        "priority": priority,
        "source_event_type": source_event_type,
        "source_event_id": source_event_id,
        "dispatch_id": dispatch_id,
    }
    missing = [field for field in CLIENT_PAYLOAD_FIELDS if field not in payload]
    if missing:
        raise ProjectHqError(f"Sender envelope is missing fields: {missing}")
    return {
        "event_type": LIFECYCLE_DISPATCH_EVENT,
        "client_payload": payload,
    }


def github_output_lines(parsed: ParsedProjectHqCommand) -> str:
    return "\n".join(
        [
            f"lifecycle={parsed.lifecycle}",
            f"worker={parsed.worker}",
            f"priority={parsed.priority}",
            f"dispatch_id={parsed.dispatch_id}",
        ]
    )


def _write_github_output(parsed: ParsedProjectHqCommand) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    rendered = github_output_lines(parsed) + "\n"
    if output_path:
        with open(output_path, "a", encoding="utf-8") as handle:
            handle.write(rendered)
        return
    sys.stdout.write(rendered)


def _parse_github(_args: argparse.Namespace) -> int:
    raw = os.environ.get("PROJECT_HQ_COMMAND")
    if raw is None:
        raise ProjectHqError("PROJECT_HQ_COMMAND is required.")
    _write_github_output(parse_project_hq_command(raw))
    return 0


def _emit_envelope(args: argparse.Namespace) -> int:
    envelope = build_lifecycle_envelope(
        repository=args.repository,
        issue_number=args.issue_number,
        issue_node_id=args.issue_node_id,
        lifecycle_event=args.lifecycle,
        worker=args.worker,
        priority=args.priority,
        parsed_dispatch_id=args.parsed_dispatch_id,
        comment_id=args.comment_id,
    )
    sys.stdout.write(json.dumps(envelope, separators=(",", ":")))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hockey Project HQ lifecycle sender helpers")
    sub = parser.add_subparsers(dest="command", required=True)

    parse_github = sub.add_parser("parse-github", help="Parse $PROJECT_HQ_COMMAND for GitHub Actions")
    parse_github.set_defaults(func=_parse_github)

    envelope = sub.add_parser("envelope", help="Emit the constrained repository_dispatch JSON")
    envelope.add_argument("--repository", required=True)
    envelope.add_argument("--issue-number", required=True)
    envelope.add_argument("--issue-node-id", required=True)
    envelope.add_argument("--lifecycle", required=True)
    envelope.add_argument("--worker", default="")
    envelope.add_argument("--priority", default="")
    envelope.add_argument("--parsed-dispatch-id", default="")
    envelope.add_argument("--comment-id", default="")
    envelope.set_defaults(func=_emit_envelope)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except ProjectHqError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
