"""Local Streamlit MVP for Hockey Moneyline.

Launch (after ``python -m pip install -e ".[dashboard]"``):

    python -m streamlit run src/hockey_moneyline/dashboard_app.py -- \\
        --snapshot data/local/nhl_schedule_snapshot.json \\
        --store data/derived/prediction_records/pregame_prediction_records.jsonl

The app only renders ``build_dashboard_view``. It does not score, fit, write
records, post receipts, or contact a sportsbook.
"""

from __future__ import annotations

import argparse
import html
import sys
from datetime import date, datetime
from typing import Any

from hockey_moneyline.dashboard import (
    DEFAULT_DISPLAY_TIMEZONE,
    DEFAULT_SNAPSHOT_RELATIVE_PATH,
    MARKET_LAYER_ABSENT,
    STATUS_NOT_SCORED,
    STATUS_RECEIPT_READY,
    STATUS_RECORDED,
    DashboardView,
    GameCard,
    build_dashboard_view,
    load_nhl_snapshot,
)
from hockey_moneyline.errors import HockeyMoneylineError
from hockey_moneyline.prediction_record_store import DEFAULT_STORE_RELATIVE_PATH

_PAGE_TITLE = "Hockey Moneyline"
_LAUNCH_HINT = (
    'python -m streamlit run src/hockey_moneyline/dashboard_app.py -- '
    "--snapshot data/local/nhl_schedule_snapshot.json "
    "--store data/derived/prediction_records/pregame_prediction_records.jsonl"
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--snapshot", default=DEFAULT_SNAPSHOT_RELATIVE_PATH)
    parser.add_argument("--store", default=DEFAULT_STORE_RELATIVE_PATH)
    parser.add_argument("--date", default="")
    parser.add_argument("--timezone", default=DEFAULT_DISPLAY_TIMEZONE)
    known, _unknown = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    return known


def _parse_cli_date(raw: str) -> date | None:
    if not raw.strip():
        return None
    return date.fromisoformat(raw.strip())


def run() -> None:
    try:
        import streamlit as st
    except ImportError as exc:
        raise SystemExit(
            "Streamlit is not installed. Install the optional dashboard extra with "
            'python -m pip install -e ".[dashboard]", then launch with '
            f"{_LAUNCH_HINT}"
        ) from exc
    _render(st)


def _render(st: Any) -> None:
    st.set_page_config(page_title=_PAGE_TITLE, layout="wide")
    _inject_styles(st)
    args = _parse_args()
    st.markdown('<p class="hm-kicker">Local read-only MVP</p>', unsafe_allow_html=True)
    st.markdown(f'<h1 class="hm-title">{_PAGE_TITLE}</h1>', unsafe_allow_html=True)
    st.caption(
        "Daily slate, frozen-model probabilities, and protected-record provenance. "
        "This page does not score games or place bets."
    )

    with st.sidebar:
        st.header("Local sources")
        snapshot_path = st.text_input("NHL snapshot JSON", value=args.snapshot, key="snapshot_path")
        store_path = st.text_input("Prediction-record store", value=args.store, key="store_path")
        st.caption("Reloading happens when you change a field. There is no background refresh.")
        st.caption(_LAUNCH_HINT)

    try:
        snapshot = load_nhl_snapshot(snapshot_path)
    except HockeyMoneylineError as exc:
        st.error(str(exc))
        st.stop()

    covered = sorted(snapshot.covered_dates)
    if args.date.strip():
        try:
            requested = _parse_cli_date(args.date)
        except ValueError:
            st.error(f"Slate date is malformed: {args.date}")
            st.stop()
        if requested not in covered:
            listed = ", ".join(day.isoformat() for day in covered)
            st.error(
                f"Selected date {requested.isoformat()} is outside the loaded NHL "
                f"snapshot coverage ({listed}). Refusing to invent a slate."
            )
            st.stop()
    default_date = _initial_date(covered, args.date)
    selected = st.date_input(
        "Slate date",
        value=default_date,
        min_value=covered[0],
        max_value=covered[-1],
        key="slate_date",
    )
    if not isinstance(selected, date):
        st.error("Slate date is malformed.")
        st.stop()

    try:
        view = build_dashboard_view(
            snapshot_path=snapshot_path,
            store_path=store_path,
            selected_date=selected,
            display_timezone=args.timezone,
        )
    except HockeyMoneylineError as exc:
        st.error(str(exc))
        st.stop()

    _render_model_status(st, view)
    _render_slate(st, view)
    _render_record_log(st, view)
    st.divider()
    st.caption(
        "Local process only. No hosted URL, account, billing, odds, edge, or staking. "
        + view.disclaimer
    )


def _initial_date(covered: list[date], raw: str) -> date:
    if raw.strip():
        requested = _parse_cli_date(raw)
        if requested in covered:
            return requested
    today = datetime.now().date()
    if today in covered:
        return today
    return covered[-1]


def _inject_styles(st: Any) -> None:
    st.markdown(
        """
        <style>
        .hm-kicker { letter-spacing: 0.08em; text-transform: uppercase;
            color: #5c6b7a; font-size: 0.78rem; margin-bottom: 0; }
        .hm-title { margin-top: 0; color: #1c2b3a; }
        .hm-status { display: inline-block; padding: 0.15rem 0.55rem;
            border-radius: 999px; font-size: 0.78rem; font-weight: 600;
            letter-spacing: 0.03em; }
        .hm-not-scored { background: #e7edf3; color: #334155; }
        .hm-recorded { background: #d9e6f5; color: #1f4e79; }
        .hm-receipt { background: #e7f0ea; color: #1d4d38; }
        .hm-prob { font-size: 1.8rem; font-weight: 650; color: #1c2b3a; margin: 0; }
        .hm-team { color: #5c6b7a; margin-bottom: 0; }
        div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 12px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _status_html(status: str) -> str:
    css = {
        STATUS_NOT_SCORED: "hm-not-scored",
        STATUS_RECORDED: "hm-recorded",
        STATUS_RECEIPT_READY: "hm-receipt",
    }.get(status, "hm-not-scored")
    return f'<span class="hm-status {css}">{html.escape(status)}</span>'


def _render_model_status(st: Any, view: DashboardView) -> None:
    status = view.model_status
    st.subheader("Frozen model")
    st.markdown(status.model_family)
    st.caption(status.prospective_status)
    left, right = st.columns(2)
    with left:
        st.markdown(
            f"**Candidate**  \n{status.feature_family}  \n"
            f"{status.feature_column_count}-column Cycle-2 freeze"
        )
        st.markdown(f"**Artifact**  \n`{status.artifact_version}`")
        st.caption("Full artifact SHA-256")
        st.code(status.artifact_sha256, language=None)
    with right:
        st.markdown(
            "**Historical forward-chaining macros**  \n"
            f"Accuracy {status.historical_accuracy}  \n"
            f"ROC-AUC {status.historical_roc_auc}  \n"
            f"Log loss {status.historical_log_loss}  \n"
            f"Brier {status.historical_brier}  \n"
            f"ECE {status.historical_ece}"
        )
        st.info(status.historical_note)
    st.caption(
        f"Snapshot observed {view.snapshot_observed_at_utc}. "
        f"Date rule: {view.selected_date_rule}. {MARKET_LAYER_ABSENT}"
    )


def _render_slate(st: Any, view: DashboardView) -> None:
    st.subheader(f"Slate · {view.selected_date.isoformat()}")
    st.caption(view.disclaimer)
    if not view.store_present:
        st.info(
            "No local prediction-record file was found. Games stay NOT SCORED "
            "until an immutable record exists. No default probability is shown."
        )
    if view.empty_slate_message:
        st.info(view.empty_slate_message)
        return
    for card in view.games:
        _render_card(st, card)


def _render_card(st: Any, card: GameCard) -> None:
    with st.container(border=True):
        header, badge = st.columns([4, 1])
        header.markdown(f"### {html.escape(card.away_team)} at {html.escape(card.home_team)}")
        badge.markdown(_status_html(card.prediction_status), unsafe_allow_html=True)
        st.caption(f"{card.scheduled_local_label}  ·  {card.scheduled_utc_label}")
        st.caption(
            "ET is America/Toronto for reading the clock. "
            "The snapshot does not include venue-local time."
        )
        st.markdown(
            f"NHL game ID `{card.nhl_game_id}`  ·  "
            f"state `{card.game_state}`  ·  schedule `{card.game_schedule_state}`  ·  "
            f"{card.game_type}"
        )
        if card.prediction_status == STATUS_NOT_SCORED:
            st.info("NOT SCORED. Model probability is not shown for this game.")
            return
        home_col, away_col = st.columns(2)
        home_col.markdown(
            f'<p class="hm-team">{html.escape(card.home_team)} home</p>',
            unsafe_allow_html=True,
        )
        home_col.markdown(
            f'<p class="hm-prob">{_percent(card.home_model_win_probability)}</p>',
            unsafe_allow_html=True,
        )
        home_col.caption("Model home win probability")
        away_col.markdown(
            f'<p class="hm-team">{html.escape(card.away_team)} away</p>',
            unsafe_allow_html=True,
        )
        away_col.markdown(
            f'<p class="hm-prob">{_percent(card.away_model_win_probability)}</p>',
            unsafe_allow_html=True,
        )
        away_col.caption("Model away win probability = 1 − home")
        if card.home_model_win_probability is not None:
            st.progress(card.home_model_win_probability, text="Share of model home-win probability")
        st.markdown(card.descriptive_model_output or "")
        st.caption("Descriptive model output only. Not a betting recommendation and not a guaranteed result.")
        _render_audit(st, card)


def _render_audit(st: Any, card: GameCard) -> None:
    audit = card.audit
    if audit is None:
        return
    with st.expander("Audit and provenance"):
        if card.receipt_blocker:
            st.warning(card.receipt_blocker)
        rows = (
            ("Frozen artifact SHA-256", audit.frozen_artifact_sha256, audit.frozen_artifact_sha256_short),
            ("Prediction timestamp UTC", audit.prediction_timestamp_utc, None),
            (
                "Source SHA-256",
                audit.source_sha256 or "not present",
                audit.source_sha256_short,
            ),
            (
                "Feature-vector SHA-256",
                audit.feature_vector_sha256,
                audit.feature_vector_sha256_short,
            ),
            (
                "Prediction-record SHA-256",
                audit.prediction_record_sha256,
                audit.prediction_record_sha256_short,
            ),
            (
                "Receipt SHA-256",
                audit.receipt_sha256 or "not present",
                audit.receipt_sha256_short,
            ),
            ("Probability kind", audit.probability_kind, None),
            ("Scheduled puck drop", audit.scheduled_puck_drop_utc, None),
            ("Pregame timing valid", str(audit.pregame_timing_valid), None),
        )
        for label, full, short in rows:
            st.markdown(f"**{label}**")
            if short:
                st.caption(f"Card short form {short}")
            st.code(str(full), language=None)


def _render_record_log(st: Any, view: DashboardView) -> None:
    st.subheader("Protected prediction records")
    st.caption(
        "Immutable local records, newest prediction timestamp first. "
        + MARKET_LAYER_ABSENT
        + " Outcomes are not read into model selection."
    )
    if not view.records:
        st.info("No local immutable prediction records.")
        return
    for row in view.records:
        with st.container(border=True):
            st.markdown(f"**{row.game_label}**  ·  `{row.game_id}`")
            st.markdown(_status_html(row.prediction_status), unsafe_allow_html=True)
            st.caption(f"Prediction timestamp {row.prediction_timestamp_utc}")
            left, right = st.columns(2)
            left.markdown(f"Home model probability **{_percent(row.home_model_win_probability)}**")
            right.markdown(f"Away model probability **{_percent(row.away_model_win_probability)}**")
            st.caption(f"Probability kind `{row.probability_kind}`")
            st.caption(f"Record SHA-256 short `{row.to_dict()['record_sha256_short']}`")
            st.code(row.record_sha256, language=None)
            if row.receipt_sha256:
                st.caption(f"Receipt SHA-256 short `{row.to_dict()['receipt_sha256_short']}`")
                st.code(row.receipt_sha256, language=None)
            elif row.receipt_blocker:
                st.warning(row.receipt_blocker)


def _percent(probability: float | None) -> str:
    if probability is None:
        return "not shown"
    return f"{probability * 100:.1f}%"


if __name__ == "__main__":
    run()
