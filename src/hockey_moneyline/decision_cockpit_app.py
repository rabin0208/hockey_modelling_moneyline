"""Streamlit daily decision cockpit for Issue #85.

Launch with explicit local sources. Market fixture input is optional and is
only for synthetic/review use until Issue #87 publishes its reviewed reader.
No network fetch, scoring, store write, or wager action occurs here.
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
    build_dashboard_view,
    load_nhl_snapshot,
)
from hockey_moneyline.decision_cockpit import (
    CANONICAL_POLICY_LABEL,
    DEFAULT_FORWARD_ODDS_STORE_RELATIVE_PATH,
    FLAT_RESEARCH_BASELINE_UNITS,
    TEMPORARY_RESEARCH_POLICY_LABEL,
    CockpitGame,
    DecisionCockpitView,
    FixtureMarketSnapshotReader,
    ForwardOddsMarketSnapshotReader,
    available_forward_odds_books,
    build_decision_cockpit_view,
    format_decimal,
    format_ev,
    format_percent,
)
from hockey_moneyline.errors import HockeyMoneylineError
from hockey_moneyline.prediction_record_store import DEFAULT_STORE_RELATIVE_PATH

_PAGE_TITLE = "Daily Hockey Decision Cockpit"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--snapshot", default=DEFAULT_SNAPSHOT_RELATIVE_PATH)
    parser.add_argument("--store", default=DEFAULT_STORE_RELATIVE_PATH)
    parser.add_argument(
        "--odds-store",
        default=DEFAULT_FORWARD_ODDS_STORE_RELATIVE_PATH,
    )
    parser.add_argument("--market-book", default="")
    parser.add_argument("--market-fixture", default="")
    parser.add_argument("--date", default="")
    parser.add_argument("--timezone", default=DEFAULT_DISPLAY_TIMEZONE)
    known, _ = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    return known


def _parse_cli_date(raw: str) -> date | None:
    return None if not raw.strip() else date.fromisoformat(raw.strip())


def run() -> None:
    try:
        import streamlit as st
    except ImportError as exc:
        raise SystemExit(
            'Streamlit is not installed. Install with python -m pip install -e ".[dashboard]".'
        ) from exc
    _render(st)


def _render(st: Any) -> None:
    st.set_page_config(page_title=_PAGE_TITLE, page_icon="🏒", layout="wide")
    _inject_styles(st)
    args = _parse_args()

    st.markdown(
        '<div class="hm-kicker">Hockey Moneyline · Research cockpit</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<h1 class="hm-title">{_PAGE_TITLE}</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hm-subtitle">Protected model × current two-sided market. '
        'Fast scan first; audit details stay one tap away.</div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Read-only inputs")
        snapshot_path = st.text_input("NHL snapshot", value=args.snapshot)
        store_path = st.text_input("Protected prediction store", value=args.store)
        odds_store_path = st.text_input(
            "Validated forward odds archive",
            value=args.odds_store,
        )
        market_fixture = st.text_input(
            "Synthetic market fixture override (optional)",
            value=args.market_fixture,
        )
        st.caption(
            "Normal mode reads Issue #87's validated local archive. "
            "Fixture override is review/test input only. No sportsbook is contacted."
        )

    try:
        snapshot = load_nhl_snapshot(snapshot_path)
    except HockeyMoneylineError as exc:
        st.error(str(exc))
        st.stop()

    covered = sorted(snapshot.covered_dates)
    requested = None
    if args.date.strip():
        try:
            requested = _parse_cli_date(args.date)
        except ValueError:
            st.error(f"Slate date is malformed: {args.date}")
            st.stop()
        if requested not in covered:
            st.error(
                "Requested date is outside the loaded NHL snapshot. "
                "Refusing to invent a slate."
            )
            st.stop()

    selected = st.date_input(
        "Slate date",
        value=_initial_date(covered, requested),
        min_value=covered[0],
        max_value=covered[-1],
        label_visibility="collapsed",
    )

    try:
        dashboard = build_dashboard_view(
            snapshot_path=snapshot_path,
            store_path=store_path,
            selected_date=selected,
            display_timezone=args.timezone,
        )
        if market_fixture.strip():
            reader = FixtureMarketSnapshotReader.from_path(market_fixture)
            with st.sidebar:
                st.info("Synthetic fixture override is active.")
        else:
            game_ids = tuple(str(game.nhl_game_id) for game in dashboard.games)
            books = list(
                available_forward_odds_books(
                    store_path=odds_store_path,
                    game_ids=game_ids,
                )
            )
            requested_book = args.market_book.strip()
            if requested_book and requested_book not in books:
                books.append(requested_book)
                books.sort()
            auto_label = "Auto — only if one source exists"
            options = [auto_label, *books]
            initial = (
                options.index(requested_book)
                if requested_book and requested_book in options
                else 0
            )
            with st.sidebar:
                selected_book = st.selectbox(
                    "Sportsbook / source",
                    options=options,
                    index=initial,
                    help=(
                        "The cockpit never chooses among multiple books automatically. "
                        "Select a source when more than one is archived."
                    ),
                )
            reader = ForwardOddsMarketSnapshotReader(
                odds_store_path,
                book_source=None if selected_book == auto_label else selected_book,
            )
        view = build_decision_cockpit_view(
            dashboard_view=dashboard,
            market_reader=reader,
        )
    except HockeyMoneylineError as exc:
        st.error(str(exc))
        st.stop()

    _render_policy_banner(st, view)
    _render_summary(st, view)
    _render_slate(st, view)
    st.markdown(
        '<div class="hm-footer">Read-only research surface · no Kelly · '
        'no automated wager placement.</div>',
        unsafe_allow_html=True,
    )


def _initial_date(covered: list[date], requested: date | None) -> date:
    if requested in covered:
        return requested  # type: ignore[return-value]
    today = datetime.now().date()
    return today if today in covered else covered[-1]


def _render_policy_banner(st: Any, view: DecisionCockpitView) -> None:
    st.markdown(
        f"""
        <div class="hm-policy">
          <div class="hm-policy-lock">CANONICAL POLICY</div>
          <div><strong>{html.escape(CANONICAL_POLICY_LABEL)}</strong><br>
          <span>Long-term threshold remains unresolved under Issue #78.</span></div>
        </div>
        <div class="hm-research-note">
          <strong>{html.escape(TEMPORARY_RESEARCH_POLICY_LABEL)}</strong><br>
          <span>BET/PASS below is a manual research reference only. BET uses a flat
          {FLAT_RESEARCH_BASELINE_UNITS:.0f}-unit baseline; no stake is written to
          the recommendation contract.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"Market input: {view.market_input_label}")


def _render_summary(st: Any, view: DecisionCockpitView) -> None:
    cols = st.columns(4)
    metrics = (
        ("Games", len(view.games)),
        ("Priced", view.priced_count),
        ("Research BET", view.research_bet_count),
        ("No data", view.no_data_count),
    )
    for column, (label, value) in zip(cols, metrics, strict=True):
        column.metric(label, value)


def _render_slate(st: Any, view: DecisionCockpitView) -> None:
    st.markdown(f"### {view.selected_date.isoformat()} slate")
    if not view.games:
        st.info("No games on this date in the loaded NHL snapshot.")
        return
    for game in view.games:
        _render_game(st, game)


def _render_game(st: Any, game: CockpitGame) -> None:
    tone = game.decision_state.casefold().replace("_", "-")
    with st.container(border=True):
        st.markdown(
            f"""
            <div class="hm-card-head">
              <div>
                <div class="hm-matchup">{html.escape(game.away_team)}
                <span>@</span> {html.escape(game.home_team)}</div>
                <div class="hm-time">{html.escape(game.scheduled_local_label)}
                · {html.escape(game.scheduled_utc_label)}</div>
              </div>
              <div class="hm-state hm-{tone}">{html.escape(game.decision_state)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if game.decision_state == "NO_DATA":
            st.markdown(
                '<div class="hm-no-data-box"><strong>NO DATA</strong> · '
                f"{html.escape(game.recommendation_rationale)}</div>",
                unsafe_allow_html=True,
            )
            _render_status_row(st, game)
            _render_details(st, game)
            return

        if game.decision_state == "BET" and game.decision_team:
            st.markdown(
                f"""
                <div class="hm-callout">
                  <div class="hm-callout-label">TEMPORARY RESEARCH REFERENCE</div>
                  <div class="hm-callout-main">{html.escape(game.decision_team)} · BET</div>
                  <div class="hm-callout-sub">Flat
                  {game.research_baseline_units:.0f}-unit research baseline ·
                  not historically validated</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="hm-callout hm-callout-pass">'
                '<div class="hm-callout-label">TEMPORARY RESEARCH REFERENCE</div>'
                '<div class="hm-callout-main">PASS</div>'
                '<div class="hm-callout-sub">No side is strictly above the '
                'edge &gt; 0 screen.</div></div>',
                unsafe_allow_html=True,
            )

        away_col, home_col = st.columns(2)
        _render_side(
            away_col,
            game.away,
            is_decision=game.decision_side == "away",
        )
        _render_side(
            home_col,
            game.home,
            is_decision=game.decision_side == "home",
        )
        _render_status_row(st, game)
        _render_details(st, game)


def _render_side(column: Any, side: Any, *, is_decision: bool) -> None:
    marker = " · RESEARCH EDGE" if is_decision else ""
    column.markdown(f"**{html.escape(side.team)}**{marker}")
    column.markdown(
        f"""
        <div class="hm-side-grid">
          <div><span>Model</span><strong>{format_percent(side.model_probability)}</strong></div>
          <div><span>Odds</span><strong>{format_decimal(side.decimal_odds)}</strong></div>
          <div><span>Market fair</span><strong>{format_percent(side.fair_market_probability)}</strong></div>
          <div><span>Edge</span><strong>{format_percent(side.edge, signed=True)}</strong></div>
          <div><span>EV / 1u</span><strong>{format_ev(side.ev_per_unit)}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_status_row(st: Any, game: CockpitGame) -> None:
    receipt = "receipt ready" if game.receipt_ready else "receipt not ready"
    source = game.book_source or "no market source"
    observed = game.odds_observed_at_utc or "no observation timestamp"
    st.markdown(
        '<div class="hm-status-row">'
        f"<span>Protected: <strong>{html.escape(game.protected_prediction_status)}</strong>"
        f" · {receipt}</span>"
        f"<span>Market: <strong>{html.escape(source)}</strong>"
        f" · {html.escape(observed)}</span>"
        "</div>",
        unsafe_allow_html=True,
    )


def _render_details(st: Any, game: CockpitGame) -> None:
    with st.expander("Audit & provenance"):
        st.markdown(f"**NHL game ID** {game.nhl_game_id}")
        st.markdown(f"**Canonical policy** {game.canonical_policy_version}")
        st.markdown(f"**Temporary reference** {game.temporary_policy_version}")
        if game.no_data_reason:
            st.markdown(f"**Fail-closed reason** {game.no_data_reason}")
        if game.overround is not None:
            st.markdown(
                f"**Market overround** {format_percent(game.overround, signed=True)}"
            )
        if game.source_provenance:
            st.markdown("**Market provenance**")
            for key, value in sorted(game.source_provenance.items()):
                st.code(f"{key}: {value}", language=None)
        audit = game.audit
        if audit is not None:
            st.markdown("**Protected prediction audit**")
            for key in (
                "prediction_timestamp_utc",
                "frozen_artifact_sha256",
                "source_sha256",
                "feature_vector_sha256",
                "prediction_record_sha256",
                "receipt_sha256",
            ):
                value = getattr(audit, key, None)
                if value:
                    st.code(f"{key}: {value}", language=None)


def _inject_styles(st: Any) -> None:
    st.markdown(
        """
        <style>
        .block-container {max-width:1180px; padding-top:2rem; padding-bottom:4rem;}
        .hm-kicker {font-size:.72rem; letter-spacing:.13em; text-transform:uppercase;
          color:#607086; font-weight:700;}
        .hm-title {font-size:2.45rem; line-height:1.04; margin:.18rem 0 .4rem;
          color:#10233d; letter-spacing:-.035em;}
        .hm-subtitle {color:#5b6878; font-size:1rem; margin-bottom:1.4rem;}
        .hm-policy {display:flex; gap:1rem; align-items:flex-start; padding:1rem 1.1rem;
          border:1px solid #d7dee8; border-radius:16px; background:#f7f9fc;
          margin:.5rem 0 .55rem;}
        .hm-policy-lock {font-size:.68rem; letter-spacing:.09em; color:#fff;
          background:#1b3556; border-radius:999px; padding:.28rem .55rem;
          white-space:nowrap; font-weight:800;}
        .hm-policy span,.hm-research-note span {color:#657286; font-size:.88rem;}
        .hm-research-note {padding:.82rem 1.05rem; border-left:4px solid #d18b24;
          background:#fff8ec; border-radius:10px; margin-bottom:.45rem; color:#5d431d;}
        .hm-card-head {display:flex; justify-content:space-between; gap:1rem;
          align-items:center; margin-bottom:.85rem;}
        .hm-matchup {font-size:1.42rem; font-weight:780; letter-spacing:-.025em;
          color:#112844;}
        .hm-matchup span {color:#8895a5; font-weight:500; padding:0 .25rem;}
        .hm-time {font-size:.82rem; color:#778497; margin-top:.2rem;}
        .hm-state {font-size:.78rem; letter-spacing:.08em; font-weight:850;
          border-radius:999px; padding:.38rem .66rem; min-width:74px; text-align:center;}
        .hm-bet {background:#dff5e8; color:#17643b; border:1px solid #a9dec0;}
        .hm-pass {background:#edf1f6; color:#46566a; border:1px solid #d3dbe5;}
        .hm-no-data {background:#fff1f0; color:#8b2e2a; border:1px solid #f0c7c4;}
        .hm-no-data-box {background:#fff7f6; color:#8b2e2a; border:1px solid #f0c7c4;
          border-radius:12px; padding:.8rem .9rem; margin:.3rem 0 .8rem;}
        .hm-callout {border-radius:14px; padding:.9rem 1rem; background:#eef8f2;
          border:1px solid #c4e5d0; margin:.3rem 0 1rem;}
        .hm-callout-pass {background:#f6f8fb; border-color:#dbe2eb;}
        .hm-callout-label {font-size:.66rem; letter-spacing:.1em; font-weight:800;
          color:#6f7c8e;}
        .hm-callout-main {font-size:1.55rem; line-height:1.1; font-weight:820;
          color:#173f2d; margin:.18rem 0;}
        .hm-callout-pass .hm-callout-main {color:#36465a;}
        .hm-callout-sub {font-size:.82rem; color:#6c7888;}
        .hm-side-grid {display:grid; grid-template-columns:repeat(5,minmax(0,1fr));
          gap:.45rem; margin:.35rem 0 .75rem;}
        .hm-side-grid div {background:#f7f9fc; border:1px solid #e3e8ef;
          border-radius:11px; padding:.56rem .6rem; min-width:0;}
        .hm-side-grid span {display:block; color:#748194; font-size:.67rem;
          margin-bottom:.1rem;}
        .hm-side-grid strong {display:block; color:#172f4d; font-size:.92rem;
          white-space:nowrap;}
        .hm-status-row {display:flex; justify-content:space-between; gap:1rem;
          color:#6b788a; font-size:.75rem; padding:.55rem 0 .1rem;
          border-top:1px solid #edf0f4;}
        .hm-footer {font-size:.78rem; color:#7a8798; margin-top:1.3rem;
          text-align:center;}
        @media (max-width:720px) {
          .block-container {padding:1rem .78rem 3rem;}
          .hm-title {font-size:1.95rem;}
          .hm-policy {display:block;}
          .hm-policy-lock {display:inline-block; margin-bottom:.55rem;}
          .hm-card-head {align-items:flex-start;}
          .hm-matchup {font-size:1.18rem;}
          .hm-side-grid {grid-template-columns:repeat(2,minmax(0,1fr));}
          .hm-status-row {display:block;}
          .hm-status-row span {display:block; margin:.2rem 0;}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    run()
