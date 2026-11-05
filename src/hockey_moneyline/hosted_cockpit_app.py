"""Private phone-first Morning Board for Issue #111.

The Streamlit entrypoint stays this module so the existing private cockpit
opens onto one morning surface. Rendering reads the persisted board only.
Prepare Today and Force live odds refresh are the only provider actions.
"""

from __future__ import annotations

import html
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from hockey_moneyline.decision_cockpit import format_decimal, format_ev, format_percent
from hockey_moneyline.hosted_cockpit import DEFAULT_HOSTED_DISPLAY_TIMEZONE
from hockey_moneyline.morning_board import (
    DECISION_NO_DATA,
    DEFAULT_ATTEMPT_PATH,
    DEFAULT_BOARD_PATH,
    DEFAULT_ODDS_STORE_PATH,
    HYDRATION_CURRENT,
    HYDRATION_EMPTY,
    HYDRATION_STALE,
    MARKET_CACHED,
    MorningBoardGame,
    MorningBoardView,
    MorningShortlistEntry,
    force_live_odds_refresh,
    load_morning_board,
    prepare_today,
)
from hockey_moneyline.prediction_record_store import DEFAULT_STORE_RELATIVE_PATH

_PAGE_TITLE = "Hockey Morning Board"


def run() -> None:
    try:
        import streamlit as st
    except ImportError as exc:
        raise SystemExit(
            'Streamlit is not installed. Install with python -m pip install -e ".[dashboard]".'
        ) from exc
    _render(st)


def _render(st: Any) -> None:
    st.set_page_config(page_title=_PAGE_TITLE, page_icon="🏒", layout="centered")
    _inject_styles(st)
    zone = ZoneInfo(DEFAULT_HOSTED_DISPLAY_TIMEZONE)
    today = datetime.now(zone).date()

    st.markdown('<div class="hm-kicker">PRIVATE MORNING BOARD</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="hm-title">Today\'s NHL decisions</h1>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hm-subtitle">One manual prepare. Frozen model, PlayNow prices, '
        "edge and EV. No wager button and no canonical bet threshold yet.</div>",
        unsafe_allow_html=True,
    )

    if st.button("Prepare Today", type="primary", use_container_width=True):
        prepare_today(
            slate_date=today,
            observed_at_utc=datetime.now(timezone.utc),
            board_path=DEFAULT_BOARD_PATH,
            attempt_path=DEFAULT_ATTEMPT_PATH,
            odds_store_path=DEFAULT_ODDS_STORE_PATH,
            protected_store_path=DEFAULT_STORE_RELATIVE_PATH,
        )

    st.caption("Secondary. One PlayNow request, including inside the 15-minute window.")
    if st.button("Force live odds refresh", use_container_width=True):
        force_live_odds_refresh(
            slate_date=today,
            observed_at_utc=datetime.now(timezone.utc),
            board_path=DEFAULT_BOARD_PATH,
            attempt_path=DEFAULT_ATTEMPT_PATH,
            odds_store_path=DEFAULT_ODDS_STORE_PATH,
            protected_store_path=DEFAULT_STORE_RELATIVE_PATH,
        )

    view = load_morning_board(
        slate_date=today,
        observed_at_utc=datetime.now(timezone.utc),
        board_path=DEFAULT_BOARD_PATH,
        attempt_path=DEFAULT_ATTEMPT_PATH,
        protected_store_path=DEFAULT_STORE_RELATIVE_PATH,
    )
    _render_status(st, view)

    if view.hydration_status == HYDRATION_EMPTY and not view.blocker_message:
        st.markdown(
            f"""
            <div class="hm-empty">
              <strong>{html.escape(today.isoformat())}</strong><br>
              Tap <strong>Prepare Today</strong>. Page load does not call PlayNow,
              the NHL, or MoneyPuck.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    show_recommended = st.toggle("Show Recommended", value=True)
    _render_shortlist(st, view, show_recommended=show_recommended)
    _render_ranked(st, view)
    st.caption(
        "Last successful board is a private local snapshot under data/derived/morning_board/. "
        "Odds stay in the Issue #87 archive at that path. Streamlit Cloud's filesystem "
        "can reset on a new runtime; this is not protected evidence and not a paid store."
    )
    st.markdown(
        '<div class="hm-footer">Research shortlist only · canonical threshold blocked · '
        "no staking · no wager · protected records stay read-only.</div>",
        unsafe_allow_html=True,
    )


def _render_status(st: Any, view: MorningBoardView) -> None:
    tone = "ok" if view.hydration_status == HYDRATION_CURRENT else "blocked"
    st.markdown(
        f'<div class="hm-banner hm-{tone}">{html.escape(view.status_banner)}</div>',
        unsafe_allow_html=True,
    )
    if view.blocker_message:
        st.error(view.blocker_message)
    if view.hydration_status == HYDRATION_STALE:
        st.warning(
            "This is yesterday's or another slate's last successful board. "
            "It is not today's board until Prepare Today succeeds."
        )
    rows = (
        ("Slate date", _slate_label(view)),
        ("Source refresh", _source_label(view)),
        ("PlayNow", _playnow_label(view)),
        ("FUT games", str(view.fut_count)),
        ("Valid PlayNow markets", str(view.priced_count)),
        ("Frozen model", view.model_status),
        ("Canonical recommendation", view.canonical_policy_label),
    )
    st.markdown('<div class="hm-status-grid">', unsafe_allow_html=True)
    for label, value in rows:
        st.markdown(
            f'<div class="hm-status-card"><span>{html.escape(label)}</span>'
            f"<strong>{html.escape(value)}</strong></div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)
    st.caption(view.protected_evidence_label)
    if view.protected_read_error:
        st.warning(view.protected_read_error)


def _render_shortlist(st: Any, view: MorningBoardView, *, show_recommended: bool) -> None:
    st.markdown("### Show Recommended")
    st.markdown(
        f'<div class="hm-rule">{html.escape(view.shortlist_heading)} · '
        f"{html.escape(view.shortlist_rule_label)}</div>",
        unsafe_allow_html=True,
    )
    if not show_recommended:
        st.info("Recommended shortlist is hidden. The full research ranking is below.")
        return
    if not view.shortlist:
        st.markdown(
            f'<div class="hm-empty">{html.escape(view.shortlist_empty_label)}</div>',
            unsafe_allow_html=True,
        )
        return
    for entry in view.shortlist:
        _render_shortlist_card(st, entry)


def _render_shortlist_card(st: Any, entry: MorningShortlistEntry) -> None:
    with st.container(border=True):
        st.markdown(
            f"""
            <div class="hm-game-head">
              <div>
                <div class="hm-matchup">{html.escape(entry.matchup)}</div>
                <div class="hm-time">{html.escape(entry.side)} · {html.escape(entry.team)}</div>
              </div>
              <div class="hm-research-badge">10 PP · UNVALIDATED</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="hm-mini-grid">
              <div><span>PlayNow decimal</span><strong>{format_decimal(entry.decimal_odds)}</strong></div>
              <div><span>Model</span><strong>{format_percent(entry.model_probability)}</strong></div>
              <div><span>Fair market</span><strong>{format_percent(entry.fair_market_probability)}</strong></div>
              <div><span>Edge</span><strong>{format_percent(entry.edge, signed=True)}</strong></div>
              <div><span>EV / 1u</span><strong>{format_ev(entry.ev_per_unit)}</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_ranked(st: Any, view: MorningBoardView) -> None:
    st.markdown("### Ranked slate")
    st.caption(view.research_ranking_label)
    if not view.games:
        st.info("No slate games are on this board.")
        return
    for row in view.games:
        _render_game(st, row)


def _render_game(st: Any, row: MorningBoardGame) -> None:
    game = row.game
    badge_class = "hm-nodata-badge" if row.decision_label == DECISION_NO_DATA else "hm-blocked-badge"
    with st.container(border=True):
        st.markdown(
            f"""
            <div class="hm-game-head">
              <div>
                <div class="hm-kicker">RANK {row.rank_index}</div>
                <div class="hm-matchup">{html.escape(game.away_team)}
                <span>@</span> {html.escape(game.home_team)}</div>
                <div class="hm-time">{html.escape(game.scheduled_local_label)} ·
                {html.escape(game.scheduled_utc_label)}</div>
              </div>
            </div>
            <div class="{badge_class}">{html.escape(row.decision_label)}</div>
            """,
            unsafe_allow_html=True,
        )
        if row.protected_evidence:
            st.caption("Protected prediction record on file · read only · not created here")
        away_col, home_col = st.columns(2)
        _render_side(away_col, game.away)
        _render_side(home_col, game.home)
        if game.is_priced:
            st.caption(
                f"Market {game.book_source or 'PlayNow'} · observed {game.odds_observed_at_utc}"
            )
        else:
            st.caption(game.recommendation_rationale or "NO DATA")
        with st.expander("Audit and provenance"):
            st.markdown(f"**NHL game ID** `{game.nhl_game_id}`")
            st.markdown(f"**Calculation** `{html.escape(game.protected_prediction_status)}`")
            if game.source_provenance:
                st.code(
                    "\n".join(
                        f"{key}: {value}"
                        for key, value in sorted(game.source_provenance.items())
                    ),
                    language=None,
                )
            if row.audit:
                for label, key in (
                    ("Observed", "observed_at_utc"),
                    ("Frozen artifact SHA-256", "model_artifact_sha256"),
                    ("Feature vector SHA-256", "feature_vector_sha256"),
                    ("Current-season prior SHA-256", "current_season_prior_source_sha256"),
                    ("MoneyPuck SHA-256", "moneypuck_source_sha256"),
                    ("Not scored", "reason"),
                ):
                    value = row.audit.get(key)
                    if value:
                        st.code(f"{label}: {value}", language=None)


def _render_side(column: Any, side: Any) -> None:
    column.markdown(f"**{html.escape(side.team)}**")
    column.markdown(
        f"""
        <div class="hm-probability">
          <span>Model win probability</span>
          <strong>{format_percent(side.model_probability)}</strong>
        </div>
        <div class="hm-mini-grid">
          <div><span>PlayNow decimal</span><strong>{format_decimal(side.decimal_odds)}</strong></div>
          <div><span>Fair market</span><strong>{format_percent(side.fair_market_probability)}</strong></div>
          <div><span>Edge</span><strong>{format_percent(side.edge, signed=True)}</strong></div>
          <div><span>EV / 1u</span><strong>{format_ev(side.ev_per_unit)}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _slate_label(view: MorningBoardView) -> str:
    if view.snapshot_slate_date is None:
        return view.requested_slate_date.isoformat()
    if view.hydration_status == HYDRATION_STALE:
        return (
            f"{view.snapshot_slate_date.isoformat()} snapshot · "
            f"today is {view.requested_slate_date.isoformat()}"
        )
    return view.snapshot_slate_date.isoformat()


def _source_label(view: MorningBoardView) -> str:
    if view.source_observed_at_utc is None:
        return "Not refreshed"
    local = view.source_observed_local or ""
    return f"{view.source_observed_at_utc} · {local}".strip(" ·")


def _playnow_label(view: MorningBoardView) -> str:
    if view.market_mode is None and view.playnow_observed_at_utc is None:
        return "No PlayNow observation"
    mode = view.market_mode or "UNAVAILABLE"
    observed = view.playnow_observed_at_utc or "observation unavailable"
    local = view.playnow_observed_local or ""
    age = view.playnow_age_label or "age unavailable"
    prefix = "CACHED" if mode == MARKET_CACHED else mode
    return f"{prefix} · age {age} · {observed} · {local}".strip(" ·")


def _inject_styles(st: Any) -> None:
    st.markdown(
        """
        <style>
        .block-container {max-width:760px; padding-top:1.1rem; padding-bottom:3rem;}
        .hm-kicker {font-size:.68rem; letter-spacing:.14em; color:#66768a; font-weight:800;}
        .hm-title {font-size:2rem; line-height:1.04; margin:.18rem 0 .35rem; color:#10233d;}
        .hm-subtitle {font-size:.92rem; color:#657286; margin-bottom:1rem;}
        .hm-banner {border-radius:12px; padding:.7rem .8rem; font-weight:750; margin:.8rem 0 .55rem;}
        .hm-banner.hm-ok {background:#e5f5ea; color:#24543c; border:1px solid #b9dec5;}
        .hm-banner.hm-blocked {background:#fff6f5; color:#8a4a45; border:1px solid #efcfcb;}
        .hm-status-grid {display:grid; grid-template-columns:1fr; gap:.45rem; margin:.2rem 0 .7rem;}
        .hm-status-card {border:1px solid #dfe5ec; border-radius:12px; padding:.62rem .68rem; background:#f8fafc;}
        .hm-status-card span {display:block; font-size:.65rem; text-transform:uppercase; letter-spacing:.08em; color:#758396;}
        .hm-status-card strong {display:block; font-size:.84rem; line-height:1.35; color:#233b58; margin-top:.16rem; overflow-wrap:anywhere;}
        .hm-rule {font-size:.78rem; font-weight:800; color:#7a5a18; background:#fffaf0; border:1px solid #ead8ad; border-radius:10px; padding:.55rem .7rem; margin:.2rem 0 .7rem;}
        .hm-game-head {display:flex; justify-content:space-between; gap:.7rem; align-items:flex-start; margin-bottom:.7rem;}
        .hm-matchup {font-size:1.2rem; font-weight:800; color:#112844; overflow-wrap:anywhere;}
        .hm-matchup span {color:#8996a7; font-weight:500;}
        .hm-time {font-size:.78rem; color:#768498; margin-top:.18rem;}
        .hm-research-badge,.hm-blocked-badge,.hm-nodata-badge {font-size:.68rem; letter-spacing:.04em; font-weight:850; border-radius:999px; padding:.38rem .55rem; display:inline-block; margin-bottom:.7rem;}
        .hm-research-badge {color:#7a5a18; background:#fffaf0; border:1px solid #ead8ad;}
        .hm-blocked-badge {color:#8a4a45; background:#fff6f5; border:1px solid #efcfcb;}
        .hm-nodata-badge {color:#5c4a12; background:#fff4d6; border:1px solid #e3c56a;}
        .hm-probability {background:#eef4fb; border:1px solid #d3e0ef; border-radius:12px; padding:.68rem; margin:.25rem 0 .45rem;}
        .hm-probability span {display:block; font-size:.66rem; color:#6c7b8f;}
        .hm-probability strong {display:block; font-size:1.35rem; color:#15395f; margin-top:.08rem;}
        .hm-mini-grid {display:grid; grid-template-columns:1fr 1fr; gap:.35rem;}
        .hm-mini-grid div {background:#f8fafc; border:1px solid #e2e8ef; border-radius:9px; padding:.48rem;}
        .hm-mini-grid span {display:block; font-size:.62rem; color:#7b8797;}
        .hm-mini-grid strong {display:block; font-size:.86rem; color:#33475f; overflow-wrap:anywhere;}
        .hm-empty {border:1px dashed #cfd8e3; background:#f8fafc; border-radius:14px; padding:1rem; color:#536479; margin:.4rem 0 .8rem;}
        .hm-footer {font-size:.72rem; color:#7b8797; text-align:center; margin-top:1.2rem;}
        @media (max-width:560px) {
          .block-container {padding:1rem .72rem 2.6rem;}
          .hm-title {font-size:1.7rem;}
          .hm-matchup {font-size:1.08rem;}
          .hm-probability strong {font-size:1.2rem;}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    run()
