"""Streamlit dashboard for NHL moneyline probabilities and historical results."""
from __future__ import annotations

import sys
import unicodedata
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import altair as alt
import pandas as pd
import requests
import streamlit as st
from nhlpy import NHLClient

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
MODEL_PATH = PROJECT_ROOT / "models" / "logistic_regression_optimized.pkl"
RESULTS_PATH = PROJECT_ROOT / "data" / "historical_accuracy_roi.csv"
BETS_PATH = PROJECT_ROOT / "data" / "historical_bet_games.csv"
LOCAL_TZ = ZoneInfo("America/Los_Angeles")
MAX_STAKE = 0.05
SLATE_CAP = 0.10

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from predict_games import (  # noqa: E402
    create_features_for_game,
    load_historical_games,
    load_model,
    parse_game_data,
)


st.set_page_config(
    page_title="Hockey Moneyline",
    page_icon="🏒",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
      --hm-bg: #071018;
      --hm-panel: #0c1824;
      --hm-line: #1d3348;
      --hm-text: #eef6ff;
      --hm-muted: #9eb0c3;
      --hm-accent: #7ad7ff;
      --hm-green: #3dd68c;
      --hm-orange: #ffb15a;
    }
    .stApp {
      background:
        radial-gradient(circle at 78% 0%, rgba(122, 215, 255, 0.14), transparent 24rem),
        linear-gradient(180deg, #102033 0, var(--hm-bg) 22rem);
      color: var(--hm-text);
    }
    [data-testid="stSidebar"] {
      background: rgba(12, 24, 36, 0.98);
      border-right: 1px solid var(--hm-line);
    }
    [data-testid="stMetric"] {
      background: rgba(12, 24, 36, 0.78);
      border: 1px solid var(--hm-line);
      border-radius: 0.75rem;
      padding: 0.85rem 1rem;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
      background: rgba(12, 24, 36, 0.76);
      border-color: var(--hm-line);
      border-radius: 0.8rem;
    }
    .hm-hero {
      margin: 0 0 1.5rem;
      padding: clamp(1.25rem, 3vw, 2.25rem);
      background:
        linear-gradient(90deg, rgba(122, 215, 255, 0.16), transparent 62%),
        linear-gradient(135deg, #0b3a5b 0%, #145a86 55%, #1a7aa8 100%);
      border: 1px solid rgba(122, 215, 255, 0.35);
      border-radius: 0.8rem;
    }
    .hm-hero h1 { margin: 0; color: white; font-size: clamp(2.2rem, 5vw, 4rem); line-height: 1; }
    .hm-hero p { margin: 0.65rem 0 0; color: rgba(255, 255, 255, 0.88); font-size: 1.05rem; }
    .hm-game-title { color: var(--hm-text); font-size: 1.2rem; font-weight: 750; }
    .hm-meta { color: var(--hm-muted); font-size: 0.83rem; }
    .hm-pick, .hm-stake {
      display: inline-block;
      margin: 0.15rem 0.5rem 0.8rem 0;
      padding: 0.28rem 0.6rem;
      border-radius: 0.3rem;
      font-size: 0.76rem;
      font-weight: 800;
      letter-spacing: 0.02em;
      text-transform: uppercase;
    }
    .hm-pick { color: #03140c; background: var(--hm-green); }
    .hm-stake { color: #1a1206; background: var(--hm-orange); }
    .hm-no-pick { color: var(--hm-muted); font-size: 0.8rem; }
    h1, h2, h3, label, [data-testid="stMetricLabel"] { color: var(--hm-text) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


def fold_name(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name))
    return "".join(ch for ch in text if not unicodedata.combining(ch)).casefold().strip()


def american_to_implied(odds: float) -> float:
    odds = float(odds)
    if odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    return 100.0 / (odds + 100.0)


def devig(home_ml: float, away_ml: float) -> tuple[float, float]:
    home = american_to_implied(home_ml)
    away = american_to_implied(away_ml)
    total = home + away
    return home / total, away / total


def format_american(odds: object) -> str:
    if pd.isna(odds):
        return "—"
    value = int(round(float(odds)))
    return f"{value:+d}"


def quarter_kelly(model_p: float, market_p: float, american_odds: float) -> float:
    """Quarter Kelly on a 50/50 blend of model and market probability, capped at 5%."""
    blend = 0.5 * float(model_p) + 0.5 * float(market_p)
    if american_odds < 0:
        net_odds = 100.0 / abs(float(american_odds))
    else:
        net_odds = float(american_odds) / 100.0
    if net_odds <= 0:
        return 0.0
    full = (blend * net_odds - (1.0 - blend)) / net_odds
    return max(0.0, min(MAX_STAKE, full / 4.0))


def cap_slate(stakes: pd.Series) -> pd.Series:
    total = float(stakes.fillna(0).sum())
    if total <= SLATE_CAP or total <= 0:
        return stakes
    return stakes * (SLATE_CAP / total)


@st.cache_resource(show_spinner=False)
def get_model():
    model, feature_columns = load_model(str(MODEL_PATH))
    return model, feature_columns


@st.cache_resource(show_spinner=False)
def get_history() -> pd.DataFrame:
    return load_historical_games()


def predict_one(model, feature_columns, history: pd.DataFrame, game: dict) -> dict | None:
    features = create_features_for_game(history, game, feature_columns)
    if features is None:
        return None
    home_prob = float(model.predict_proba(features)[0][1])
    return {
        "game_id": game.get("game_id"),
        "date": game["date"],
        "away_team": game["away_team_name"],
        "home_team": game["home_team_name"],
        "p_home": home_prob,
        "p_away": 1.0 - home_prob,
        "start_time": game.get("start_time") or "",
        "home_score": game.get("home_score"),
        "away_score": game.get("away_score"),
    }


def games_on_date(target: date) -> list[dict]:
    today = date.today()
    if target >= today:
        client = NHLClient(debug=False)
        schedule = client.schedule.daily_schedule(date=target.isoformat())
        raw_games = schedule.get("games", []) if isinstance(schedule, dict) else []
        parsed = []
        for game in raw_games:
            state = str(game.get("gameState") or "").upper()
            if state in {"OFF", "FINAL", "LIVE", "CRIT", "OVER"}:
                continue
            game["gameDate"] = target.isoformat()
            row = parse_game_data(game)
            if row:
                parsed.append(row)
        return parsed

    history = get_history()
    rows = history[(history["date"] == target) & history["home_wins"].notna()]
    parsed = []
    for row in rows.itertuples(index=False):
        parsed.append(
            {
                "game_id": row.game_id,
                "date": target,
                "home_team_id": int(row.home_team_id),
                "home_team_name": row.home_team_name,
                "away_team_id": int(row.away_team_id),
                "away_team_name": row.away_team_name,
                "home_score": row.home_score,
                "away_score": row.away_score,
                "start_time": "",
            }
        )
    return parsed


def fetch_espn_odds(target: date) -> pd.DataFrame:
    response = requests.get(
        "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard",
        params={"dates": target.strftime("%Y%m%d")},
        timeout=30,
    )
    response.raise_for_status()
    rows = []
    for event in response.json().get("events") or []:
        competition = event["competitions"][0]
        names = {side["homeAway"]: side["team"]["displayName"] for side in competition["competitors"]}
        event_id = event["id"]
        odds_response = requests.get(
            f"https://sports.core.api.espn.com/v2/sports/hockey/leagues/nhl/events/{event_id}/competitions/{event_id}/odds",
            timeout=30,
        )
        if odds_response.status_code != 200:
            continue
        items = odds_response.json().get("items") or []
        if not items:
            continue
        item = items[0]
        home_ml = (item.get("homeTeamOdds") or {}).get("moneyLine")
        away_ml = (item.get("awayTeamOdds") or {}).get("moneyLine")
        if home_ml is None or away_ml is None:
            continue
        rows.append(
            {
                "home_key": fold_name(names.get("home", "")),
                "away_key": fold_name(names.get("away", "")),
                "home_ml": float(home_ml),
                "away_ml": float(away_ml),
            }
        )
    return pd.DataFrame(rows)


def attach_market(predictions: pd.DataFrame, target: date) -> pd.DataFrame:
    frame = predictions.copy()
    frame["home_ml"] = pd.NA
    frame["away_ml"] = pd.NA
    frame["p_home_market"] = pd.NA
    frame["p_away_market"] = pd.NA
    if target < date.today() or frame.empty:
        return frame
    odds = fetch_espn_odds(target)
    if odds.empty:
        return frame
    lookup = {(row.home_key, row.away_key): row for row in odds.itertuples(index=False)}
    for idx, game in frame.iterrows():
        match = lookup.get((fold_name(game["home_team"]), fold_name(game["away_team"])))
        if match is None:
            continue
        home_fair, away_fair = devig(match.home_ml, match.away_ml)
        frame.at[idx, "home_ml"] = match.home_ml
        frame.at[idx, "away_ml"] = match.away_ml
        frame.at[idx, "p_home_market"] = home_fair
        frame.at[idx, "p_away_market"] = away_fair
    return frame


def add_recommendations(frame: pd.DataFrame, edge_threshold: float) -> pd.DataFrame:
    games = frame.copy()
    games["bet_side"] = pd.NA
    games["bet_team"] = pd.NA
    games["bet_edge"] = pd.NA
    games["bet_odds"] = pd.NA
    games["stake_frac"] = 0.0
    for idx, game in games.iterrows():
        if pd.isna(game.get("p_home_market")):
            continue
        home_edge = float(game["p_home"]) - float(game["p_home_market"])
        away_edge = float(game["p_away"]) - float(game["p_away_market"])
        if home_edge >= away_edge and home_edge >= edge_threshold:
            games.at[idx, "bet_side"] = "home"
            games.at[idx, "bet_team"] = game["home_team"]
            games.at[idx, "bet_edge"] = home_edge
            games.at[idx, "bet_odds"] = game["home_ml"]
            games.at[idx, "stake_frac"] = quarter_kelly(game["p_home"], game["p_home_market"], game["home_ml"])
        elif away_edge >= edge_threshold:
            games.at[idx, "bet_side"] = "away"
            games.at[idx, "bet_team"] = game["away_team"]
            games.at[idx, "bet_edge"] = away_edge
            games.at[idx, "bet_odds"] = game["away_ml"]
            games.at[idx, "stake_frac"] = quarter_kelly(game["p_away"], game["p_away_market"], game["away_ml"])
    recommended = games["bet_side"].notna()
    if recommended.any():
        games.loc[recommended, "stake_frac"] = cap_slate(games.loc[recommended, "stake_frac"])
    return games.sort_values(["bet_edge", "start_time"], ascending=[False, True], na_position="last")


@st.cache_data(ttl=3 * 3600, show_spinner=False)
def load_slate(target: date) -> pd.DataFrame:
    model, feature_columns = get_model()
    history = get_history()
    rows = []
    for game in games_on_date(target):
        prediction = predict_one(model, feature_columns, history, game)
        if prediction:
            rows.append(prediction)
    return pd.DataFrame(rows)


def local_start(value: object) -> str:
    if not value or pd.isna(value):
        return ""
    try:
        stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(LOCAL_TZ)
    except ValueError:
        return ""
    return stamp.strftime("%-I:%M %p PT")


def render_team(name: str, model_prob: float, market_prob: object, odds: object, show_market: bool) -> None:
    st.subheader(name)
    edge = None if not show_market or pd.isna(market_prob) else float(model_prob) - float(market_prob)
    st.metric(
        "Model win probability",
        f"{model_prob:.1%}",
        delta=None if edge is None else f"{edge:+.1%} vs market",
    )
    st.progress(min(max(model_prob, 0.0), 1.0), text="Logistic regression")
    if show_market:
        st.metric("Sportsbook fair probability", "—" if pd.isna(market_prob) else f"{float(market_prob):.1%}")
        if pd.notna(market_prob):
            st.progress(min(max(float(market_prob), 0.0), 1.0), text=f"ESPN {format_american(odds)}")


def render_slate(target: date, recommendations_only: bool, edge_threshold: float, bankroll: float) -> None:
    is_past = target < date.today()
    with st.spinner(f"Scoring the {target:%B %-d} slate…"):
        try:
            scored = load_slate(target)
            if not is_past and not scored.empty:
                scored = attach_market(scored, target)
                scored = add_recommendations(scored, edge_threshold)
        except Exception as exc:
            st.error(str(exc))
            return

    if scored.empty:
        st.info("No NHL games were found for this date.")
        return

    recommendations = scored.iloc[0:0]
    if not is_past and "bet_side" in scored.columns:
        recommendations = scored[scored["bet_side"].notna()]
    visible = recommendations if recommendations_only and not is_past else scored
    matched = 0 if is_past or "home_ml" not in scored.columns else int(scored["home_ml"].notna().sum())
    slate_stake = 0.0 if recommendations.empty else float(recommendations["stake_frac"].fillna(0).sum())

    metrics = st.columns(4)
    metrics[0].metric("Games", len(scored))
    metrics[1].metric("Market matches", "—" if is_past else f"{matched}/{len(scored)}")
    metrics[2].metric("Recommended bets", "—" if is_past else len(recommendations))
    if is_past or bankroll <= 0:
        metrics[3].metric("Slate stake", "—" if is_past else f"{slate_stake:.1%}")
    else:
        metrics[3].metric("Slate stake", f"${slate_stake * bankroll:,.0f}", delta=f"{slate_stake:.1%}")

    if visible.empty:
        st.info(f"No bets clear the {edge_threshold:.0%} edge threshold.")
        return

    st.subheader(f"{target:%A, %B %-d} matchups")
    if is_past:
        st.caption("Completed games. Probabilities come from the model trained on the full history, so this is not the held-out test. Use Results for that.")
    else:
        st.caption(
            f"Showing {len(visible)} of {len(scored)} games. "
            f"A bet needs at least {edge_threshold:.0%} edge over the vig-free ESPN line. "
            "Stake is quarter Kelly on a 50/50 model and market blend, capped at 5% per bet and 10% for the slate."
        )

    for _, game in visible.iterrows():
        with st.container(border=True):
            title, meta = st.columns([4, 1])
            title.markdown(
                f"<div class='hm-game-title'>{game['away_team']} @ {game['home_team']}</div>",
                unsafe_allow_html=True,
            )
            if is_past and pd.notna(game.get("away_score")) and pd.notna(game.get("home_score")):
                meta.markdown(
                    f"<div class='hm-meta'>Final {int(game['away_score'])}–{int(game['home_score'])}</div>",
                    unsafe_allow_html=True,
                )
            else:
                meta.markdown(
                    f"<div class='hm-meta'>{local_start(game.get('start_time'))}</div>",
                    unsafe_allow_html=True,
                )

            if is_past:
                st.markdown("<span class='hm-no-pick'>Historical slate — model probabilities only.</span>", unsafe_allow_html=True)
            elif pd.notna(game.get("bet_side")):
                stake = float(game.get("stake_frac") or 0)
                stake_label = f"Stake: {stake:.1%} of bankroll"
                if bankroll > 0 and stake > 0:
                    stake_label += f" · ${stake * bankroll:,.0f}"
                elif stake <= 0:
                    stake_label = "Stake: skip — not positive after the market blend"
                st.markdown(
                    f"<span class='hm-pick'>Model edge: {game['bet_team']} "
                    f"{format_american(game['bet_odds'])} · {float(game['bet_edge']):.1%}</span>"
                    f"<span class='hm-stake'>{stake_label}</span>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<span class='hm-no-pick'>No side clears the selected edge threshold.</span>",
                    unsafe_allow_html=True,
                )

            away_col, home_col = st.columns(2, gap="large")
            with away_col:
                render_team(
                    game["away_team"],
                    float(game["p_away"]),
                    game.get("p_away_market"),
                    game.get("away_ml"),
                    show_market=not is_past,
                )
            with home_col:
                render_team(
                    game["home_team"],
                    float(game["p_home"]),
                    game.get("p_home_market"),
                    game.get("home_ml"),
                    show_market=not is_past,
                )


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin(["true", "1"])


def market_columns(market: str) -> tuple[str, str, str]:
    if market == "open":
        return "open_odds_valid", "open_edge", "open_profit"
    return "odds_valid", "edge", "profit"


def season_roi(games: pd.DataFrame, threshold: float, market: str = "kaggle") -> pd.DataFrame:
    """Accuracy on every game; ROI only on predicted winners that clear the edge."""
    valid_col, edge_col, profit_col = market_columns(market)
    rows = []
    for season, group in games.groupby("season", sort=False):
        picks = group["home_win"] == group["pick_home"]
        bets = group[as_bool(group[valid_col]) & group[edge_col].notna()]
        if threshold > 0:
            bets = bets[bets[edge_col] >= threshold]
        profit = float(bets[profit_col].sum()) if len(bets) else float("nan")
        rows.append(
            {
                "season": season,
                "games": len(group),
                "accuracy": float(picks.mean()),
                "bets": len(bets),
                "hit_rate": float((bets[profit_col] > 0).mean()) if len(bets) else float("nan"),
                "profit": profit,
                "roi": profit / len(bets) if len(bets) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def same_ticket_roi(games: pd.DataFrame, threshold: float) -> pd.DataFrame | None:
    """Reprice the closing bets at the opening line, on games that have both."""
    if "open_odds_valid" not in games.columns:
        return None
    both = games[as_bool(games["odds_valid"]) & as_bool(games["open_odds_valid"])].copy()
    both = both[both["edge"].notna() & both["open_profit"].notna()]
    if threshold > 0:
        both = both[both["edge"] >= threshold]
    if both.empty:
        return None
    return both


def render_results() -> None:
    st.subheader("Leave-one-season-out results")
    edge_percent = st.slider(
        "Minimum edge",
        min_value=0,
        max_value=20,
        value=0,
        step=1,
        format="%d%%",
        key="results_edge_percent",
        help="0% bets every predicted winner that has a moneyline. A higher value keeps only picks that beat the line by at least that much.",
    )
    market = st.radio(
        "Edge vs",
        options=["Closing (AN consensus)", "Opening (AN Open)"],
        horizontal=True,
        help=(
            "Closing is the Action Network consensus moneyline around puck drop. "
            "Opening is book Open, from the 2023-24 season on, and pays at those opening prices."
        ),
    )
    threshold = edge_percent / 100.0
    use_open = market.startswith("Opening")
    if use_open:
        st.caption(
            "Each season is scored by a logistic regression trained on the other seasons. "
            f"ROI is a $1 bet on the predicted winner when its edge over the vig-free opening line "
            f"is at least {edge_percent}%. Payout uses the opening moneyline for that side. "
            "Action Network Open coverage starts in 2023-24."
        )
    else:
        st.caption(
            "Each season is scored by a logistic regression trained on the other seasons. "
            "Accuracy uses every game. ROI is a $1 bet on the predicted winner when its edge "
            f"over the vig-free closing line is at least {edge_percent}%. "
            "Payout uses the closing moneyline for that side."
        )
    if not BETS_PATH.is_file():
        st.info("Results are not ready yet. Run python src/evaluate_historical.py.")
        return

    games = pd.read_csv(BETS_PATH)
    if use_open and "open_edge" not in games.columns:
        st.warning(
            "Opening odds are not in the results file. Run "
            "`python src/fetch_opening_odds.py`, then `python src/evaluate_historical.py`."
        )
        return

    priced = same_ticket_roi(games, threshold)
    if priced is not None:
        close_roi = float(priced["profit"].sum()) / len(priced)
        opening_roi = float(priced["open_profit"].sum()) / len(priced)
        st.caption(
            f"Same closing bets that also have an opening price: "
            f"closing ROI {close_roi:+.1%}, opening price {opening_roi:+.1%} "
            f"({len(priced)} bets)."
        )

    selected = "open" if use_open else "close"
    by_season = season_roi(games, threshold, selected)
    overall = season_roi(games.assign(season="Overall"), threshold, selected)
    table = pd.concat([by_season, overall], ignore_index=True)
    display = table.copy()
    display["accuracy"] = display["accuracy"].map(lambda value: f"{value:.1%}")
    display["hit_rate"] = display["hit_rate"].map(lambda value: "—" if pd.isna(value) else f"{value:.1%}")
    display["profit"] = display["profit"].map(lambda value: "—" if pd.isna(value) else f"{value:+.1f}")
    display["roi"] = display["roi"].map(lambda value: "—" if pd.isna(value) else f"{value:+.1%}")
    display = display.rename(
        columns={
            "season": "Season",
            "games": "Games",
            "accuracy": "Accuracy",
            "bets": "Bets",
            "hit_rate": "Bet hit rate",
            "profit": "Profit ($1)",
            "roi": "ROI",
        }
    )
    st.dataframe(display, use_container_width=True, hide_index=True)

    chart_data = by_season.dropna(subset=["roi"])
    if chart_data.empty:
        st.info(f"No bets clear a {edge_percent}% edge.")
        return
    chart = (
        alt.Chart(chart_data)
        .mark_bar()
        .encode(
            x=alt.X("season:N", title="Season", sort=list(chart_data["season"])),
            y=alt.Y("roi:Q", title="ROI", axis=alt.Axis(format=".0%")),
            tooltip=[
                alt.Tooltip("season:N", title="Season"),
                alt.Tooltip("bets:Q", title="Bets"),
                alt.Tooltip("roi:Q", title="ROI", format=".1%"),
                alt.Tooltip("profit:Q", title="Profit", format="+.1f"),
            ],
            color=alt.value("#7ad7ff"),
        )
        .properties(height=280, title=f"{market} ROI by season at {edge_percent}% minimum edge")
    )
    st.altair_chart(chart, use_container_width=True)


st.markdown(
    """
    <section class="hm-hero">
      <h1>Hockey Moneyline</h1>
      <p>NHL win probabilities from the logistic model, set next to the sportsbook line.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

today = date.today()
if "target_date" not in st.session_state:
    st.session_state["target_date"] = today


def choose_tomorrow() -> None:
    st.session_state["target_date"] = today + timedelta(days=1)


with st.sidebar:
    st.header("Controls")
    target_date = st.date_input(
        "Game date",
        key="target_date",
        min_value=date(2025, 10, 1),
        max_value=date(2027, 6, 30),
    )
    is_past = target_date < today
    recommendations_only = st.toggle("Recommendations only", value=False, disabled=is_past)
    edge_percent = st.slider(
        "Minimum edge",
        min_value=0,
        max_value=20,
        value=5,
        step=1,
        format="%d%%",
        disabled=is_past,
    )
    bankroll = st.number_input("Bankroll ($)", min_value=0, value=1000, step=100, disabled=is_past)
    if st.button("Refresh data", type="primary", use_container_width=True):
        load_slate.clear()
        st.rerun()
    if target_date == today:
        st.button("View tomorrow", use_container_width=True, on_click=choose_tomorrow)
    st.divider()
    st.caption("Model: logistic regression, C=0.001, trained on every completed game in the feature file.")

slate_tab, results_tab = st.tabs(["Daily slate", "Results"])
with slate_tab:
    render_slate(target_date, recommendations_only, edge_percent / 100.0, float(bankroll))
with results_tab:
    render_results()
