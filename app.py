"""
Discovery Insight Engine — Streamlit dashboard.

Primary question: Why don't Blinkit users explore new categories?

IA (parity with reference, not a clone):
  Findings Board · Category Opportunities · Validation Desk ·
  Live Pipeline · Try-it Console · Evidence Lab · Methodology · Admin

Run from repo root:
  streamlit run app.py
"""

from __future__ import annotations

import html
import json
import os
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from discovery_engine.env_loader import load_env

load_env()

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"

PRIMARY_QUESTION = "Why don't Blinkit users explore new categories?"

SENTIMENT_COLORS = {
    "Positive": "#12B76A",
    "Neutral": "#98A2B3",
    "Negative": "#F04438",
}
SEGMENT_COLORS = {
    "Routine Buyers": "#12B76A",
    "Explorers": "#2E90FA",
    "Price Sensitive": "#F79009",
    "Impulse Buyers": "#6172F3",
}
PRIORITY_ORDER = {"High": 0, "Medium": 1, "Low": 2}
PRIORITY_COLORS = {"High": "#F04438", "Medium": "#F79009", "Low": "#2E90FA"}
SIGNAL_COLORS = {
    "stuck_in_routine": "#F79009",
    "want_to_explore_blocked": "#F04438",
    "explored_new": "#12B76A",
    "unclear": "#98A2B3",
    "noise": "#D0D5DD",
}

PAGES = [
    "Findings Board",
    "Category Opportunities",
    "Validation Desk",
    "Live Pipeline",
    "Try-it Console",
    "Evidence Lab",
    "Methodology",
    "Admin",
]

PAGE_ICONS = {
    "Findings Board": "✦",
    "Category Opportunities": "↑",
    "Validation Desk": "☑",
    "Live Pipeline": "⟳",
    "Try-it Console": "▷",
    "Evidence Lab": "▣",
    "Methodology": "?",
    "Admin": "⚙",
}

PLOTLY_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "toImageButtonOptions": {"format": "png", "filename": "discovery_chart"},
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_sentiment() -> pd.DataFrame:
    """Load sentiment.csv with required columns validated."""
    path = OUTPUT / "sentiment.csv"
    cols = ["Review", "Sentiment", "Confidence Score"]
    if not path.exists():
        return pd.DataFrame(columns=cols)
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not read sentiment.csv: {exc}")
        return pd.DataFrame(columns=cols)
    for col in cols:
        if col not in df.columns:
            df[col] = pd.NA if col != "Confidence Score" else 0.0
    df["Sentiment"] = df["Sentiment"].astype(str).str.strip().str.capitalize()
    df["Confidence Score"] = pd.to_numeric(df["Confidence Score"], errors="coerce").fillna(0.0)
    df["Review"] = df["Review"].fillna("").astype(str)
    return df[cols]


@st.cache_data(show_spinner=False)
def load_themes() -> pd.DataFrame:
    """Load themes.csv and derive display names for charts/tables."""
    path = OUTPUT / "themes.csv"
    empty_cols = [
        "Theme name",
        "Number of reviews",
        "Representative keywords",
        "Representative reviews",
        "Display name",
    ]
    if not path.exists():
        return pd.DataFrame(columns=empty_cols)
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not read themes.csv: {exc}")
        return pd.DataFrame(columns=empty_cols)
    if "Theme name" not in df.columns:
        df["Theme name"] = "Untitled"
    if "Number of reviews" not in df.columns:
        df["Number of reviews"] = 0
    if "Representative keywords" not in df.columns:
        df["Representative keywords"] = ""
    if "Representative reviews" not in df.columns:
        df["Representative reviews"] = ""
    df["Number of reviews"] = pd.to_numeric(df["Number of reviews"], errors="coerce").fillna(0).astype(int)
    df["Display name"] = df.apply(_theme_display_name, axis=1)
    return df.sort_values("Number of reviews", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_segments() -> pd.DataFrame:
    """Load user_segments.csv."""
    path = OUTPUT / "user_segments.csv"
    cols = ["Review", "Segment", "Cluster ID", "Prototype Similarity", "Label Rationale"]
    if not path.exists():
        return pd.DataFrame(columns=cols)
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not read user_segments.csv: {exc}")
        return pd.DataFrame(columns=cols)
    for col in cols:
        if col not in df.columns:
            df[col] = pd.NA
    return df


@st.cache_data(show_spinner=False)
def load_insights() -> dict:
    """Load insights.json; return an empty payload on missing/corrupt files."""
    path = OUTPUT / "insights.json"
    empty = {"insights": [], "research_questions": [], "meta": {}}
    if not path.exists():
        return empty
    try:
        with path.open(encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not read insights.json: {exc}")
        return empty
    if not isinstance(payload, dict):
        return empty
    payload.setdefault("insights", [])
    payload.setdefault("research_questions", [])
    payload.setdefault("meta", {})
    if not isinstance(payload["insights"], list):
        payload["insights"] = []
    return payload


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime if path.exists() else 0.0
    except OSError:
        return 0.0


@st.cache_data(show_spinner=False)
def _load_synthesis_cached(mtime: float) -> dict:
    path = OUTPUT / "synthesis.json"
    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not read synthesis.json: {exc}")
        return {}


def load_synthesis() -> dict:
    """Load synthesis.json; cache key includes mtime so regenerations refresh the UI."""
    return _load_synthesis_cached(_mtime(OUTPUT / "synthesis.json"))


def ensure_synthesis(*, force: bool = False) -> dict:
    """Return synthesis with category opportunities, generating from tags if needed."""
    syn = load_synthesis()
    ops = (syn or {}).get("category_opportunities") or []
    if syn and ops and not force:
        return syn

    tags_path = OUTPUT / "exploration_tags.csv"
    merged_path = DATA_PROC / "merged_reviews.csv"
    if not tags_path.exists() and merged_path.exists():
        try:
            from analysis.exploration import save_tags, tag_corpus

            df = pd.read_csv(merged_path)
            text_col = "text" if "text" in df.columns else "Review"
            save_tags(tag_corpus(df, text_col=text_col), tags_path)
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Could not build exploration tags: {exc}")

    if not tags_path.exists():
        return syn or {}

    try:
        from llm.synthesis import generate_synthesis

        syn = generate_synthesis(tags_path, OUTPUT / "synthesis.json", polish=False)
        st.cache_data.clear()
        return syn if isinstance(syn, dict) else load_synthesis()
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not generate synthesis: {exc}")
        return syn or {}


@st.cache_data(show_spinner=False)
def _load_exploration_cached(mtime: float) -> pd.DataFrame:
    path = OUTPUT / "exploration_tags.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not read exploration_tags.csv: {exc}")
        return pd.DataFrame()


def load_exploration() -> pd.DataFrame:
    return _load_exploration_cached(_mtime(OUTPUT / "exploration_tags.csv"))


@st.cache_data(show_spinner=False)
def _load_merged_cached(mtime: float) -> pd.DataFrame:
    path = DATA_PROC / "merged_reviews.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not read merged_reviews.csv: {exc}")
        return pd.DataFrame()


def load_merged() -> pd.DataFrame:
    return _load_merged_cached(_mtime(DATA_PROC / "merged_reviews.csv"))


def _theme_display_name(row: pd.Series) -> str:
    raw = str(row.get("Theme name", "") or "").strip()
    keywords = str(row.get("Representative keywords", "") or "")
    parts = [p.strip() for p in keywords.split(",") if p.strip() and p.strip() not in {"", "/"}]
    if parts:
        label = " · ".join(parts[:3])
    else:
        label = " / ".join([p.strip() for p in raw.split("/") if p.strip()]) or "Untitled theme"
    if len(label) > 48:
        label = label[:45].rstrip() + "…"
    return label


def is_dark() -> bool:
    """Permanent dark theme."""
    return True


def theme_tokens() -> dict[str, str]:
    if is_dark():
        return {
            "ink": "#F5F7FA",
            "muted": "#98A2B3",
            "line": "#2A3340",
            "panel": "#121821",
            "bg": "#0B0F14",
            "accent": "#32D583",
            "accent_soft": "#12261C",
            "grid": "#243041",
            "pie_stroke": "#121821",
            "sidebar_top": "#070B10",
            "sidebar_mid": "#0E141C",
            "sidebar_bot": "#121A24",
        }
    return {
        "ink": "#101828",
        "muted": "#667085",
        "line": "#EAECF0",
        "panel": "#FFFFFF",
        "bg": "#F8FAFC",
        "accent": "#039855",
        "accent_soft": "#ECFDF3",
        "grid": "#EEF2F6",
        "pie_stroke": "#FFFFFF",
        "sidebar_top": "#0B1F17",
        "sidebar_mid": "#0F2A1F",
        "sidebar_bot": "#143528",
    }


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


def _chart_layout(fig: go.Figure, *, height: int | None = None) -> go.Figure:
    """Apply readable axis/legend fonts for the active light/dark palette."""
    tok = theme_tokens()
    ink = tok["ink"]
    muted = tok["muted"]
    fig.update_layout(
        template="plotly_white" if not is_dark() else "plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans, Inter, sans-serif", color=ink, size=13),
        title_font=dict(color=ink),
        legend=dict(font=dict(color=ink)),
        margin=dict(t=28, b=28, l=24, r=24),
        hoverlabel=dict(
            bgcolor=tok["panel"],
            bordercolor=tok["line"],
            font=dict(family="Plus Jakarta Sans, sans-serif", color=ink, size=12),
        ),
        xaxis=dict(
            color=muted,
            title_font=dict(color=muted),
            tickfont=dict(color=ink, size=12),
            gridcolor=tok["grid"],
            zerolinecolor=tok["grid"],
            linecolor=tok["line"],
        ),
        yaxis=dict(
            color=ink,
            title_font=dict(color=muted),
            tickfont=dict(color=ink, size=12),
            gridcolor=tok["grid"],
            zerolinecolor=tok["grid"],
            linecolor=tok["line"],
        ),
    )
    # Outside bar labels / pie slice text
    fig.update_traces(
        selector=dict(type="bar"),
        textfont=dict(color=ink, size=12, family="Plus Jakarta Sans, sans-serif"),
    )
    fig.update_traces(
        selector=dict(type="pie"),
        textfont=dict(color=ink, size=13, family="Plus Jakarta Sans, sans-serif"),
    )
    if height is not None:
        fig.update_layout(height=height)
    return fig


def _show_chart(fig: go.Figure) -> None:
    """Render Plotly without Streamlit theme override (keeps light-mode labels readable)."""
    st.plotly_chart(
        fig,
        use_container_width=True,
        config=PLOTLY_CONFIG,
        theme=None,
    )


def sentiment_pie(df: pd.DataFrame) -> go.Figure:
    tok = theme_tokens()
    counts = (
        df["Sentiment"]
        .value_counts()
        .reindex(["Positive", "Neutral", "Negative"])
        .fillna(0)
        .astype(int)
    )
    fig = go.Figure(
        data=[
            go.Pie(
                labels=counts.index.tolist(),
                values=counts.values.tolist(),
                hole=0.62,
                marker=dict(
                    colors=[SENTIMENT_COLORS[k] for k in counts.index],
                    line=dict(color=tok["pie_stroke"], width=3),
                ),
                textinfo="label+percent",
                textfont=dict(size=13, family="Plus Jakarta Sans, sans-serif"),
                hovertemplate="<b>%{label}</b><br>%{value} reviews (%{percent})<extra></extra>",
                pull=[0.02, 0.0, 0.02],
            )
        ]
    )
    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.14,
            x=0.5,
            xanchor="center",
            font=dict(color=tok["ink"], size=12),
        ),
        annotations=[
            dict(
                text=f"<b>{int(counts.sum())}</b><br><span style='font-size:12px'>reviews</span>",
                x=0.5,
                y=0.5,
                font=dict(size=18, color=tok["ink"], family="Sora, sans-serif"),
                showarrow=False,
            )
        ],
    )
    fig = _chart_layout(fig, height=400)
    fig.update_traces(
        textfont=dict(color=tok["ink"], size=13, family="Plus Jakarta Sans, sans-serif"),
        selector=dict(type="pie"),
    )
    return fig


def theme_frequency_chart(themes: pd.DataFrame, top_n: int = 10) -> go.Figure:
    tok = theme_tokens()
    data = themes.head(top_n).iloc[::-1]
    fig = go.Figure(
        go.Bar(
            x=data["Number of reviews"],
            y=data["Display name"],
            orientation="h",
            marker=dict(
                color=tok["accent"],
                cornerradius=8,
                line=dict(width=0),
            ),
            hovertemplate="<b>%{y}</b><br>%{x} reviews<extra></extra>",
            text=data["Number of reviews"],
            textposition="outside",
            textfont=dict(color=tok["ink"], size=12),
            cliponaxis=False,
        )
    )
    fig = _chart_layout(fig, height=max(340, 40 * len(data) + 90))
    fig.update_layout(
        xaxis_title="Reviews",
        yaxis_title="",
        yaxis_automargin=True,
    )
    return fig


def segment_bar(segments: pd.DataFrame) -> go.Figure:
    tok = theme_tokens()
    counts = segments["Segment"].value_counts().reset_index()
    counts.columns = ["Segment", "Count"]
    fig = px.bar(
        counts,
        x="Segment",
        y="Count",
        text="Count",
        color="Segment",
        color_discrete_map=SEGMENT_COLORS,
    )
    fig.update_traces(
        textposition="outside",
        marker_line_width=0,
        width=0.55,
        cliponaxis=False,
        textfont=dict(color=tok["ink"], size=12),
    )
    fig = _chart_layout(fig, height=400)
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Users")
    return fig


def opportunity_rank_chart(ranked: pd.DataFrame) -> go.Figure:
    tok = theme_tokens()
    data = ranked.iloc[::-1]
    fig = go.Figure(
        go.Bar(
            x=data["Score"],
            y=data["Title"],
            orientation="h",
            marker=dict(
                color=[PRIORITY_COLORS.get(p, "#98A2B3") for p in data["Priority"]],
                cornerradius=8,
            ),
            customdata=data[["Priority", "Opportunity"]],
            hovertemplate=(
                "<b>%{y}</b><br>Priority: %{customdata[0]}<br>"
                "Score: %{x}<br>%{customdata[1]}<extra></extra>"
            ),
            text=data["Priority"],
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(color="#FFFFFF", size=12),
        )
    )
    fig = _chart_layout(fig, height=max(380, 56 * len(data) + 90))
    # Keep inside labels white on colored bars (overrides _chart_layout bar textfont)
    fig.update_traces(textfont=dict(color="#FFFFFF", size=12), selector=dict(type="bar"))
    fig.update_layout(xaxis_title="Opportunity score", xaxis_range=[0, 105], yaxis_automargin=True)
    return fig


def confidence_bar(sentiment: pd.DataFrame) -> go.Figure:
    tok = theme_tokens()
    if "Confidence Score" not in sentiment.columns or sentiment.empty:
        fig = go.Figure()
        fig.update_layout(title="No confidence data")
        return _chart_layout(fig, height=360)
    conf = (
        sentiment.groupby("Sentiment", as_index=False)["Confidence Score"]
        .mean()
        .sort_values("Confidence Score", ascending=False)
    )
    fig = px.bar(
        conf,
        x="Sentiment",
        y="Confidence Score",
        color="Sentiment",
        color_discrete_map=SENTIMENT_COLORS,
        text=conf["Confidence Score"].map(lambda x: f"{x:.2f}"),
    )
    fig.update_traces(
        textposition="outside",
        width=0.55,
        cliponaxis=False,
        textfont=dict(color=tok["ink"], size=12),
    )
    fig = _chart_layout(fig, height=360)
    fig.update_layout(
        showlegend=False,
        xaxis_title="",
        yaxis_title="Avg confidence",
        yaxis_range=[0, 1.08],
    )
    return fig


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def inject_styles() -> None:
    tok = theme_tokens()
    mode = "dark" if is_dark() else "light"
    st.markdown(
        f"""
        <style data-discovery-theme="{mode}">
          /* theme:{mode} */
          @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Sora:wght@500;600;700&display=swap');

          :root {{
            --ink: {tok["ink"]};
            --muted: {tok["muted"]};
            --line: {tok["line"]};
            --panel: {tok["panel"]};
            --bg: {tok["bg"]};
            --accent: {tok["accent"]};
            --accent-soft: {tok["accent_soft"]};
            --warn: #F79009;
            --danger: #F04438;
            --shadow: {"0 8px 24px rgba(0,0,0,0.35)" if is_dark() else "0 8px 24px rgba(16,24,40,0.06)"};
          }}

          html, body, [class*="css"] {{
            font-family: "Plus Jakarta Sans", Inter, sans-serif !important;
          }}

          .stApp {{
            background:
              radial-gradient(900px 420px at 0% -10%, {"rgba(50,213,131,0.12)" if is_dark() else "rgba(3,152,85,0.10)"} 0%, transparent 55%),
              radial-gradient(700px 360px at 100% 0%, {"rgba(46,144,250,0.10)" if is_dark() else "rgba(46,144,250,0.08)"} 0%, transparent 50%),
              var(--bg) !important;
            color: var(--ink) !important;
            /* Override Streamlit theme tokens so light mode stays readable after dark */
            --text-color: {tok["ink"]} !important;
            --secondary-text-color: {tok["muted"]} !important;
            --text-color-strong: {tok["ink"]} !important;
            --background-color: {tok["bg"]} !important;
            --secondary-background-color: {tok["panel"]} !important;
          }}

          /* Main pane text (do not inherit sidebar white text) */
          [data-testid="stAppViewContainer"] {{
            color: {tok["ink"]} !important;
            background: transparent !important;
          }}
          [data-testid="stAppViewContainer"] p,
          [data-testid="stAppViewContainer"] li,
          [data-testid="stAppViewContainer"] span,
          [data-testid="stAppViewContainer"] label,
          [data-testid="stAppViewContainer"] h1,
          [data-testid="stAppViewContainer"] h2,
          [data-testid="stAppViewContainer"] h3,
          [data-testid="stAppViewContainer"] h4,
          [data-testid="stMarkdownContainer"],
          [data-testid="stMarkdownContainer"] p,
          [data-testid="stMarkdownContainer"] span,
          [data-testid="stCaptionContainer"],
          [data-testid="stCaptionContainer"] p,
          [data-testid="stWidgetLabel"] p,
          [data-testid="stWidgetLabel"] label,
          .stMarkdown, .stCaption, .stText,
          div[data-baseweb="input"] input,
          div[data-baseweb="textarea"] textarea,
          div[data-baseweb="select"] > div,
          [data-testid="stSelectbox"] div,
          [data-testid="stMultiSelect"] div,
          [data-testid="stTextInput"] input,
          [data-testid="stNumberInput"] input {{
            color: {tok["ink"]} !important;
          }}
          [data-testid="stCaptionContainer"],
          [data-testid="stCaptionContainer"] p {{
            color: {tok["muted"]} !important;
          }}
          div[data-baseweb="input"] input,
          div[data-baseweb="textarea"] textarea,
          [data-testid="stTextInput"] input {{
            background-color: {tok["panel"]} !important;
            -webkit-text-fill-color: {tok["ink"]} !important;
          }}
          [data-testid="stDataFrame"] * {{
            color: {tok["ink"]} !important;
          }}
          div[data-testid="stDownloadButton"] button,
          div[data-testid="stButton"] button {{
            color: {tok["ink"]} !important;
            background-color: {tok["panel"]} !important;
          }}

          [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {tok["sidebar_top"]} 0%, {tok["sidebar_mid"]} 50%, {tok["sidebar_bot"]} 100%) !important;
            border-right: 1px solid rgba(255,255,255,0.06);
          }}
          [data-testid="stSidebar"],
          [data-testid="stSidebar"] p,
          [data-testid="stSidebar"] span,
          [data-testid="stSidebar"] label,
          [data-testid="stSidebar"] .stMarkdown {{
            color: #F5F7FA !important;
          }}
          [data-testid="stSidebar"] .stRadio label {{
            padding: 0.62rem 0.8rem;
            border-radius: 12px;
            margin-bottom: 0.18rem;
            border: 1px solid transparent;
            transition: background 160ms ease, border-color 160ms ease;
            color: #F5F7FA !important;
          }}
          [data-testid="stSidebar"] .stRadio label:hover {{
            background: rgba(255,255,255,0.06);
            border-color: rgba(255,255,255,0.08);
          }}
          [data-testid="stSidebar"] [data-baseweb="radio"] div[role="radiogroup"] label[data-checked="true"],
          [data-testid="stSidebar"] label:has(input:checked) {{
            background: rgba(50,213,131,0.16) !important;
            border-color: rgba(50,213,131,0.35) !important;
          }}
          [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
          [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{
            color: #D0D5DD !important;
          }}
          [data-testid="stSidebar"] div[data-testid="stDownloadButton"] button,
          [data-testid="stSidebar"] div[data-testid="stButton"] button {{
            color: #F5F7FA !important;
            background-color: rgba(255,255,255,0.08) !important;
            border-color: rgba(255,255,255,0.18) !important;
          }}

          h1, h2, h3, .hero-title, .insight-title, .metric-value {{
            font-family: "Sora", "Plus Jakarta Sans", sans-serif !important;
            letter-spacing: -0.03em;
          }}
          .block-container {{ padding-top: 1.35rem; max-width: 1200px; }}

          .hero-kicker {{
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: var(--accent);
            background: var(--accent-soft);
            padding: 0.32rem 0.7rem;
            border-radius: 999px;
            margin-bottom: 0.65rem;
            border: 1px solid color-mix(in srgb, var(--accent) 25%, transparent);
          }}
          .hero-title {{
            font-size: 2.05rem;
            font-weight: 700;
            line-height: 1.12;
            margin: 0 0 0.4rem 0;
            color: {tok["ink"]} !important;
            animation: fadeRise 420ms ease both;
          }}
          .hero-sub {{
            color: {tok["muted"]} !important;
            font-size: 1rem;
            line-height: 1.55;
            margin-bottom: 1.25rem;
            max-width: 46rem;
            animation: fadeRise 520ms ease both;
          }}

          .metric-card {{
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 1.05rem 1.15rem 1rem;
            box-shadow: var(--shadow);
            height: 100%;
            position: relative;
            overflow: hidden;
            transition: transform 180ms ease, box-shadow 180ms ease;
            animation: fadeRise 480ms ease both;
          }}
          .metric-card:hover {{
            transform: translateY(-2px);
            box-shadow: {"0 14px 32px rgba(0,0,0,0.45)" if is_dark() else "0 14px 32px rgba(16,24,40,0.10)"};
          }}
          .metric-card::before {{
            content: "";
            position: absolute;
            inset: 0 auto auto 0;
            width: 100%;
            height: 3px;
            background: linear-gradient(90deg, var(--accent), transparent 70%);
            opacity: 0.85;
          }}
          .metric-label {{
            font-size: 0.74rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: {tok["muted"]} !important;
            margin-bottom: 0.45rem;
          }}
          .metric-value {{
            font-size: 2.05rem;
            font-weight: 700;
            line-height: 1;
            color: {tok["ink"]} !important;
            font-variant-numeric: tabular-nums;
          }}
          .metric-value[data-count] {{
            animation: countPulse 700ms ease;
          }}
          .metric-hint {{ margin-top: 0.45rem; font-size: 0.82rem; color: {tok["muted"]} !important; }}

          .panel {{
            background: {tok["panel"]} !important;
            border: 1px solid {tok["line"]} !important;
            border-radius: 18px;
            padding: 1.15rem 1.25rem 1.05rem;
            margin-bottom: 1rem;
            box-shadow: var(--shadow);
            animation: fadeRise 560ms ease both;
            color: {tok["ink"]} !important;
          }}
          .panel h3 {{
            margin: 0 0 0.28rem 0 !important;
            font-size: 1.05rem !important;
            color: {tok["ink"]} !important;
          }}
          .panel-sub {{ color: {tok["muted"]} !important; font-size: 0.9rem; margin-bottom: 0.85rem; }}

          .toolbar {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            align-items: center;
            margin: 0.35rem 0 0.9rem 0;
          }}

          .insight-card {{
            background: {tok["panel"]} !important;
            border: 1px solid {tok["line"]} !important;
            border-left: 4px solid var(--accent);
            border-radius: 16px;
            padding: 1.05rem 1.2rem;
            margin-bottom: 0.85rem;
            box-shadow: var(--shadow);
            animation: fadeRise 500ms ease both;
            color: {tok["ink"]} !important;
          }}
          .insight-card.priority-high {{ border-left-color: var(--danger); }}
          .insight-card.priority-medium {{ border-left-color: var(--warn); }}
          .insight-card.priority-low {{ border-left-color: #2E90FA; }}
          .insight-title {{
            font-size: 1.08rem;
            font-weight: 700;
            margin: 0.25rem 0 0.45rem 0;
            color: {tok["ink"]} !important;
          }}
          .badge {{
            display: inline-block;
            font-size: 0.7rem;
            font-weight: 800;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            padding: 0.2rem 0.55rem;
            border-radius: 999px;
            background: {"#1D2939" if is_dark() else "#F2F4F7"};
            color: {"#D0D5DD" if is_dark() else "#344054"} !important;
          }}
          .badge.high {{ background: {"#3B1219" if is_dark() else "#FEE4E2"}; color: {"#FDA29B" if is_dark() else "#B42318"} !important; }}
          .badge.medium {{ background: {"#3B2A0E" if is_dark() else "#FEF0C7"}; color: {"#FEC84B" if is_dark() else "#B54708"} !important; }}
          .badge.low {{ background: {"#102A56" if is_dark() else "#D1E9FF"}; color: {"#84CAFF" if is_dark() else "#175CD3"} !important; }}
          .field-label {{
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: {tok["muted"]} !important;
            margin-top: 0.55rem;
          }}
          .field-body {{ color: {tok["ink"]} !important; font-size: 0.94rem; line-height: 1.5; }}

          .flow-wrap {{
            font-family: "Plus Jakarta Sans", sans-serif;
            color: {tok["ink"]} !important;
            max-width: 640px;
            margin: 0 auto;
          }}
          .simple-flow {{
            display: flex;
            flex-direction: column;
            align-items: stretch;
            gap: 0;
          }}
          .simple-step {{
            border: 1px solid {tok["line"]};
            border-radius: 16px;
            background: {tok["panel"]};
            padding: 1rem 1.15rem;
            box-shadow: var(--shadow);
          }}
          .simple-step .n {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 1.7rem;
            height: 1.7rem;
            border-radius: 999px;
            background: {tok["accent"]};
            color: #fff !important;
            font-family: "Sora", sans-serif;
            font-size: 0.78rem;
            font-weight: 700;
            margin-bottom: 0.45rem;
          }}
          .simple-step .t {{
            font-family: "Sora", sans-serif;
            font-size: 1.05rem;
            font-weight: 700;
            color: {tok["ink"]} !important;
            margin: 0 0 0.3rem 0;
          }}
          .simple-step .d {{
            color: {tok["muted"]} !important;
            font-size: 0.92rem;
            line-height: 1.45;
            margin: 0;
          }}
          .simple-step .pills {{
            margin-top: 0.55rem;
          }}
          .simple-arrow {{
            text-align: center;
            color: {tok["accent"]} !important;
            font-size: 1.25rem;
            font-weight: 700;
            line-height: 1;
            padding: 0.25rem 0;
          }}
          .flow-note {{
            margin-top: 0.85rem;
            color: {tok["muted"]} !important;
            font-size: 0.82rem;
            text-align: center;
          }}
          .flow-pill {{
            display: inline-block;
            margin: 0.15rem 0.25rem 0.15rem 0;
            padding: 0.18rem 0.55rem;
            border-radius: 999px;
            background: {tok["accent_soft"]};
            color: {tok["accent"]} !important;
            font-size: 0.75rem;
            font-weight: 700;
          }}
          div[data-testid="stDataFrame"] {{
            border: 1px solid var(--line);
            border-radius: 14px;
            overflow: hidden;
            background: var(--panel);
          }}

          div[data-testid="stDownloadButton"] button,
          div[data-testid="stButton"] button {{
            border-radius: 11px !important;
            font-weight: 600 !important;
            border: 1px solid var(--line) !important;
          }}

          .sidebar-brand {{
            font-family: "Sora", sans-serif;
            font-size: 1.15rem;
            font-weight: 700;
            letter-spacing: -0.03em;
            margin-bottom: 0.15rem;
          }}
          .sidebar-meta {{
            color: #98A2B3 !important;
            font-size: 0.82rem;
            margin-bottom: 0.85rem;
          }}
          .nav-hint {{
            font-size: 0.7rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #98A2B3 !important;
            margin: 0.4rem 0 0.35rem 0;
            font-weight: 700;
          }}

          @keyframes fadeRise {{
            from {{ opacity: 0; transform: translateY(8px); }}
            to {{ opacity: 1; transform: translateY(0); }}
          }}
          @keyframes countPulse {{
            0% {{ opacity: 0.35; transform: translateY(6px) scale(0.96); }}
            60% {{ opacity: 1; transform: translateY(-1px) scale(1.02); }}
            100% {{ opacity: 1; transform: translateY(0) scale(1); }}
          }}

          /* Streamlit chrome */
          header[data-testid="stHeader"] {{ background: transparent; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="hero-kicker">{html.escape(PRIMARY_QUESTION)}</div>
        <div class="hero-title">{html.escape(title)}</div>
        <div class="hero-sub">{html.escape(subtitle)}</div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, hint: str = "", accent: str | None = None) -> None:
    value_style = f' style="color:{accent};"' if accent else ""
    safe_label = html.escape(label)
    safe_value = html.escape(str(value))
    safe_hint = html.escape(hint)
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">{safe_label}</div>
          <div class="metric-value" data-count="1"{value_style}>{safe_value}</div>
          <div class="metric-hint">{safe_hint}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def download_csv_button(df: pd.DataFrame, filename: str, label: str = "Download CSV") -> None:
    if df is None or df.empty:
        st.download_button(
            label=label,
            data="",
            file_name=filename,
            mime="text/csv",
            disabled=True,
            use_container_width=False,
        )
        return
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        use_container_width=False,
    )


def panel_start(title: str, subtitle: str = "") -> None:
    sub = f'<div class="panel-sub">{html.escape(subtitle)}</div>' if subtitle else ""
    st.markdown(
        f'<div class="panel"><h3>{html.escape(title)}</h3>{sub}',
        unsafe_allow_html=True,
    )


def panel_end() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def filter_dataframe_search(df: pd.DataFrame, query: str, columns: list[str]) -> pd.DataFrame:
    q = (query or "").strip().lower()
    if not q or df.empty:
        return df
    mask = False
    for col in columns:
        if col in df.columns:
            mask = mask | df[col].astype(str).str.lower().str.contains(q, na=False, regex=False)
    return df[mask] if isinstance(mask, pd.Series) else df


def rank_opportunities(insights: list[dict]) -> pd.DataFrame:
    rows = []
    for i, item in enumerate(insights):
        priority = str(item.get("Priority", "Medium")).strip().title()
        base = {"High": 90, "Medium": 65, "Low": 40}.get(priority, 55)
        score = base - i * 2
        rows.append(
            {
                "Rank": 0,
                "Title": item.get("Title", "Untitled"),
                "Priority": priority,
                "Opportunity": item.get("Opportunity", ""),
                "Recommendation": item.get("Recommendation", ""),
                "Business Impact": item.get("Business Impact", ""),
                "Score": score,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(
        by=["Priority", "Score"],
        key=lambda s: s.map(PRIORITY_ORDER) if s.name == "Priority" else s,
        ascending=[True, False],
    ).reset_index(drop=True)
    df["Rank"] = df.index + 1
    return df


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def render_overview(sentiment: pd.DataFrame, themes: pd.DataFrame, segments: pd.DataFrame, insights: dict) -> None:
    page_header(
        "Overview",
        "A single view of review volume, sentiment mix, emerging themes, and ranked product opportunities.",
    )

    total = len(sentiment)
    pos = int((sentiment["Sentiment"] == "Positive").sum()) if total else 0
    neg = int((sentiment["Sentiment"] == "Negative").sum()) if total else 0
    neu = int((sentiment["Sentiment"] == "Neutral").sum()) if total else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Total Reviews", f"{total:,}", "From Play Store sample")
    with c2:
        pct = f"{(pos / total * 100):.0f}% of corpus" if total else "—"
        metric_card("Positive", f"{pos:,}", pct, SENTIMENT_COLORS["Positive"])
    with c3:
        pct = f"{(neg / total * 100):.0f}% of corpus" if total else "—"
        metric_card("Negative", f"{neg:,}", pct, SENTIMENT_COLORS["Negative"])
    with c4:
        pct = f"{(neu / total * 100):.0f}% of corpus" if total else "—"
        metric_card("Neutral", f"{neu:,}", pct, SENTIMENT_COLORS["Neutral"])

    st.write("")
    left, right = st.columns(2)
    with left:
        panel_start("Sentiment mix", "Share of Positive, Neutral, and Negative reviews.")
        if total:
            _show_chart(sentiment_pie(sentiment))
        else:
            st.info("Run `python -m analysis.sentiment` to generate sentiment.csv.")
        panel_end()

    with right:
        panel_start("Theme frequency", "Top themes by number of supporting reviews.")
        if not themes.empty:
            _show_chart(theme_frequency_chart(themes, top_n=8))
        else:
            st.info("Run `python -m analysis.themes` to generate themes.csv.")
        panel_end()

    m1, m2, m3 = st.columns(3)
    with m1:
        metric_card("Themes discovered", f"{len(themes):,}", "BERTopic clusters")
    with m2:
        n_seg = segments["Segment"].nunique() if not segments.empty else 0
        metric_card("User segments", f"{n_seg:,}", "KMeans behavioral groups")
    with m3:
        n_ins = len(insights.get("insights", []))
        metric_card("Product insights", f"{n_ins:,}", "LLM research answers")

    st.write("")
    d1, d2, d3 = st.columns(3)
    with d1:
        download_csv_button(sentiment, "sentiment.csv", "Download sentiment CSV")
    with d2:
        download_csv_button(themes.drop(columns=["Display name"], errors="ignore"), "themes.csv", "Download themes CSV")
    with d3:
        download_csv_button(segments, "user_segments.csv", "Download segments CSV")


def render_themes(themes: pd.DataFrame) -> None:
    page_header(
        "Top Themes",
        "Dominant discussion themes mined from reviews, with keywords and representative evidence.",
    )
    if themes.empty:
        st.warning("No themes found. Generate with `python -m analysis.themes`.")
        return

    theme_names = themes["Display name"].tolist()
    if "theme_filter" not in st.session_state:
        st.session_state.theme_filter = theme_names
    # Drop stale selections after data refresh
    st.session_state.theme_filter = [t for t in st.session_state.theme_filter if t in theme_names]
    if not st.session_state.theme_filter:
        st.session_state.theme_filter = theme_names

    f1, f2, f3 = st.columns([1.4, 1.4, 0.8])
    with f1:
        search = st.text_input(
            "Search themes",
            placeholder="Search by theme, keywords, or evidence…",
            key="theme_search",
        )
    with f2:
        selected = st.multiselect(
            "Filter themes",
            options=theme_names,
            key="theme_filter",
            help="Select one or more themes to focus the chart and table.",
        )
    with f3:
        st.write("")
        st.write("")
        download_csv_button(
            themes.drop(columns=["Display name"], errors="ignore"),
            "themes.csv",
            "Download CSV",
        )

    filtered = themes[themes["Display name"].isin(selected)] if selected else themes.iloc[0:0]
    filtered = filter_dataframe_search(
        filtered,
        search,
        ["Display name", "Theme name", "Representative keywords", "Representative reviews"],
    )

    min_reviews = int(themes["Number of reviews"].min()) if len(themes) else 0
    max_reviews = int(themes["Number of reviews"].max()) if len(themes) else 1
    if max_reviews <= min_reviews:
        max_reviews = min_reviews + 1
    review_range = st.slider(
        "Minimum reviews",
        min_value=min_reviews,
        max_value=max_reviews,
        value=min_reviews,
        key="theme_min_reviews",
    )
    filtered = filtered[filtered["Number of reviews"] >= review_range]

    panel_start("Theme frequency", f"Showing {len(filtered)} of {len(themes)} themes.")
    if filtered.empty:
        st.info("No themes match the current search / filters.")
    else:
        _show_chart(theme_frequency_chart(filtered, top_n=max(12, len(filtered))))
    panel_end()

    st.subheader("Theme detail")
    show = filtered.copy()
    show = show.rename(
        columns={
            "Display name": "Theme",
            "Number of reviews": "Reviews",
            "Representative keywords": "Keywords",
            "Representative reviews": "Evidence",
        }
    )
    st.dataframe(
        show[["Theme", "Reviews", "Keywords", "Evidence"]] if not show.empty else show,
        use_container_width=True,
        hide_index=True,
        height=420,
    )


def render_sentiment(sentiment: pd.DataFrame) -> None:
    page_header(
        "Sentiment",
        "Classifier output from the Hugging Face sentiment-analysis pipeline.",
    )
    if sentiment.empty:
        st.warning("No sentiment data. Generate with `python -m analysis.sentiment`.")
        return

    total = len(sentiment)
    pos = int((sentiment["Sentiment"] == "Positive").sum())
    neg = int((sentiment["Sentiment"] == "Negative").sum())
    neu = int((sentiment["Sentiment"] == "Neutral").sum())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Total Reviews", f"{total:,}")
    with c2:
        metric_card("Positive", f"{pos:,}", accent=SENTIMENT_COLORS["Positive"])
    with c3:
        metric_card("Negative", f"{neg:,}", accent=SENTIMENT_COLORS["Negative"])
    with c4:
        metric_card("Neutral", f"{neu:,}", accent=SENTIMENT_COLORS["Neutral"])

    st.write("")
    left, right = st.columns([1.1, 1])
    with left:
        panel_start("Sentiment pie chart", "Distribution across Positive / Neutral / Negative.")
        _show_chart(sentiment_pie(sentiment))
        panel_end()
    with right:
        avg_conf = float(sentiment["Confidence Score"].mean()) if "Confidence Score" in sentiment.columns else 0.0
        panel_start("Confidence", "Mean model confidence by class.")
        _show_chart(confidence_bar(sentiment))
        st.caption(f"Overall mean confidence: {avg_conf:.3f}")
        panel_end()

    s1, s2, s3 = st.columns([1.6, 1.1, 0.8])
    with s1:
        search = st.text_input(
            "Search reviews",
            placeholder="Search review text…",
            key="sentiment_search",
        )
    with s2:
        sentiment_filter = st.multiselect(
            "Filter sentiment",
            options=["Positive", "Neutral", "Negative"],
            default=["Positive", "Neutral", "Negative"],
            key="sentiment_filter",
        )
    with s3:
        st.write("")
        st.write("")
        download_csv_button(sentiment, "sentiment.csv", "Download CSV")

    view = sentiment[sentiment["Sentiment"].isin(sentiment_filter)] if sentiment_filter else sentiment.iloc[0:0]
    view = filter_dataframe_search(view, search, ["Review", "Sentiment"])

    st.subheader("Classified reviews")
    st.caption(f"{len(view):,} of {len(sentiment):,} reviews")
    st.dataframe(
        view[["Review", "Sentiment", "Confidence Score"]],
        use_container_width=True,
        hide_index=True,
        height=360,
    )


def render_segments(segments: pd.DataFrame) -> None:
    page_header(
        "User Segments",
        "Behavioral clusters: Routine Buyers, Explorers, Price Sensitive, and Impulse Buyers.",
    )
    if segments.empty:
        st.warning("No segments found. Generate with `python -m analysis.segments`.")
        return

    counts = segments["Segment"].value_counts()
    cols = st.columns(min(4, max(1, len(counts))))
    for i, (name, count) in enumerate(counts.items()):
        with cols[i % len(cols)]:
            metric_card(str(name), f"{int(count):,}", "assigned reviews", SEGMENT_COLORS.get(str(name)))

    st.write("")
    left, right = st.columns([1.1, 1])
    with left:
        panel_start("Segment distribution", "Review count by user segment.")
        _show_chart(segment_bar(segments))
        panel_end()
    with right:
        panel_start("Segment notes", "How each cluster was labeled.")
        for name, group in segments.groupby("Segment"):
            rationale = str(group["Label Rationale"].iloc[0]) if "Label Rationale" in group.columns else ""
            short = rationale[:220] + ("…" if len(rationale) > 220 else "")
            st.markdown(f"**{name}** · {len(group)} reviews")
            st.caption(short)
        panel_end()

    s1, s2, s3 = st.columns([1.6, 1.1, 0.8])
    with s1:
        search = st.text_input(
            "Search assignments",
            placeholder="Search review text or segment…",
            key="segment_search",
        )
    with s2:
        seg_opts = sorted(segments["Segment"].dropna().unique().tolist())
        seg_filter = st.multiselect(
            "Filter segments",
            options=seg_opts,
            default=seg_opts,
            key="segment_filter",
        )
    with s3:
        st.write("")
        st.write("")
        download_csv_button(segments, "user_segments.csv", "Download CSV")

    view = segments[segments["Segment"].isin(seg_filter)] if seg_filter else segments.iloc[0:0]
    view = filter_dataframe_search(view, search, ["Review", "Segment", "Label Rationale"])

    st.subheader("Segment assignments")
    st.caption(f"{len(view):,} of {len(segments):,} assignments")
    cols_show = [c for c in ["Review", "Segment", "Cluster ID", "Prototype Similarity"] if c in view.columns]
    st.dataframe(view[cols_show], use_container_width=True, hide_index=True, height=360)


def render_insights(insights_payload: dict) -> None:
    page_header(
        "Generated Product Insights",
        "LLM answers to the seven discovery research questions, grounded in themes and segments.",
    )
    insights = insights_payload.get("insights") or []
    if not insights:
        st.warning("No insights found. Generate with `python -m llm.insights`.")
        return

    meta = insights_payload.get("meta") or {}
    st.caption(
        f"Source: {meta.get('source', 'llm')} · model: {meta.get('model', '—')} · "
        f"{len(insights)} insights"
    )

    i1, i2, i3 = st.columns([1.6, 1.0, 0.8])
    with i1:
        search = st.text_input(
            "Search insights",
            placeholder="Search title, evidence, opportunity…",
            key="insight_search",
        )
    with i2:
        priorities = sorted(
            {str(x.get("Priority", "Medium")).strip().title() for x in insights},
            key=lambda p: PRIORITY_ORDER.get(p, 9),
        )
        priority_filter = st.multiselect(
            "Filter priority",
            options=priorities or ["High", "Medium", "Low"],
            default=priorities or ["High", "Medium", "Low"],
            key="insight_priority_filter",
        )
    with i3:
        st.write("")
        st.write("")
        export_df = pd.DataFrame(insights)
        download_csv_button(export_df, "insights.csv", "Download CSV")

    q = (search or "").strip().lower()
    for item in insights:
        priority = str(item.get("Priority", "Medium")).strip().title()
        if priority_filter and priority not in priority_filter:
            continue
        blob = " ".join(
            str(item.get(k, ""))
            for k in ("Title", "Question", "Evidence", "Business Impact", "Opportunity", "Recommendation")
        ).lower()
        if q and q not in blob:
            continue
        css = f"priority-{priority.lower()}"
        badge = priority.lower()
        st.markdown(
            f"""
            <div class="insight-card {css}">
              <span class="badge {badge}">{html.escape(priority)} priority</span>
              <div class="insight-title">{html.escape(str(item.get("Title", "Untitled")))}</div>
              <div class="field-label">Research question</div>
              <div class="field-body">{html.escape(str(item.get("Question", "")))}</div>
              <div class="field-label">Evidence</div>
              <div class="field-body">{html.escape(str(item.get("Evidence", "")))}</div>
              <div class="field-label">Business impact</div>
              <div class="field-body">{html.escape(str(item.get("Business Impact", "")))}</div>
              <div class="field-label">Opportunity</div>
              <div class="field-body">{html.escape(str(item.get("Opportunity", "")))}</div>
              <div class="field-label">Recommendation</div>
              <div class="field-body">{html.escape(str(item.get("Recommendation", "")))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_opportunities(insights_payload: dict) -> None:
    page_header(
        "Opportunity Ranking",
        "Prioritized product opportunities ordered by urgency and strategic impact.",
    )
    insights = insights_payload.get("insights") or []
    ranked = rank_opportunities(insights)
    if ranked.empty:
        st.warning("No opportunities to rank. Generate insights first.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Ranked opportunities", f"{len(ranked):,}")
    with c2:
        metric_card("High priority", f"{int((ranked['Priority'] == 'High').sum()):,}", accent=PRIORITY_COLORS["High"])
    with c3:
        metric_card("Medium / Low", f"{int((ranked['Priority'] != 'High').sum()):,}")

    o1, o2 = st.columns([2.2, 0.8])
    with o1:
        search = st.text_input(
            "Search opportunities",
            placeholder="Search title, opportunity, or recommendation…",
            key="opportunity_search",
        )
    with o2:
        st.write("")
        st.write("")
        download_csv_button(ranked, "opportunity_ranking.csv", "Download CSV")

    view = filter_dataframe_search(
        ranked,
        search,
        ["Title", "Opportunity", "Recommendation", "Business Impact", "Priority"],
    )

    panel_start("Opportunity scoreboard", "Score reflects priority band with stable ordering within band.")
    if view.empty:
        st.info("No opportunities match the current search.")
    else:
        _show_chart(opportunity_rank_chart(view))
    panel_end()

    st.subheader("Ranked backlog")
    for _, row in view.iterrows():
        badge = str(row["Priority"]).lower()
        st.markdown(
            f"""
            <div class="insight-card priority-{badge}">
              <span class="badge {badge}">#{int(row["Rank"])} · {html.escape(str(row["Priority"]))}</span>
              <div class="insight-title">{html.escape(str(row["Title"]))}</div>
              <div class="field-label">Opportunity</div>
              <div class="field-body">{html.escape(str(row["Opportunity"]))}</div>
              <div class="field-label">Recommendation</div>
              <div class="field-body">{html.escape(str(row["Recommendation"]))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _workflow_counts() -> dict[str, int]:
    """Live artifact counts for the methodology workflow diagram."""
    counts = {
        "play_raw": 0,
        "app_store_raw": 0,
        "reddit_raw": 0,
        "youtube_raw": 0,
        "merged": 0,
        "cleaned": 0,
        "sentiment": 0,
        "themes": 0,
        "segments": 0,
        "exploration": 0,
        "exploration_relevant": 0,
        "insights": 0,
        "category_ops": 0,
        "gold": 0,
    }
    play = DATA_RAW / "blinkit_play_reviews.csv"
    app_store = DATA_RAW / "app_store_reviews.csv"
    reddit = DATA_RAW / "reddit_posts.csv"
    youtube = DATA_RAW / "youtube_comments.csv"
    merged = DATA_PROC / "merged_reviews.csv"
    cleaned = DATA_PROC / "preprocessed_reviews.csv"
    gold = ROOT / "data" / "gold" / "gold_labels.jsonl"
    try:
        if play.exists():
            counts["play_raw"] = len(pd.read_csv(play))
        if app_store.exists():
            counts["app_store_raw"] = len(pd.read_csv(app_store))
        if reddit.exists():
            counts["reddit_raw"] = len(pd.read_csv(reddit))
        if youtube.exists():
            counts["youtube_raw"] = len(pd.read_csv(youtube))
        if merged.exists():
            counts["merged"] = len(pd.read_csv(merged))
        if cleaned.exists():
            counts["cleaned"] = len(pd.read_csv(cleaned))
        if (OUTPUT / "sentiment.csv").exists():
            counts["sentiment"] = len(pd.read_csv(OUTPUT / "sentiment.csv"))
        if (OUTPUT / "themes.csv").exists():
            counts["themes"] = len(pd.read_csv(OUTPUT / "themes.csv"))
        if (OUTPUT / "user_segments.csv").exists():
            counts["segments"] = len(pd.read_csv(OUTPUT / "user_segments.csv"))
        if (OUTPUT / "exploration_tags.csv").exists():
            edf = pd.read_csv(OUTPUT / "exploration_tags.csv")
            counts["exploration"] = len(edf)
            if "is_relevant" in edf.columns:
                counts["exploration_relevant"] = int(edf["is_relevant"].sum())
        if (OUTPUT / "insights.json").exists():
            payload = json.loads((OUTPUT / "insights.json").read_text(encoding="utf-8"))
            counts["insights"] = len(payload.get("insights") or [])
        if (OUTPUT / "synthesis.json").exists():
            syn = json.loads((OUTPUT / "synthesis.json").read_text(encoding="utf-8"))
            counts["category_ops"] = len(syn.get("category_opportunities") or [])
        if gold.exists():
            counts["gold"] = sum(1 for _ in gold.open(encoding="utf-8") if _.strip())
    except Exception:
        pass
    return counts


def render_end_to_end_workflow() -> None:
    """Simple, user-friendly downward flowchart of the research pipeline."""
    c = _workflow_counts()
    sources = (
        f"Play {c['play_raw']:,} · App Store {c['app_store_raw']:,} · "
        f"Reddit {c['reddit_raw']:,} · YouTube {c['youtube_raw']:,}"
    )

    def step(num: str, title: str, detail: str, pills: str = "") -> str:
        pill_html = f'<div class="pills">{pills}</div>' if pills else ""
        return f"""
    <div class="simple-step">
      <div class="n">{html.escape(num)}</div>
      <div class="t">{html.escape(title)}</div>
      <p class="d">{html.escape(detail)}</p>
      {pill_html}
    </div>"""

    arrow = '<div class="simple-arrow">↓</div>'

    html_block = f"""
<div class="flow-wrap">
  <div class="simple-flow">
    {step("1", "Collect feedback", "Gather what people say about Blinkit from app stores and online communities.", f'<span class="flow-pill">{html.escape(sources)}</span>')}
    {arrow}
    {step("2", "Combine into one dataset", "Merge every source into a single clean review list, without duplicates.", f'<span class="flow-pill">{c["merged"]:,} reviews ready</span>')}
    {arrow}
    {step("3", "Find patterns", "Spot recurring themes, sentiment, and shopper groups in the feedback.", f'<span class="flow-pill">{c["themes"]:,} themes</span><span class="flow-pill">{c["sentiment"]:,} sentiment labels</span><span class="flow-pill">{c["segments"]:,} segments</span>')}
    {arrow}
    {step("4", "Tag exploration barriers", "Flag reviews that explain why people stick to familiar categories.", f'<span class="flow-pill">{c["exploration_relevant"]:,} relevant of {c["exploration"]:,}</span>')}
    {arrow}
    {step("5", "Turn findings into actions", "Summarize barriers, jobs-to-be-done, experiments, and category opportunities.", f'<span class="flow-pill">{c["category_ops"]:,} category opportunities</span>')}
    {arrow}
    {step("6", "Explore in the dashboard", "Use Findings, Opportunities, and Validation to review evidence and decide what to test next.", '<span class="flow-pill">Findings Board</span><span class="flow-pill">Category Opportunities</span><span class="flow-pill">Validation Desk</span>')}
  </div>
  <div class="flow-note">Refresh data anytime with <code>python main.py --skip-collect</code>, then click Refresh in the sidebar.</div>
</div>
"""

    panel_start(
        "How the insight engine works",
        "A simple 6-step path from raw feedback to product decisions.",
    )
    st.markdown(html_block, unsafe_allow_html=True)
    panel_end()


def render_methodology() -> None:
    """Explain data workflow, theme mining, insight generation, and validation."""
    page_header(
        "Methodology",
        "A plain-language look at how feedback becomes exploration insights.",
    )

    render_end_to_end_workflow()

    panel_start(
        "1. How your workflow gathers and analyzes data",
        "End-to-end path from raw feedback to dashboard-ready outputs.",
    )
    st.markdown(
        """
**Collection**
- **Google Play reviews** are scraped for the Blinkit app (`com.grofers.customerapp`) via `google-play-scraper` and stored as `data/raw/blinkit_play_reviews.csv` (Review, Rating, Date, Helpful Count).
- **Reddit posts** (optional) are collected with PRAW when `REDDIT_*` credentials exist, then merged into the same clean corpus.

**Cleaning & preparation** (`python main.py`)
1. Soft-dedupe (casefold + whitespace) so empty and exact duplicates are removed without over-collapsing near-duplicates.
2. NLP preprocess (normalize text; keep a fallback for short/emoji-only reviews).
3. Write `data/processed/preprocessed_reviews.csv`.

**Analysis stages (in order)**
| Stage | Method | Output |
| --- | --- | --- |
| Embeddings | TF-IDF (default) or sentence-transformers | `data/processed/review_embeddings.npy` |
| Themes | BERTopic on embeddings | `output/themes.csv` |
| Sentiment | Hugging Face `sentiment-analysis` pipeline (3-class) | `output/sentiment.csv` |
| Segments | KMeans (k=4) + prototype labeling | `output/user_segments.csv` |
| Insights | LLM grounded on themes/sentiment/segments (heuristic fallback) | `output/insights.json` |

The Streamlit dashboard and Theme Explorer API read these artifacts — they do not re-scrape on every page load.
        """
    )
    panel_end()

    panel_start(
        "2. How themes are identified",
        "Unsupervised topic discovery over the review corpus.",
    )
    st.markdown(
        """
**Approach:** BERTopic clusters review embeddings, then names each cluster from its top keywords.

**Steps**
1. Encode each cleaned review into a vector (TF-IDF + truncated SVD by default for speed/reliability; optional MiniLM backend).
2. Fit BERTopic (UMAP → HDBSCAN → c-TF-IDF topic words).
3. For every topic, export:
   - **Theme name** — top representative keywords
   - **Number of reviews** assigned to the topic
   - **Representative keywords**
   - **Representative reviews** — short evidence snippets

**Why this design**
- Scales from hundreds to thousands of reviews without hand-coding categories.
- Keywords + example reviews keep themes **inspectable** on the Top Themes page.
- Outlier / mixed reviews are kept as an explicit bucket rather than forced into a weak topic.
        """
    )
    panel_end()

    panel_start(
        "3. How insights are generated",
        "Structured answers to seven discovery research questions.",
    )
    st.markdown(
        """
**Research questions answered**
1. Why do users repeatedly buy from the same categories?
2. What prevents users from exploring new categories?
3. How do users discover products today?
4. What role do habits play?
5. What frustrations emerge repeatedly?
6. Which user segments experiment more?
7. What unmet needs emerge?

**Generation path** (`llm/insights.py`)
1. Build a compact **evidence context** from `themes.csv`, `sentiment.csv`, and `user_segments.csv` (theme sizes, keywords, sentiment mix, segment counts).
2. Call an OpenAI-compatible LLM (Groq-compatible when `OPENAI_API_KEY` is a `gsk_` key) with a JSON schema requiring, for each insight:
   - Title, Evidence, Business Impact, Opportunity, Priority, Recommendation
3. If the LLM is unavailable or returns invalid JSON, generate **grounded fallback insights** from the same CSV evidence (no invented quotes).

**Dashboard use**
- Product Insights shows the full cards.
- Opportunity Ranking sorts by Priority (High → Medium → Low) for a backlog view.
        """
    )
    panel_end()

    panel_start(
        "4. How you validated the quality of the insights",
        "Phase 0 quality gates plus runtime checks that keep insights tied to evidence.",
    )
    st.markdown(
        """
**Offline / Phase 0 validation** (see `docs/METRICS.md`, `docs/LABELING_GUIDE.md`)
- **Gold labels** (`data/gold/gold_labels.jsonl`) define barrier/category/spam expectations.
- **Quote grounding rate** — insights or themes must cite evidence that appears in source text (target ≥ 0.95 on held gold).
- **Barrier / category F1** — multi-label offline metrics before promoting a model version.
- **Eval harness** — `python -m scripts.run_eval` / `python -m scripts.validate_gold` for schema and metric checks.

**Runtime / pipeline validation**
- Insights are generated only after themes exist; sentiment and segments are attached when present.
- LLM outputs are schema-normalized (required fields, Priority ∈ High/Medium/Low).
- Invalid or empty LLM responses fall back to heuristic insights built from the same CSVs — so the dashboard never depends on an ungrounded free-form answer.
- Sentiment uses a calibrated 3-class model with confidence scores; segments require ≥ 4 reviews and publish a **label rationale** per cluster.
- Soft-dedupe + non-empty review filters reduce spam/empty noise before modeling.

**Human review loop**
- Theme Explorer / Streamlit tables expose representative reviews so product partners can spot-check themes and insight Evidence fields against real quotes before acting on recommendations.
        """
    )
    panel_end()

    st.caption("Entrypoints: `python main.py` · `python -m streamlit run app.py` · docs in `docs/`.")


# ---------------------------------------------------------------------------
# New IA pages (primary question oriented)
# ---------------------------------------------------------------------------


def render_findings_board(synthesis: dict, insights: dict) -> None:
    page_header(
        "Findings Board",
        "Synthesis for category exploration — barriers, JTBD, unmet needs, and testable experiments.",
    )
    if not synthesis:
        st.info("Run `python main.py --skip-collect` (or full collect) to generate `output/synthesis.json`.")
        if insights.get("insights"):
            st.markdown("#### Legacy insight cards still available")
            render_insights(insights)
        return

    corpus = synthesis.get("corpus") or {}
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Corpus", f"{corpus.get('total_reviews', 0):,}", "multi-source items")
    with c2:
        metric_card(
            "Exploration-relevant",
            f"{corpus.get('exploration_relevant', 0):,}",
            f"rate {corpus.get('relevance_rate', 0):.0%}",
        )
    with c3:
        metric_card("Barriers ranked", str(len(synthesis.get("barriers_ranked") or [])), "friction taxonomy")
    with c4:
        metric_card(
            "Experiments",
            str(len(synthesis.get("testable_experiments") or [])),
            "ready to instrument",
        )

    panel_start("Executive summary", "Grounded in exploration-tagged corpus counts")
    st.markdown(synthesis.get("executive_summary") or "_No summary yet._")
    panel_end()

    barriers = synthesis.get("barriers_ranked") or []
    if barriers:
        panel_start("Barriers ranked", "Why users avoid new categories")
        bdf = pd.DataFrame(barriers)
        fig = go.Figure(
            go.Bar(
                x=bdf["mentions"],
                y=[str(b).replace("_", " ") for b in bdf["barrier"]],
                orientation="h",
                marker_color="#039855",
                text=bdf["mentions"],
                textposition="outside",
            )
        )
        fig.update_layout(yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=10, b=10))
        _show_chart(_chart_layout(fig, height=max(280, 40 * len(bdf))))
        panel_end()

    col_a, col_b = st.columns(2)
    with col_a:
        panel_start("Jobs to be done", "Situations → outcomes for exploration")
        for job in synthesis.get("jobs_to_be_done") or []:
            st.markdown(f"**{job.get('job', '')}**")
            st.caption(
                f"Situation: {job.get('situation', '')} · Outcome: {job.get('desired_outcome', '')}"
            )
            st.markdown(f"_Workaround today:_ {job.get('current_workaround', '')}")
            st.markdown("---")
        panel_end()
    with col_b:
        panel_start("Unmet needs", "Prioritized gaps the product must close")
        for need in synthesis.get("unmet_needs") or []:
            st.markdown(f"**[{need.get('priority', 'P1')}] {need.get('need', '')}**")
            st.caption(need.get("pain", ""))
            st.markdown(f"<span style='opacity:.7'>Evidence: {html.escape(str(need.get('evidence', '')))}</span>", unsafe_allow_html=True)
            st.markdown("---")
        panel_end()

    panel_start("Hypotheses", "Falsifiable statements tied to barriers")
    for h in synthesis.get("hypotheses") or []:
        st.markdown(f"**{h.get('id')}** — {h.get('statement')}")
        st.caption(
            f"Barrier: {str(h.get('linked_barrier', '')).replace('_', ' ')} · "
            f"mentions={h.get('evidence_mentions', 0)} · status={h.get('status', 'open')}"
        )
    panel_end()

    panel_start("Testable experiments", "Interventions with primary metric + guardrail")
    for exp in synthesis.get("testable_experiments") or []:
        with st.expander(f"{exp.get('id')} · {exp.get('name')}", expanded=False):
            st.markdown(f"**Intervention:** {exp.get('intervention')}")
            st.markdown(f"**Primary metric:** {exp.get('primary_metric')}")
            st.markdown(f"**Guardrail:** {exp.get('guardrail')}")
            st.caption(
                f"Barrier: {str(exp.get('barrier', '')).replace('_', ' ')} · "
                f"Hypothesis: {exp.get('hypothesis_link')} · {exp.get('sample_size_note')}"
            )
    panel_end()


def render_category_opportunities(synthesis: dict) -> None:
    page_header(
        "Category Opportunities",
        "Where blocked exploration intent concentrates — ranked for product bets.",
    )
    ops = (synthesis or {}).get("category_opportunities") or []
    if not ops:
        with st.spinner("Building category opportunities from exploration tags…"):
            synthesis = ensure_synthesis(force=True)
            ops = (synthesis or {}).get("category_opportunities") or []
    if not ops:
        st.warning(
            "Still no category opportunities. Ensure `data/processed/merged_reviews.csv` exists, "
            "then click **Refresh data** in the sidebar or run `python -m llm.synthesis --no-polish`."
        )
        if st.button("Generate synthesis now", type="primary"):
            with st.spinner("Generating…"):
                ensure_synthesis(force=True)
            st.cache_data.clear()
            st.rerun()
        return
    df = pd.DataFrame(ops)
    c1, c2 = st.columns([1.2, 1])
    with c1:
        panel_start("Opportunity score by category")
        fig = go.Figure(
            go.Bar(
                x=df["opportunity_score"],
                y=df["category"],
                orientation="h",
                marker_color="#2E90FA",
                text=df["opportunity_score"],
                textposition="outside",
            )
        )
        fig.update_layout(yaxis=dict(autorange="reversed"))
        _show_chart(_chart_layout(fig, height=360))
        panel_end()
    with c2:
        panel_start("Why these categories")
        for row in ops:
            st.markdown(
                f"**#{row.get('rank')} {row.get('category')}** · score {row.get('opportunity_score')}"
            )
            st.caption(
                f"{row.get('why_now')} · attack `{str(row.get('primary_barrier_to_attack', '')).replace('_', ' ')}` "
                f"via `{row.get('suggested_experiment')}`"
            )
            st.markdown("---")
        panel_end()
    download_csv_button(df, "category_opportunities.csv", "Download category opportunities")


def build_hypothesis_triangulation(
    exploration: pd.DataFrame,
    synthesis: dict,
) -> pd.DataFrame:
    """Per-hypothesis evidence counts + cross-source triangulation from tagged corpus."""
    hyps = (synthesis or {}).get("hypotheses") or []
    if not hyps:
        return pd.DataFrame()

    df = exploration.copy()
    if df.empty:
        return pd.DataFrame()

    text_col = "text" if "text" in df.columns else ("Review" if "Review" in df.columns else None)
    has_source = "source" in df.columns
    has_barriers = "barriers" in df.columns

    rows = []
    for h in hyps:
        barrier = str(h.get("linked_barrier") or "").strip()
        hid = str(h.get("id") or "")
        statement = str(h.get("statement") or "")

        if has_barriers and barrier:
            mask = df["barriers"].fillna("").astype(str).str.contains(
                rf"(?:^|\|){re.escape(barrier)}(?:\||$)",
                regex=True,
                na=False,
            )
            matched = df.loc[mask]
        else:
            matched = df.iloc[0:0]

        n = int(len(matched))
        source_counts: dict[str, int] = {}
        if has_source and n:
            source_counts = {
                str(k): int(v) for k, v in matched["source"].fillna("unknown").value_counts().items()
            }
        n_sources = len(source_counts)

        if n >= 50 and n_sources >= 3:
            strength = "Strong"
        elif n >= 15 and n_sources >= 2:
            strength = "Moderate"
        elif n >= 5:
            strength = "Emerging"
        else:
            strength = "Weak"

        quote = ""
        if n and text_col:
            # Prefer a relevant blocked/stuck example when available
            prefer = matched
            if "exploration_signal" in matched.columns:
                prefer = matched[
                    matched["exploration_signal"].isin(
                        ["want_to_explore_blocked", "stuck_in_routine", "explored_new"]
                    )
                ]
                if prefer.empty:
                    prefer = matched
            sample = str(prefer.iloc[0][text_col] or "").strip().replace("\n", " ")
            quote = sample[:160] + ("…" if len(sample) > 160 else "")

        breakdown = ", ".join(f"{k}:{v}" for k, v in sorted(source_counts.items(), key=lambda x: -x[1]))
        rows.append(
            {
                "Hypothesis": hid,
                "Statement": statement,
                "Linked barrier": barrier.replace("_", " "),
                "Evidence mentions": n,
                "Sources with evidence": n_sources,
                "Source breakdown": breakdown or "—",
                "Triangulation": strength,
                "Example quote": quote or "—",
                "Status": str(h.get("status") or "open"),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        by=["Evidence mentions", "Sources with evidence"],
        ascending=[False, False],
    ).reset_index(drop=True)


def render_validation_desk(exploration: pd.DataFrame, merged: pd.DataFrame, synthesis: dict) -> None:
    page_header(
        "Validation Desk",
        "Check relevance filters, signal mix, and sample evidence before trusting synthesis.",
    )
    if exploration.empty:
        st.warning("Missing `output/exploration_tags.csv`. Run exploration tagging via `main.py`.")
        return

    corpus = (synthesis or {}).get("corpus") or {}
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Tagged rows", f"{len(exploration):,}", "exploration_tags.csv")
    with c2:
        rel = int(exploration["is_relevant"].sum()) if "is_relevant" in exploration.columns else 0
        metric_card("Relevant", f"{rel:,}", f"{rel / max(len(exploration), 1):.0%} of corpus")
    with c3:
        sources = corpus.get("by_source") or (
            exploration["source"].value_counts().to_dict() if "source" in exploration.columns else {}
        )
        metric_card("Sources", str(len(sources)), ", ".join(list(sources.keys())[:4]) or "—")

    # Per-hypothesis evidence & triangulation
    hyp_table = build_hypothesis_triangulation(exploration, synthesis or {})
    panel_start(
        "Per-hypothesis evidence & triangulation",
        "Mentions of each linked barrier, spread across sources, with an example quote.",
    )
    if hyp_table.empty:
        st.info("No hypotheses in synthesis yet. Generate synthesis to populate this table.")
    else:
        strong = int((hyp_table["Triangulation"] == "Strong").sum())
        moderate = int((hyp_table["Triangulation"] == "Moderate").sum())
        m1, m2, m3 = st.columns(3)
        with m1:
            metric_card("Hypotheses", str(len(hyp_table)), "from synthesis.json")
        with m2:
            metric_card("Strong triangulation", str(strong), "≥50 mentions · ≥3 sources")
        with m3:
            metric_card("Moderate triangulation", str(moderate), "≥15 mentions · ≥2 sources")

        st.dataframe(
            hyp_table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Statement": st.column_config.TextColumn("Statement", width="large"),
                "Example quote": st.column_config.TextColumn("Example quote", width="large"),
                "Evidence mentions": st.column_config.NumberColumn(format="%d"),
                "Sources with evidence": st.column_config.NumberColumn(format="%d"),
            },
        )
        download_csv_button(
            hyp_table,
            "hypothesis_evidence_triangulation.csv",
            "Download hypothesis evidence table",
        )
    panel_end()

    if "exploration_signal" in exploration.columns:
        panel_start("Exploration signal mix")
        counts = exploration["exploration_signal"].value_counts()
        fig = go.Figure(
            go.Pie(
                labels=[str(x).replace("_", " ") for x in counts.index],
                values=counts.values,
                hole=0.55,
                marker=dict(colors=[SIGNAL_COLORS.get(str(k), "#98A2B3") for k in counts.index]),
            )
        )
        _show_chart(_chart_layout(fig, height=360))
        panel_end()

    if not merged.empty and "source" in merged.columns:
        panel_start("Multi-source corpus coverage")
        src = merged["source"].value_counts().reset_index()
        src.columns = ["source", "count"]
        fig = px.bar(src, x="source", y="count", color="source")
        fig.update_layout(showlegend=False)
        _show_chart(_chart_layout(fig, height=320))
        panel_end()

    panel_start("Sample tagged evidence", "Spot-check relevance and barriers")
    view = exploration.copy()
    if "is_relevant" in view.columns:
        only_rel = st.checkbox("Relevant only", value=True)
        if only_rel:
            view = view[view["is_relevant"] == True]  # noqa: E712
    signal_opts = (
        sorted(view["exploration_signal"].dropna().unique().tolist())
        if "exploration_signal" in view.columns
        else []
    )
    pick = st.multiselect("Signals", signal_opts, default=signal_opts[:3] if signal_opts else [])
    if pick and "exploration_signal" in view.columns:
        view = view[view["exploration_signal"].isin(pick)]
    text_col = "text" if "text" in view.columns else ("Review" if "Review" in view.columns else None)
    cols = [
        c
        for c in [
            text_col,
            "source",
            "exploration_signal",
            "barriers",
            "categories_mentioned",
            "relevance_reason",
        ]
        if c and c in view.columns
    ]
    st.dataframe(view[cols].head(80) if cols else view.head(80), use_container_width=True, hide_index=True)
    download_csv_button(view.head(500), "validation_sample.csv", "Download sample")
    panel_end()


def _file_meta(path: Path) -> dict[str, str]:
    if not path.exists():
        return {"status": "missing", "mtime": "—", "size": "—"}
    st_info = path.stat()
    mtime = datetime.fromtimestamp(st_info.st_mtime).strftime("%Y-%m-%d %H:%M")
    size = f"{st_info.st_size / 1024:.1f} KB"
    return {"status": "ready", "mtime": mtime, "size": size}


def render_live_pipeline() -> None:
    page_header(
        "Live Pipeline",
        "Artifact freshness across collect → merge → tag → synthesis. Re-run from the terminal.",
    )
    stages = [
        ("Play Store raw", DATA_RAW / "blinkit_play_reviews.csv"),
        ("App Store raw", DATA_RAW / "app_store_reviews.csv"),
        ("Reddit raw", DATA_RAW / "reddit_posts.csv"),
        ("YouTube raw", DATA_RAW / "youtube_comments.csv"),
        ("Merged corpus", DATA_PROC / "merged_reviews.csv"),
        ("Cleaned corpus", DATA_PROC / "preprocessed_reviews.csv"),
        ("Themes", OUTPUT / "themes.csv"),
        ("Sentiment", OUTPUT / "sentiment.csv"),
        ("Segments", OUTPUT / "user_segments.csv"),
        ("Exploration tags", OUTPUT / "exploration_tags.csv"),
        ("Insights JSON", OUTPUT / "insights.json"),
        ("Synthesis JSON", OUTPUT / "synthesis.json"),
    ]
    rows = []
    for name, path in stages:
        meta = _file_meta(path)
        rows.append({"Stage": name, "Path": str(path.relative_to(ROOT)), **meta})
    df = pd.DataFrame(rows)
    ready = int((df["status"] == "ready").sum())
    c1, c2 = st.columns(2)
    with c1:
        metric_card("Artifacts ready", f"{ready}/{len(df)}", "files on disk")
    with c2:
        metric_card("Primary question", "Category exploration", PRIMARY_QUESTION[:48] + "…")

    panel_start("Stage board")
    st.dataframe(df, use_container_width=True, hide_index=True)
    panel_end()

    st.markdown(
        """
```bash
# Full refresh (network collectors)
python main.py --play-count 500

# Offline reuse of existing raw CSVs
python main.py --skip-collect

# Tag + synthesize only (after merge exists)
python -m analysis.exploration
python -m llm.synthesis --no-polish
```
        """
    )


def render_try_it_console() -> None:
    page_header(
        "Try-it Console",
        "Paste a review and see exploration signal, barriers, and category cues instantly.",
    )
    from analysis.exploration import tag_review

    samples = [
        "I always reorder the same milk and bread. Never browse other categories after work.",
        "Wanted to try electronics but prices look marked up and I don't trust quality for phones.",
        "Bought pet food for the first time on Blinkit — delivery was fine but hard to find in search.",
        "Great app!",
    ]
    pick = st.selectbox("Load sample", ["(blank)"] + samples)
    default = "" if pick == "(blank)" else pick
    text = st.text_area("Review text", value=default, height=140)
    if st.button("Tag review", type="primary"):
        result = tag_review(text)
        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("Relevant", "yes" if result["is_relevant"] else "no", result["relevance_reason"][:80])
        with c2:
            metric_card("Signal", str(result["exploration_signal"]).replace("_", " "), "")
        with c3:
            metric_card("Barriers", str(result["barriers"] or "—").replace("|", ", ").replace("_", " "), result.get("categories_mentioned") or "no categories")
        st.json(result)


def render_evidence_lab(
    sentiment: pd.DataFrame,
    themes: pd.DataFrame,
    segments: pd.DataFrame,
    insights: dict,
) -> None:
    page_header(
        "Evidence Lab",
        "Drill into themes, sentiment, segments, and legacy insight cards that feed the synthesis.",
    )
    tab1, tab2, tab3, tab4 = st.tabs(["Themes", "Sentiment", "Segments", "Legacy insights"])
    with tab1:
        if themes.empty:
            st.info("No themes.csv yet.")
        else:
            _show_chart(theme_frequency_chart(themes))
            st.dataframe(
                themes[["Display name", "Number of reviews", "Representative keywords"]].head(25)
                if "Display name" in themes.columns
                else themes.head(25),
                use_container_width=True,
                hide_index=True,
            )
    with tab2:
        if sentiment.empty:
            st.info("No sentiment.csv yet.")
        else:
            c1, c2 = st.columns([1, 1.2])
            with c1:
                _show_chart(sentiment_pie(sentiment))
            with c2:
                st.dataframe(sentiment.head(40), use_container_width=True, hide_index=True)
    with tab3:
        if segments.empty:
            st.info("No user_segments.csv yet.")
        else:
            counts = segments["Segment"].value_counts() if "Segment" in segments.columns else None
            if counts is not None:
                fig = go.Figure(
                    go.Bar(
                        x=counts.index.tolist(),
                        y=counts.values.tolist(),
                        marker_color=[SEGMENT_COLORS.get(str(x), "#2E90FA") for x in counts.index],
                    )
                )
                _show_chart(_chart_layout(fig, height=320))
            st.dataframe(segments.head(40), use_container_width=True, hide_index=True)
    with tab4:
        cards = insights.get("insights") or []
        if not cards:
            st.info("No insights.json cards yet.")
        else:
            for item in cards[:12]:
                st.markdown(f"**[{item.get('Priority', '')}] {item.get('Title', 'Untitled')}**")
                st.caption(str(item.get("Opportunity") or item.get("Insight") or "")[:240])
                st.markdown("---")


def render_admin() -> None:
    page_header("Admin", "Protected controls for cache flush and environment visibility.")
    expected = os.getenv("ADMIN_DASHBOARD_PASSWORD", "blinkit-research").strip()
    if "admin_ok" not in st.session_state:
        st.session_state.admin_ok = False
    if not st.session_state.admin_ok:
        pwd = st.text_input("Admin password", type="password")
        if st.button("Unlock"):
            if pwd == expected:
                st.session_state.admin_ok = True
                st.rerun()
            else:
                st.error("Incorrect password")
        st.caption("Set `ADMIN_DASHBOARD_PASSWORD` in `.env` (default for local demos: blinkit-research).")
        return

    st.success("Admin unlocked for this session.")
    if st.button("Clear Streamlit cache"):
        st.cache_data.clear()
        st.success("Cache cleared")
    if st.button("Lock admin"):
        st.session_state.admin_ok = False
        st.rerun()

    panel_start("Environment (non-secret)")
    keys = [
        "PHASE1_EMBEDDING",
        "OPENAI_MODEL",
        "GROQ_MODEL",
        "YOUTUBE_API_KEY",
        "REDDIT_CLIENT_ID",
        "GROQ_API_KEY",
        "OPENAI_API_KEY",
    ]
    rows = []
    for k in keys:
        val = os.getenv(k, "")
        rows.append({"key": k, "set": "yes" if val.strip() else "no"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    panel_end()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="Category Discovery Engine",
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = True

    with st.sidebar:
        st.markdown('<div class="sidebar-brand">◈ Category Discovery</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sidebar-meta">Why don’t users explore new categories?</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="nav-hint">Workspace</div>', unsafe_allow_html=True)
        page_labels = [f"{PAGE_ICONS[p]}  {p}" for p in PAGES]
        choice = st.radio("Navigate", page_labels, label_visibility="collapsed", key="nav_page")
        page = choice.split("  ", 1)[-1].strip()

        st.markdown("---")
        st.caption("Data · `output/` + `data/processed/`")
        if st.button("Refresh data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    inject_styles()

    sentiment = load_sentiment()
    themes = load_themes()
    segments = load_segments()
    insights = load_insights()
    exploration = load_exploration()
    merged = load_merged()
    # Auto-heal empty/stale synthesis so Category Opportunities never dead-ends
    if page in {"Findings Board", "Category Opportunities", "Validation Desk"}:
        synthesis = ensure_synthesis()
    else:
        synthesis = load_synthesis()

    if page == "Findings Board":
        render_findings_board(synthesis, insights)
    elif page == "Category Opportunities":
        render_category_opportunities(synthesis)
    elif page == "Validation Desk":
        render_validation_desk(exploration, merged, synthesis)
    elif page == "Live Pipeline":
        render_live_pipeline()
    elif page == "Try-it Console":
        render_try_it_console()
    elif page == "Evidence Lab":
        render_evidence_lab(sentiment, themes, segments, insights)
    elif page == "Methodology":
        render_methodology()
    elif page == "Admin":
        render_admin()


if __name__ == "__main__":
    main()
