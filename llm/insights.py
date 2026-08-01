"""
Generate product insights with an LLM from theme + sentiment analysis.

Reads:
  output/themes.csv
  output/sentiment.csv   (optional but recommended)
  output/user_segments.csv (optional — improves segment insights)

Answers:
  - Why do users repeatedly buy from the same categories?
  - What prevents users from exploring new categories?
  - How do users discover products today?
  - What role do habits play?
  - What frustrations emerge repeatedly?
  - Which user segments experiment more?
  - What unmet needs emerge?

Each insight JSON object has:
  Title, Evidence, Business Impact, Opportunity, Priority, Recommendation
  (+ Question for traceability)

Writes:
  output/insights.json

Usage (from repo root):
  python -m llm.insights
  python -m llm.insights --themes output/themes.csv --sentiment output/sentiment.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from discovery_engine.env_loader import load_env  # noqa: E402

load_env()

DEFAULT_THEMES = ROOT / "output" / "themes.csv"
DEFAULT_SENTIMENT = ROOT / "output" / "sentiment.csv"
DEFAULT_SEGMENTS = ROOT / "output" / "user_segments.csv"
DEFAULT_OUT = ROOT / "output" / "insights.json"

RESEARCH_QUESTIONS = [
    "Why do users repeatedly buy from the same categories?",
    "What prevents users from exploring new categories?",
    "How do users discover products today?",
    "What role do habits play?",
    "What frustrations emerge repeatedly?",
    "Which user segments experiment more?",
    "What unmet needs emerge?",
]

INSIGHT_KEYS = (
    "Title",
    "Evidence",
    "Business Impact",
    "Opportunity",
    "Priority",
    "Recommendation",
)


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def build_context(
    themes: pd.DataFrame,
    sentiment: pd.DataFrame | None,
    segments: pd.DataFrame | None,
) -> str:
    parts: list[str] = []
    parts.append("=== THEMES (BERTopic) ===")
    for _, row in themes.iterrows():
        parts.append(
            f"- {row.get('Theme name')} | n={row.get('Number of reviews')} | "
            f"keywords={row.get('Representative keywords')} | "
            f"examples={str(row.get('Representative reviews', ''))[:300]}"
        )

    if sentiment is not None and len(sentiment):
        parts.append("\n=== SENTIMENT DISTRIBUTION ===")
        counts = sentiment["Sentiment"].value_counts(dropna=False).to_dict()
        parts.append(str(counts))
        neg = sentiment[sentiment["Sentiment"].astype(str).str.lower() == "negative"]
        if len(neg):
            parts.append("Sample negative reviews:")
            for t in neg["Review"].astype(str).head(8).tolist():
                parts.append(f"  - {t[:220]}")
        pos = sentiment[sentiment["Sentiment"].astype(str).str.lower() == "positive"]
        if len(pos):
            parts.append("Sample positive reviews:")
            for t in pos["Review"].astype(str).head(5).tolist():
                parts.append(f"  - {t[:220]}")
    else:
        parts.append("\n=== SENTIMENT ===\n(not available)")

    if segments is not None and len(segments):
        parts.append("\n=== USER SEGMENTS (KMeans) ===")
        parts.append(str(segments["Segment"].value_counts().to_dict()))
        for seg, grp in segments.groupby("Segment"):
            rationale = str(grp["Label Rationale"].iloc[0])[:400] if "Label Rationale" in grp else ""
            parts.append(f"[{seg}] n={len(grp)} | {rationale}")
            for t in grp["Review"].astype(str).head(3).tolist():
                parts.append(f"  - {t[:200]}")
    else:
        parts.append("\n=== USER SEGMENTS ===\n(not available)")

    return "\n".join(parts)


def _system_prompt() -> str:
    return (
        "You are a product strategist for a quick-commerce grocery app (Blinkit-like). "
        "Using ONLY the provided analysis evidence, produce actionable product insights. "
        "Do not invent quotes that are not grounded in the evidence. "
        "Priority must be one of: High, Medium, Low. "
        "Return valid JSON only."
    )


def _user_prompt(context: str) -> str:
    questions = "\n".join(f"{i+1}. {q}" for i, q in enumerate(RESEARCH_QUESTIONS))
    return f"""
Analysis evidence:
{context}

Answer EACH of these research questions with exactly one insight object:
{questions}

Return a JSON object with key "insights" whose value is an array of objects.
Each object MUST have these keys:
  "Question": the research question being answered
  "Title": short insight title
  "Evidence": concrete evidence from themes/sentiment/segments (cite keywords, counts, quotes)
  "Business Impact": why this matters for MAC new-category exploration / retention / growth
  "Opportunity": product opportunity statement
  "Priority": High | Medium | Low
  "Recommendation": specific next product action

Produce exactly {len(RESEARCH_QUESTIONS)} insight objects, one per question, in the same order.
""".strip()


def generate_with_openai(context: str) -> tuple[list[dict[str, Any]], str]:
    """
    Call an OpenAI-compatible chat API and return normalized insights.

    Returns:
        (insights, model_name_used)

    Raises:
        RuntimeError / ValueError / json.JSONDecodeError on failure.
    """
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing in .env")

    # Groq keys look like gsk_... and need the Groq OpenAI-compatible base URL
    base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
    model = os.getenv("OPENAI_MODEL", "").strip()
    if api_key.startswith("gsk_"):
        base_url = base_url or "https://api.groq.com/openai/v1"
        # Ignore OpenAI-only model names left in .env
        if not model or model.startswith("gpt-") or model.startswith("o1") or model.startswith("o3"):
            model = "llama-3.3-70b-versatile"
    else:
        model = model or "gpt-4o-mini"

    client = OpenAI(api_key=api_key, base_url=base_url)
    resp = client.chat.completions.create(
        model=model,
        temperature=0.3,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": _user_prompt(context)},
        ],
    )
    if not resp.choices:
        raise ValueError("LLM returned no choices")
    raw = resp.choices[0].message.content or "{}"
    payload = json.loads(raw)
    insights = payload.get("insights")
    if not isinstance(insights, list) or not insights:
        raise ValueError("LLM JSON missing non-empty 'insights' array")
    normalized = []
    for i, item in enumerate(insights):
        if not isinstance(item, dict):
            raise ValueError(f"Insight at index {i} is not an object")
        normalized.append(_normalize_insight(item, i))
    return normalized, model


def _normalize_insight(item: dict[str, Any], index: int) -> dict[str, Any]:
    q = item.get("Question") or (
        RESEARCH_QUESTIONS[index] if index < len(RESEARCH_QUESTIONS) else ""
    )
    out = {"Question": q}
    for key in INSIGHT_KEYS:
        val = item.get(key) or item.get(key.lower()) or item.get(key.replace(" ", "_"))
        out[key] = str(val).strip() if val is not None else ""
    if out["Priority"] not in {"High", "Medium", "Low"}:
        p = out["Priority"].title()
        out["Priority"] = p if p in {"High", "Medium", "Low"} else "Medium"
    return out


def generate_fallback(
    themes: pd.DataFrame,
    sentiment: pd.DataFrame | None,
    segments: pd.DataFrame | None,
) -> list[dict[str, Any]]:
    """Heuristic insights when OpenAI is unavailable — still grounded in local CSVs."""
    top_themes = themes.sort_values("Number of reviews", ascending=False).head(5)
    theme_lines = []
    for _, r in top_themes.iterrows():
        theme_lines.append(
            f"{r['Theme name']} (n={r['Number of reviews']}; keywords={r.get('Representative keywords')})"
        )
    theme_blob = "; ".join(theme_lines) if theme_lines else "limited theme signal"

    sent_summary = "sentiment file missing"
    if sentiment is not None and len(sentiment):
        sent_summary = str(sentiment["Sentiment"].value_counts().to_dict())

    explorers_n = 0
    routine_n = 0
    if segments is not None and len(segments):
        vc = segments["Segment"].value_counts()
        explorers_n = int(vc.get("Explorers", 0))
        routine_n = int(vc.get("Routine Buyers", 0))

    templates = [
        {
            "Question": RESEARCH_QUESTIONS[0],
            "Title": "Repeat buying is reinforced by convenience loops",
            "Evidence": f"Top themes skew to generic praise and reorder-adjacent language: {theme_blob}",
            "Business Impact": "Same-category lock-in caps basket diversity and lifetime value.",
            "Opportunity": "Break reorder autopilot with curated adjacent-category nudges.",
            "Priority": "High",
            "Recommendation": "After checkout of staples, show a single 'try next' SKU from an adjacent category with social proof.",
        },
        {
            "Question": RESEARCH_QUESTIONS[1],
            "Title": "Trust, fees, and coverage block exploration",
            "Evidence": f"Themes mentioning charges/coverage/poor experience appear in: {theme_blob}. Sentiment mix: {sent_summary}",
            "Business Impact": "Perceived risk and cost stop trial of unfamiliar categories.",
            "Opportunity": "Reduce trial risk with guarantees, clearer fees, and coverage transparency.",
            "Priority": "High",
            "Recommendation": "Add first-time category trial credits and upfront fee/coverage messaging on discovery surfaces.",
        },
        {
            "Question": RESEARCH_QUESTIONS[2],
            "Title": "Discovery is still search- and homepage-led",
            "Evidence": f"Feedback clusters around app usability, location, and delivery rather than guided browse: {theme_blob}",
            "Business Impact": "Weak discovery paths keep users in known aisles.",
            "Opportunity": "Make unfamiliar categories visible in the default journey.",
            "Priority": "Medium",
            "Recommendation": "Ship a 'new for you' rail seeded by complementary categories to the user's last order.",
        },
        {
            "Question": RESEARCH_QUESTIONS[3],
            "Title": "Habits dominate over intentional exploration",
            "Evidence": f"Segment mix shows Routine Buyers={routine_n}, Explorers={explorers_n}. Themes: {theme_blob}",
            "Business Impact": "Habit loops suppress category breadth and MAC new-category rates.",
            "Opportunity": "Design habit-friendly experiments that feel low-effort.",
            "Priority": "High",
            "Recommendation": "Attach micro-trials to reorder flows (one-tap add-on) instead of requiring browse sessions.",
        },
        {
            "Question": RESEARCH_QUESTIONS[4],
            "Title": "Recurring frustrations center on delivery reliability and charges",
            "Evidence": f"Recurring theme keywords include delivery/charge/location issues: {theme_blob}. Sentiment: {sent_summary}",
            "Business Impact": "Unresolved friction reduces willingness to try new categories after a bad experience.",
            "Opportunity": "Close the post-incident recovery loop to restore exploration confidence.",
            "Priority": "High",
            "Recommendation": "Trigger recovery offers and clearer refund SLAs after partial delivery or surge-fee complaints.",
        },
        {
            "Question": RESEARCH_QUESTIONS[5],
            "Title": "Explorers are the minority but highest-leverage segment",
            "Evidence": f"From segments: Explorers={explorers_n}, Routine Buyers={routine_n}, full mix available in user_segments.csv",
            "Business Impact": "Concentrating discovery experiments on Explorers raises learning speed and conversion.",
            "Opportunity": "Scale Explorer behaviors into Routine Buyers via low-risk trials.",
            "Priority": "Medium",
            "Recommendation": "Target Explorers first with specialty/new-arrival campaigns; measure lift before broad rollout.",
        },
        {
            "Question": RESEARCH_QUESTIONS[6],
            "Title": "Unmet needs: fee clarity, coverage, and trustworthy trial info",
            "Evidence": f"Unmet-need signals in themes (charges, area coverage, quality/service variance): {theme_blob}",
            "Business Impact": "Missing pre-purchase info keeps users from experimenting beyond staples.",
            "Opportunity": "Instrument PDP/discovery with the info users ask for before trying something new.",
            "Priority": "Medium",
            "Recommendation": "Prioritize unit-price, fee breakdown, coverage ETA, and freshness cues on category entry pages.",
        },
    ]
    return templates


def write_insights(insights: list[dict[str, Any]], path: Path, meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta,
        "research_questions": RESEARCH_QUESTIONS,
        "insights": insights,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM product insights from themes + sentiment")
    parser.add_argument("--themes", type=Path, default=DEFAULT_THEMES)
    parser.add_argument("--sentiment", type=Path, default=DEFAULT_SENTIMENT)
    parser.add_argument("--segments", type=Path, default=DEFAULT_SEGMENTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--fallback-only",
        action="store_true",
        help="Skip OpenAI and write heuristic insights from CSVs",
    )
    args = parser.parse_args()

    if not args.themes.exists():
        print(f"Missing themes file: {args.themes}", file=sys.stderr)
        print("Run: python -m analysis.themes", file=sys.stderr)
        return 1

    themes = pd.read_csv(args.themes)
    sentiment = _read_csv(args.sentiment)
    segments = _read_csv(args.segments)
    if sentiment is None:
        print(f"Warning: sentiment file not found ({args.sentiment}); continuing without it")
    if segments is None:
        print(f"Warning: segments file not found ({args.segments}); continuing without it")

    context = build_context(themes, sentiment, segments)
    source = "openai"
    model_used = None
    try:
        if args.fallback_only:
            raise RuntimeError("fallback-only flag set")
        print("Generating insights with OpenAI...")
        insights, model_used = generate_with_openai(context)
    except Exception as e:
        print(f"OpenAI path unavailable ({e}); using grounded fallback insights")
        insights = generate_fallback(themes, sentiment, segments)
        source = "fallback"

    # Ensure one insight per research question when possible
    if len(insights) < len(RESEARCH_QUESTIONS):
        print("Warning: fewer insights than research questions")

    meta = {
        "source": source,
        "themes_path": str(args.themes),
        "sentiment_path": str(args.sentiment) if sentiment is not None else None,
        "segments_path": str(args.segments) if segments is not None else None,
        "model": model_used if source == "openai" else None,
        "n_insights": len(insights),
    }
    write_insights(insights, args.out, meta)

    print(f"\nWrote {len(insights)} insights -> {args.out.resolve()}")
    for item in insights:
        print(f"- [{item.get('Priority')}] {item.get('Title')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
