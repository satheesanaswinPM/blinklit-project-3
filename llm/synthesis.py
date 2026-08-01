"""
Synthesis layer for the primary research question:

  Why don't Blinkit users explore new categories?

Produces output/synthesis.json with:
  - executive_summary
  - barriers_ranked
  - jobs_to_be_done
  - unmet_needs
  - category_opportunities
  - hypotheses
  - testable_experiments
"""

from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from env_loader import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
TAGS_PATH = ROOT / "output" / "exploration_tags.csv"
THEMES_PATH = ROOT / "output" / "themes.csv"
SEGMENTS_PATH = ROOT / "output" / "user_segments.csv"
OUT_PATH = ROOT / "output" / "synthesis.json"

PRIMARY_QUESTION = "Why don't Blinkit users explore new categories?"

# Seed JTBD / experiments — grounded in barrier taxonomy, refined by corpus counts.
JTBD_SEED = [
    {
        "job": "When I need something outside my usual grocery list, I want to trust the assortment and quality so I can order without second-guessing.",
        "situation": "Expanding beyond staples into beauty, electronics, home, or pet",
        "desired_outcome": "Confidence that the category is as reliable as grocery",
        "current_workaround": "Order only staples on Blinkit; buy elsewhere for other categories",
    },
    {
        "job": "When I'm browsing under time pressure, I want new categories to surface in context so I don't have to know what to search.",
        "situation": "Short session, habitual reorder path",
        "desired_outcome": "Serendipitous but relevant discovery without leaving the flow",
        "current_workaround": "Stay on the same SKUs / search only known brands",
    },
    {
        "job": "When trying a new category for the first time, I want clear price and return signals so the risk feels bounded.",
        "situation": "First purchase in electronics / beauty / pharmacy-adjacent",
        "desired_outcome": "Transparent fees, quality cues, easy undo",
        "current_workaround": "Abandon cart or stick to grocery where risk is known",
    },
]

EXPERIMENT_SEED = [
    {
        "id": "exp_discover_rail",
        "name": "Category discovery rail after grocery add",
        "hypothesis_link": "H1",
        "barrier": "hard_to_discover_in_app",
        "intervention": "After adding a staple, show a contextual 'also useful tonight' rail from an adjacent category with trust badges.",
        "primary_metric": "Attach rate of non-grocery SKU in same session",
        "guardrail": "Checkout completion rate must not drop >2%",
        "sample_size_note": "Power for 15% relative lift in attach; ~2 weeks on high-traffic cities",
    },
    {
        "id": "exp_first_buy_guarantee",
        "name": "First-buy quality guarantee for new categories",
        "hypothesis_link": "H2",
        "barrier": "dont_trust_quality_for_new_category",
        "intervention": "Offer one-tap easy return + quality score for first purchase in beauty/electronics/home.",
        "primary_metric": "First-time category conversion among grocery-only cohorts",
        "guardrail": "Return rate spike <3pp vs control",
        "sample_size_note": "Stratify by city tier; exclude surge-heavy hours",
    },
    {
        "id": "exp_price_clarity",
        "name": "Upfront fee clarity on new-category PDPs",
        "hypothesis_link": "H3",
        "barrier": "price_uncertainty_new_category",
        "intervention": "Show all-in price (item + fees) and 'vs offline' cue on first-time category PDPs.",
        "primary_metric": "PDP→ATC for tagged new-category sessions",
        "guardrail": "No increase in support tickets tagged 'hidden fees'",
        "sample_size_note": "A/B on electronics + beauty first",
    },
    {
        "id": "exp_break_routine",
        "name": "Routine break prompt for power reorderers",
        "hypothesis_link": "H4",
        "barrier": "habit_reorder_only",
        "intervention": "For users with 5+ identical reorder carts, surface one curated 'try something new' pack with free delivery on first try.",
        "primary_metric": "Share of sessions with ≥1 new category SKU",
        "guardrail": "Reorder NPS / time-to-checkout",
        "sample_size_note": "Target stuck_in_routine segment only",
    },
]

CATEGORY_COPY = {
    "grocery": "Core habit category — use as springboard, not the growth lever.",
    "electronics": "High intent + high distrust; quality and authenticity cues matter most.",
    "beauty": "Discovery-friendly but needs sampling / return confidence.",
    "home": "Assortment gaps and 'not for Blinkit' mental model.",
    "pet": "Niche but sticky once tried; trust on freshness/expiry.",
    "pharmacy": "Regulatory/trust barriers; careful messaging.",
    "snacks": "Adjacent to grocery — easiest expansion wedge.",
}


def _barrier_counts(df: pd.DataFrame) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    for cell in df.get("barriers", pd.Series(dtype=str)).fillna(""):
        for b in str(cell).split("|"):
            b = b.strip()
            if b:
                counter[b] += 1
    total = max(sum(counter.values()), 1)
    ranked = [
        {
            "barrier": name,
            "mentions": int(n),
            "share": round(n / total, 3),
            "severity": "high" if n / total >= 0.25 else "medium" if n / total >= 0.1 else "low",
        }
        for name, n in counter.most_common()
    ]
    return ranked


def _category_opportunities(df: pd.DataFrame, barriers: list[dict]) -> list[dict[str, Any]]:
    cat_counter: Counter[str] = Counter()
    blocked = df[df["exploration_signal"] == "want_to_explore_blocked"] if "exploration_signal" in df.columns else df
    for cell in blocked.get("categories_mentioned", pd.Series(dtype=str)).fillna(""):
        for c in str(cell).split("|"):
            c = c.strip()
            if c and c != "grocery":
                cat_counter[c] += 1
    top_barrier = barriers[0]["barrier"] if barriers else "hard_to_discover_in_app"
    ops = []
    for i, (cat, n) in enumerate(cat_counter.most_common(8), start=1):
        ops.append(
            {
                "rank": i,
                "category": cat,
                "blocked_mentions": int(n),
                "opportunity_score": round(min(100, 40 + n * 2), 1),
                "why_now": CATEGORY_COPY.get(cat, "Emerging signal in exploration-tagged corpus."),
                "primary_barrier_to_attack": top_barrier,
                "suggested_experiment": next(
                    (e["id"] for e in EXPERIMENT_SEED if e["barrier"] == top_barrier),
                    EXPERIMENT_SEED[0]["id"],
                ),
            }
        )
    if not ops:
        # Fallback opportunities so dashboard isn't empty on thin corpora
        for i, cat in enumerate(["electronics", "beauty", "snacks", "home"], start=1):
            ops.append(
                {
                    "rank": i,
                    "category": cat,
                    "blocked_mentions": 0,
                    "opportunity_score": 55 - i * 5,
                    "why_now": CATEGORY_COPY.get(cat, ""),
                    "primary_barrier_to_attack": top_barrier,
                    "suggested_experiment": EXPERIMENT_SEED[0]["id"],
                }
            )
    return ops


def _hypotheses(barriers: list[dict]) -> list[dict[str, Any]]:
    mapping = {
        "hard_to_discover_in_app": (
            "H1",
            "Users fail to explore new categories because in-app discovery does not surface adjacent categories during grocery sessions.",
        ),
        "dont_trust_quality_for_new_category": (
            "H2",
            "Users avoid new categories because quality/authenticity risk feels higher than grocery.",
        ),
        "price_uncertainty_new_category": (
            "H3",
            "Users hesitate on new categories when all-in price and fees are unclear relative to offline.",
        ),
        "habit_reorder_only": (
            "H4",
            "Power users stay in reorder loops that never expose non-grocery demand.",
        ),
        "assortment_gap": (
            "H5",
            "Exploration intent dies when expected SKUs are missing or out of stock.",
        ),
        "coverage_or_eta": (
            "H6",
            "ETA/coverage anxiety reduces willingness to try unfamiliar categories.",
        ),
    }
    hyps = []
    used = set()
    for b in barriers[:5]:
        key = b["barrier"]
        if key in mapping and key not in used:
            hid, text = mapping[key]
            hyps.append(
                {
                    "id": hid,
                    "statement": text,
                    "linked_barrier": key,
                    "evidence_mentions": b["mentions"],
                    "status": "open",
                }
            )
            used.add(key)
    # Always include core set
    for key, (hid, text) in mapping.items():
        if key not in used and hid in {"H1", "H2", "H3", "H4"}:
            hyps.append(
                {
                    "id": hid,
                    "statement": text,
                    "linked_barrier": key,
                    "evidence_mentions": 0,
                    "status": "open",
                }
            )
            used.add(key)
    return sorted(hyps, key=lambda h: h["id"])


def _unmet_needs(barriers: list[dict], signal_counts: dict) -> list[dict[str, Any]]:
    needs = [
        {
            "need": "Visible, trusted paths into non-grocery categories during habitual grocery sessions",
            "pain": "Discovery is search-dependent; reorderers never see adjacency",
            "evidence": f"stuck_in_routine={signal_counts.get('stuck_in_routine', 0)}, discover barrier mentions",
            "priority": "P0",
        },
        {
            "need": "Risk reducers (returns, authenticity, freshness) for first buys outside grocery",
            "pain": "Quality distrust blocks trial even when assortment exists",
            "evidence": "dont_trust_quality_for_new_category in barrier ranking",
            "priority": "P0",
        },
        {
            "need": "All-in price clarity before commitment on unfamiliar categories",
            "pain": "Fee/markup surprise kills exploration mid-funnel",
            "evidence": "price_uncertainty_new_category ranking",
            "priority": "P1",
        },
        {
            "need": "Assortment confidence signals (in stock, alternatives) for expansion categories",
            "pain": "Empty or sparse shelves teach users 'Blinkit isn't for this'",
            "evidence": "assortment_gap + want_to_explore_blocked",
            "priority": "P1",
        },
    ]
    # Reorder by whether matching barrier is top
    top = {b["barrier"] for b in barriers[:3]}
    for n in needs:
        if "discover" in n["need"].lower() and "hard_to_discover_in_app" in top:
            n["priority"] = "P0"
        if "Risk" in n["need"] and "dont_trust_quality_for_new_category" in top:
            n["priority"] = "P0"
    return needs


def build_synthesis_from_corpus(
    tags: pd.DataFrame,
    themes: pd.DataFrame | None = None,
    segments: pd.DataFrame | None = None,
) -> dict[str, Any]:
    relevant = tags[tags.get("is_relevant", False) == True] if "is_relevant" in tags.columns else tags
    signal_counts = (
        tags["exploration_signal"].value_counts().to_dict()
        if "exploration_signal" in tags.columns
        else {}
    )
    barriers = _barrier_counts(relevant if len(relevant) else tags)
    hyps = _hypotheses(barriers)
    cat_ops = _category_opportunities(tags, barriers)
    unmet = _unmet_needs(barriers, signal_counts)

    n = len(tags)
    n_rel = int(tags["is_relevant"].sum()) if "is_relevant" in tags.columns else n
    sources = (
        tags["source"].value_counts().to_dict() if "source" in tags.columns else {}
    )

    top_barriers_txt = ", ".join(b["barrier"].replace("_", " ") for b in barriers[:3]) or "discovery and trust"
    exec_summary = (
        f"Across {n:,} multi-source reviews ({n_rel:,} exploration-relevant), "
        f"users mostly stay in grocery routines or bounce when discovery, quality trust, or price clarity fail. "
        f"Dominant barriers: {top_barriers_txt}. "
        f"Category expansion opportunity concentrates in "
        f"{', '.join(c['category'] for c in cat_ops[:3]) or 'adjacent non-grocery categories'}. "
        f"Recommended next step: run discovery-rail + first-buy guarantee experiments on the top blocked categories."
    )

    segment_note = ""
    if segments is not None and len(segments) and "segment" in segments.columns:
        seg_counts = segments["segment"].value_counts().head(4).to_dict()
        segment_note = f"Behavioral clusters observed: {seg_counts}"

    theme_note = ""
    if themes is not None and len(themes) and "theme" in themes.columns:
        theme_note = "Top themes: " + ", ".join(
            str(t) for t in themes["theme"].value_counts().head(5).index.tolist()
        )

    return {
        "primary_question": PRIMARY_QUESTION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "corpus": {
            "total_reviews": n,
            "exploration_relevant": n_rel,
            "relevance_rate": round(n_rel / max(n, 1), 3),
            "by_source": {str(k): int(v) for k, v in sources.items()},
            "exploration_signals": {str(k): int(v) for k, v in signal_counts.items()},
            "notes": [x for x in [segment_note, theme_note] if x],
        },
        "executive_summary": exec_summary,
        "barriers_ranked": barriers,
        "jobs_to_be_done": JTBD_SEED,
        "unmet_needs": unmet,
        "hypotheses": hyps,
        "testable_experiments": EXPERIMENT_SEED,
        "category_opportunities": cat_ops,
        "method": {
            "tagging": "rule-based exploration tagger (offline, explainable)",
            "synthesis": "corpus-count grounded templates + optional LLM polish",
            "parity_note": "Aligned to discovery-engine research IA; local implementation differs in tagging and UI",
        },
    }


def _llm_polish(synthesis: dict[str, Any]) -> dict[str, Any]:
    """Optional short LLM rewrite of executive_summary only."""
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        synthesis["llm_polish"] = False
        return synthesis
    try:
        from openai import OpenAI

        base = None
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        if os.getenv("GROQ_API_KEY"):
            base = "https://api.groq.com/openai/v1"
            model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
            api_key = os.getenv("GROQ_API_KEY")
        client = OpenAI(api_key=api_key, base_url=base)
        prompt = (
            f"Primary question: {PRIMARY_QUESTION}\n"
            f"Draft summary:\n{synthesis['executive_summary']}\n\n"
            "Rewrite in 4-5 crisp sentences for a PM audience. Keep numbers. No markdown."
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a product research synthesizer for Blinkit category discovery."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=400,
        )
        text = (resp.choices[0].message.content or "").strip()
        if text:
            synthesis["executive_summary"] = text
            synthesis["llm_polish"] = True
            synthesis["llm_model"] = model
        else:
            synthesis["llm_polish"] = False
    except Exception as exc:  # noqa: BLE001
        synthesis["llm_polish"] = False
        synthesis["llm_error"] = str(exc)[:200]
    return synthesis


def generate_synthesis(
    tags_path: Path = TAGS_PATH,
    out_path: Path = OUT_PATH,
    polish: bool = True,
) -> dict[str, Any]:
    if not tags_path.exists():
        raise FileNotFoundError(f"Missing exploration tags: {tags_path}")
    tags = pd.read_csv(tags_path)
    themes = pd.read_csv(THEMES_PATH) if THEMES_PATH.exists() else None
    segments = pd.read_csv(SEGMENTS_PATH) if SEGMENTS_PATH.exists() else None
    synthesis = build_synthesis_from_corpus(tags, themes, segments)
    if polish:
        synthesis = _llm_polish(synthesis)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(synthesis, indent=2, ensure_ascii=False), encoding="utf-8")
    return synthesis


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Build synthesis.json for category exploration")
    parser.add_argument("--tags", type=Path, default=TAGS_PATH)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--no-polish", action="store_true")
    args = parser.parse_args()
    syn = generate_synthesis(args.tags, args.out, polish=not args.no_polish)
    print(f"Wrote {args.out}")
    print(f"Relevant: {syn['corpus']['exploration_relevant']} / {syn['corpus']['total_reviews']}")
    print(f"Barriers: {len(syn['barriers_ranked'])} | Categories: {len(syn['category_opportunities'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
