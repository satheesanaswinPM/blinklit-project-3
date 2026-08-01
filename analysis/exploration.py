"""
Exploration tagging for category-discovery research.

Primary question:
  Why don't Blinkit users explore new categories?

Labels each review with:
  - is_relevant (category-exploration related vs noise)
  - exploration_signal: stuck_in_routine | want_to_explore_blocked | explored_new | unclear
  - barriers (multi-label strings)
  - categories_mentioned
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = ROOT / "data" / "processed" / "merged_reviews.csv"
DEFAULT_OUT = ROOT / "output" / "exploration_tags.csv"

CATEGORY_PATTERNS: dict[str, tuple[str, ...]] = {
    "grocery": ("grocery", "kirana", "vegetables", "fruits", "milk", "dairy", "atta", "rice", "dal"),
    "electronics": ("electronics", "phone", "iphone", "earphone", "charger", "laptop", "ps5", "gadget"),
    "beauty": ("beauty", "skincare", "makeup", "cosmetic", "serum", "shampoo"),
    "home": ("home", "kitchen", "cleaning", "detergent", "furniture", "appliance"),
    "pet": ("pet", "dog food", "cat food", "litter"),
    "pharmacy": ("medicine", "pharmacy", "otc", "thermometer"),
    "snacks": ("snacks", "chips", "cold drink", "ice cream", "beverage", "munchies"),
}

BARRIER_PATTERNS: dict[str, tuple[str, ...]] = {
    "hard_to_discover_in_app": ("can't find", "cant find", "hard to find", "search", "not showing", "discover", "browse"),
    "price_uncertainty_new_category": ("expensive", "costly", "markup", "surge", "handling charge", "delivery fee", "price"),
    "dont_trust_quality_for_new_category": ("quality", "fresh", "expired", "fake", "return", "refund", "damaged", "trust"),
    "habit_reorder_only": ("always order", "same", "usual", "reorder", "regular", "only grocery", "habit"),
    "coverage_or_eta": ("not available", "my area", "location", "late", "delay", "slot"),
    "assortment_gap": ("not available", "out of stock", "missing", "don't have", "dont have", "limited"),
}

EXPLORE_INTENT = (
    "try new",
    "new category",
    "first time",
    "explore",
    "something new",
    "other category",
    "non grocery",
    "electronics",
    "beauty",
    "pet food",
)
EXPLORED = ("bought", "ordered", "tried", "got my", "purchased", "delivered")
ROUTINE = ("always", "usual", "same", "reorder", "every day", "daily", "regular")
NOISE = ("good app", "nice app", "best app", "love it", "awesome", "ok", "okay", "super", "excellent service")


def _find_labels(text: str, patterns: dict[str, tuple[str, ...]]) -> list[str]:
    t = text.lower()
    hits = []
    for label, kws in patterns.items():
        if any(kw in t for kw in kws):
            hits.append(label)
    return hits


def tag_review(text: str) -> dict:
    """Heuristic tagger — fast, offline, explainable."""
    raw = (text or "").strip()
    t = raw.lower()
    categories = _find_labels(t, CATEGORY_PATTERNS)
    barriers = _find_labels(t, BARRIER_PATTERNS)

    short_noise = len(t.split()) <= 4 and any(n in t for n in NOISE)
    has_explore_intent = any(k in t for k in EXPLORE_INTENT) or len(categories) >= 2
    has_explored = has_explore_intent and any(k in t for k in EXPLORED)
    has_routine = any(k in t for k in ROUTINE) or "habit_reorder_only" in barriers
    has_block = bool(barriers) and (
        has_explore_intent
        or any(
            b in barriers
            for b in (
                "hard_to_discover_in_app",
                "price_uncertainty_new_category",
                "dont_trust_quality_for_new_category",
                "assortment_gap",
            )
        )
    )

    if short_noise and not categories and not barriers:
        return {
            "is_relevant": False,
            "exploration_signal": "noise",
            "barriers": "",
            "categories_mentioned": "",
            "relevance_reason": "short generic praise/complaint without category cues",
        }

    if has_explored and categories:
        signal = "explored_new"
        relevant = True
        reason = "mentions trying/buying beyond routine categories"
    elif has_block and (has_explore_intent or categories or barriers):
        signal = "want_to_explore_blocked"
        relevant = True
        reason = "intent or category talk with friction/barrier cues"
    elif has_routine and not has_explore_intent:
        signal = "stuck_in_routine"
        relevant = True
        reason = "routine/reorder language without exploration intent"
    elif categories or barriers or has_explore_intent:
        signal = "unclear"
        relevant = True
        reason = "category or barrier cues without clear funnel stage"
    else:
        signal = "noise"
        relevant = False
        reason = "no category-exploration signal"

    return {
        "is_relevant": relevant,
        "exploration_signal": signal,
        "barriers": "|".join(barriers),
        "categories_mentioned": "|".join(categories),
        "relevance_reason": reason,
    }


def tag_corpus(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    if text_col not in df.columns:
        raise ValueError(f"Missing text column {text_col}")
    tags = df[text_col].astype(str).map(tag_review)
    tag_df = pd.DataFrame(list(tags))
    out = pd.concat([df.reset_index(drop=True), tag_df], axis=1)
    return out


def save_tags(df: pd.DataFrame, path: Path = DEFAULT_OUT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    return path


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Tag reviews for category exploration")
    parser.add_argument("--in", dest="inp", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if not args.inp.exists():
        print(f"Missing input: {args.inp}")
        return 1
    df = pd.read_csv(args.inp)
    tagged = tag_corpus(df)
    save_tags(tagged, args.out)
    print(f"Tagged {len(tagged)} -> {args.out}")
    print(tagged["exploration_signal"].value_counts().to_string())
    print(f"Relevant: {int(tagged['is_relevant'].sum())} / {len(tagged)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
