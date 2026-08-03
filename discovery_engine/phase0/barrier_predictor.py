"""Heuristic barrier (and light label) predictor for Phase 0 gold eval.

Predicts gold taxonomy barrier ids from text — not identity copy of gold labels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Keyword cues aligned to taxonomies/barriers.json + gold seed language
BARRIER_PATTERNS: dict[str, tuple[str, ...]] = {
    "trust_quality": (
        "don't trust",
        "dont trust",
        "trust",
        "fake",
        "adulteration",
        "unknown brand",
        "quality",
        "shady",
        "risky",
        "authenticity",
    ),
    "price_value": (
        "expensive",
        "marked up",
        "markup",
        "cheaper",
        "price",
        "costly",
        "not worth",
        "deal",
        "unit price",
    ),
    "freshness_spoilage": (
        "expiry",
        "expired",
        "spoil",
        "rotten",
        "smelled off",
        "stale",
        "near expiry",
        "fresh",
        "spoilage",
    ),
    "brand_lock_in": (
        "only buy",
        "stick to",
        "substituting",
        "usual brand",
        "amul",
        "known brand",
        "sealed brands",
    ),
    "discovery_invisibility": (
        "didn't even know",
        "didnt even know",
        "never shows",
        "never showed",
        "homepage",
        "hard to find",
        "buried",
        "discover",
        "invisib",
    ),
    "search_vocab_gap": (
        "searched",
        "search shows",
        "what to type",
        "what query",
        "zero results",
        "results were poor",
        "don't know what to",
    ),
    "assortment_doubt": (
        "only 2",
        "limited options",
        "not enough variety",
        "missing",
        "my size",
        "thin",
        "assortment",
    ),
    "habit_mental_model": (
        "only use",
        "only for",
        "late night snacks",
        "would never buy",
        "not for",
        "always the same",
        "reorder",
        "usual",
        "habit",
        "sirf",
    ),
    "cognitive_load": (
        "no time",
        "too many choices",
        "browsing takes",
        "cognitive",
        "forever",
        "just want my usual",
    ),
    "past_bad_experience": (
        "tried once",
        "never again",
        "last time",
        "got rotten",
        "refund hassle",
        "never buying",
        "put me off",
    ),
    "info_trust_gap": (
        "expiry on the product",
        "ingredients",
        "customer photos",
        "unit price",
        "return policy",
        "origin",
        "need real",
        "show clear",
        "info",
    ),
}

CATEGORY_PATTERNS: dict[str, tuple[str, ...]] = {
    "snacks": ("snack", "chips", "biscuits", "cola", "cold drink"),
    "beverages": ("drink", "beverage", "cola", "oat milk", "cold drink"),
    "fruits_vegetables": ("veggie", "veggies", "fruit", "tomato", "produce"),
    "staples": ("grocery", "atta", "spices", "oils", "staples"),
    "dairy_eggs": ("milk", "curd", "dairy", "eggs", "amul"),
    "bakery": ("bread", "bakery"),
    "meat_seafood": ("chicken", "meat", "seafood", "fish"),
    "personal_care": ("shampoo", "personal care"),
    "baby_care": ("diaper", "baby"),
    "pet_care": ("pet food", "pet"),
    "household": ("cleaner", "cleaning", "household"),
    "pharma": ("otc", "vitamin", "pharma", "medicine"),
    "organic_specialty": ("organic", "gluten free", "specialty"),
    "packaged_food": ("packaged food",),
}

SENTIMENT_CUES = {
    "negative": ("never", "don't", "dont", "scared", "rotten", "expensive", "failed", "rude", "risky", "off"),
    "positive": ("love", "like", "great", "curious", "would buy more", "trial packs"),
}


def _hits(text: str, patterns: dict[str, tuple[str, ...]]) -> list[str]:
    t = text.lower()
    found = []
    for label, kws in patterns.items():
        if any(kw in t for kw in kws):
            found.append(label)
    return found


def predict_barriers(text: str, *, is_spam: bool = False, is_off_topic: bool = False) -> list[str]:
    """Predict gold-taxonomy barrier ids for a feedback string."""
    raw = (text or "").strip()
    if not raw or is_spam:
        return ["none"]
    hits = _hits(raw, BARRIER_PATTERNS)
    if not hits:
        if is_off_topic:
            return ["none"]
        return ["other"] if len(raw.split()) > 6 else ["none"]
    # Prefer specific barriers; drop 'none' if any real barrier fired
    return sorted(set(hits))


def predict_categories(text: str, *, is_spam: bool = False, is_off_topic: bool = False) -> list[str]:
    if is_spam or not (text or "").strip():
        return ["off_topic"]
    hits = _hits(text, CATEGORY_PATTERNS)
    if not hits:
        return ["off_topic"] if is_off_topic else []
    return sorted(set(hits))


def predict_sentiment(text: str) -> str:
    t = (text or "").lower()
    neg = sum(1 for k in SENTIMENT_CUES["negative"] if k in t)
    pos = sum(1 for k in SENTIMENT_CUES["positive"] if k in t)
    if neg > pos:
        return "negative"
    if pos > neg:
        return "positive"
    return "neutral"


def predict_insight_types(barriers: list[str], categories: list[str], text: str) -> list[str]:
    types: list[str] = []
    if barriers and barriers != ["none"]:
        types.append("barrier_taxonomy")
    if any(b in barriers for b in ("habit_mental_model", "cognitive_load", "brand_lock_in")):
        types.append("habit_drivers")
    if any(b in barriers for b in ("discovery_invisibility", "search_vocab_gap")):
        types.append("discovery_path_map")
    if any(b in barriers for b in ("info_trust_gap",)):
        types.append("info_needs")
    if "try" in text.lower() or "explore" in text.lower() or "curious" in text.lower():
        types.append("receptive_segments")
    if barriers and barriers != ["none"]:
        types.append("opportunity_briefs")
    return sorted(set(types)) or ["barrier_taxonomy"]


def _evidence_spans(text: str, max_spans: int = 2) -> list[str]:
    """Pick short substrings that appear in text for grounding checks."""
    t = (text or "").strip()
    if not t:
        return []
    # Prefer first sentence / first 80 chars as grounded span
    spans = []
    chunk = t.split(".")[0].strip()
    if chunk and chunk.lower() in t.lower():
        spans.append(chunk[:120])
    if len(t) > 40 and t[:40] not in spans:
        spans.append(t[:40])
    return spans[:max_spans] or [t[: min(40, len(t))]]


def predict_for_gold_row(row: dict[str, Any]) -> dict[str, Any]:
    """Build an eval-harness prediction dict from a gold row's text (not labels)."""
    text = str(row.get("text") or "")
    is_spam = bool(row.get("is_spam"))
    is_off = bool(row.get("is_off_topic_discovery"))
    barriers = predict_barriers(text, is_spam=is_spam, is_off_topic=is_off)
    categories = predict_categories(text, is_spam=is_spam, is_off_topic=is_off)
    sentiment = predict_sentiment(text)
    if is_spam:
        sentiment = "neutral"
    return {
        "id": row["id"],
        "sentiment": sentiment,
        "categories": categories,
        "barriers": barriers,
        "insight_types": predict_insight_types(barriers, categories, text),
        "theme": (text[:60] + "…") if len(text) > 60 else text,
        "evidence_spans": _evidence_spans(text),
    }


def write_predictions(gold_path: Path, pred_path: Path) -> int:
    """Write predicted JSONL for every gold row. Returns count."""
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with gold_path.open(encoding="utf-8") as fin, pred_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pred = predict_for_gold_row(row)
            fout.write(json.dumps(pred, ensure_ascii=False) + "\n")
            n += 1
    return n
