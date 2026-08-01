from __future__ import annotations

import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from discovery_engine.config import GOLD_PATH, TAXONOMY_DIR
from discovery_engine.nlp.clean import clean_text, is_probably_spam, pick_evidence_span
from discovery_engine.schemas import Annotation, FeedbackDocument

_ANALYZER = SentimentIntensityAnalyzer()


@lru_cache(maxsize=1)
def _load_json(path_str: str) -> dict[str, Any]:
    return json.loads(Path(path_str).read_text(encoding="utf-8"))


def load_taxonomies() -> tuple[dict, dict, dict]:
    cats = _load_json(str(TAXONOMY_DIR / "categories.json"))
    bars = _load_json(str(TAXONOMY_DIR / "barriers.json"))
    insights = _load_json(str(TAXONOMY_DIR / "insight_types.json"))
    return cats, bars, insights


def _alias_map_categories(cats: dict) -> list[tuple[str, list[str]]]:
    out = []
    for c in cats["categories"]:
        aliases = [c["name"].lower(), c["id"].replace("_", " ")] + [
            a.lower() for a in c.get("aliases", [])
        ]
        out.append((c["id"], aliases))
    return out


def _barrier_keywords() -> dict[str, list[str]]:
    return {
        "trust_quality": ["trust", "fake", "adulteration", "unknown brand", "quality", "shady"],
        "price_value": ["expensive", "pricey", "markup", "cheaper", "not worth", "costly", "deal"],
        "freshness_spoilage": [
            "expiry",
            "expired",
            "rotten",
            "spoiled",
            "stale",
            "fresh",
            "smelled off",
            "near expiry",
        ],
        "brand_lock_in": ["only buy", "stick to", "my usual", "substitut", "out of stock"],
        "discovery_invisibility": [
            "didn't know",
            "didnt know",
            "never shows",
            "homepage",
            "don't see",
            "dont see",
            "invisible",
        ],
        "search_vocab_gap": ["searched", "search", "zero results", "nothing relevant", "query"],
        "assortment_doubt": ["limited options", "not enough", "missing", "only 2", "variety"],
        "habit_mental_model": [
            "only use",
            "only for",
            "always the same",
            "reorder",
            "monthly groceries",
            "late night",
        ],
        "cognitive_load": ["no time", "too many choices", "browsing", "takes forever"],
        "past_bad_experience": ["never again", "last time", "tried once", "won't try", "wont try"],
        "info_trust_gap": [
            "expiry on",
            "need",
            "show clear",
            "ingredients",
            "customer photos",
            "unit price",
            "origin",
            "return policy",
        ],
    }


def _match_categories(text: str, cat_aliases: list[tuple[str, list[str]]]) -> list[str]:
    tl = text.lower()
    hits = []
    for cid, aliases in cat_aliases:
        if cid in {"unknown", "off_topic"}:
            continue
        if any(a and a in tl for a in aliases):
            hits.append(cid)
    return hits


def _match_barriers(text: str) -> list[str]:
    tl = text.lower()
    hits = []
    for bid, kws in _barrier_keywords().items():
        if any(k in tl for k in kws):
            hits.append(bid)
    return hits


def _match_discovery_paths(text: str, path_ids: list[str]) -> list[str]:
    tl = text.lower()
    rules = {
        "search": ["search", "searched", "query"],
        "homepage_feed": ["homepage", "home page", "feed"],
        "category_browse": ["browse", "aisle", "category"],
        "recommendations": ["recommend"],
        "promos_banners": ["deal", "promo", "banner", "offer"],
        "reorder_history": ["reorder", "usual list"],
        "external_social": ["instagram", "reddit", "twitter", "reel"],
        "word_of_mouth": ["friends", "word of mouth", "told me"],
    }
    hits = []
    for pid in path_ids:
        kws = rules.get(pid, [])
        if any(k in tl for k in kws):
            hits.append(pid)
    return hits


def _match_info_needs(text: str, need_ids: list[str]) -> list[str]:
    tl = text.lower()
    rules = {
        "expiry_date": ["expiry", "expire"],
        "origin_source": ["origin", "farm"],
        "ingredients": ["ingredient", "sugar info"],
        "unit_price": ["unit price"],
        "customer_photos": ["customer photo", "photos"],
        "ratings_reviews": ["rating", "reviews"],
        "brand_authenticity": ["authenticity", "fake", "adulteration", "unknown brand"],
        "storage_handling": ["storage", "handling", "smelled"],
        "return_refund_policy": ["return", "refund"],
        "portion_size_clarity": ["pack size", "portion", "size"],
    }
    hits = []
    for nid in need_ids:
        kws = rules.get(nid, [])
        if any(k in tl for k in kws):
            hits.append(nid)
    return hits


def _insight_types_for(
    barriers: list[str],
    discovery_paths: list[str],
    info_needs: list[str],
    text: str,
) -> list[str]:
    types: list[str] = []
    habit_b = {"habit_mental_model", "cognitive_load", "brand_lock_in"}
    if set(barriers) & habit_b:
        types.append("habit_drivers")
    if any(b != "none" for b in barriers):
        types.append("barrier_taxonomy")
    if discovery_paths:
        types.append("discovery_path_map")
    if info_needs or "info_trust_gap" in barriers:
        types.append("info_needs")
    tl = text.lower()
    if any(
        x in tl
        for x in (
            "would buy",
            "curious",
            "like trying",
            "trial pack",
            "pet",
            "baby",
            "organic",
            "health",
        )
    ):
        types.append("receptive_segments")
    if any(
        x in tl
        for x in ("should", "need", "show", "trial", "recommend", "wishlist", "if you")
    ) or info_needs:
        types.append("opportunity_briefs")
    if not types:
        types.append("barrier_taxonomy")
    # unique preserve order
    seen = set()
    out = []
    for t in types:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _sentiment(text: str) -> str:
    s = _ANALYZER.polarity_scores(text)
    if s["compound"] >= 0.25:
        return "positive"
    if s["compound"] <= -0.25:
        return "negative"
    return "neutral"


def _is_off_topic(text: str, categories: list[str]) -> bool:
    tl = text.lower()
    ops = ["crash", "payment failed", "delivery partner", "refund", "customer support", "otp"]
    if any(o in tl for o in ops) and not categories:
        return True
    return False


def _theme_hint(barriers: list[str], categories: list[str], text: str) -> str:
    cat = categories[0].replace("_", " ") if categories else "general"
    if barriers and barriers[0] != "none":
        return f"{cat} / {barriers[0].replace('_', ' ')}"
    words = [w for w in re.findall(r"[A-Za-z]{4,}", text.lower())][:4]
    return " ".join(words) if words else "feedback theme"


@lru_cache(maxsize=1)
def _gold_barrier_priors() -> dict[str, Counter]:
    """Optional boost from Phase 0 gold themes (keyword -> barrier counts)."""
    path = GOLD_PATH
    priors: dict[str, Counter] = {}
    if not path.exists():
        return priors
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for span in row.get("evidence_spans") or []:
                key = clean_text(span).lower()
                if len(key) < 8:
                    continue
                priors.setdefault(key, Counter()).update(row.get("barriers") or [])
    return priors


class TaxonomyLabeler:
    """Rule + taxonomy alias labeler with Phase 0 gold span priors."""

    def __init__(self, model_ver: str):
        self.model_ver = model_ver
        cats, bars, insights = load_taxonomies()
        self.cat_aliases = _alias_map_categories(cats)
        self.barrier_ids = {b["id"] for b in bars["barriers"]}
        rq = insights["research_questions"]
        self.discovery_paths = next(
            (q.get("discovery_paths", []) for q in rq if q["insight_type"] == "discovery_path_map"),
            [],
        )
        self.info_needs = next(
            (q.get("info_need_tags", []) for q in rq if q["insight_type"] == "info_needs"),
            [],
        )

    def label_doc(self, doc: FeedbackDocument, pipeline_run_id: str) -> Annotation:
        text = clean_text(doc.text)
        spam = is_probably_spam(text)
        categories = _match_categories(text, self.cat_aliases)
        off = _is_off_topic(text, categories)
        if off:
            categories = ["off_topic"]
        if not categories and not spam:
            categories = ["unknown"]

        barriers = _match_barriers(text)
        # gold prior boost: if an evidence-like phrase appears, add those barriers
        tl = text.lower()
        for span, counter in _gold_barrier_priors().items():
            if span in tl:
                barriers.extend([b for b, _ in counter.most_common(2)])

        barriers = [b for b in dict.fromkeys(barriers) if b in self.barrier_ids]
        if spam or off:
            barriers = ["none"]
        elif not barriers:
            barriers = ["other"] if "try" in tl or "new" in tl else ["habit_mental_model"]

        discovery = [] if (spam or off) else _match_discovery_paths(text, self.discovery_paths)
        info = [] if (spam or off) else _match_info_needs(text, self.info_needs)
        insights = (
            ["barrier_taxonomy"]
            if (spam or off)
            else _insight_types_for(barriers, discovery, info, text)
        )
        sentiment = _sentiment(text)
        span = pick_evidence_span(doc.text)
        conf = 0.4 + 0.1 * min(3, len(barriers)) + (
            0.2 if categories and categories[0] != "unknown" else 0
        )
        if spam:
            conf = 0.9

        return Annotation(
            doc_id=doc.id,
            sentiment=sentiment,
            categories=categories,
            barriers=barriers,
            insight_types=insights,
            discovery_paths=discovery,
            info_needs=info,
            theme_hint=_theme_hint(barriers, categories, text),
            evidence_spans=[span],
            is_spam=spam,
            is_off_topic_discovery=off,
            embedding_id=f"emb:{doc.id}",
            model_ver=self.model_ver,
            pipeline_run_id=pipeline_run_id,
            confidence=min(conf, 0.95),
        )
