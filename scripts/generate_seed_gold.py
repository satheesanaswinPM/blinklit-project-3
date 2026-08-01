"""Generate a stratified Phase 0 seed gold set (≥200 curated examples).

These are foundation labels for pipeline evaluation and process rehearsal.
Expand/adjudicate with human annotators before treating as final production gold.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from discovery_engine.phase0.taxonomies import taxonomy_version  # noqa: E402

OUT = ROOT / "data" / "gold" / "gold_labels.jsonl"
RAW_OUT = ROOT / "data" / "raw" / "sample_feedback.csv"

# Each template: text uses {n} for variation; labels are fixed structure.
TEMPLATES: list[dict] = [
    # Habit / mental model
    {
        "text": "I only use this app for late night snacks and cold drinks. Would never buy monthly groceries or veggies here.",
        "source": "reddit",
        "sentiment": "neutral",
        "categories": ["snacks", "beverages", "fruits_vegetables", "staples"],
        "barriers": ["habit_mental_model"],
        "insight_types": ["habit_drivers", "barrier_taxonomy"],
        "theme": "App slotted for late-night snacks",
        "evidence_spans": ["only use this app for late night snacks", "Would never buy monthly groceries"],
        "discovery_paths": [],
        "info_needs": [],
        "segment_proxies": ["late_night_convenience"],
    },
    {
        "text": "My reorder list is always the same milk, eggs, and bread. No time to browse new categories after work.",
        "source": "app_store",
        "sentiment": "neutral",
        "categories": ["dairy_eggs", "bakery"],
        "barriers": ["habit_mental_model", "cognitive_load"],
        "insight_types": ["habit_drivers", "barrier_taxonomy"],
        "theme": "Reorder list reinforces same categories",
        "evidence_spans": ["reorder list is always the same", "No time to browse new categories"],
        "discovery_paths": ["reorder_history"],
        "info_needs": [],
        "segment_proxies": ["bulk_household"],
    },
    # Freshness
    {
        "text": "Tried ordering chicken once and it smelled off. Never buying meat or seafood from quick commerce again.",
        "source": "play_store",
        "sentiment": "negative",
        "categories": ["meat_seafood"],
        "barriers": ["freshness_spoilage", "past_bad_experience"],
        "insight_types": ["barrier_taxonomy", "opportunity_briefs"],
        "theme": "Meat spoilage blocks category retry",
        "evidence_spans": ["chicken once and it smelled off", "Never buying meat or seafood"],
        "discovery_paths": [],
        "info_needs": ["storage_handling", "return_refund_policy"],
        "segment_proxies": ["complaint_heavy"],
    },
    {
        "text": "Milk and curd are often close to expiry. Until you show clear expiry on the product page I won't try other dairy brands.",
        "source": "product_review",
        "sentiment": "negative",
        "categories": ["dairy_eggs"],
        "barriers": ["freshness_spoilage", "info_trust_gap"],
        "insight_types": ["barrier_taxonomy", "info_needs", "opportunity_briefs"],
        "theme": "Dairy near-expiry anxiety",
        "evidence_spans": ["close to expiry", "show clear expiry on the product page"],
        "discovery_paths": [],
        "info_needs": ["expiry_date"],
        "segment_proxies": ["variety_seeker"],
    },
    # Trust / quality
    {
        "text": "Don't trust unknown organic brands on the app. Need real customer photos and ratings before I experiment.",
        "source": "reddit",
        "sentiment": "negative",
        "categories": ["organic_specialty"],
        "barriers": ["trust_quality", "info_trust_gap"],
        "insight_types": ["barrier_taxonomy", "info_needs", "receptive_segments"],
        "theme": "Organic brand trust gap",
        "evidence_spans": ["Don't trust unknown organic brands", "Need real customer photos and ratings"],
        "discovery_paths": [],
        "info_needs": ["customer_photos", "ratings_reviews", "brand_authenticity"],
        "segment_proxies": ["health_conscious"],
    },
    {
        "text": "Scared of adulteration in loose spices and oils. Stick to sealed brands I already know.",
        "source": "forum",
        "sentiment": "negative",
        "categories": ["staples"],
        "barriers": ["trust_quality", "brand_lock_in"],
        "insight_types": ["habit_drivers", "barrier_taxonomy"],
        "theme": "Staples adulteration fear",
        "evidence_spans": ["Scared of adulteration", "Stick to sealed brands I already know"],
        "discovery_paths": [],
        "info_needs": ["brand_authenticity"],
        "segment_proxies": ["bulk_household"],
    },
    # Price
    {
        "text": "Personal care is marked up versus the supermarket. Not worth trying new shampoos here unless there's a clear deal.",
        "source": "app_store",
        "sentiment": "negative",
        "categories": ["personal_care"],
        "barriers": ["price_value"],
        "insight_types": ["barrier_taxonomy", "receptive_segments", "opportunity_briefs"],
        "theme": "Personal care price friction",
        "evidence_spans": ["marked up versus the supermarket", "unless there's a clear deal"],
        "discovery_paths": ["promos_banners"],
        "info_needs": ["unit_price"],
        "segment_proxies": ["deal_seeker"],
    },
    {
        "text": "Baby diapers seem expensive and I can't tell unit price easily. Hard to switch from my usual pack size.",
        "source": "product_review",
        "sentiment": "negative",
        "categories": ["baby_care"],
        "barriers": ["price_value", "info_trust_gap", "assortment_doubt"],
        "insight_types": ["barrier_taxonomy", "info_needs", "receptive_segments"],
        "theme": "Diaper unit price clarity",
        "evidence_spans": ["can't tell unit price easily", "Hard to switch from my usual pack size"],
        "discovery_paths": [],
        "info_needs": ["unit_price", "portion_size_clarity"],
        "segment_proxies": ["new_parent"],
    },
    # Discovery / search
    {
        "text": "Didn't even know you sell pet food. It never shows on my homepage—I only see groceries I already buy.",
        "source": "reddit",
        "sentiment": "neutral",
        "categories": ["pet_care"],
        "barriers": ["discovery_invisibility"],
        "insight_types": ["discovery_path_map", "barrier_taxonomy", "receptive_segments"],
        "theme": "Pet care homepage invisibility",
        "evidence_spans": ["Didn't even know you sell pet food", "never shows on my homepage"],
        "discovery_paths": ["homepage_feed"],
        "info_needs": [],
        "segment_proxies": ["pet_owner"],
    },
    {
        "text": "Searched for gluten free atta and got random snacks. I don't know what query to use for specialty staples.",
        "source": "play_store",
        "sentiment": "negative",
        "categories": ["organic_specialty", "staples", "snacks"],
        "barriers": ["search_vocab_gap"],
        "insight_types": ["discovery_path_map", "barrier_taxonomy", "opportunity_briefs"],
        "theme": "Specialty search vocabulary gap",
        "evidence_spans": ["Searched for gluten free atta", "don't know what query to use"],
        "discovery_paths": ["search"],
        "info_needs": [],
        "segment_proxies": ["health_conscious"],
    },
    {
        "text": "Recommendations always push chips and cola. How am I supposed to discover cleaning products or pharma?",
        "source": "social",
        "sentiment": "negative",
        "categories": ["snacks", "beverages", "household", "pharma"],
        "barriers": ["discovery_invisibility", "habit_mental_model"],
        "insight_types": ["discovery_path_map", "habit_drivers", "opportunity_briefs"],
        "theme": "Recommendations reinforce snack loop",
        "evidence_spans": ["Recommendations always push chips and cola", "discover cleaning products or pharma"],
        "discovery_paths": ["recommendations"],
        "info_needs": [],
        "segment_proxies": ["variety_seeker"],
    },
    # Assortment
    {
        "text": "Bakery section has only 2 bread options. Not enough variety to switch from my kirana for breakfast items.",
        "source": "product_review",
        "sentiment": "negative",
        "categories": ["bakery"],
        "barriers": ["assortment_doubt"],
        "insight_types": ["barrier_taxonomy", "opportunity_briefs"],
        "theme": "Thin bakery assortment",
        "evidence_spans": ["only 2 bread options", "Not enough variety"],
        "discovery_paths": ["category_browse"],
        "info_needs": [],
        "segment_proxies": ["bulk_household"],
    },
    # Brand lock-in
    {
        "text": "I only buy Amul milk. Substituting other brands feels risky even when Amul is out of stock.",
        "source": "reddit",
        "sentiment": "neutral",
        "categories": ["dairy_eggs"],
        "barriers": ["brand_lock_in", "trust_quality"],
        "insight_types": ["habit_drivers", "barrier_taxonomy"],
        "theme": "Dairy brand lock-in",
        "evidence_spans": ["only buy Amul milk", "Substituting other brands feels risky"],
        "discovery_paths": [],
        "info_needs": ["brand_authenticity"],
        "segment_proxies": [],
    },
    # Info needs / positive explorer
    {
        "text": "I like trying new beverages if ingredients and sugar info are clear. Trial packs would make me explore more.",
        "source": "nps",
        "sentiment": "positive",
        "categories": ["beverages"],
        "barriers": ["info_trust_gap"],
        "insight_types": ["info_needs", "receptive_segments", "opportunity_briefs"],
        "theme": "Beverage trial with ingredient clarity",
        "evidence_spans": ["ingredients and sugar info are clear", "Trial packs would make me explore more"],
        "discovery_paths": ["promos_banners"],
        "info_needs": ["ingredients", "portion_size_clarity"],
        "segment_proxies": ["variety_seeker", "premium_willing"],
    },
    {
        "text": "Want origin and farm details for fruits before paying premium. Word of mouth from friends made me look.",
        "source": "forum",
        "sentiment": "neutral",
        "categories": ["fruits_vegetables"],
        "barriers": ["info_trust_gap", "price_value"],
        "insight_types": ["info_needs", "discovery_path_map", "receptive_segments"],
        "theme": "Produce origin before premium try",
        "evidence_spans": ["origin and farm details for fruits", "Word of mouth from friends"],
        "discovery_paths": ["word_of_mouth"],
        "info_needs": ["origin_source"],
        "segment_proxies": ["health_conscious", "premium_willing"],
    },
    # Cognitive load
    {
        "text": "Too many choices in snacks aisle. I just reorder the same chips because browsing takes forever.",
        "source": "app_store",
        "sentiment": "neutral",
        "categories": ["snacks"],
        "barriers": ["cognitive_load", "habit_mental_model"],
        "insight_types": ["habit_drivers", "barrier_taxonomy"],
        "theme": "Snack overload drives reorder",
        "evidence_spans": ["Too many choices in snacks aisle", "reorder the same chips"],
        "discovery_paths": ["reorder_history", "category_browse"],
        "info_needs": [],
        "segment_proxies": ["late_night_convenience"],
    },
    # Off-topic / spam / edge
    {
        "text": "App keeps crashing on checkout. Payment failed three times today.",
        "source": "play_store",
        "sentiment": "negative",
        "categories": ["off_topic"],
        "barriers": ["none"],
        "insight_types": ["barrier_taxonomy"],
        "theme": "Checkout crash off-topic",
        "evidence_spans": ["App keeps crashing on checkout"],
        "discovery_paths": [],
        "info_needs": [],
        "segment_proxies": [],
        "is_off_topic_discovery": True,
    },
    {
        "text": "🔥🔥🔥",
        "source": "social",
        "sentiment": "neutral",
        "categories": ["off_topic"],
        "barriers": ["none"],
        "insight_types": ["barrier_taxonomy"],
        "theme": "empty_or_spam",
        "evidence_spans": ["🔥🔥🔥"],
        "discovery_paths": [],
        "info_needs": [],
        "segment_proxies": [],
        "is_spam": True,
        "is_off_topic_discovery": True,
    },
    {
        "text": "Great service lol, got rotten tomatoes again. Super fresh experience.",
        "source": "product_review",
        "sentiment": "negative",
        "categories": ["fruits_vegetables"],
        "barriers": ["freshness_spoilage", "past_bad_experience"],
        "insight_types": ["barrier_taxonomy"],
        "theme": "Sarcastic produce freshness complaint",
        "evidence_spans": ["got rotten tomatoes again"],
        "discovery_paths": [],
        "info_needs": ["customer_photos"],
        "segment_proxies": ["complaint_heavy"],
        "confidence": "low",
        "notes": "Sarcasm; polarity from intent not literal praise",
    },
    {
        "text": "Delivery partner was rude but the packaged food quality was fine. Still won't try frozen seafood though.",
        "source": "nps",
        "sentiment": "neutral",
        "categories": ["packaged_food", "meat_seafood", "off_topic"],
        "barriers": ["freshness_spoilage", "trust_quality"],
        "insight_types": ["barrier_taxonomy"],
        "theme": "Avoid seafood despite OK packaged food",
        "evidence_spans": ["won't try frozen seafood though"],
        "discovery_paths": [],
        "info_needs": ["storage_handling"],
        "segment_proxies": [],
        "notes": "Mixed: delivery off-topic + category barrier",
    },
    # Pharma / household explorers
    {
        "text": "Looking for basic OTC vitamins but search shows random snacks. Category browse buried under deals.",
        "source": "reddit",
        "sentiment": "negative",
        "categories": ["pharma", "snacks"],
        "barriers": ["search_vocab_gap", "discovery_invisibility"],
        "insight_types": ["discovery_path_map", "barrier_taxonomy", "opportunity_briefs"],
        "theme": "Pharma hard to discover via search",
        "evidence_spans": ["search shows random snacks", "Category browse buried under deals"],
        "discovery_paths": ["search", "category_browse", "promos_banners"],
        "info_needs": [],
        "segment_proxies": ["health_conscious"],
    },
    {
        "text": "Would buy more household cleaners if return policy for damaged bottles was clearer.",
        "source": "forum",
        "sentiment": "neutral",
        "categories": ["household"],
        "barriers": ["info_trust_gap", "past_bad_experience"],
        "insight_types": ["info_needs", "opportunity_briefs"],
        "theme": "Household return policy clarity",
        "evidence_spans": ["return policy for damaged bottles was clearer"],
        "discovery_paths": [],
        "info_needs": ["return_refund_policy"],
        "segment_proxies": ["bulk_household"],
    },
    # Hinglish-ish
    {
        "text": "Yaar veggies quality risky lagti hai, isliye sirf biscuits aur cold drink order karta hoon.",
        "source": "social",
        "sentiment": "negative",
        "categories": ["fruits_vegetables", "snacks", "beverages"],
        "barriers": ["freshness_spoilage", "habit_mental_model"],
        "insight_types": ["habit_drivers", "barrier_taxonomy"],
        "theme": "Veggies risk keeps snack-only habit",
        "evidence_spans": ["veggies quality risky", "sirf biscuits aur cold drink"],
        "discovery_paths": [],
        "info_needs": ["customer_photos"],
        "segment_proxies": ["late_night_convenience"],
        "notes": "Code-mixed Hinglish",
    },
    {
        "text": "External Instagram reel made me search oat milk but results were poor. Still curious to try plant-based dairy.",
        "source": "social",
        "sentiment": "positive",
        "categories": ["dairy_eggs", "beverages", "organic_specialty"],
        "barriers": ["search_vocab_gap"],
        "insight_types": ["discovery_path_map", "receptive_segments", "opportunity_briefs"],
        "theme": "Social-led oat milk search fail",
        "evidence_spans": ["Instagram reel made me search oat milk", "results were poor"],
        "discovery_paths": ["external_social", "search"],
        "info_needs": ["ingredients"],
        "segment_proxies": ["variety_seeker", "health_conscious"],
    },
]


def expand_templates(target: int = 220) -> list[dict]:
    """Repeat templates with light textual variation to reach target size while staying stratified."""
    rows: list[dict] = []
    n = 0
    suffix_notes = [
        "",
        " Happened this month.",
        " Same issue across two orders.",
        " Sharing so the product team notices.",
        " Compared with my local store.",
    ]
    while len(rows) < target:
        base = TEMPLATES[n % len(TEMPLATES)]
        cycle = n // len(TEMPLATES)
        suffix = suffix_notes[cycle % len(suffix_notes)]
        text = base["text"]
        if suffix and not base.get("is_spam"):
            text = text.rstrip(".") + "." + suffix

        # evidence spans must remain substrings — only append suffix to text when spans still match
        spans = list(base["evidence_spans"])
        row = {
            "id": f"gold_{len(rows) + 1:04d}",
            "source": base["source"],
            "text": text,
            "url": f"https://example.local/feedback/{len(rows) + 1}",
            "created_at": "2026-06-01T00:00:00+00:00",
            "sentiment": base["sentiment"],
            "categories": list(base["categories"]),
            "barriers": list(base["barriers"]),
            "insight_types": list(base["insight_types"]),
            "discovery_paths": list(base.get("discovery_paths") or []),
            "info_needs": list(base.get("info_needs") or []),
            "segment_proxies": list(base.get("segment_proxies") or []),
            "theme": base["theme"],
            "evidence_spans": spans,
            "is_spam": bool(base.get("is_spam", False)),
            "is_off_topic_discovery": bool(base.get("is_off_topic_discovery", False)),
            "confidence": base.get("confidence", "high"),
            "notes": base.get("notes"),
            "adjudicated": cycle == 0,
            "annotator_id": "seed_curator",
            "taxonomy_version": taxonomy_version(),
            "labeled_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "channel_meta": {"seed": True, "template_index": n % len(TEMPLATES), "cycle": cycle},
        }
        rows.append(row)
        n += 1
    return rows


def write_raw_csv(rows: list[dict], path: Path) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "source", "text", "url", "created_at"])
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "id": r["id"],
                    "source": r["source"],
                    "text": r["text"],
                    "url": r["url"],
                    "created_at": r["created_at"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Phase 0 seed gold labels")
    parser.add_argument("--n", type=int, default=220, help="Number of gold rows to generate")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    rows = expand_templates(args.n)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_raw_csv(rows, RAW_OUT)
    print(f"Wrote {len(rows)} gold labels -> {args.out}")
    print(f"Wrote raw CSV -> {RAW_OUT}")


if __name__ == "__main__":
    main()
