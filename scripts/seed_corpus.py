"""Create multi-source sample feedback CSVs for Phase 1 collectors."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
GOLD = ROOT / "data" / "gold" / "gold_labels.jsonl"

EXTRA = [
    {
        "id": "extra_app_001",
        "source": "app_store",
        "text": "Love the speed but I only ever reorder milk and eggs. The app never nudges me toward personal care.",
        "url": "https://example.local/app/1",
        "created_at": "2026-05-10T10:00:00+00:00",
    },
    {
        "id": "extra_play_001",
        "source": "play_store",
        "text": "Search for protein powder shows snacks. Hard to discover wellness and pharma categories.",
        "url": "https://example.local/play/1",
        "created_at": "2026-05-11T10:00:00+00:00",
    },
    {
        "id": "extra_reddit_001",
        "source": "reddit",
        "text": "Anyone else scared to buy meat on Blinkit-style apps after one bad freshness experience? I stuck to packaged snacks since.",
        "url": "https://example.local/reddit/1",
        "created_at": "2026-05-12T10:00:00+00:00",
    },
    {
        "id": "extra_review_001",
        "source": "product_review",
        "text": "Organic honey looks premium but without origin details and customer photos I won't try a new brand.",
        "url": "https://example.local/review/1",
        "created_at": "2026-05-13T10:00:00+00:00",
    },
    {
        "id": "extra_forum_001",
        "source": "forum",
        "text": "Homepage is all deals on chips. Cleaning supplies are invisible unless you already know the category path.",
        "url": "https://example.local/forum/1",
        "created_at": "2026-05-14T10:00:00+00:00",
    },
    {
        "id": "extra_app_002",
        "source": "app_store",
        "text": "Too many snack choices. I just reorder the same biscuits because browsing takes forever after work.",
        "url": "https://example.local/app/2",
        "created_at": "2026-05-15T10:00:00+00:00",
    },
    {
        "id": "extra_play_002",
        "source": "play_store",
        "text": "Baby wipes are fine but unit price is unclear so I won't switch pack sizes or try another brand.",
        "url": "https://example.local/play/2",
        "created_at": "2026-05-16T10:00:00+00:00",
    },
    {
        "id": "extra_reddit_002",
        "source": "reddit",
        "text": "Didn't know they sell pet litter. Recommendations only push cola and chips on my homepage feed.",
        "url": "https://example.local/reddit/2",
        "created_at": "2026-05-17T10:00:00+00:00",
    },
    {
        "id": "extra_review_002",
        "source": "product_review",
        "text": "Curd arrived close to expiry again. Until expiry is clear on the product page I avoid new dairy brands.",
        "url": "https://example.local/review/2",
        "created_at": "2026-05-18T10:00:00+00:00",
    },
    {
        "id": "extra_forum_002",
        "source": "forum",
        "text": "Bakery has very limited options. Not enough variety to move my breakfast bread order from the kirana.",
        "url": "https://example.local/forum/2",
        "created_at": "2026-05-19T10:00:00+00:00",
    },
    {
        "id": "extra_social_001",
        "source": "social",
        "text": "Instagram reel made me search oat milk but results were poor. Still curious if plant-based dairy gets better.",
        "url": "https://example.local/social/1",
        "created_at": "2026-05-20T10:00:00+00:00",
    },
    {
        "id": "extra_nps_001",
        "source": "nps",
        "text": "I like trying new beverages when ingredients are clear. Trial packs would make me explore beyond my usual cola.",
        "url": "https://example.local/nps/1",
        "created_at": "2026-05-21T10:00:00+00:00",
    },
    {
        "id": "extra_app_003",
        "source": "app_store",
        "text": "Personal care feels marked up versus the supermarket. Need a clear deal before trying a new shampoo here.",
        "url": "https://example.local/app/3",
        "created_at": "2026-05-22T10:00:00+00:00",
    },
    {
        "id": "extra_reddit_003",
        "source": "reddit",
        "text": "I only buy one milk brand. Substituting feels risky even when my usual SKU is out of stock.",
        "url": "https://example.local/reddit/3",
        "created_at": "2026-05-23T10:00:00+00:00",
    },
    {
        "id": "extra_play_003",
        "source": "play_store",
        "text": "App crash on checkout is annoying. Separately, I still won't try seafood after one spoiled order.",
        "url": "https://example.local/play/3",
        "created_at": "2026-05-24T10:00:00+00:00",
    },
    {
        "id": "extra_review_003",
        "source": "product_review",
        "text": "Would buy more household cleaners if the return policy for damaged bottles was clearer.",
        "url": "https://example.local/review/3",
        "created_at": "2026-05-25T10:00:00+00:00",
    },
    {
        "id": "extra_forum_003",
        "source": "forum",
        "text": "Friends told me to try premium fruit boxes but I need farm origin details before paying more.",
        "url": "https://example.local/forum/3",
        "created_at": "2026-05-26T10:00:00+00:00",
    },
    {
        "id": "extra_social_002",
        "source": "social",
        "text": "Yaar veggies quality risky lagti hai, isliye mostly biscuits aur cold drink hi order karta hoon.",
        "url": "https://example.local/social/2",
        "created_at": "2026-05-27T10:00:00+00:00",
    },
    {
        "id": "extra_nps_002",
        "source": "nps",
        "text": "Category browse for vitamins is buried under deals. Search keeps showing random packaged food.",
        "url": "https://example.local/nps/2",
        "created_at": "2026-05-28T10:00:00+00:00",
    },
    {
        "id": "extra_reddit_004",
        "source": "reddit",
        "text": "This app is only for late night snacks in my head. I would never do monthly staples shopping here.",
        "url": "https://example.local/reddit/4",
        "created_at": "2026-05-29T10:00:00+00:00",
    },
]


def load_gold_as_docs() -> list[dict]:
    rows = []
    if not GOLD.exists():
        return rows
    with GOLD.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            g = json.loads(line)
            rows.append(
                {
                    "id": g["id"],
                    "source": g["source"],
                    "text": g["text"],
                    "url": g.get("url"),
                    "created_at": g.get("created_at"),
                }
            )
    return rows


def write_split_csvs(docs: list[dict]) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for p in RAW.glob("*.csv"):
        p.unlink()

    by_source: dict[str, list[dict]] = {}
    for d in docs:
        by_source.setdefault(d["source"], []).append(d)

    fields = ["id", "source", "text", "url", "created_at"]
    for source, items in by_source.items():
        path = RAW / f"{source}_feedback.csv"
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in items:
                w.writerow({k: row.get(k) for k in fields})
        print(f"Wrote {len(items)} -> {path}")


def seed_corpus(limit_gold: int = 180) -> int:
    docs = load_gold_as_docs()
    # Prefer unique texts (Phase 0 seed expands templates with suffixes)
    uniq: list[dict] = []
    seen_text: set[str] = set()
    for d in docs:
        key = " ".join((d.get("text") or "").lower().split())
        # strip trailing boilerplate suffixes used in gold expansion
        for suffix in (
            " happened this month.",
            " same issue across two orders.",
            " sharing so the product team notices.",
            " compared with my local store.",
        ):
            if key.endswith(suffix):
                key = key[: -len(suffix)].rstrip(" .")
                break
        if key in seen_text:
            continue
        seen_text.add(key)
        uniq.append(d)
        if len(uniq) >= limit_gold:
            break

    seen_ids = {d["id"] for d in uniq}
    for e in EXTRA:
        if e["id"] not in seen_ids:
            uniq.append(e)
    write_split_csvs(uniq)
    print(f"Total documents seeded: {len(uniq)}")
    return len(uniq)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-gold", type=int, default=180)
    args = parser.parse_args()
    seed_corpus(limit_gold=args.limit_gold)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
