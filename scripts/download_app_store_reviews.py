"""
Download Apple App Store customer reviews via the public RSS feed.

Columns (unified-ready): id, source, date, rating, text, author, url, scraped_at

Usage:
  python -m scripts.download_app_store_reviews
  python -m scripts.download_app_store_reviews --pages 10 --country in
"""

from __future__ import annotations

import argparse
import csv
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "data" / "raw" / "app_store_reviews.csv"
BLINKIT_IOS_ID = "960335206"


def fetch_page(app_id: str, country: str, page: int) -> list[dict]:
    url = (
        f"https://itunes.apple.com/{country}/rss/customerreviews/"
        f"page={page}/id={app_id}/sortBy=mostRecent/json"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "blinkit-discovery-engine/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    entries = payload.get("feed", {}).get("entry", [])
    # First entry is often the app metadata
    reviews = []
    for entry in entries:
        if "im:rating" not in entry:
            continue
        review_id = (entry.get("id", {}) or {}).get("label", "")
        title = (entry.get("title", {}) or {}).get("label", "")
        body = (entry.get("content", {}) or {}).get("label", "")
        text = f"{title}. {body}".strip(". ").strip()
        author = ((entry.get("author") or {}).get("name") or {}).get("label", "")
        rating = (entry.get("im:rating") or {}).get("label", "")
        updated = (entry.get("updated") or {}).get("label", "")
        link = ""
        for l in entry.get("link", []) if isinstance(entry.get("link"), list) else []:
            if isinstance(l, dict) and l.get("attributes", {}).get("rel") == "related":
                link = l.get("attributes", {}).get("href", "")
        reviews.append(
            {
                "id": f"app_store_{review_id}",
                "source": "app_store",
                "date": updated,
                "rating": rating,
                "text": text,
                "author": author,
                "url": link or f"https://apps.apple.com/{country}/app/id{app_id}",
                "scraped_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            }
        )
    return reviews


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["id", "source", "date", "rating", "text", "author", "url", "scraped_at"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Blinkit App Store reviews (RSS)")
    parser.add_argument("--app-id", default=BLINKIT_IOS_ID)
    parser.add_argument("--country", default="in")
    parser.add_argument("--pages", type=int, default=10, help="RSS pages (~50 reviews each)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    seen: set[str] = set()
    rows: list[dict] = []
    for page in range(1, max(1, args.pages) + 1):
        try:
            batch = fetch_page(args.app_id, args.country, page)
        except Exception as exc:  # noqa: BLE001
            print(f"Page {page} failed: {exc}")
            break
        if not batch:
            break
        for r in batch:
            if r["id"] in seen or not r["text"]:
                continue
            seen.add(r["id"])
            rows.append(r)
        print(f"Page {page}: +{len(batch)} (total unique {len(rows)})")

    write_csv(rows, args.out)
    print(f"Saved {len(rows)} App Store reviews -> {args.out.resolve()}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
