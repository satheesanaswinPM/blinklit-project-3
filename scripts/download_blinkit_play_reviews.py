"""
Download Google Play Store reviews for Blinkit and save to CSV.

Columns: Review, Rating, Date, Helpful Count

Usage (from repo root):
  python -m scripts.download_blinkit_play_reviews
  python -m scripts.download_blinkit_play_reviews --count 500
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

from google_play_scraper import Sort, reviews

REPO_ROOT = Path(__file__).resolve().parents[1]
BLINKIT_APP_ID = "com.grofers.customerapp"
DEFAULT_OUT = REPO_ROOT / "data" / "raw" / "blinkit_play_reviews.csv"
BATCH_SIZE = 200  # google-play-scraper max per request is typically ~200


def fetch_reviews(
    app_id: str,
    *,
    total: int,
    lang: str = "en",
    country: str = "in",
) -> list[dict]:
    """Fetch up to `total` reviews, newest first."""
    collected: list[dict] = []
    token = None

    while len(collected) < total:
        batch_count = min(BATCH_SIZE, total - len(collected))
        batch, token = reviews(
            app_id,
            lang=lang,
            country=country,
            sort=Sort.NEWEST,
            count=batch_count,
            continuation_token=token,
        )
        if not batch:
            break
        collected.extend(batch)
        if token is None:
            break

    return collected[:total]


def to_rows(raw_reviews: list[dict]) -> list[dict]:
    rows = []
    for r in raw_reviews:
        at = r.get("at")
        if isinstance(at, datetime):
            date_str = at.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        else:
            date_str = str(at) if at is not None else ""

        rows.append(
            {
                "Review": (r.get("content") or "").strip(),
                "Rating": r.get("score"),
                "Date": date_str,
                "Helpful Count": r.get("thumbsUpCount", 0),
            }
        )
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["Review", "Rating", "Date", "Helpful Count"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Blinkit Google Play Store reviews to CSV"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=200,
        help="Number of reviews to download (default: 200)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output CSV path (default: {DEFAULT_OUT})",
    )
    parser.add_argument("--lang", default="en", help="Review language (default: en)")
    parser.add_argument("--country", default="in", help="Store country (default: in)")
    parser.add_argument(
        "--app-id",
        default=BLINKIT_APP_ID,
        help=f"Play Store app id (default: {BLINKIT_APP_ID})",
    )
    args = parser.parse_args()

    print(f"Fetching up to {args.count} reviews for {args.app_id} ({args.country}/{args.lang})...")
    raw = fetch_reviews(
        args.app_id,
        total=args.count,
        lang=args.lang,
        country=args.country,
    )
    rows = to_rows(raw)
    rows = [r for r in rows if r["Review"]]

    write_csv(rows, args.out)
    print(f"Saved {len(rows)} reviews -> {args.out.resolve()}")


if __name__ == "__main__":
    main()
