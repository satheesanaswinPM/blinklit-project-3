"""
Download YouTube comments mentioning Blinkit via Data API v3.

Requires YOUTUBE_API_KEY in .env. If missing, exits 0 after writing/keeping
an empty-or-existing CSV so the multi-source merge can still run.

Usage:
  python -m scripts.download_youtube_comments --max-videos 8 --max-comments 80
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import urllib.parse
import urllib.request
import json

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from discovery_engine.env_loader import load_env  # noqa: E402

load_env()

DEFAULT_OUT = REPO_ROOT / "data" / "raw" / "youtube_comments.csv"


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "blinkit-discovery-engine/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def search_videos(api_key: str, query: str, max_videos: int) -> list[str]:
    q = urllib.parse.urlencode(
        {
            "part": "snippet",
            "q": query,
            "type": "video",
            "maxResults": max(1, min(max_videos, 25)),
            "key": api_key,
        }
    )
    data = _get_json(f"https://www.googleapis.com/youtube/v3/search?{q}")
    ids = []
    for item in data.get("items", []):
        vid = ((item.get("id") or {}).get("videoId")) or ""
        if vid:
            ids.append(vid)
    return ids


def fetch_comments(api_key: str, video_id: str, max_comments: int) -> list[dict]:
    rows: list[dict] = []
    page_token = None
    while len(rows) < max_comments:
        params = {
            "part": "snippet",
            "videoId": video_id,
            "maxResults": min(100, max_comments - len(rows)),
            "textFormat": "plainText",
            "key": api_key,
        }
        if page_token:
            params["pageToken"] = page_token
        q = urllib.parse.urlencode(params)
        try:
            data = _get_json(f"https://www.googleapis.com/youtube/v3/commentThreads?{q}")
        except Exception as exc:  # noqa: BLE001
            print(f"Comments failed for {video_id}: {exc}")
            break
        for item in data.get("items", []):
            sn = (((item.get("snippet") or {}).get("topLevelComment") or {}).get("snippet") or {})
            text = (sn.get("textDisplay") or "").strip()
            if not text:
                continue
            cid = (item.get("id") or f"{video_id}_{len(rows)}")
            rows.append(
                {
                    "id": f"youtube_{cid}",
                    "source": "youtube",
                    "date": sn.get("publishedAt", ""),
                    "rating": "",
                    "text": text,
                    "author": sn.get("authorDisplayName", ""),
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                    "scraped_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                }
            )
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["id", "source", "date", "rating", "text", "author", "url", "scraped_at"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download YouTube comments about Blinkit")
    parser.add_argument("--query", default="Blinkit review OR Blinkit delivery OR Blinkit app")
    parser.add_argument("--max-videos", type=int, default=8)
    parser.add_argument("--max-comments", type=int, default=60)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    api_key = os.getenv("YOUTUBE_API_KEY", "").strip()
    if not api_key:
        print("YOUTUBE_API_KEY missing — skipping YouTube collection.")
        if not args.out.exists():
            write_csv([], args.out)
        return 0

    video_ids = search_videos(api_key, args.query, args.max_videos)
    print(f"Found {len(video_ids)} videos")
    rows: list[dict] = []
    seen: set[str] = set()
    for vid in video_ids:
        batch = fetch_comments(api_key, vid, args.max_comments)
        for r in batch:
            if r["id"] in seen:
                continue
            seen.add(r["id"])
            rows.append(r)
        print(f"  {vid}: +{len(batch)} (total {len(rows)})")

    write_csv(rows, args.out)
    print(f"Saved {len(rows)} YouTube comments -> {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
