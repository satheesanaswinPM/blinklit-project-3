"""
Merge multi-source feedback into a unified corpus for exploration analysis.

Sources (when present):
  - data/raw/blinkit_play_reviews.csv
  - data/raw/app_store_reviews.csv
  - data/raw/reddit_posts.csv / reddit_feedback.csv
  - data/raw/youtube_comments.csv

Output:
  data/processed/merged_reviews.csv
  Columns: id, source, date, rating, text, author, url, scraped_at
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "merged_reviews.csv"

UNIFIED_COLS = ["id", "source", "date", "rating", "text", "author", "url", "scraped_at"]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _hash_id(source: str, text: str) -> str:
    h = hashlib.sha256(re.sub(r"\s+", " ", text.strip().lower()).encode("utf-8")).hexdigest()[:12]
    return f"{source}_{h}"


def _from_play(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    rows = []
    for i, r in df.iterrows():
        text = str(r.get("Review") or "").strip()
        if not text:
            continue
        rows.append(
            {
                "id": _hash_id("play_store", text),
                "source": "play_store",
                "date": str(r.get("Date") or ""),
                "rating": r.get("Rating", ""),
                "text": text,
                "author": "",
                "url": "",
                "scraped_at": _now(),
            }
        )
    return pd.DataFrame(rows)


def _from_unified(path: Path, default_source: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "text" not in df.columns and "Review" in df.columns:
        df = df.rename(columns={"Review": "text"})
    if "text" not in df.columns and "Body" in df.columns:
        title = df["Title"].fillna("") if "Title" in df.columns else ""
        body = df["Body"].fillna("")
        comments = df["Comments"].fillna("") if "Comments" in df.columns else ""
        df["text"] = (title.astype(str) + " " + body.astype(str) + " " + comments.astype(str)).str.strip()
    out = []
    for _, r in df.iterrows():
        text = str(r.get("text") or "").strip()
        if not text:
            continue
        source = str(r.get("source") or default_source).strip() or default_source
        rid = str(r.get("id") or "").strip() or _hash_id(source, text)
        out.append(
            {
                "id": rid,
                "source": source,
                "date": str(r.get("date") or r.get("Date") or r.get("Created") or ""),
                "rating": r.get("rating", r.get("Rating", "")),
                "text": text,
                "author": str(r.get("author") or r.get("Author") or ""),
                "url": str(r.get("url") or r.get("URL") or ""),
                "scraped_at": str(r.get("scraped_at") or _now()),
            }
        )
    return pd.DataFrame(out)


def merge_sources(
    *,
    play_path: Path | None = None,
    app_store_path: Path | None = None,
    reddit_path: Path | None = None,
    youtube_path: Path | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    play_path = play_path or (RAW / "blinkit_play_reviews.csv")
    app_store_path = app_store_path or (RAW / "app_store_reviews.csv")
    youtube_path = youtube_path or (RAW / "youtube_comments.csv")
    reddit_candidates = [
        reddit_path,
        RAW / "reddit_posts.csv",
        RAW / "reddit_feedback.csv",
    ]

    if play_path.exists():
        frames.append(_from_play(play_path))
    if app_store_path.exists():
        frames.append(_from_unified(app_store_path, "app_store"))
    if youtube_path.exists():
        frames.append(_from_unified(youtube_path, "youtube"))
    for rp in reddit_candidates:
        if rp is not None and Path(rp).exists():
            frames.append(_from_unified(Path(rp), "reddit"))
            break

    if not frames:
        return pd.DataFrame(columns=UNIFIED_COLS)

    df = pd.concat(frames, ignore_index=True)
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]
    # Soft exact dedupe across sources
    key = df["text"].str.casefold().str.replace(r"\s+", " ", regex=True)
    df = df.loc[~key.duplicated()].reset_index(drop=True)
    return df[UNIFIED_COLS]


def save_merged(df: pd.DataFrame, path: Path = OUT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    return path


if __name__ == "__main__":
    merged = merge_sources()
    out = save_merged(merged)
    print(f"Merged {len(merged)} items -> {out}")
    if len(merged):
        print(merged["source"].value_counts().to_string())
