"""CSV ingest for Phase 1 SQLite pipeline.

Supports both the schema used by `scripts/run_pipeline` seed corpora
(`id,source,text,...`) and collector outputs (`Review`, Reddit `Title`/`Body`).
"""

from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path
from typing import Iterable

from discovery_engine.config import RAW_DIR
from discovery_engine.schemas import FeedbackDocument
from discovery_engine.storage.db import utc_now

VALID_SOURCES = {
    "app_store",
    "play_store",
    "reddit",
    "product_review",
    "forum",
    "social",
    "nps",
    "other",
}


def content_hash(text: str) -> str:
    """Stable hash for deduplication (whitespace-normalized, lowercased)."""
    return hashlib.sha256(re.sub(r"\s+", " ", text.strip().lower()).encode("utf-8")).hexdigest()


def _extract_text(row: dict) -> str:
    """Pull primary feedback text from heterogeneous CSV schemas."""
    for key in ("text", "Review", "review", "body", "Body", "content", "Content"):
        val = (row.get(key) or "").strip()
        if val:
            return val
    title = (row.get("Title") or row.get("title") or "").strip()
    body = (row.get("Body") or row.get("body") or "").strip()
    comments = (row.get("Comments") or row.get("comments") or "").strip()
    combined = " ".join(p for p in (title, body, comments) if p).strip()
    return combined


def _infer_source_from_stem(stem: str) -> str | None:
    lower = stem.lower()
    if "blinkit_play" in lower or "play_store" in lower or lower.startswith("play"):
        return "play_store"
    if "reddit" in lower:
        return "reddit"
    if "app_store" in lower:
        return "app_store"
    for src in VALID_SOURCES:
        if lower.startswith(src):
            return src
    return None


def _row_to_doc(row: dict, fallback_source: str | None = None) -> FeedbackDocument | None:
    text = _extract_text(row)
    if not text:
        return None
    source = (row.get("source") or fallback_source or "other").strip()
    if source not in VALID_SOURCES:
        source = "other"
    doc_id = (row.get("id") or "").strip() or f"{source}_{content_hash(text)[:12]}"
    meta = {"content_hash": content_hash(text)}
    if row.get("channel_meta"):
        meta["raw_channel_meta"] = row["channel_meta"]
    if row.get("Rating") is not None and str(row.get("Rating")).strip() != "":
        meta["rating"] = row.get("Rating")
    return FeedbackDocument(
        id=doc_id,
        source=source,
        text=text,
        url=(row.get("url") or row.get("URL") or None),
        created_at=(row.get("created_at") or row.get("Date") or row.get("Created") or None),
        scraped_at=utc_now(),
        channel_meta=meta,
    )


def load_csv(path: Path, fallback_source: str | None = None) -> list[FeedbackDocument]:
    """Load feedback documents from a single CSV file."""
    docs: list[FeedbackDocument] = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc = _row_to_doc(row, fallback_source=fallback_source)
            if doc:
                docs.append(doc)
    return docs


def load_raw_dir(raw_dir: Path | None = None) -> list[FeedbackDocument]:
    """Load and dedupe all `*.csv` files under the raw data directory."""
    directory = raw_dir or RAW_DIR
    if not directory.exists():
        return []
    docs: list[FeedbackDocument] = []
    seen_hash: set[str] = set()
    for path in sorted(directory.glob("*.csv")):
        fallback = _infer_source_from_stem(path.stem)
        for doc in load_csv(path, fallback_source=fallback):
            h = doc.channel_meta.get("content_hash")
            if h in seen_hash:
                continue
            seen_hash.add(h)
            docs.append(doc)
    return docs


def dedupe_docs(docs: Iterable[FeedbackDocument]) -> list[FeedbackDocument]:
    """Drop documents that share the same content hash (first wins)."""
    out: list[FeedbackDocument] = []
    seen: set[str] = set()
    for d in docs:
        h = d.channel_meta.get("content_hash") or content_hash(d.text)
        if h in seen:
            continue
        seen.add(h)
        out.append(d)
    return out
