from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

from discovery_engine.config import DB_PATH, PROCESSED_DIR
from discovery_engine.schemas import Annotation, FeedbackDocument, ThemeCard


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def db_session(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    conn = connect(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS pipeline_runs (
  id TEXT PRIMARY KEY,
  started_at TEXT NOT NULL,
  finished_at TEXT,
  status TEXT NOT NULL,
  n_docs INTEGER DEFAULT 0,
  n_themes INTEGER DEFAULT 0,
  model_ver TEXT NOT NULL,
  notes TEXT
);

CREATE TABLE IF NOT EXISTS feedback_documents (
  id TEXT PRIMARY KEY,
  source TEXT NOT NULL,
  text TEXT NOT NULL,
  url TEXT,
  created_at TEXT,
  scraped_at TEXT,
  channel_meta TEXT NOT NULL DEFAULT '{}',
  content_hash TEXT,
  pipeline_run_id TEXT
);

CREATE TABLE IF NOT EXISTS annotations (
  doc_id TEXT PRIMARY KEY,
  sentiment TEXT NOT NULL,
  categories TEXT NOT NULL,
  barriers TEXT NOT NULL,
  insight_types TEXT NOT NULL,
  discovery_paths TEXT NOT NULL,
  info_needs TEXT NOT NULL,
  theme_hint TEXT,
  evidence_spans TEXT NOT NULL,
  is_spam INTEGER NOT NULL DEFAULT 0,
  is_off_topic_discovery INTEGER NOT NULL DEFAULT 0,
  embedding_id TEXT,
  model_ver TEXT NOT NULL,
  pipeline_run_id TEXT NOT NULL,
  confidence REAL NOT NULL DEFAULT 0,
  FOREIGN KEY(doc_id) REFERENCES feedback_documents(id)
);

CREATE TABLE IF NOT EXISTS theme_cards (
  id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  volume INTEGER NOT NULL,
  sentiment_mix TEXT NOT NULL,
  barriers TEXT NOT NULL,
  categories TEXT NOT NULL,
  insight_types TEXT NOT NULL,
  discovery_paths TEXT NOT NULL,
  info_needs TEXT NOT NULL,
  evidence TEXT NOT NULL,
  pipeline_run_id TEXT NOT NULL,
  model_ver TEXT NOT NULL,
  coherence REAL NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_docs_source ON feedback_documents(source);
CREATE INDEX IF NOT EXISTS idx_themes_run ON theme_cards(pipeline_run_id);
"""


def init_db(db_path: Path | None = None) -> Path:
    path = db_path or DB_PATH
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with db_session(path) as conn:
        conn.executescript(SCHEMA_SQL)
    return path


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _loads(s: str | None, default: Any) -> Any:
    if not s:
        return default
    return json.loads(s)


def clear_run_outputs(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM theme_cards")
    conn.execute("DELETE FROM annotations")
    conn.execute("DELETE FROM feedback_documents")


def insert_pipeline_run(
    conn: sqlite3.Connection,
    run_id: str,
    model_ver: str,
    status: str = "running",
    notes: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO pipeline_runs (id, started_at, status, model_ver, notes)
        VALUES (?, ?, ?, ?, ?)
        """,
        (run_id, utc_now(), status, model_ver, notes),
    )


def finish_pipeline_run(
    conn: sqlite3.Connection,
    run_id: str,
    status: str,
    n_docs: int,
    n_themes: int,
) -> None:
    conn.execute(
        """
        UPDATE pipeline_runs
        SET finished_at = ?, status = ?, n_docs = ?, n_themes = ?
        WHERE id = ?
        """,
        (utc_now(), status, n_docs, n_themes, run_id),
    )


def upsert_documents(conn: sqlite3.Connection, docs: list[FeedbackDocument], run_id: str) -> int:
    n = 0
    for d in docs:
        conn.execute(
            """
            INSERT OR REPLACE INTO feedback_documents
            (id, source, text, url, created_at, scraped_at, channel_meta, content_hash, pipeline_run_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                d.id,
                d.source,
                d.text,
                d.url,
                d.created_at,
                d.scraped_at or utc_now(),
                _dumps(d.channel_meta),
                d.channel_meta.get("content_hash"),
                run_id,
            ),
        )
        n += 1
    return n


def upsert_annotations(conn: sqlite3.Connection, anns: list[Annotation]) -> int:
    for a in anns:
        conn.execute(
            """
            INSERT OR REPLACE INTO annotations
            (doc_id, sentiment, categories, barriers, insight_types, discovery_paths, info_needs,
             theme_hint, evidence_spans, is_spam, is_off_topic_discovery, embedding_id,
             model_ver, pipeline_run_id, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                a.doc_id,
                a.sentiment,
                _dumps(a.categories),
                _dumps(a.barriers),
                _dumps(a.insight_types),
                _dumps(a.discovery_paths),
                _dumps(a.info_needs),
                a.theme_hint,
                _dumps(a.evidence_spans),
                int(a.is_spam),
                int(a.is_off_topic_discovery),
                a.embedding_id,
                a.model_ver,
                a.pipeline_run_id,
                a.confidence,
            ),
        )
    return len(anns)


def upsert_themes(conn: sqlite3.Connection, themes: list[ThemeCard]) -> int:
    for t in themes:
        conn.execute(
            """
            INSERT OR REPLACE INTO theme_cards
            (id, title, volume, sentiment_mix, barriers, categories, insight_types,
             discovery_paths, info_needs, evidence, pipeline_run_id, model_ver, coherence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                t.id,
                t.title,
                t.volume,
                _dumps(t.sentiment_mix),
                _dumps(t.barriers),
                _dumps(t.categories),
                _dumps(t.insight_types),
                _dumps(t.discovery_paths),
                _dumps(t.info_needs),
                _dumps([e.model_dump() for e in t.evidence]),
                t.pipeline_run_id,
                t.model_ver,
                t.coherence,
            ),
        )
    return len(themes)


def latest_successful_run(conn: sqlite3.Connection) -> Optional[sqlite3.Row]:
    return conn.execute(
        """
        SELECT * FROM pipeline_runs
        WHERE status = 'success'
        ORDER BY finished_at DESC
        LIMIT 1
        """
    ).fetchone()


def list_themes(
    conn: sqlite3.Connection,
    *,
    insight_type: str | None = None,
    barrier: str | None = None,
    source: str | None = None,
    q: str | None = None,
) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM theme_cards ORDER BY volume DESC, title ASC"
    ).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        theme = {
            "id": r["id"],
            "title": r["title"],
            "volume": r["volume"],
            "sentiment_mix": _loads(r["sentiment_mix"], {}),
            "barriers": _loads(r["barriers"], []),
            "categories": _loads(r["categories"], []),
            "insight_types": _loads(r["insight_types"], []),
            "discovery_paths": _loads(r["discovery_paths"], []),
            "info_needs": _loads(r["info_needs"], []),
            "evidence": _loads(r["evidence"], []),
            "pipeline_run_id": r["pipeline_run_id"],
            "model_ver": r["model_ver"],
            "coherence": r["coherence"],
        }
        if insight_type and insight_type not in theme["insight_types"]:
            continue
        if barrier and barrier not in theme["barriers"]:
            continue
        if source:
            if not any(e.get("source") == source for e in theme["evidence"]):
                continue
        if q:
            ql = q.lower()
            blob = " ".join(
                [
                    theme["title"],
                    " ".join(theme["barriers"]),
                    " ".join(theme["categories"]),
                    " ".join(e.get("span", "") for e in theme["evidence"]),
                ]
            ).lower()
            if ql not in blob:
                continue
        out.append(theme)
    return out


def get_theme(conn: sqlite3.Connection, theme_id: str) -> Optional[dict[str, Any]]:
    themes = list_themes(conn)
    for t in themes:
        if t["id"] == theme_id:
            return t
    return None


def list_documents(
    conn: sqlite3.Connection,
    *,
    source: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    if source:
        rows = conn.execute(
            """
            SELECT d.*, a.sentiment, a.barriers, a.categories, a.insight_types, a.evidence_spans,
                   a.is_spam, a.is_off_topic_discovery
            FROM feedback_documents d
            LEFT JOIN annotations a ON a.doc_id = d.id
            WHERE d.source = ?
            ORDER BY d.id
            LIMIT ?
            """,
            (source, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT d.*, a.sentiment, a.barriers, a.categories, a.insight_types, a.evidence_spans,
                   a.is_spam, a.is_off_topic_discovery
            FROM feedback_documents d
            LEFT JOIN annotations a ON a.doc_id = d.id
            ORDER BY d.id
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    out = []
    for r in rows:
        out.append(
            {
                "id": r["id"],
                "source": r["source"],
                "text": r["text"],
                "url": r["url"],
                "created_at": r["created_at"],
                "sentiment": r["sentiment"],
                "barriers": _loads(r["barriers"], []),
                "categories": _loads(r["categories"], []),
                "insight_types": _loads(r["insight_types"], []),
                "evidence_spans": _loads(r["evidence_spans"], []),
                "is_spam": bool(r["is_spam"]) if r["is_spam"] is not None else False,
                "is_off_topic_discovery": bool(r["is_off_topic_discovery"])
                if r["is_off_topic_discovery"] is not None
                else False,
            }
        )
    return out


def stats_summary(conn: sqlite3.Connection) -> dict[str, Any]:
    run = latest_successful_run(conn)
    n_docs = conn.execute("SELECT COUNT(*) AS c FROM feedback_documents").fetchone()["c"]
    n_ann = conn.execute("SELECT COUNT(*) AS c FROM annotations").fetchone()["c"]
    n_themes = conn.execute("SELECT COUNT(*) AS c FROM theme_cards").fetchone()["c"]
    by_source = {
        r["source"]: r["c"]
        for r in conn.execute(
            "SELECT source, COUNT(*) AS c FROM feedback_documents GROUP BY source"
        ).fetchall()
    }
    insight_counts: dict[str, int] = {}
    for r in conn.execute("SELECT insight_types FROM theme_cards").fetchall():
        for it in _loads(r["insight_types"], []):
            insight_counts[it] = insight_counts.get(it, 0) + 1
    barrier_counts: dict[str, int] = {}
    for r in conn.execute("SELECT barriers FROM theme_cards").fetchall():
        for b in _loads(r["barriers"], []):
            barrier_counts[b] = barrier_counts.get(b, 0) + 1
    return {
        "n_docs": n_docs,
        "n_annotations": n_ann,
        "n_themes": n_themes,
        "by_source": by_source,
        "themes_by_insight_type": insight_counts,
        "themes_by_barrier": barrier_counts,
        "latest_run": dict(run) if run else None,
        "db_path": str(DB_PATH),
    }
