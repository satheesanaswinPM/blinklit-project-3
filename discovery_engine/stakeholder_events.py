"""Append-only stakeholder analytics for the Streamlit research console."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVENTS_PATH = ROOT / "output" / "stakeholder_events.jsonl"


def log_event(
    event_type: str,
    *,
    path: Path | None = None,
    **properties: Any,
) -> dict[str, Any]:
    """Append one event to JSONL. Returns the written record."""
    target = path or DEFAULT_EVENTS_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "event": event_type,
        **{k: v for k, v in properties.items() if v is not None},
    }
    with target.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def load_events(path: Path | None = None, limit: int | None = None) -> list[dict[str, Any]]:
    target = path or DEFAULT_EVENTS_PATH
    if not target.exists():
        return []
    rows: list[dict[str, Any]] = []
    with target.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if limit is not None and limit > 0:
        return rows[-limit:]
    return rows


def summarize_events(path: Path | None = None) -> dict[str, Any]:
    rows = load_events(path)
    page_views = Counter()
    mvp_views = Counter()
    mvp_actions = Counter()
    other = Counter()
    for r in rows:
        ev = str(r.get("event") or "")
        if ev == "page_view":
            page_views[str(r.get("page") or "unknown")] += 1
        elif ev == "mvp_tab_view":
            mvp_views[str(r.get("mvp") or "unknown")] += 1
        elif ev == "mvp_action":
            mvp_actions[str(r.get("action") or "unknown")] += 1
        else:
            other[ev or "unknown"] += 1
    return {
        "total_events": len(rows),
        "page_views": dict(page_views.most_common()),
        "mvp_tab_views": dict(mvp_views.most_common()),
        "mvp_actions": dict(mvp_actions.most_common()),
        "other_events": dict(other.most_common()),
        "last_event_at": rows[-1].get("ts") if rows else None,
    }
