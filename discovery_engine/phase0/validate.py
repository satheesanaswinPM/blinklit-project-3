from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .taxonomies import (
    barrier_ids,
    category_ids,
    discovery_path_ids,
    info_need_ids,
    insight_type_ids,
    segment_proxy_ids,
)

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
VALID_SENTIMENT = {"positive", "neutral", "negative"}
VALID_CONFIDENCE = {"low", "medium", "high"}


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def span_in_text(span: str, text: str) -> bool:
    if span in text:
        return True
    return _norm_ws(span) in _norm_ws(text)


@dataclass
class ValidationIssue:
    doc_id: str
    field: str
    message: str
    severity: str = "error"  # error | warning


@dataclass
class ValidationReport:
    n_docs: int = 0
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0


def validate_record(doc: dict[str, Any]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    doc_id = str(doc.get("id", "<missing>"))

    def err(field: str, msg: str) -> None:
        issues.append(ValidationIssue(doc_id, field, msg, "error"))

    def warn(field: str, msg: str) -> None:
        issues.append(ValidationIssue(doc_id, field, msg, "warning"))

    required = [
        "id",
        "source",
        "text",
        "sentiment",
        "categories",
        "barriers",
        "insight_types",
        "theme",
        "evidence_spans",
        "is_spam",
        "is_off_topic_discovery",
        "confidence",
        "taxonomy_version",
        "labeled_at",
    ]
    for key in required:
        if key not in doc:
            err(key, "missing required field")

    if "source" in doc and doc["source"] not in VALID_SOURCES:
        err("source", f"invalid source: {doc['source']}")

    if "sentiment" in doc and doc["sentiment"] not in VALID_SENTIMENT:
        err("sentiment", f"invalid sentiment: {doc['sentiment']}")

    if "confidence" in doc and doc["confidence"] not in VALID_CONFIDENCE:
        err("confidence", f"invalid confidence: {doc['confidence']}")

    text = doc.get("text") or ""
    if not isinstance(text, str) or not text.strip():
        err("text", "text must be non-empty")

    cats = doc.get("categories")
    if not isinstance(cats, list) or not cats:
        err("categories", "must be non-empty list")
    else:
        known = category_ids()
        for c in cats:
            if c not in known:
                err("categories", f"unknown category id: {c}")

    barriers = doc.get("barriers")
    if not isinstance(barriers, list) or not barriers:
        err("barriers", "must be non-empty list")
    else:
        known_b = barrier_ids()
        for b in barriers:
            if b not in known_b:
                err("barriers", f"unknown barrier id: {b}")
        if "other" in barriers and not (doc.get("notes") or "").strip():
            err("barriers", "'other' requires notes explaining the gap")
        if "none" in barriers and len(set(barriers) - {"none"}) > 0:
            warn("barriers", "'none' combined with other barriers is unusual")

    insights = doc.get("insight_types")
    if not isinstance(insights, list) or not insights:
        err("insight_types", "must be non-empty list")
    else:
        known_i = insight_type_ids()
        for i in insights:
            if i not in known_i:
                err("insight_types", f"unknown insight_type: {i}")

    for field_name, allowed in (
        ("discovery_paths", discovery_path_ids()),
        ("info_needs", info_need_ids()),
        ("segment_proxies", segment_proxy_ids()),
    ):
        values = doc.get(field_name) or []
        if not isinstance(values, list):
            err(field_name, "must be a list")
            continue
        for v in values:
            if v not in allowed:
                err(field_name, f"unknown id: {v}")

    spans = doc.get("evidence_spans")
    is_spam = bool(doc.get("is_spam"))
    if not isinstance(spans, list) or (not spans and not is_spam):
        err("evidence_spans", "non-spam docs need ≥1 evidence span")
    elif isinstance(spans, list):
        for span in spans:
            if not isinstance(span, str) or not span.strip():
                err("evidence_spans", "span must be non-empty string")
            elif text and not span_in_text(span, text):
                err("evidence_spans", f"span not found in text: {span[:80]!r}")

    theme = doc.get("theme")
    if isinstance(theme, str) and len(theme.split()) > 10:
        warn("theme", "theme longer than 10 words; prefer ≤8")

    if doc.get("is_off_topic_discovery") and "off_topic" not in (cats or []):
        warn("categories", "off-topic discovery docs usually include category 'off_topic'")

    return issues


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from e
    return rows


def validate_gold_file(path: Path) -> ValidationReport:
    rows = load_jsonl(path)
    report = ValidationReport(n_docs=len(rows))
    seen_ids: set[str] = set()
    for doc in rows:
        doc_id = str(doc.get("id", ""))
        if doc_id in seen_ids:
            report.issues.append(
                ValidationIssue(doc_id, "id", "duplicate id in gold file", "error")
            )
        seen_ids.add(doc_id)
        report.issues.extend(validate_record(doc))
    return report


def stratification_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from collections import Counter

    sources = Counter(r.get("source") for r in rows)
    sentiments = Counter(r.get("sentiment") for r in rows)
    barriers: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    insights: Counter[str] = Counter()
    for r in rows:
        barriers.update(r.get("barriers") or [])
        categories.update(r.get("categories") or [])
        insights.update(r.get("insight_types") or [])
    return {
        "n": len(rows),
        "by_source": dict(sources),
        "by_sentiment": dict(sentiments),
        "by_barrier": dict(barriers),
        "by_category": dict(categories),
        "by_insight_type": dict(insights),
        "spam": sum(1 for r in rows if r.get("is_spam")),
        "off_topic_discovery": sum(1 for r in rows if r.get("is_off_topic_discovery")),
        "adjudicated": sum(1 for r in rows if r.get("adjudicated")),
    }
