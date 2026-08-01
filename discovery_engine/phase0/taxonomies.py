from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from discovery_engine.config import TAXONOMY_DIR


@lru_cache(maxsize=1)
def load_categories() -> dict[str, Any]:
    return json.loads((TAXONOMY_DIR / "categories.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_barriers() -> dict[str, Any]:
    return json.loads((TAXONOMY_DIR / "barriers.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_insight_types() -> dict[str, Any]:
    return json.loads((TAXONOMY_DIR / "insight_types.json").read_text(encoding="utf-8"))


def category_ids() -> set[str]:
    return {c["id"] for c in load_categories()["categories"]}


def barrier_ids() -> set[str]:
    return {b["id"] for b in load_barriers()["barriers"]}


def insight_type_ids() -> set[str]:
    return {q["insight_type"] for q in load_insight_types()["research_questions"]}


def discovery_path_ids() -> set[str]:
    for q in load_insight_types()["research_questions"]:
        if q["insight_type"] == "discovery_path_map":
            return set(q.get("discovery_paths", []))
    return set()


def info_need_ids() -> set[str]:
    for q in load_insight_types()["research_questions"]:
        if q["insight_type"] == "info_needs":
            return set(q.get("info_need_tags", []))
    return set()


def segment_proxy_ids() -> set[str]:
    for q in load_insight_types()["research_questions"]:
        if q["insight_type"] == "receptive_segments":
            return set(q.get("segment_proxies", []))
    return set()


def taxonomy_version() -> str:
    return load_categories()["version"]
