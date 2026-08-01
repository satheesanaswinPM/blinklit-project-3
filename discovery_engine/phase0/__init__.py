"""Phase 0 — taxonomies, gold validation, offline eval."""

from discovery_engine.phase0.eval_harness import evaluate, load_and_evaluate, write_report
from discovery_engine.phase0.taxonomies import (
    barrier_ids,
    category_ids,
    insight_type_ids,
    load_barriers,
    load_categories,
    load_insight_types,
    taxonomy_version,
)
from discovery_engine.phase0.validate import (
    load_jsonl,
    stratification_stats,
    validate_gold_file,
    validate_record,
)

__all__ = [
    "barrier_ids",
    "category_ids",
    "evaluate",
    "insight_type_ids",
    "load_and_evaluate",
    "load_barriers",
    "load_categories",
    "load_insight_types",
    "load_jsonl",
    "stratification_stats",
    "taxonomy_version",
    "validate_gold_file",
    "validate_record",
    "write_report",
]
