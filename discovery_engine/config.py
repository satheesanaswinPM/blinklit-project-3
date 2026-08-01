from __future__ import annotations

from pathlib import Path

# discovery_engine/ -> repo root
PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent

TAXONOMY_DIR = REPO_ROOT / "taxonomies"
SCHEMA_DIR = REPO_ROOT / "schemas"
DOCS_DIR = REPO_ROOT / "docs"
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
GOLD_DIR = DATA_DIR / "gold"
GOLD_PATH = GOLD_DIR / "gold_labels.jsonl"
EVAL_DIR = DATA_DIR / "eval_reports"
FIXTURES_DIR = DATA_DIR / "fixtures"
DB_PATH = PROCESSED_DIR / "insights.db"
ENV_PATH = REPO_ROOT / ".env"
OUTPUT_DIR = REPO_ROOT / "output"
