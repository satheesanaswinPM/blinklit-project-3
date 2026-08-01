from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from discovery_engine.phase0.validate import stratification_stats, validate_gold_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Phase 0 gold label JSONL")
    parser.add_argument(
        "--gold",
        type=Path,
        default=ROOT / "data" / "gold" / "gold_labels.jsonl",
    )
    parser.add_argument(
        "--stats",
        type=Path,
        default=ROOT / "data" / "gold" / "gold_stats.json",
        help="Write stratification stats JSON",
    )
    parser.add_argument("--min-rows", type=int, default=200)
    args = parser.parse_args()

    if not args.gold.exists():
        print(f"ERROR: gold file not found: {args.gold}")
        return 1

    report = validate_gold_file(args.gold)
    from discovery_engine.phase0.validate import load_jsonl

    rows = load_jsonl(args.gold)
    stats = stratification_stats(rows)
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"Validated {report.n_docs} documents from {args.gold}")
    print(f"Errors: {len(report.errors)} | Warnings: {len(report.warnings)}")
    for issue in report.errors[:30]:
        print(f"  [error] {issue.doc_id}.{issue.field}: {issue.message}")
    for issue in report.warnings[:20]:
        print(f"  [warn]  {issue.doc_id}.{issue.field}: {issue.message}")
    if report.n_docs < args.min_rows:
        print(f"ERROR: need ≥ {args.min_rows} rows, found {report.n_docs}")
        return 1
    if not report.ok:
        return 1
    print("OK - gold set passed validation")
    print(f"Stats written -> {args.stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
