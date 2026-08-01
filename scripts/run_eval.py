from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from discovery_engine.phase0.eval_harness import load_and_evaluate, write_report  # noqa: E402
from discovery_engine.phase0.validate import load_jsonl  # noqa: E402


def make_identity_predictions(gold_path: Path, pred_path: Path) -> None:
    """Baseline predictions = gold labels (sanity check that metrics hit ~1.0)."""
    rows = load_jsonl(gold_path)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    with pred_path.open("w", encoding="utf-8") as f:
        for r in rows:
            pred = {
                "id": r["id"],
                "sentiment": r["sentiment"],
                "categories": r["categories"],
                "barriers": r["barriers"],
                "insight_types": r["insight_types"],
                "theme": r["theme"],
                "evidence_spans": r["evidence_spans"],
            }
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 0 offline eval harness")
    parser.add_argument(
        "--gold",
        type=Path,
        default=ROOT / "data" / "gold" / "gold_labels.jsonl",
    )
    parser.add_argument(
        "--pred",
        type=Path,
        default=None,
        help="Predictions JSONL. If omitted, writes identity baseline predictions.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data" / "eval_reports",
    )
    parser.add_argument("--grounding-k", type=int, default=1)
    parser.add_argument("--theme-k", type=int, default=5)
    args = parser.parse_args()

    pred_path = args.pred
    if pred_path is None:
        pred_path = args.out_dir / "identity_predictions.jsonl"
        make_identity_predictions(args.gold, pred_path)
        print(f"No --pred given; wrote identity baseline -> {pred_path}")

    result = load_and_evaluate(
        args.gold,
        pred_path,
        grounding_k=args.grounding_k,
        theme_k=args.theme_k,
    )
    out_json = args.out_dir / "latest_eval.json"
    out_md = args.out_dir / "latest_eval.md"
    write_report(result, out_json, out_md)
    print(json.dumps({k: result[k] for k in result if k not in {"barrier", "category", "insight_type", "grounding_failures", "missing_predictions", "extra_predictions"}}, indent=2))
    print(f"Barrier micro-F1: {result['barrier']['micro_f1']:.3f}")
    print(f"Reports -> {out_json} , {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
