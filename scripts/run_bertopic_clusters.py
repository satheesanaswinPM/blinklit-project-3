"""
Cluster reviews with BERTopic and write cluster summaries.

Usage (from repo root):
  python -m scripts.run_bertopic_clusters --in data/raw/blinkit_play_reviews.csv --text-col Review --preprocess --no-openai
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from discovery_engine.env_loader import load_env  # noqa: E402

load_env()

from discovery_engine.nlp.bertopic_cluster import (  # noqa: E402
    cluster_and_summarize,
    load_reviews_from_csv,
    summaries_to_frame,
)
from discovery_engine.nlp.preprocess import preprocess_corpus  # noqa: E402

OUT_DIR = ROOT / "data" / "processed"


def main() -> int:
    parser = argparse.ArgumentParser(description="BERTopic clustering + cluster summaries")
    parser.add_argument("--in", dest="inp", type=Path, required=True)
    parser.add_argument("--text-col", default=None)
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--min-cluster-size", type=int, default=3)
    parser.add_argument("--nr-topics", default="auto")
    parser.add_argument("--embedding-backend", choices=["tfidf", "st"], default="tfidf")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--no-openai", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    if not args.inp.exists():
        print(f"Input not found: {args.inp}", file=sys.stderr)
        return 1

    reviews = load_reviews_from_csv(str(args.inp), text_col=args.text_col)
    raw_count = len(reviews)
    if args.preprocess:
        reviews = preprocess_corpus(reviews, dedupe=True)
        print(f"Preprocessed: {raw_count} -> {len(reviews)} reviews")
    else:
        reviews = [r.strip() for r in reviews if str(r).strip()]
        print(f"Loaded {len(reviews)} non-empty reviews")

    nr_topics: str | int = "auto" if str(args.nr_topics).lower() == "auto" else int(args.nr_topics)

    print(f"Fitting BERTopic (embedding backend: {args.embedding_backend})...")
    summaries, assignments, _model = cluster_and_summarize(
        reviews,
        embedding_backend=args.embedding_backend,
        embedding_model=args.embedding_model,
        min_cluster_size=args.min_cluster_size,
        nr_topics=nr_topics,
        use_openai_summary=not args.no_openai,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "bertopic_cluster_summaries.csv"
    assign_path = args.out_dir / "bertopic_cluster_assignments.csv"
    json_path = args.out_dir / "bertopic_cluster_summaries.json"

    summaries_to_frame(summaries).to_csv(summary_path, index=False, encoding="utf-8")
    assignments.to_csv(assign_path, index=False, encoding="utf-8")
    json_path.write_text(
        json.dumps([s.to_dict() for s in summaries], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nFound {len(summaries)} clusters (including outliers if present):\n")
    for s in summaries:
        print(f"[{s.cluster_id}] n={s.size}  {s.label}")
        print(f"    {s.summary}\n")

    print(f"Summaries   -> {summary_path}")
    print(f"Assignments -> {assign_path}")
    print(f"JSON        -> {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
