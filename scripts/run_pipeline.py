from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from discovery_engine.pipeline import run_pipeline  # noqa: E402
from scripts.seed_corpus import seed_corpus  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed corpus (optional) + run Phase 1 NLP pipeline")
    parser.add_argument("--skip-seed", action="store_true", help="Do not regenerate raw CSVs")
    parser.add_argument("--limit-gold", type=int, default=180)
    parser.add_argument("--min-cluster-size", type=int, default=2)
    parser.add_argument("--n-clusters", type=int, default=None)
    args = parser.parse_args()

    if not args.skip_seed:
        print("Seeding sample corpus...")
        seed_corpus(limit_gold=args.limit_gold)

    print("Running pipeline...")
    result = run_pipeline(
        min_cluster_size=args.min_cluster_size,
        n_clusters=args.n_clusters,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
