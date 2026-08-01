from __future__ import annotations

"""Minimal IAA helper for dual-annotated JSONL pairs.

Expected files: annotator_a.jsonl and annotator_b.jsonl with the same ids.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from discovery_engine.phase0.validate import load_jsonl  # noqa: E402


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def cohens_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    classes = sorted(set(labels_a) | set(labels_b))
    n = len(labels_a)
    if n == 0:
        return 0.0
    # confusion counts
    idx = {c: i for i, c in enumerate(classes)}
    mat = [[0 for _ in classes] for _ in classes]
    for a, b in zip(labels_a, labels_b):
        mat[idx[a]][idx[b]] += 1
    po = sum(mat[i][i] for i in range(len(classes))) / n
    pe = 0.0
    for i in range(len(classes)):
        row = sum(mat[i][j] for j in range(len(classes))) / n
        col = sum(mat[j][i] for j in range(len(classes))) / n
        pe += row * col
    if pe == 1:
        return 1.0
    return (po - pe) / (1 - pe)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute simple IAA between two annotators")
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    args = parser.parse_args()

    a = {r["id"]: r for r in load_jsonl(args.a)}
    b = {r["id"]: r for r in load_jsonl(args.b)}
    ids = sorted(set(a) & set(b))
    if not ids:
        print("No overlapping ids")
        return 1

    barrier_j = [jaccard(set(a[i].get("barriers") or []), set(b[i].get("barriers") or [])) for i in ids]
    cat_j = [jaccard(set(a[i].get("categories") or []), set(b[i].get("categories") or [])) for i in ids]
    sent_a = [a[i]["sentiment"] for i in ids]
    sent_b = [b[i]["sentiment"] for i in ids]
    theme_exact = sum(1 for i in ids if a[i].get("theme") == b[i].get("theme")) / len(ids)

    report = {
        "n_paired": len(ids),
        "mean_barrier_jaccard": sum(barrier_j) / len(barrier_j),
        "mean_category_jaccard": sum(cat_j) / len(cat_j),
        "sentiment_cohens_kappa": cohens_kappa(sent_a, sent_b),
        "theme_exact_agreement": theme_exact,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
