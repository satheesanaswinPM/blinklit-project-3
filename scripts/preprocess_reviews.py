"""
CLI: preprocess a CSV review column and write cleaned output.

Usage (from repo root):
  python -m scripts.preprocess_reviews --in data/raw/blinkit_play_reviews.csv --text-col Review
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from discovery_engine.nlp.preprocess import normalize_for_dedupe, preprocess_text  # noqa: E402

DEFAULT_OUT = ROOT / "data" / "processed" / "preprocessed_reviews.csv"


def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess feedback CSV text column")
    parser.add_argument("--in", dest="inp", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--text-col", default="Review")
    parser.add_argument("--no-dedupe", action="store_true")
    args = parser.parse_args()

    if not args.inp.exists():
        print(f"Input not found: {args.inp}", file=sys.stderr)
        return 1

    df = pd.read_csv(args.inp)
    if args.text_col not in df.columns:
        print(f"Column {args.text_col!r} not in CSV. Available: {list(df.columns)}", file=sys.stderr)
        return 1

    texts = df[args.text_col].fillna("").astype(str).tolist()
    if not args.no_dedupe:
        seen: set[str] = set()
        keep_idx: list[int] = []
        for i, t in enumerate(texts):
            key = normalize_for_dedupe(t)
            if not key or key in seen:
                continue
            seen.add(key)
            keep_idx.append(i)
        df = df.iloc[keep_idx].reset_index(drop=True)
        texts = [texts[i] for i in keep_idx]
        print(f"Deduped: kept {len(df)} / original rows")

    df["cleaned_text"] = [preprocess_text(t) for t in texts]
    before = len(df)
    df = df[df["cleaned_text"].str.len() > 0].reset_index(drop=True)
    print(f"Cleaned: {len(df)} rows (dropped {before - len(df)} empty)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"Wrote -> {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
