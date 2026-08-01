"""
Detect major discussion themes with BERTopic.

Loads (or computes) review embeddings, fits BERTopic, and writes:

  output/themes.csv

Columns:
  Theme name
  Number of reviews
  Representative keywords
  Representative reviews

Usage (from repo root):
  python -m analysis.themes
  python -m analysis.themes --in data/raw/blinkit_play_reviews.csv --text-col Review
  python -m analysis.themes --embeddings data/processed/review_embeddings.npy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from discovery_engine.env_loader import load_env  # noqa: E402
from discovery_engine.nlp.bertopic_cluster import (  # noqa: E402
    _build_topic_model,
    load_reviews_from_csv,
)
from discovery_engine.nlp.embed_cache import load_or_compute_embeddings  # noqa: E402

load_env()

DEFAULT_IN = ROOT / "data" / "raw" / "blinkit_play_reviews.csv"
DEFAULT_OUT = ROOT / "output" / "themes.csv"
DEFAULT_EMB_CACHE = ROOT / "data" / "processed" / "review_embeddings.npy"


def detect_themes(
    docs: list[str],
    embeddings: np.ndarray,
    embedder: object,
    *,
    min_cluster_size: int = 3,
    nr_topics: str | int = "auto",
    top_n_keywords: int = 8,
    n_representatives: int = 3,
) -> pd.DataFrame:
    """Fit BERTopic on precomputed embeddings and return theme table."""
    if len(docs) < 2:
        raise ValueError("Need at least 2 reviews to detect themes")

    effective_min = min(min_cluster_size, max(2, len(docs) // 4))
    topic_model = _build_topic_model(
        embedder,
        min_cluster_size=effective_min,
        nr_topics=nr_topics,
        n_docs=len(docs),
    )
    topics, _ = topic_model.fit_transform(docs, embeddings=embeddings)
    topic_info = topic_model.get_topic_info()

    rows: list[dict] = []
    for _, row in topic_info.iterrows():
        topic_id = int(row["Topic"])
        indices = [i for i, t in enumerate(topics) if int(t) == topic_id]
        examples = [docs[i].replace("\n", " ").strip() for i in indices[:n_representatives]]

        if topic_id == -1:
            theme_name = "Outliers / mixed"
            keywords: list[str] = []
        else:
            words = topic_model.get_topic(topic_id) or []
            keywords = [w for w, _ in words[: top_n_keywords * 2] if w and str(w).strip()]
            keywords = keywords[:top_n_keywords]
            theme_name = " / ".join(keywords[:4]) if keywords else f"Theme {topic_id}"

        rows.append(
            {
                "Theme name": theme_name,
                "Number of reviews": len(indices),
                "Representative keywords": ", ".join(keywords),
                "Representative reviews": " | ".join(examples),
            }
        )

    # Major themes first (exclude empty); outliers last
    frame = pd.DataFrame(rows)
    frame["_outlier"] = frame["Theme name"].eq("Outliers / mixed")
    frame = frame.sort_values(
        by=["_outlier", "Number of reviews"],
        ascending=[True, False],
    ).drop(columns=["_outlier"])
    return frame.reset_index(drop=True)


def save_themes(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "Theme name",
        "Number of reviews",
        "Representative keywords",
        "Representative reviews",
    ]
    df[cols].to_csv(path, index=False, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="BERTopic theme detection -> output/themes.csv")
    parser.add_argument("--in", dest="inp", type=Path, default=DEFAULT_IN, help="Reviews CSV")
    parser.add_argument("--text-col", default=None, help="Text column (auto-detect if omitted)")
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=None,
        help="Optional .npy embeddings (rows aligned with reviews after empty-drop)",
    )
    parser.add_argument(
        "--save-embeddings",
        type=Path,
        default=DEFAULT_EMB_CACHE,
        help="Where to cache computed embeddings (.npy)",
    )
    parser.add_argument("--embedding-backend", choices=["tfidf", "st"], default="tfidf")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--min-cluster-size", type=int, default=3)
    parser.add_argument("--nr-topics", default="auto")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not args.inp.exists():
        print(f"Input not found: {args.inp}", file=sys.stderr)
        return 1

    reviews = load_reviews_from_csv(str(args.inp), text_col=args.text_col)
    docs = [r.strip() for r in reviews if str(r).strip()]
    print(f"Loaded {len(docs)} non-empty reviews from {args.inp}")

    embedder, embeddings = load_or_compute_embeddings(
        docs,
        embeddings_path=args.embeddings,
        embedding_backend=args.embedding_backend,
        embedding_model=args.embedding_model,
        save_embeddings=None if args.embeddings else args.save_embeddings,
    )

    nr_topics: str | int = "auto" if str(args.nr_topics).lower() == "auto" else int(args.nr_topics)
    print("Fitting BERTopic...")
    themes = detect_themes(
        docs,
        embeddings,
        embedder,
        min_cluster_size=args.min_cluster_size,
        nr_topics=nr_topics,
    )
    save_themes(themes, args.out)

    print(f"\nDetected {len(themes)} themes:\n")
    for _, row in themes.iterrows():
        print(f"- {row['Theme name']}  (n={row['Number of reviews']})")
        if row["Representative keywords"]:
            print(f"  keywords: {row['Representative keywords']}")
    print(f"\nSaved -> {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
