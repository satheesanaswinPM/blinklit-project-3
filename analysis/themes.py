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
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from discovery_engine.env_loader import load_env  # noqa: E402

load_env()

DEFAULT_IN = ROOT / "data" / "raw" / "blinkit_play_reviews.csv"
DEFAULT_OUT = ROOT / "output" / "themes.csv"
DEFAULT_EMB_CACHE = ROOT / "data" / "processed" / "review_embeddings.npy"

# Short / generic praise themes that add little product-discovery signal
_GENERIC_TOKENS = {
    "good",
    "nice",
    "best",
    "ok",
    "okay",
    "app",
    "service",
    "sarvice",
    "secives",
    "delivery",
    "dilevery",
    "fast",
    "super",
    "excellent",
    "awesome",
    "love",
    "great",
    "perfect",
    "amazing",
    "wow",
    "ooo",
    "5star",
    "star",
    "working",
    "performance",
    "application",
}


def _tokenize_theme_blob(text: str) -> list[str]:
    return [t for t in re.split(r"[^a-zA-Z0-9]+", (text or "").lower()) if t]


def is_low_information_theme(row: pd.Series | dict) -> bool:
    """True for generic praise / spammy clusters that shouldn't drive product insights."""
    name = str(row.get("Theme name", "") or "")
    keywords = str(row.get("Representative keywords", "") or "")
    n = int(pd.to_numeric(row.get("Number of reviews", 0), errors="coerce") or 0)
    if name.strip() in {"", "Outliers / mixed", "Untitled"}:
        return name.strip() in {"", "Untitled"}

    tokens = _tokenize_theme_blob(name + " " + keywords)
    if not tokens:
        return True
    generic = sum(1 for t in tokens if t in _GENERIC_TOKENS or t.isdigit() or len(t) <= 2)
    generic_share = generic / max(len(tokens), 1)
    unique_content = {t for t in tokens if t not in _GENERIC_TOKENS and not t.isdigit() and len(t) > 2}
    # Drop if almost entirely generic praise, or tiny unique content with high volume of "good/nice"
    if generic_share >= 0.75 and len(unique_content) <= 1:
        return True
    if len(unique_content) == 0 and n >= 10:
        return True
    # Repeated single-token themes like "good good / good"
    if len(set(tokens)) <= 2 and all(t in _GENERIC_TOKENS or len(t) <= 2 for t in set(tokens)):
        return True
    return False


def filter_low_information_themes(df: pd.DataFrame) -> pd.DataFrame:
    """Remove low-signal themes; keep outliers row if present."""
    if df is None or df.empty:
        return df
    keep_mask = ~df.apply(is_low_information_theme, axis=1)
    # Always keep explicit outliers bucket for transparency
    if "Theme name" in df.columns:
        keep_mask = keep_mask | df["Theme name"].astype(str).eq("Outliers / mixed")
    out = df.loc[keep_mask].copy()
    if "Number of reviews" in out.columns:
        out = out.sort_values("Number of reviews", ascending=False)
    return out.reset_index(drop=True)


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
    from discovery_engine.nlp.bertopic_cluster import _build_topic_model

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

    frame = pd.DataFrame(rows)
    frame["_outlier"] = frame["Theme name"].eq("Outliers / mixed")
    frame = frame.sort_values(
        by=["_outlier", "Number of reviews"],
        ascending=[True, False],
    ).drop(columns=["_outlier"])
    frame = filter_low_information_themes(frame.reset_index(drop=True))
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

    from discovery_engine.nlp.bertopic_cluster import load_reviews_from_csv
    from discovery_engine.nlp.embed_cache import load_or_compute_embeddings

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
