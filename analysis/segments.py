"""
Cluster reviews into user segments with KMeans on embeddings.

Segments:
  - Routine Buyers
  - Explorers
  - Price Sensitive
  - Impulse Buyers

Each cluster is labeled by similarity to segment prototypes + keyword evidence.
The CSV includes a rationale explaining why that label was assigned.

Output:
  output/user_segments.csv

Usage (from repo root):
  python -m analysis.segments
  python -m analysis.segments --in data/raw/blinkit_play_reviews.csv --text-col Review
  python -m analysis.segments --embeddings data/processed/review_embeddings.npy
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from discovery_engine.env_loader import load_env  # noqa: E402
from discovery_engine.nlp.bertopic_cluster import _encode_docs, load_reviews_from_csv  # noqa: E402
from discovery_engine.nlp.embed_cache import (  # noqa: E402
    load_or_compute_embeddings as _shared_load_or_compute_embeddings,
)

load_env()

DEFAULT_IN = ROOT / "data" / "raw" / "blinkit_play_reviews.csv"
DEFAULT_OUT = ROOT / "output" / "user_segments.csv"
DEFAULT_EMB_CACHE = ROOT / "data" / "processed" / "review_embeddings.npy"

SEGMENTS = (
    "Routine Buyers",
    "Explorers",
    "Price Sensitive",
    "Impulse Buyers",
)

SEGMENT_PROTOTYPES: dict[str, list[str]] = {
    "Routine Buyers": [
        "I always reorder the same milk eggs and bread every week.",
        "I stick to my usual brands and never try new categories.",
        "This app is only for my regular grocery staples and habit buys.",
    ],
    "Explorers": [
        "I like trying new products categories and discovering organic specialty items.",
        "I browse recommendations to find new brands I have never bought before.",
        "Curious to experiment with new beverages and unfamiliar categories.",
    ],
    "Price Sensitive": [
        "Too expensive compared to the supermarket I wait for deals and discounts.",
        "Need clear unit price and offers before I switch brands.",
        "Marked up prices make me avoid trying anything new here.",
    ],
    "Impulse Buyers": [
        "Late night snacks and cold drinks only when I suddenly crave something.",
        "I open the app for quick munchies and impulse treats.",
        "Saw a promo and ordered chips and cola immediately.",
    ],
}

SEGMENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "Routine Buyers": (
        "reorder",
        "usual",
        "always",
        "same",
        "regular",
        "habit",
        "stick",
        "every week",
        "staples",
        "only buy",
    ),
    "Explorers": (
        "try",
        "trying",
        "new",
        "discover",
        "explore",
        "curious",
        "organic",
        "variety",
        "browse",
        "recommend",
        "experiment",
    ),
    "Price Sensitive": (
        "expensive",
        "price",
        "cheap",
        "deal",
        "discount",
        "offer",
        "costly",
        "markup",
        "worth",
        "rs",
        "charge",
        "money",
    ),
    "Impulse Buyers": (
        "late night",
        "snack",
        "snacks",
        "craving",
        "sudden",
        "chips",
        "cola",
        "munchies",
        "quick",
        "impulse",
        "tonight",
    ),
}


def load_or_compute_embeddings(
    docs: list[str],
    *,
    embeddings_path: Path | None,
    embedding_backend: str,
    embedding_model: str,
    save_embeddings: Path | None,
) -> np.ndarray:
    """Load or compute L2-normalized embeddings for KMeans segmentation."""
    _embedder, emb = _shared_load_or_compute_embeddings(
        docs,
        embeddings_path=embeddings_path,
        embedding_backend=embedding_backend,
        embedding_model=embedding_model,
        save_embeddings=save_embeddings,
        normalize_vectors=True,
    )
    return emb


def _keyword_hits(texts: list[str], keywords: tuple[str, ...]) -> list[tuple[str, int]]:
    blob = " ".join(t.lower() for t in texts)
    hits = []
    for kw in keywords:
        count = blob.count(kw)
        if count:
            hits.append((kw, count))
    hits.sort(key=lambda x: (-x[1], x[0]))
    return hits


def _top_terms(texts: list[str], n: int = 8) -> list[str]:
    stop = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "to",
        "of",
        "in",
        "on",
        "for",
        "is",
        "it",
        "this",
        "that",
        "was",
        "are",
        "with",
        "my",
        "i",
        "me",
        "you",
        "app",
        "very",
        "good",
        "nice",
    }
    counts: Counter[str] = Counter()
    for t in texts:
        for w in "".join(ch.lower() if ch.isalnum() else " " for ch in t).split():
            if len(w) < 3 or w in stop:
                continue
            counts[w] += 1
    return [w for w, _ in counts.most_common(n)]


def label_clusters(
    docs: list[str],
    labels: np.ndarray,
    *,
    embedding_backend: str,
    embedding_model: str,
) -> dict[int, dict]:
    """
    Map each KMeans cluster id -> segment label + rationale.

    Jointly embeds reviews + prototype texts so cosine similarity is valid,
    then explains the choice with keyword evidence.
    """
    proto_names = list(SEGMENTS)
    proto_texts: list[str] = []
    proto_slices: dict[str, tuple[int, int]] = {}
    for name in proto_names:
        start = len(proto_texts)
        proto_texts.extend(SEGMENT_PROTOTYPES[name])
        proto_slices[name] = (start, len(proto_texts))

    joint_docs = list(docs) + proto_texts
    _embedder, joint_emb = _encode_docs(joint_docs, embedding_backend, embedding_model)
    joint_emb = normalize(np.asarray(joint_emb))
    n = len(docs)

    proto_mat = normalize(
        np.vstack(
            [
                joint_emb[n + proto_slices[name][0] : n + proto_slices[name][1]].mean(axis=0)
                for name in proto_names
            ]
        )
    )

    mapping: dict[int, dict] = {}
    used_labels: set[str] = set()
    cluster_ids = sorted(set(int(x) for x in labels))

    pending: list[tuple[int, np.ndarray, list[str]]] = []
    for cid in cluster_ids:
        idx = np.where(labels == cid)[0]
        member_docs = [docs[i] for i in idx]
        centroid = normalize(joint_emb[idx].mean(axis=0, keepdims=True))[0]
        pending.append((cid, centroid, member_docs))

    while pending:
        best_cid = None
        best_label = None
        best_score = -1.0
        best_docs: list[str] = []
        best_cos = 0.0

        for cid, centroid, member_docs in pending:
            sims = cosine_similarity(centroid.reshape(1, -1), proto_mat)[0]
            for j, name in enumerate(proto_names):
                if name in used_labels and len(used_labels) < len(SEGMENTS):
                    continue
                kw_score = sum(c for _, c in _keyword_hits(member_docs, SEGMENT_KEYWORDS[name]))
                blended = float(sims[j]) + 0.05 * kw_score
                if blended > best_score:
                    best_score = blended
                    best_cos = float(sims[j])
                    best_cid = cid
                    best_label = name
                    best_docs = member_docs

        if best_label is None:
            cid, centroid, member_docs = pending[0]
            sims = cosine_similarity(centroid.reshape(1, -1), proto_mat)[0]
            j = int(np.argmax(sims))
            best_cid = cid
            best_label = proto_names[j]
            best_cos = float(sims[j])
            best_docs = member_docs

        assert best_cid is not None and best_label is not None
        kw_hits = _keyword_hits(best_docs, SEGMENT_KEYWORDS[best_label])
        top_kw = ", ".join(f"{k}({c})" for k, c in kw_hits[:5]) or "few direct keyword hits"
        top_terms = ", ".join(_top_terms(best_docs, 6)) or "n/a"
        example = best_docs[0].replace("\n", " ").strip()
        if len(example) > 140:
            example = example[:137] + "..."

        rationale = (
            f"Cluster {best_cid} (n={len(best_docs)}) labeled '{best_label}' because its "
            f"centroid is closest to that segment's prototype reviews "
            f"(cosine={best_cos:.3f}). Supporting cues: {top_kw}. "
            f"Frequent terms: {top_terms}. "
            f"Example review: \"{example}\""
        )
        mapping[best_cid] = {
            "segment": best_label,
            "similarity": best_cos,
            "rationale": rationale,
            "size": len(best_docs),
        }
        used_labels.add(best_label)
        pending = [p for p in pending if p[0] != best_cid]

    return mapping


def cluster_segments(
    docs: list[str],
    embeddings: np.ndarray,
    *,
    embedding_backend: str,
    embedding_model: str,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Cluster reviews into four behavioral segments with KMeans.

    Embeddings are L2-normalized before clustering so cosine geometry
    is consistent regardless of how the caller prepared the matrix.
    """
    if len(docs) < 4:
        raise ValueError(f"Need at least 4 reviews for 4 segments; got {len(docs)}")
    if embeddings.shape[0] != len(docs):
        raise ValueError(
            f"Embeddings rows ({embeddings.shape[0]}) != docs ({len(docs)})"
        )

    emb = normalize(np.asarray(embeddings, dtype=float))
    print("Running KMeans (k=4)...")
    km = KMeans(n_clusters=4, random_state=random_state, n_init=10)
    labels = km.fit_predict(emb)
    mapping = label_clusters(
        docs,
        labels,
        embedding_backend=embedding_backend,
        embedding_model=embedding_model,
    )

    rows = []
    for i, doc in enumerate(docs):
        cid = int(labels[i])
        info = mapping[cid]
        rows.append(
            {
                "Review": doc,
                "Segment": info["segment"],
                "Cluster ID": cid,
                "Prototype Similarity": round(info["similarity"], 4),
                "Label Rationale": info["rationale"],
            }
        )
    return pd.DataFrame(rows)


def save_segments(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "Review",
        "Segment",
        "Cluster ID",
        "Prototype Similarity",
        "Label Rationale",
    ]
    df[cols].to_csv(path, index=False, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="KMeans user segments -> output/user_segments.csv")
    parser.add_argument("--in", dest="inp", type=Path, default=DEFAULT_IN)
    parser.add_argument("--text-col", default=None)
    parser.add_argument("--embeddings", type=Path, default=None)
    parser.add_argument("--save-embeddings", type=Path, default=DEFAULT_EMB_CACHE)
    parser.add_argument("--embedding-backend", choices=["tfidf", "st"], default="tfidf")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not args.inp.exists():
        print(f"Input not found: {args.inp}", file=sys.stderr)
        return 1

    reviews = load_reviews_from_csv(str(args.inp), text_col=args.text_col)
    docs = [r.strip() for r in reviews if str(r).strip()]
    print(f"Loaded {len(docs)} non-empty reviews from {args.inp}")

    embeddings = load_or_compute_embeddings(
        docs,
        embeddings_path=args.embeddings,
        embedding_backend=args.embedding_backend,
        embedding_model=args.embedding_model,
        save_embeddings=None if args.embeddings else args.save_embeddings,
    )

    df = cluster_segments(
        docs,
        embeddings,
        embedding_backend=args.embedding_backend,
        embedding_model=args.embedding_model,
    )
    save_segments(df, args.out)

    print("\nSegment sizes:")
    print(df["Segment"].value_counts().to_string())
    print("\nLabel rationales:")
    for seg, grp in df.groupby("Segment", sort=False):
        print(f"\n[{seg}]")
        text = str(grp["Label Rationale"].iloc[0]).encode("ascii", errors="replace").decode("ascii")
        print(f"  {text}")
    print(f"\nSaved -> {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
