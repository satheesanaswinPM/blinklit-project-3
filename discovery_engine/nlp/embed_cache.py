"""
Shared embedding load / compute / cache helpers.

Used by analysis.themes, analysis.segments, and main.py so embedding
row counts, normalization, and cache invalidation stay consistent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.preprocessing import normalize

from discovery_engine.nlp.bertopic_cluster import _encode_docs


def load_or_compute_embeddings(
    docs: list[str],
    *,
    embeddings_path: Path | None = None,
    embedding_backend: str = "tfidf",
    embedding_model: str = "all-MiniLM-L6-v2",
    save_embeddings: Path | None = None,
    normalize_vectors: bool = False,
) -> tuple[Any, np.ndarray]:
    """
    Load embeddings from `.npy` or compute them for `docs`.

    Returns:
        (embedder, embeddings) where embedder is suitable for BERTopic
        (or a lightweight TF-IDF stub when loading from cache).

    Raises:
        FileNotFoundError: cached path missing
        ValueError: row count mismatch between cache and docs
    """
    if not docs:
        raise ValueError("Cannot embed an empty document list")

    if embeddings_path is not None:
        if not embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
        emb = np.asarray(np.load(embeddings_path), dtype=float)
        if emb.ndim != 2:
            raise ValueError(f"Expected 2D embeddings array, got shape {emb.shape}")
        if emb.shape[0] != len(docs):
            raise ValueError(
                f"Embeddings rows ({emb.shape[0]}) != number of reviews ({len(docs)}). "
                "Recompute embeddings after cleaning."
            )
        # Stub embedder so BERTopic does not download MiniLM when using a cache.
        # Prefer matching the requested backend for consistency.
        embedder, _ = _encode_docs(
            docs[:2] if len(docs) >= 2 else docs,
            embedding_backend if embedding_backend == "tfidf" else "tfidf",
            embedding_model,
        )
        if normalize_vectors:
            emb = normalize(emb)
        print(f"Loaded embeddings {emb.shape} from {embeddings_path}")
        return embedder, emb

    print(f"Computing embeddings (backend={embedding_backend})...")
    embedder, emb = _encode_docs(docs, embedding_backend, embedding_model)
    emb = np.asarray(emb, dtype=float)
    if normalize_vectors:
        emb = normalize(emb)

    if save_embeddings is not None:
        save_embeddings.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_embeddings, emb)
        print(f"Saved embeddings {emb.shape} -> {save_embeddings}")

    return embedder, emb
