"""
Cluster review embeddings with BERTopic and summarize each cluster.

Default embedding backend is local TF-IDF+SVD (no HuggingFace download).
Set embedding_backend='st' (or CLI --embedding-backend st) to use
sentence-transformers (all-MiniLM-L6-v2).

Summaries combine c-TF-IDF keywords + representative reviews, with optional
OpenAI one-liners when OPENAI_API_KEY is set.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


@dataclass
class ClusterSummary:
    cluster_id: int
    label: str
    keywords: list[str]
    size: int
    summary: str
    representative_reviews: list[str] = field(default_factory=list)
    review_indices: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TfidfSVDEmbedder:
    """Local embedder with BERTopic BaseEmbedder interface (no HF download)."""

    def __init__(self, max_features: int = 8000, n_components: int = 64):
        self.max_features = max_features
        self.n_components = n_components
        self._vectorizer: TfidfVectorizer | None = None
        self._svd: TruncatedSVD | None = None

    def embed(self, documents: list[str], verbose: bool = False) -> np.ndarray:  # noqa: ARG002
        self._vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, 2),
            min_df=1,
            stop_words="english",
        )
        x = self._vectorizer.fit_transform(documents)
        n_comp = min(self.n_components, max(2, x.shape[1] - 1), max(2, len(documents) - 1))
        self._svd = TruncatedSVD(n_components=n_comp, random_state=42)
        return normalize(self._svd.fit_transform(x))


def _make_tfidf_backend() -> Any:
    """Wrap TF-IDF embedder as a BERTopic backend to avoid default MiniLM download."""
    try:
        from bertopic.backend import BaseEmbedder
    except ImportError:
        return TfidfSVDEmbedder()

    class _Backend(BaseEmbedder):
        def __init__(self):
            super().__init__()
            self._impl = TfidfSVDEmbedder()

        def embed(self, documents: list[str], verbose: bool = False) -> np.ndarray:
            return self._impl.embed(documents, verbose=verbose)

    return _Backend()


def _encode_docs(docs: list[str], embedding_backend: str, embedding_model: str) -> tuple[Any, np.ndarray]:
    """Return (embedding_model_for_bertopic, embeddings_matrix)."""
    if embedding_backend == "st":
        try:
            from discovery_engine.env_loader import load_env

            load_env()
        except Exception:
            pass
        from sentence_transformers import SentenceTransformer

        st = SentenceTransformer(embedding_model)
        emb = np.asarray(st.encode(docs, show_progress_bar=False))
        return st, emb

    backend = _make_tfidf_backend()
    emb = backend.embed(docs)
    return backend, emb


def _build_topic_model(
    embedder: Any,
    *,
    min_cluster_size: int = 3,
    min_samples: int = 1,
    nr_topics: str | int | None = "auto",
    n_docs: int = 50,
):
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    n_neighbors = min(15, max(2, min(min_cluster_size + 2, n_docs - 1)))
    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=min(5, max(2, n_docs - 2)),
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)

    return BERTopic(
        embedding_model=embedder,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        nr_topics=nr_topics,
        calculate_probabilities=False,
        verbose=False,
    )


def _keyword_summary(keywords: list[str], examples: list[str]) -> str:
    kw = ", ".join(keywords[:8]) if keywords else "general feedback"
    if examples:
        snippet = examples[0].replace("\n", " ").strip()
        if len(snippet) > 160:
            snippet = snippet[:157] + "..."
        return f"Users discuss {kw}. Example: \"{snippet}\""
    return f"Users discuss {kw}."


def _openai_summary(keywords: list[str], examples: list[str]) -> Optional[str]:
    try:
        from discovery_engine.env_loader import load_env

        load_env()
    except Exception:
        pass
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        joined = "\n".join(f"- {e[:300]}" for e in examples[:5])
        prompt = (
            "Summarize this cluster of quick-commerce app reviews in 1-2 sentences. "
            "Focus on the user problem or theme.\n"
            f"Keywords: {', '.join(keywords[:10])}\n"
            f"Reviews:\n{joined}"
        )
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=120,
        )
        text = (resp.choices[0].message.content or "").strip()
        return text or None
    except Exception:
        return None


def cluster_and_summarize(
    reviews: list[str],
    *,
    embedding_backend: str = "tfidf",
    embedding_model: str = "all-MiniLM-L6-v2",
    min_cluster_size: int = 3,
    nr_topics: str | int | None = "auto",
    top_n_words: int = 8,
    n_representatives: int = 3,
    use_openai_summary: bool = True,
) -> tuple[list[ClusterSummary], pd.DataFrame, Any]:
    """
    Fit BERTopic on reviews and return cluster summaries + assignment table.
    """
    docs = [str(r).strip() for r in reviews if str(r).strip()]
    if len(docs) < max(2, min_cluster_size):
        raise ValueError(
            f"Need at least {max(2, min_cluster_size)} non-empty reviews; got {len(docs)}"
        )

    effective_min = min(min_cluster_size, max(2, len(docs) // 4))
    embedder, embeddings = _encode_docs(docs, embedding_backend, embedding_model)
    topic_model = _build_topic_model(
        embedder,
        min_cluster_size=effective_min,
        nr_topics=nr_topics,
        n_docs=len(docs),
    )
    topics, _ = topic_model.fit_transform(docs, embeddings=embeddings)

    topic_info = topic_model.get_topic_info()
    summaries: list[ClusterSummary] = []

    for _, row in topic_info.iterrows():
        topic_id = int(row["Topic"])
        if topic_id == -1:
            label = "Outliers / mixed"
            keywords: list[str] = []
        else:
            words = topic_model.get_topic(topic_id) or []
            keywords = [w for w, _ in words[:top_n_words]]
            label = "_".join(keywords[:4]) if keywords else f"topic_{topic_id}"

        indices = [i for i, t in enumerate(topics) if int(t) == topic_id]
        examples = [docs[i] for i in indices[:n_representatives]]

        summary = _keyword_summary(keywords, examples)
        if use_openai_summary and topic_id != -1:
            llm = _openai_summary(keywords, examples)
            if llm:
                summary = llm

        summaries.append(
            ClusterSummary(
                cluster_id=topic_id,
                label=label,
                keywords=keywords,
                size=len(indices),
                summary=summary,
                representative_reviews=examples,
                review_indices=indices,
            )
        )

    summaries.sort(key=lambda s: (s.cluster_id == -1, -s.size, s.cluster_id))

    label_by_id = {s.cluster_id: s.label for s in summaries}
    assignments = pd.DataFrame(
        {
            "review": docs,
            "cluster_id": [int(t) for t in topics],
            "cluster_label": [label_by_id.get(int(t), str(t)) for t in topics],
        }
    )
    return summaries, assignments, topic_model


def summaries_to_frame(summaries: list[ClusterSummary]) -> pd.DataFrame:
    rows = []
    for s in summaries:
        rows.append(
            {
                "cluster_id": s.cluster_id,
                "label": s.label,
                "size": s.size,
                "keywords": ", ".join(s.keywords),
                "summary": s.summary,
                "representative_reviews": " || ".join(s.representative_reviews),
            }
        )
    return pd.DataFrame(rows)


def load_reviews_from_csv(
    path: str,
    text_col: str | None = None,
    cleaned_col: str = "cleaned_text",
) -> list[str]:
    df = pd.read_csv(path)
    if text_col and text_col in df.columns:
        col = text_col
    elif cleaned_col in df.columns:
        col = cleaned_col
    elif "Review" in df.columns:
        col = "Review"
    elif "Body" in df.columns:
        col = "Body"
    elif "text" in df.columns:
        col = "text"
    else:
        raise ValueError(f"No text column found. Columns: {list(df.columns)}")
    return df[col].fillna("").astype(str).tolist()
