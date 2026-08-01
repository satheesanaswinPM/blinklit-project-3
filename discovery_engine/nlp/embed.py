from __future__ import annotations

import os
from typing import Literal

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


class Embedder:
    """TF-IDF+SVD by default; optional sentence-transformers via PHASE1_EMBEDDING=st."""

    def __init__(self, backend: Literal["tfidf", "st"] | None = None):
        env = os.getenv("PHASE1_EMBEDDING", "").lower()
        self.backend = backend or ("st" if env == "st" else "tfidf")
        self._vectorizer: TfidfVectorizer | None = None
        self._svd: TruncatedSVD | None = None
        self._st_model = None

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        if self.backend == "st":
            return self._st_encode(texts)
        return self._tfidf_fit_transform(texts)

    def transform(self, texts: list[str]) -> np.ndarray:
        if self.backend == "st":
            return self._st_encode(texts)
        assert self._vectorizer is not None and self._svd is not None
        x = self._vectorizer.transform(texts)
        emb = self._svd.transform(x)
        return normalize(emb)

    def _tfidf_fit_transform(self, texts: list[str]) -> np.ndarray:
        self._vectorizer = TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 2),
            min_df=1,
            stop_words="english",
        )
        x = self._vectorizer.fit_transform(texts)
        n_components = min(128, max(2, x.shape[1] - 1), max(2, len(texts) - 1))
        self._svd = TruncatedSVD(n_components=n_components, random_state=42)
        emb = self._svd.fit_transform(x)
        return normalize(emb)

    def _st_encode(self, texts: list[str]) -> np.ndarray:
        try:
            from discovery_engine.env_loader import load_env

            load_env()
        except Exception:
            pass
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers not installed. pip install sentence-transformers "
                "or unset PHASE1_EMBEDDING=st"
            ) from e
        if self._st_model is None:
            self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
        emb = self._st_model.encode(texts, show_progress_bar=False)
        return normalize(np.asarray(emb))
