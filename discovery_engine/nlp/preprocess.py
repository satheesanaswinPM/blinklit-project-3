"""
Text preprocessing for feedback reviews.

- Remove emojis
- Remove punctuation
- Remove stop words
- Lemmatize tokens
- Drop duplicate reviews

Usage:
  from discovery_engine.nlp.preprocess import preprocess_text, dedupe_reviews, preprocess_corpus

  clean = preprocess_text("Great app!! 😀 Never buying meat again...")
  unique = dedupe_reviews(["Nice", "nice", "Nice!!"])
"""

from __future__ import annotations

import re
import string
from functools import lru_cache
from typing import Iterable

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Broad emoji / pictograph ranges (covers most common symbols without extra deps)
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001F900-\U0001F9FF"
    "\U0001FA70-\U0001FAFF"
    "\U00002600-\U000026FF"
    "\U0000FE0F"  # variation selector
    "\U0000200D"  # ZWJ
    "]+",
    flags=re.UNICODE,
)

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


@lru_cache(maxsize=1)
def ensure_nltk_data() -> None:
    """Download required NLTK resources once (idempotent)."""
    resources = {
        "tokenizers/punkt": "punkt",
        "tokenizers/punkt_tab": "punkt_tab",
        "taggers/averaged_perceptron_tagger": "averaged_perceptron_tagger",
        "taggers/averaged_perceptron_tagger_eng": "averaged_perceptron_tagger_eng",
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
    }
    for path, name in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


@lru_cache(maxsize=1)
def _stopwords() -> frozenset[str]:
    ensure_nltk_data()
    return frozenset(stopwords.words("english"))


@lru_cache(maxsize=1)
def _lemmatizer() -> WordNetLemmatizer:
    ensure_nltk_data()
    return WordNetLemmatizer()


def _to_wordnet_pos(tag: str):
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("N"):
        return wordnet.NOUN
    if tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def remove_emojis(text: str) -> str:
    return _EMOJI_RE.sub(" ", text or "")


def remove_punctuation(text: str) -> str:
    return (text or "").translate(_PUNCT_TABLE)


def remove_stopwords(tokens: Iterable[str], *, language: str = "english") -> list[str]:
    stops = _stopwords() if language == "english" else frozenset(stopwords.words(language))
    return [t for t in tokens if t not in stops and t.strip()]


def lemmatize_tokens(tokens: Iterable[str]) -> list[str]:
    ensure_nltk_data()
    lemmatizer = _lemmatizer()
    token_list = list(tokens)
    if not token_list:
        return []
    try:
        tagged = nltk.pos_tag(token_list)
    except LookupError:
        return [lemmatizer.lemmatize(t) for t in token_list]
    return [lemmatizer.lemmatize(tok, _to_wordnet_pos(pos)) for tok, pos in tagged]


def normalize_for_dedupe(text: str) -> str:
    """Aggressive normalization used only for duplicate detection."""
    t = remove_emojis(text or "")
    t = _URL_RE.sub(" ", t)
    t = t.lower()
    t = remove_punctuation(t)
    t = _WHITESPACE_RE.sub(" ", t).strip()
    return t


def preprocess_text(
    text: str,
    *,
    remove_emoji: bool = True,
    remove_punct: bool = True,
    remove_stops: bool = True,
    lemmatize: bool = True,
    return_tokens: bool = False,
) -> str | list[str]:
    """
    Full preprocessing pipeline for a single review/post body.

    Returns a cleaned string (space-joined lemmas) by default, or tokens if
    ``return_tokens=True``.
    """
    ensure_nltk_data()
    t = text or ""
    t = t.replace("\u00a0", " ")
    t = _URL_RE.sub(" ", t)
    if remove_emoji:
        t = remove_emojis(t)
    t = t.lower()
    if remove_punct:
        t = remove_punctuation(t)
    t = _WHITESPACE_RE.sub(" ", t).strip()
    if not t:
        return [] if return_tokens else ""

    try:
        tokens = word_tokenize(t)
    except LookupError:
        tokens = t.split()

    tokens = [tok for tok in tokens if tok.isalnum()]
    if remove_stops:
        tokens = remove_stopwords(tokens)
    if lemmatize:
        tokens = lemmatize_tokens(tokens)

    if return_tokens:
        return tokens
    return " ".join(tokens)


def dedupe_reviews(
    texts: Iterable[str],
    *,
    keep: str = "first",
) -> list[str]:
    """
    Remove duplicate reviews after emoji/punct/case normalization.

    ``keep``: 'first' or 'last' occurrence.
    """
    items = list(texts)
    if keep not in {"first", "last"}:
        raise ValueError("keep must be 'first' or 'last'")

    if keep == "last":
        items = list(reversed(items))

    seen: set[str] = set()
    unique: list[str] = []
    for text in items:
        key = normalize_for_dedupe(text)
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(text)

    if keep == "last":
        unique.reverse()
    return unique


def preprocess_corpus(
    texts: Iterable[str],
    *,
    dedupe: bool = True,
    **preprocess_kwargs,
) -> list[str]:
    """
    Dedupe (optional) then preprocess each review.
    Empty strings after cleaning are dropped.
    """
    items = list(texts)
    if dedupe:
        items = dedupe_reviews(items)

    out: list[str] = []
    for text in items:
        cleaned = preprocess_text(text, **preprocess_kwargs)
        if isinstance(cleaned, list):
            cleaned = " ".join(cleaned)
        if cleaned:
            out.append(cleaned)
    return out
