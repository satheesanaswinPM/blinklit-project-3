"""
Text helpers for the insight pipeline.

- ``clean_text`` / ``is_probably_spam`` / ``pick_evidence_span``: lightweight ops
  used during labeling and ThemeCard grounding (keeps original wording for quotes).
- Full NLP preprocessing (emoji, punct, stopwords, lemma, dedupe) lives in
  ``discovery_engine.nlp.preprocess``.
"""

from __future__ import annotations

import re

from discovery_engine.nlp.preprocess import remove_emojis


def clean_text(text: str) -> str:
    t = text.replace("\u00a0", " ")
    t = remove_emojis(t)
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def is_probably_spam(text: str) -> bool:
    t = clean_text(text)
    if not t:
        return True
    alpha = sum(ch.isalpha() for ch in t)
    if len(t) <= 8 and alpha < 3:
        return True
    if alpha / max(len(t), 1) < 0.2 and len(t) < 40:
        return True
    return False


def pick_evidence_span(text: str, max_words: int = 18) -> str:
    """Pick a grounded evidence span that is an exact substring of `text`."""
    t = text.strip()
    if not t:
        return text
    keywords = (
        "never",
        "don't",
        "dont",
        "won't",
        "wont",
        "only",
        "search",
        "expiry",
        "fresh",
        "trust",
        "expensive",
        "try",
        "discover",
        "homepage",
        "recommend",
    )
    parts = re.split(r"(?<=[.!?])\s+", t)
    ranked = sorted(
        parts,
        key=lambda p: sum(1 for k in keywords if k in p.lower()),
        reverse=True,
    )
    candidate = ranked[0] if ranked else t
    words = candidate.split()
    if len(words) > max_words:
        candidate = " ".join(words[:max_words])
        if candidate not in t:
            candidate = ranked[0] if ranked else t
    if candidate not in t:
        candidate = t[: min(120, len(t))]
    return candidate
