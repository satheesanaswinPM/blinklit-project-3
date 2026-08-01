from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from discovery_engine.nlp.clean import pick_evidence_span
from discovery_engine.schemas import Annotation, EvidenceQuote, FeedbackDocument, ThemeCard


def cluster_embeddings(
    embeddings: np.ndarray,
    *,
    min_cluster_size: int = 2,
    n_clusters: int | None = None,
) -> np.ndarray:
    """Cluster embeddings into a target number of themes."""
    n = embeddings.shape[0]
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.array([0], dtype=int)

    if n_clusters is None:
        # Aim for interpretable theme count for MVP corpora
        n_clusters = int(np.clip(n // max(min_cluster_size, 2), 4, 16))
        n_clusters = min(n_clusters, n)

    if n_clusters <= 1:
        return np.zeros(n, dtype=int)

    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="cosine",
        linkage="average",
    )
    return clustering.fit_predict(embeddings)


def _top_n(counter: Counter, n: int = 5) -> list[str]:
    return [k for k, _ in counter.most_common(n) if k and k != "none"]


def _quote(d: FeedbackDocument, a: Annotation) -> EvidenceQuote:
    span = a.evidence_spans[0] if a.evidence_spans else pick_evidence_span(d.text)
    if span not in d.text:
        span = pick_evidence_span(d.text)
    if span not in d.text:
        span = d.text[: min(120, len(d.text))]
    return EvidenceQuote(
        doc_id=d.id,
        source=d.source,
        text=d.text,
        url=d.url,
        span=span,
        sentiment=a.sentiment,
    )


def _card_from_members(
    members: list[tuple[FeedbackDocument, Annotation]],
    *,
    theme_id: str,
    title: str | None,
    pipeline_run_id: str,
    model_ver: str,
    min_evidence: int,
) -> ThemeCard | None:
    if len(members) < min_evidence:
        return None

    barrier_c: Counter = Counter()
    cat_c: Counter = Counter()
    insight_c: Counter = Counter()
    path_c: Counter = Counter()
    info_c: Counter = Counter()
    sent_c: Counter = Counter()
    hint_c: Counter = Counter()
    evidence: list[EvidenceQuote] = []
    seen_docs: set[str] = set()

    for d, a in members:
        barrier_c.update(a.barriers)
        cat_c.update([c for c in a.categories if c not in {"off_topic", "unknown"}])
        insight_c.update(a.insight_types)
        path_c.update(a.discovery_paths)
        info_c.update(a.info_needs)
        sent_c.update([a.sentiment])
        if a.theme_hint:
            hint_c.update([a.theme_hint])
        if d.id in seen_docs:
            continue
        seen_docs.add(d.id)
        evidence.append(_quote(d, a))

    if len(evidence) < min_evidence:
        return None

    top_barrier = _top_n(barrier_c, 1)
    top_cat = _top_n(cat_c, 1)
    if title:
        resolved = title
    elif top_barrier and top_cat:
        resolved = f"{top_cat[0].replace('_', ' ').title()}: {top_barrier[0].replace('_', ' ')}"
    elif hint_c:
        resolved = hint_c.most_common(1)[0][0]
    else:
        resolved = theme_id

    return ThemeCard(
        id=theme_id,
        title=resolved,
        volume=len(members),
        sentiment_mix=dict(sent_c),
        barriers=_top_n(barrier_c, 5),
        categories=_top_n(cat_c, 5),
        insight_types=_top_n(insight_c, 6),
        discovery_paths=_top_n(path_c, 5),
        info_needs=_top_n(info_c, 5),
        evidence=evidence[:8],
        pipeline_run_id=pipeline_run_id,
        model_ver=model_ver,
        coherence=min(1.0, len(evidence) / max(len(members), 1)),
    )


def build_theme_cards(
    docs: list[FeedbackDocument],
    anns: list[Annotation],
    labels: np.ndarray,
    *,
    pipeline_run_id: str,
    model_ver: str,
    min_evidence: int = 2,
) -> list[ThemeCard]:
    doc_by_id = {d.id: d for d in docs}
    ann_by_id = {a.doc_id: a for a in anns}
    groups: dict[int, list[str]] = defaultdict(list)
    for doc, lab in zip(docs, labels):
        groups[int(lab)].append(doc.id)

    themes: list[ThemeCard] = []
    theme_idx = 0
    for lab, doc_ids in sorted(groups.items(), key=lambda x: -len(x[1])):
        if lab < 0:
            continue
        members = []
        for did in doc_ids:
            a = ann_by_id.get(did)
            d = doc_by_id.get(did)
            if not a or not d or a.is_spam or a.is_off_topic_discovery:
                continue
            members.append((d, a))
        theme_idx += 1
        card = _card_from_members(
            members,
            theme_id=f"theme_{theme_idx:03d}",
            title=None,
            pipeline_run_id=pipeline_run_id,
            model_ver=model_ver,
            min_evidence=min_evidence,
        )
        if card:
            themes.append(card)

    # Fallback: barrier-primary themes if embedding clusters produced too few cards
    if len(themes) < 4:
        by_barrier: dict[str, list[tuple[FeedbackDocument, Annotation]]] = defaultdict(list)
        for d, a in zip(docs, anns):
            if a.is_spam or a.is_off_topic_discovery:
                continue
            primary = next((b for b in a.barriers if b != "none"), None)
            if not primary:
                continue
            by_barrier[primary].append((d, a))
        themes = []
        for i, (barrier, members) in enumerate(
            sorted(by_barrier.items(), key=lambda x: -len(x[1])), start=1
        ):
            card = _card_from_members(
                members,
                theme_id=f"theme_{i:03d}",
                title=f"Barrier: {barrier.replace('_', ' ')}",
                pipeline_run_id=pipeline_run_id,
                model_ver=model_ver,
                min_evidence=min_evidence,
            )
            if card:
                themes.append(card)
    return themes
