from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np

from discovery_engine import MODEL_VER
from discovery_engine.collectors.csv_loader import dedupe_docs, load_raw_dir
from discovery_engine.config import DB_PATH, RAW_DIR
from discovery_engine.nlp.clean import clean_text
from discovery_engine.nlp.cluster import build_theme_cards, cluster_embeddings
from discovery_engine.nlp.embed import Embedder
from discovery_engine.nlp.label import TaxonomyLabeler
from discovery_engine.storage import db


def run_pipeline(
    *,
    raw_dir: Path | None = None,
    db_path: Path | None = None,
    replace: bool = True,
    min_cluster_size: int = 2,
    n_clusters: int | None = None,
) -> dict:
    raw_dir = raw_dir or RAW_DIR
    db_path = db_path or DB_PATH
    db.init_db(db_path)

    docs = dedupe_docs(load_raw_dir(raw_dir))
    if not docs:
        raise FileNotFoundError(f"No CSV feedback found in {raw_dir}")

    run_id = f"run_{uuid.uuid4().hex[:10]}"
    labeler = TaxonomyLabeler(MODEL_VER)

    with db.db_session(db_path) as conn:
        db.insert_pipeline_run(conn, run_id, MODEL_VER, status="running")
        if replace:
            db.clear_run_outputs(conn)
        db.upsert_documents(conn, docs, run_id)

        anns = [labeler.label_doc(d, run_id) for d in docs]
        db.upsert_annotations(conn, anns)

        usable_idx = [
            i
            for i, a in enumerate(anns)
            if not a.is_spam and not a.is_off_topic_discovery
        ]
        themes = []
        if usable_idx:
            texts = [clean_text(docs[i].text) for i in usable_idx]
            embedder = Embedder()
            emb = embedder.fit_transform(texts)
            labels_local = cluster_embeddings(
                emb,
                min_cluster_size=min_cluster_size,
                n_clusters=n_clusters,
            )
            full_labels = np.full(len(docs), -1, dtype=int)
            for local_i, doc_i in enumerate(usable_idx):
                full_labels[doc_i] = int(labels_local[local_i])

            themes = build_theme_cards(
                docs,
                anns,
                full_labels,
                pipeline_run_id=run_id,
                model_ver=MODEL_VER,
                min_evidence=2,
            )
            db.upsert_themes(conn, themes)

        db.finish_pipeline_run(conn, run_id, "success", n_docs=len(docs), n_themes=len(themes))

    return {
        "run_id": run_id,
        "n_docs": len(docs),
        "n_annotations": len(docs),
        "n_themes": len(themes),
        "model_ver": MODEL_VER,
        "db_path": str(db_path),
        "embedding_backend": Embedder().backend,
    }
