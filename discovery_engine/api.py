from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from discovery_engine import MODEL_VER, __version__
from discovery_engine.config import DB_PATH
from discovery_engine.storage import db

app = FastAPI(
    title="Discovery Insight Engine — Phase 1",
    description="Theme Explorer API for quick-commerce category discovery insights.",
    version=__version__,
)

app.add_middleware(
    CORSMiddleware,
    # Local Theme Explorer (Vite) + Streamlit; credentials disabled so "*" is valid.
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    db.init_db(DB_PATH)


@app.get("/health")
def health() -> dict:
    return {"ok": True, "version": __version__, "model_ver": MODEL_VER}


@app.get("/stats")
def stats() -> dict:
    with db.db_session() as conn:
        return db.stats_summary(conn)


@app.get("/themes")
def themes(
    insight_type: Optional[str] = Query(None, description="e.g. habit_drivers, barrier_taxonomy"),
    barrier: Optional[str] = None,
    source: Optional[str] = None,
    q: Optional[str] = None,
) -> dict:
    with db.db_session() as conn:
        items = db.list_themes(
            conn,
            insight_type=insight_type,
            barrier=barrier,
            source=source,
            q=q,
        )
        run = db.latest_successful_run(conn)
    return {
        "count": len(items),
        "pipeline_run": dict(run) if run else None,
        "themes": items,
    }


@app.get("/themes/{theme_id}")
def theme_detail(theme_id: str) -> dict:
    with db.db_session() as conn:
        theme = db.get_theme(conn, theme_id)
    if not theme:
        raise HTTPException(status_code=404, detail="theme not found")
    return theme


@app.get("/documents")
def documents(
    source: Optional[str] = None,
    limit: int = Query(100, ge=1, le=500),
) -> dict:
    with db.db_session() as conn:
        items = db.list_documents(conn, source=source, limit=limit)
    return {"count": len(items), "documents": items}


@app.get("/insights/overview")
def insights_overview() -> dict:
    """Answers the Phase 1 research questions at a glance using ThemeCards."""
    with db.db_session() as conn:
        all_themes = db.list_themes(conn)
        stats = db.stats_summary(conn)

    def top_for(insight: str, n: int = 5) -> list[dict]:
        matched = [t for t in all_themes if insight in t["insight_types"]]
        matched.sort(key=lambda t: t["volume"], reverse=True)
        return [
            {
                "id": t["id"],
                "title": t["title"],
                "volume": t["volume"],
                "barriers": t["barriers"],
                "evidence_preview": [e["span"] for e in t["evidence"][:2]],
            }
            for t in matched[:n]
        ]

    return {
        "habit_drivers": top_for("habit_drivers"),
        "barriers": top_for("barrier_taxonomy"),
        "discovery_paths": top_for("discovery_path_map"),
        "info_needs": top_for("info_needs"),
        "receptive_segments": top_for("receptive_segments"),
        "opportunity_briefs": top_for("opportunity_briefs"),
        "stats": stats,
    }
