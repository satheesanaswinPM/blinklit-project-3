"""
End-to-end Discovery Insight Engine pipeline.

Primary question: Why don't Blinkit users explore new categories?

Stages:
  1–4  Collect Play / App Store / Reddit / YouTube
  5    Merge multi-source corpus
  6    Clean data
  7–10 Embeddings → themes → sentiment → segments
  11   Exploration tagging
  12   Product insights (legacy RQ board)
  13   Synthesis (JTBD, unmet needs, experiments, category ops)

Usage (from repo root):
  python main.py
  python main.py --play-count 100 --reddit-limit 15
  python main.py --skip-collect   # reuse existing raw CSVs (no network)
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from discovery_engine.env_loader import load_env  # noqa: E402

load_env()

# ---------------------------------------------------------------------------
# Paths & defaults
# ---------------------------------------------------------------------------

PLAY_CSV = ROOT / "data" / "raw" / "blinkit_play_reviews.csv"
APP_STORE_CSV = ROOT / "data" / "raw" / "app_store_reviews.csv"
REDDIT_CSV = ROOT / "data" / "raw" / "reddit_posts.csv"
YOUTUBE_CSV = ROOT / "data" / "raw" / "youtube_comments.csv"
MERGED_CSV = ROOT / "data" / "processed" / "merged_reviews.csv"
CLEANED_CSV = ROOT / "data" / "processed" / "preprocessed_reviews.csv"
EMBEDDINGS_NPY = ROOT / "data" / "processed" / "review_embeddings.npy"
THEMES_CSV = ROOT / "output" / "themes.csv"
SENTIMENT_CSV = ROOT / "output" / "sentiment.csv"
SEGMENTS_CSV = ROOT / "output" / "user_segments.csv"
EXPLORATION_CSV = ROOT / "output" / "exploration_tags.csv"
INSIGHTS_JSON = ROOT / "output" / "insights.json"
SYNTHESIS_JSON = ROOT / "output" / "synthesis.json"

EMBEDDING_BACKEND = os.getenv("PHASE1_EMBEDDING", "tfidf").strip() or "tfidf"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOTAL_STAGES = 13


@dataclass
class PipelineContext:
    """Mutable state shared across pipeline stages."""

    play_count: int = 200
    reddit_limit: int = 25
    skip_collect: bool = False
    continue_on_error: bool = False
    embedding_backend: str = EMBEDDING_BACKEND
    # Parallel lists: cleaned docs used for embed/theme/segment/sentiment
    docs: list[str] = field(default_factory=list)
    raw_docs: list[str] = field(default_factory=list)
    embeddings: np.ndarray | None = None
    failed_stages: list[str] = field(default_factory=list)
    skipped_stages: list[str] = field(default_factory=list)


def _banner(stage_no: int, total: int, title: str) -> None:
    print("\n" + "=" * 72)
    print(f"[{stage_no}/{total}] {title}")
    print("=" * 72)


def _ok(msg: str) -> None:
    print(f"[OK] {msg}")


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}", file=sys.stderr)


def _series_or_empty(df: pd.DataFrame, column: str) -> pd.Series:
    """Return a string Series for `column`, or empty strings if missing."""
    if column in df.columns:
        return df[column].fillna("").astype(str)
    return pd.Series([""] * len(df), index=df.index, dtype=str)


def run_stage(
    ctx: PipelineContext,
    *,
    stage_no: int,
    total: int,
    title: str,
    fn: Callable[[], Any],
    critical: bool = True,
) -> Any:
    """Execute one pipeline stage with progress logging and exception handling."""
    _banner(stage_no, total, title)
    try:
        result = fn()
        _ok(f"Completed: {title}")
        return result
    except Exception as exc:  # noqa: BLE001 — surface any stage failure cleanly
        _fail(f"Stage failed: {title}")
        _fail(f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        ctx.failed_stages.append(title)
        if critical and not ctx.continue_on_error:
            raise
        _warn(f"Continuing after failure in: {title}")
        return None


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def stage_collect_play(ctx: PipelineContext) -> Path:
    """Download Blinkit Play Store reviews, or reuse an existing CSV."""
    if ctx.skip_collect:
        if PLAY_CSV.exists():
            _warn(f"Skipping Play scrape; using existing {PLAY_CSV}")
            return PLAY_CSV
        raise FileNotFoundError(
            f"--skip-collect set but Play CSV not found: {PLAY_CSV}. "
            "Run without --skip-collect or place a CSV at that path."
        )

    from scripts.download_blinkit_play_reviews import fetch_reviews, to_rows, write_csv

    print(f"Fetching up to {ctx.play_count} Blinkit Play Store reviews...")
    raw = fetch_reviews("com.grofers.customerapp", total=ctx.play_count, lang="en", country="in")
    rows = [r for r in to_rows(raw) if (r.get("Review") or "").strip()]
    write_csv(rows, PLAY_CSV)
    print(f"Saved {len(rows)} reviews -> {PLAY_CSV}")
    if not rows:
        raise RuntimeError("Google Play collector returned 0 non-empty reviews")
    return PLAY_CSV


def stage_collect_app_store(ctx: PipelineContext) -> Path | None:
    """Download App Store RSS reviews, or reuse existing CSV / seed feedback."""
    if ctx.skip_collect:
        if APP_STORE_CSV.exists():
            _warn(f"Skipping App Store scrape; using existing {APP_STORE_CSV}")
            return APP_STORE_CSV
        seed = ROOT / "data" / "raw" / "app_store_feedback.csv"
        if seed.exists():
            _warn(f"Skipping App Store scrape; seed feedback present at {seed}")
            return seed
        _warn("Skipping App Store scrape (--skip-collect and no app_store CSV)")
        ctx.skipped_stages.append("Collect App Store reviews")
        return None

    from scripts.download_app_store_reviews import fetch_page, write_csv, BLINKIT_IOS_ID

    print("Fetching Blinkit App Store reviews (RSS)...")
    rows: list[dict] = []
    for page in range(1, 11):
        try:
            rows.extend(fetch_page(BLINKIT_IOS_ID, "in", page))
        except Exception as exc:  # noqa: BLE001
            _warn(f"App Store page {page} failed: {exc}")
            break
    if rows:
        write_csv(rows, APP_STORE_CSV)
        print(f"Saved {len(rows)} App Store reviews -> {APP_STORE_CSV}")
        return APP_STORE_CSV
    _warn("App Store collector returned 0 reviews; seed feedback may still be merged")
    ctx.skipped_stages.append("Collect App Store reviews")
    return APP_STORE_CSV if APP_STORE_CSV.exists() else None


def stage_collect_reddit(ctx: PipelineContext) -> Path | None:
    """Download Reddit posts when credentials exist; otherwise skip."""
    if ctx.skip_collect:
        if REDDIT_CSV.exists():
            _warn(f"Skipping Reddit scrape; using existing {REDDIT_CSV}")
            return REDDIT_CSV
        seed = ROOT / "data" / "raw" / "reddit_feedback.csv"
        if seed.exists():
            _warn(f"Skipping Reddit scrape; seed feedback present at {seed}")
            return seed
        _warn("Skipping Reddit scrape (--skip-collect and no existing reddit CSV)")
        ctx.skipped_stages.append("Collect Reddit posts")
        return None

    client_id = os.getenv("REDDIT_CLIENT_ID", "").strip()
    client_secret = os.getenv("REDDIT_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        _warn(
            "Reddit credentials missing in .env "
            "(REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET). Skipping Reddit collection."
        )
        ctx.skipped_stages.append("Collect Reddit posts")
        return REDDIT_CSV if REDDIT_CSV.exists() else None

    from scripts.download_reddit_posts import (
        DEFAULT_SUBREDDITS,
        fetch_posts,
        make_reddit,
        write_csv,
    )

    print(f"Fetching Reddit posts (limit={ctx.reddit_limit}/subreddit)...")
    reddit = make_reddit()
    reddit.read_only = True
    rows = fetch_posts(
        reddit,
        list(DEFAULT_SUBREDDITS),
        limit_per_sub=ctx.reddit_limit,
        sort="hot",
        max_comments=30,
    )
    write_csv(rows, REDDIT_CSV)
    print(f"Saved {len(rows)} posts -> {REDDIT_CSV}")
    return REDDIT_CSV


def stage_collect_youtube(ctx: PipelineContext) -> Path | None:
    """Download YouTube comments when YOUTUBE_API_KEY is set."""
    if ctx.skip_collect:
        if YOUTUBE_CSV.exists():
            _warn(f"Skipping YouTube scrape; using existing {YOUTUBE_CSV}")
            return YOUTUBE_CSV
        _warn("Skipping YouTube scrape (--skip-collect)")
        ctx.skipped_stages.append("Collect YouTube comments")
        return None

    if not os.getenv("YOUTUBE_API_KEY", "").strip():
        _warn("YOUTUBE_API_KEY missing; skipping YouTube collection")
        ctx.skipped_stages.append("Collect YouTube comments")
        return YOUTUBE_CSV if YOUTUBE_CSV.exists() else None

    from scripts.download_youtube_comments import main as yt_main

    code = yt_main()
    if code != 0:
        _warn("YouTube collector exited non-zero")
        ctx.skipped_stages.append("Collect YouTube comments")
        return YOUTUBE_CSV if YOUTUBE_CSV.exists() else None
    return YOUTUBE_CSV if YOUTUBE_CSV.exists() else None


def stage_merge_corpus(ctx: PipelineContext) -> Path:
    """Unify Play / App Store / Reddit / YouTube / seed channels."""
    from discovery_engine.corpus.merge import merge_sources, save_merged

    merged = merge_sources()
    if merged.empty:
        raise RuntimeError("Merged corpus is empty — run collectors or add seed feedback CSVs")
    path = save_merged(merged, MERGED_CSV)
    print(f"Merged {len(merged)} items -> {path}")
    print(merged["source"].value_counts().to_string())
    return path


def stage_clean_data(ctx: PipelineContext) -> Path:
    """Light-clean the multi-source corpus into embedding-ready docs.

    Prefers merged_reviews.csv; falls back to Play (+ Reddit) if merge missing.
    Soft exact-text dedupe (whitespace + case fold only).
    """
    from discovery_engine.nlp.preprocess import preprocess_text

    frames: list[pd.DataFrame] = []
    if MERGED_CSV.exists():
        merged = pd.read_csv(MERGED_CSV)
        part = pd.DataFrame(
            {
                "Review": _series_or_empty(merged, "text"),
                "source": _series_or_empty(merged, "source"),
                "raw_text": _series_or_empty(merged, "text"),
            }
        )
        frames.append(part)
        print(f"Cleaning from merged corpus ({len(part)} rows)")
    else:
        if not PLAY_CSV.exists():
            raise FileNotFoundError(f"Play reviews not found: {PLAY_CSV}")
        play = pd.read_csv(PLAY_CSV)
        if "Review" not in play.columns:
            raise ValueError(f"Expected 'Review' column in {PLAY_CSV}; got {list(play.columns)}")
        play_part = play.copy()
        play_part["source"] = "play_store"
        play_part["raw_text"] = _series_or_empty(play_part, "Review")
        frames.append(play_part)
        if REDDIT_CSV.exists():
            try:
                reddit = pd.read_csv(REDDIT_CSV)
            except Exception as exc:  # noqa: BLE001
                _warn(f"Could not read Reddit CSV ({exc}); continuing with Play only")
                reddit = None
            if reddit is not None and len(reddit):
                title = _series_or_empty(reddit, "Title")
                body = _series_or_empty(reddit, "Body")
                comments = _series_or_empty(reddit, "Comments")
                combined = (title + " " + body + " " + comments).str.strip()
                reddit_part = pd.DataFrame(
                    {"Review": combined, "source": "reddit", "raw_text": combined}
                )
                frames.append(reddit_part)
                print(f"Merged {len(reddit_part)} Reddit rows into clean corpus")

    df = pd.concat(frames, ignore_index=True, sort=False)
    texts = df["raw_text"].fillna("").astype(str).tolist()

    # Soft dedupe: only drop empty / exact same text after casefold+whitespace
    seen: set[str] = set()
    keep_idx: list[int] = []
    for i, t in enumerate(texts):
        key = " ".join(t.casefold().split())
        if not key or key in seen:
            continue
        seen.add(key)
        keep_idx.append(i)
    df = df.iloc[keep_idx].reset_index(drop=True)
    texts = [texts[i] for i in keep_idx]
    print(f"Soft-deduped: kept {len(df)} rows")

    print("Cleaning / normalizing text...")
    cleaned_rows: list[str] = []
    for t in texts:
        cleaned = str(preprocess_text(t) or "").strip()
        if not cleaned:
            # Keep emoji/short reviews in the corpus for sentiment/volume
            cleaned = " ".join(t.casefold().split())
        cleaned_rows.append(cleaned)
    df["cleaned_text"] = cleaned_rows
    df["Review"] = df["raw_text"].astype(str)

    mask = df["Review"].astype(str).str.strip().str.len() > 0
    dropped = int((~mask).sum())
    df = df.loc[mask].reset_index(drop=True)
    print(f"Cleaned: {len(df)} rows (dropped {dropped} empty)")

    CLEANED_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEANED_CSV, index=False, encoding="utf-8")
    print(f"Wrote cleaned corpus -> {CLEANED_CSV}")

    ctx.docs = df["cleaned_text"].astype(str).tolist()
    ctx.raw_docs = df["Review"].astype(str).tolist()
    if len(ctx.docs) < 2:
        raise RuntimeError("Cleaned corpus has fewer than 2 documents")
    return CLEANED_CSV


def _ensure_docs(ctx: PipelineContext) -> None:
    """Reload cleaned docs into context if a prior stage did not populate them."""
    if ctx.docs and ctx.raw_docs and len(ctx.docs) == len(ctx.raw_docs):
        return
    if not CLEANED_CSV.exists():
        raise FileNotFoundError(f"Cleaned data not found: {CLEANED_CSV}")
    df = pd.read_csv(CLEANED_CSV)
    if "cleaned_text" in df.columns:
        ctx.docs = [str(x).strip() for x in df["cleaned_text"].tolist() if str(x).strip()]
    else:
        col = "Review" if "Review" in df.columns else df.columns[0]
        ctx.docs = [str(x).strip() for x in df[col].tolist() if str(x).strip()]
    if "Review" in df.columns:
        # Align raw reviews to non-empty cleaned rows
        tmp = df.copy()
        if "cleaned_text" in tmp.columns:
            tmp = tmp[tmp["cleaned_text"].astype(str).str.len() > 0]
        ctx.raw_docs = tmp["Review"].astype(str).tolist()
    else:
        ctx.raw_docs = list(ctx.docs)
    if len(ctx.raw_docs) != len(ctx.docs):
        # Fall back to cleaned text for both to keep lengths aligned
        ctx.raw_docs = list(ctx.docs)


def stage_generate_embeddings(ctx: PipelineContext) -> Path:
    """Compute and cache review embeddings for theme + segment stages."""
    from discovery_engine.nlp.embed_cache import load_or_compute_embeddings

    _ensure_docs(ctx)
    # Always recompute in the orchestrator so cache cannot drift from cleaned docs
    _embedder, emb = load_or_compute_embeddings(
        ctx.docs,
        embeddings_path=None,
        embedding_backend=ctx.embedding_backend,
        embedding_model=EMBEDDING_MODEL,
        save_embeddings=EMBEDDINGS_NPY,
        normalize_vectors=False,
    )
    ctx.embeddings = emb
    print(f"Embeddings ready: {emb.shape}")
    return EMBEDDINGS_NPY


def stage_detect_themes(ctx: PipelineContext) -> Path:
    """Fit BERTopic and write `output/themes.csv`."""
    from analysis.themes import detect_themes, save_themes
    from discovery_engine.nlp.embed_cache import load_or_compute_embeddings

    _ensure_docs(ctx)
    if ctx.embeddings is None or ctx.embeddings.shape[0] != len(ctx.docs):
        embedder, emb = load_or_compute_embeddings(
            ctx.docs,
            embeddings_path=EMBEDDINGS_NPY if EMBEDDINGS_NPY.exists() else None,
            embedding_backend=ctx.embedding_backend,
            embedding_model=EMBEDDING_MODEL,
            save_embeddings=None if EMBEDDINGS_NPY.exists() else EMBEDDINGS_NPY,
        )
        ctx.embeddings = emb
    else:
        embedder, _ = load_or_compute_embeddings(
            ctx.docs[:2] if len(ctx.docs) >= 2 else ctx.docs,
            embeddings_path=None,
            embedding_backend="tfidf",
            embedding_model=EMBEDDING_MODEL,
            save_embeddings=None,
        )

    print("Fitting BERTopic...")
    themes = detect_themes(
        ctx.docs,
        ctx.embeddings,
        embedder,
        min_cluster_size=3,
        nr_topics="auto",
    )
    save_themes(themes, THEMES_CSV)
    print(f"Detected {len(themes)} themes -> {THEMES_CSV}")
    for _, row in themes.head(8).iterrows():
        name = str(row["Theme name"]).encode("ascii", errors="replace").decode("ascii")
        print(f"  - {name} (n={row['Number of reviews']})")
    return THEMES_CSV


def stage_analyze_sentiment(ctx: PipelineContext) -> Path:
    """Classify sentiment for the same corpus used by themes/segments."""
    from analysis.sentiment import classify_reviews, save_sentiment

    _ensure_docs(ctx)
    reviews = ctx.raw_docs or ctx.docs
    print(f"Classifying sentiment for {len(reviews)} reviews...")
    out = classify_reviews(reviews, batch_size=8)
    save_sentiment(out, SENTIMENT_CSV)
    counts = out["Sentiment"].value_counts().to_dict() if len(out) else {}
    print(f"Sentiment counts: {counts}")
    print(f"Saved -> {SENTIMENT_CSV}")
    return SENTIMENT_CSV


def stage_segment_users(ctx: PipelineContext) -> Path:
    """KMeans user segments → `output/user_segments.csv`."""
    from analysis.segments import cluster_segments, save_segments
    from discovery_engine.nlp.embed_cache import load_or_compute_embeddings

    _ensure_docs(ctx)
    if ctx.embeddings is None or ctx.embeddings.shape[0] != len(ctx.docs):
        _embedder, emb = load_or_compute_embeddings(
            ctx.docs,
            embeddings_path=EMBEDDINGS_NPY if EMBEDDINGS_NPY.exists() else None,
            embedding_backend=ctx.embedding_backend,
            embedding_model=EMBEDDING_MODEL,
            save_embeddings=None,
            normalize_vectors=True,
        )
        ctx.embeddings = emb

    print(f"Segmenting {len(ctx.docs)} users (KMeans k=4)...")
    segments = cluster_segments(
        ctx.docs,
        ctx.embeddings,
        embedding_backend=ctx.embedding_backend,
        embedding_model=EMBEDDING_MODEL,
    )
    save_segments(segments, SEGMENTS_CSV)
    print("Segment sizes:")
    print(segments["Segment"].value_counts().to_string())
    print(f"Saved -> {SEGMENTS_CSV}")
    return SEGMENTS_CSV


def stage_generate_insights(ctx: PipelineContext) -> Path:
    """LLM (or fallback) product insights → `output/insights.json`."""
    from llm.insights import (
        build_context,
        generate_fallback,
        generate_with_openai,
        write_insights,
    )

    if not THEMES_CSV.exists():
        raise FileNotFoundError(
            f"Themes not found: {THEMES_CSV}. Theme detection must succeed first."
        )

    themes = pd.read_csv(THEMES_CSV)
    sentiment = pd.read_csv(SENTIMENT_CSV) if SENTIMENT_CSV.exists() else None
    segments = pd.read_csv(SEGMENTS_CSV) if SEGMENTS_CSV.exists() else None
    if sentiment is None:
        _warn("sentiment.csv missing; insights will run without it")
    if segments is None:
        _warn("user_segments.csv missing; insights will run without it")

    context = build_context(themes, sentiment, segments)
    source = "openai"
    model_used: str | None = None
    try:
        print("Generating insights with LLM...")
        insights, model_used = generate_with_openai(context)
    except Exception as exc:  # noqa: BLE001
        _warn(f"LLM unavailable ({exc}); using grounded fallback insights")
        insights = generate_fallback(themes, sentiment, segments)
        source = "fallback"
        model_used = None

    meta = {
        "source": source,
        "themes_path": str(THEMES_CSV),
        "sentiment_path": str(SENTIMENT_CSV) if sentiment is not None else None,
        "segments_path": str(SEGMENTS_CSV) if segments is not None else None,
        "model": model_used if source == "openai" else None,
        "n_insights": len(insights),
        "pipeline": "main.py",
    }
    write_insights(insights, INSIGHTS_JSON, meta)
    print(f"Wrote {len(insights)} insights -> {INSIGHTS_JSON}")
    for item in insights:
        title = str(item.get("Title", "")).encode("ascii", errors="replace").decode("ascii")
        print(f"  - [{item.get('Priority')}] {title}")
    return INSIGHTS_JSON


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_pipeline(ctx: PipelineContext) -> int:
    """Run all eight stages and print a summary. Returns process exit code."""
    print("Discovery Insight Engine - full pipeline")
    print(f"Root: {ROOT}")
    print(f"Embedding backend: {ctx.embedding_backend}")
    if ctx.skip_collect:
        print("Collectors: skipped (reuse existing raw CSVs)")

    run_stage(
        ctx,
        stage_no=1,
        total=TOTAL_STAGES,
        title="Collect Google Play reviews",
        fn=lambda: stage_collect_play(ctx),
        critical=True,
    )
    run_stage(
        ctx,
        stage_no=2,
        total=TOTAL_STAGES,
        title="Collect Reddit posts",
        fn=lambda: stage_collect_reddit(ctx),
        critical=False,
    )
    run_stage(
        ctx,
        stage_no=3,
        total=TOTAL_STAGES,
        title="Clean data",
        fn=lambda: stage_clean_data(ctx),
        critical=True,
    )
    run_stage(
        ctx,
        stage_no=4,
        total=TOTAL_STAGES,
        title="Generate embeddings",
        fn=lambda: stage_generate_embeddings(ctx),
        critical=True,
    )
    run_stage(
        ctx,
        stage_no=5,
        total=TOTAL_STAGES,
        title="Detect themes",
        fn=lambda: stage_detect_themes(ctx),
        critical=True,
    )
    run_stage(
        ctx,
        stage_no=6,
        total=TOTAL_STAGES,
        title="Analyze sentiment",
        fn=lambda: stage_analyze_sentiment(ctx),
        critical=True,
    )
    run_stage(
        ctx,
        stage_no=7,
        total=TOTAL_STAGES,
        title="Segment users",
        fn=lambda: stage_segment_users(ctx),
        critical=True,
    )
    run_stage(
        ctx,
        stage_no=8,
        total=TOTAL_STAGES,
        title="Generate product insights",
        fn=lambda: stage_generate_insights(ctx),
        critical=True,
    )

    print("\n" + "=" * 72)
    print("PIPELINE SUMMARY")
    print("=" * 72)
    if ctx.skipped_stages:
        print("Skipped:")
        for name in ctx.skipped_stages:
            print(f"  - {name}")
    if ctx.failed_stages:
        print("Failed:")
        for name in ctx.failed_stages:
            print(f"  - {name}")
        print("Pipeline finished with errors.")
        return 1

    print("All stages completed successfully.")
    print(f"  Themes     -> {THEMES_CSV}")
    print(f"  Sentiment  -> {SENTIMENT_CSV}")
    print(f"  Segments   -> {SEGMENTS_CSV}")
    print(f"  Insights   -> {INSIGHTS_JSON}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full discovery insight pipeline (collect → insights)."
    )
    parser.add_argument("--play-count", type=int, default=200, help="Google Play reviews to fetch")
    parser.add_argument("--reddit-limit", type=int, default=25, help="Reddit posts per subreddit")
    parser.add_argument(
        "--skip-collect",
        action="store_true",
        help="Skip live collectors and reuse existing raw CSVs (no network)",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue later stages even if a critical stage fails",
    )
    parser.add_argument(
        "--embedding-backend",
        choices=["tfidf", "st"],
        default=EMBEDDING_BACKEND,
        help="Embedding backend (default: PHASE1_EMBEDDING or tfidf)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ctx = PipelineContext(
        play_count=args.play_count,
        reddit_limit=args.reddit_limit,
        skip_collect=args.skip_collect,
        continue_on_error=args.continue_on_error,
        embedding_backend=args.embedding_backend,
    )
    try:
        return run_pipeline(ctx)
    except Exception as exc:  # noqa: BLE001
        _fail(f"Pipeline aborted: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
