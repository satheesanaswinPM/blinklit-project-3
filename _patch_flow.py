from pathlib import Path

p = Path("app.py")
t = p.read_text(encoding="utf-8")
start = t.find("def render_end_to_end_workflow")
end = t.find("\ndef render_methodology")
assert start > 0 and end > start, (start, end)

new = '''def render_end_to_end_workflow() -> None:
    """Layered ASCII flowchart (main steps only) with live artifact counts."""
    c = _workflow_counts()
    play = f"{c['play_raw']:,}"
    app = f"{c['app_store_raw']:,}"
    reddit = f"{c['reddit_raw']:,}"
    youtube = f"{c['youtube_raw']:,}"
    merged = f"{c['merged']:,}"
    tagged = f"{c['exploration']:,}"
    relevant = f"{c['exploration_relevant']:,}"
    themes = f"{c['themes']:,}"
    sentiment = f"{c['sentiment']:,}"
    segments = f"{c['segments']:,}"
    cat_ops = f"{c['category_ops']:,}"
    insights = f"{c['insights']:,}"

    diagram = f"""
┌──────────────────────────────────────────────────────────────────────────────────┐
│                     LAYER 1 · MULTI-SOURCE SCRAPERS  (scripts/)                     │
│                                                                                    │
│  Apple App Store ─ RSS feed ──────────►  app_store_reviews.csv      ({app:>6})     │
│  Google Play ──── Play scraper ───────►  blinkit_play_reviews.csv   ({play:>6})     │
│  Reddit ───────── PRAW (optional) ────►  reddit_posts.csv           ({reddit:>6})     │
│  YouTube ──────── Data API v3 ────────►  youtube_comments.csv        ({youtube:>6})     │
│  Trustpilot / MouthShut ──────────────►  (DROPPED: bot-walled / JS-rendered)       │
│  LinkedIn / X / TikTok  ── excluded                                                │
│         │  soft-dedupe · non-empty text filter · pagination                        │
└─────────┼──────────────────────────────────────────────────────────────────────┘
          ▼
   ┌─────────────┐   merge.py    normalize to 8 fields, drop noise/dupes
   │  merge.py   ├────────────►  data/processed/merged_reviews.csv   ({merged} items)
   └─────────────┘               id · source · date · rating · text · author · url · scraped_at
          │
          ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                     LAYER 2 · TWO-PASS AI ANALYSIS                                  │
│                                                                                    │
│   PASS 1  exploration.py  ── per-item exploration tags ─────────────────────────┐  │
│      relevance filter · exploration_signal · barriers · categories_mentioned     │  │
│      themes ({themes}) · sentiment ({sentiment}) · segments ({segments})                        │  │
│                                              │                                    │  │
│                                              ▼                                    │  │
│                    output/exploration_tags.csv  ({tagged} · {relevant} relevant)     │  │
│                                              │                                    │  │
│   PASS 2  synthesis.py  ◄────────────────────┘                                   │  │
│      ┌───────────────────────────┐      ┌──────────────────────────────────┐    │  │
│      │ PYTHON (deterministic)    │      │ LLM (optional · language only)    │    │  │
│      │ barriers · signal dist    │  +   │ exec summary polish · narrative   │    │  │
│      │ category opportunity rank │      │ JTBD / unmet needs wording        │    │  │
│      │ hypothesis evidence counts│      │ experiment framing                │    │  │
│      └───────────────────────────┘      └──────────────────────────────────┘    │  │
│                       └──────────────┬────────────────┘                          │  │
│                                      ▼                                            │  │
│              output/synthesis.json  ({cat_ops} category opportunities)             │  │
│              output/insights.json   ({insights} legacy insight cards)              │  │
│                                      │                                            │  │
│   VALIDATE  Validation Desk  ◄───────┤   relevance mix · sample evidence ·        │  │
│                                      │   source triangulation                      │  │
│                                      ▼                                            │  │
│   EXPORT  CSV / JSON downloads from dashboard tabs                                │  │
└──────────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│               LAYER 3 · STREAMLIT DASHBOARD  (app.py)  — reads pre-computed files   │
│                                                                                    │
│   Findings Board · Category Opportunities · Validation Desk · Live Pipeline        │
│   Try-it Console · Evidence Lab · Methodology · Admin (password-gated)             │
│                                       │                           │                │
│                                       ▼                           ▼                │
│                            output/* + data/processed/      main.py (offline run)   │
└──────────────────────────────────────────────────────────────────────────────────┘
""".strip("\\n")

    panel_start(
        "Pipeline architecture",
        "Main control flow only — multi-source scrape → merge → two-pass analysis → dashboard.",
    )
    st.markdown(
        f'<div class="flow-wrap"><pre class="ascii-flow">{html.escape(diagram)}</pre></div>',
        unsafe_allow_html=True,
    )
    panel_end()


'''

# Fix accidental double-escape in strip
new = new.replace('.strip("\\\\n")', '.strip("\\n")')

p.write_text(t[:start] + new + t[end + 1 :], encoding="utf-8")
print("ok", len(new))
