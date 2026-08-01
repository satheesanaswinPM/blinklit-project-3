# Blinkit Grad Project — Discovery Insight Engine

Unified Phase 0 + Phase 1 codebase for turning multi-channel quick-commerce feedback into grounded category-discovery insights.

| Doc | Purpose |
| --- | --- |
| [docs/problemstatement.md](./docs/problemstatement.md) | Problem & opportunity |
| [docs/architecture.md](./docs/architecture.md) | Phase-wise architecture |
| [docs/edge-cases.md](./docs/edge-cases.md) | Edge cases |
| [docs/LABELING_GUIDE.md](./docs/LABELING_GUIDE.md) | Gold labeling rules |
| [docs/METRICS.md](./docs/METRICS.md) | Offline metrics |

## Layout

```
.
├── discovery_engine/     # Python package (phase0 + phase1)
│   ├── phase0/           # taxonomies loader, gold validate, eval harness
│   ├── collectors/       # CSV ingest
│   ├── nlp/              # clean, preprocess, embed, label, cluster, BERTopic
│   ├── storage/          # SQLite
│   ├── pipeline.py       # Phase 1 end-to-end batch
│   └── api.py            # FastAPI Theme Explorer
├── scripts/              # CLIs for both phases
├── taxonomies/           # shared category / barrier / insight JSON
├── schemas/              # gold label schema
├── data/
│   ├── gold/             # gold_labels.jsonl
│   ├── raw/              # Play / Reddit / sample CSVs
│   ├── processed/        # DB, BERTopic outputs, preprocessed CSV
│   └── fixtures/
├── frontend/             # React Theme Explorer
├── docs/
├── .env.example
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
# fill HF_TOKEN, OPENAI_API_KEY, REDDIT_* as needed
```

## Phase 0 — foundation

```bash
python -m scripts.generate_seed_gold --n 220
python -m scripts.validate_gold
python -m scripts.run_eval
```

## Phase 1 — collectors → NLP → Theme Explorer

```bash
# optional live collectors
python -m scripts.download_blinkit_play_reviews --count 200
python -m scripts.download_reddit_posts --limit 25

# preprocess + BERTopic (optional deep dive)
python -m scripts.preprocess_reviews --in data/raw/blinkit_play_reviews.csv --text-col Review
python -m scripts.run_bertopic_clusters --in data/raw/blinkit_play_reviews.csv --text-col Review --preprocess --no-openai

# seed sample corpus + insight pipeline → SQLite
python -m scripts.run_pipeline

# API
uvicorn discovery_engine.api:app --reload

# UI (other terminal)
cd frontend
npm install
npm run dev
```

## Theme analysis (BERTopic)

```bash
python -m analysis.themes --in data/raw/blinkit_play_reviews.csv --text-col Review
# writes output/themes.csv
# caches embeddings to data/processed/review_embeddings.npy
```

## Sentiment analysis (HuggingFace)

```bash
python -m analysis.sentiment --in data/raw/blinkit_play_reviews.csv --text-col Review
# writes output/sentiment.csv  (Review, Sentiment, Confidence Score)
# uses HF_TOKEN from .env when set
```

## User segments (KMeans)

```bash
python -m analysis.segments --in data/raw/blinkit_play_reviews.csv --text-col Review
# writes output/user_segments.csv
# Segments: Routine Buyers | Explorers | Price Sensitive | Impulse Buyers
```

## Product insights (LLM)

```bash
python -m llm.insights
# reads output/themes.csv + output/sentiment.csv (+ user_segments.csv if present)
# writes output/insights.json
# uses OPENAI_API_KEY from .env; falls back to grounded heuristics if unavailable
```

## Full pipeline

```bash
python main.py
python main.py --play-count 100 --reddit-limit 15
python main.py --skip-collect   # reuse existing data/raw CSVs
```

Runs in order: Play collect → Reddit collect → clean → embeddings → themes → sentiment → segments → insights. Prints progress after each stage; Reddit is skipped (with a warning) if credentials are missing.

## Streamlit dashboard

```bash
python -m streamlit run app.py
# or: streamlit run app.py  (if streamlit is on PATH)
```

Left sidebar navigates Overview, Top Themes, Sentiment, User Segments, Product Insights, Opportunity Ranking, and Methodology. Charts use Plotly; data is read from `output/`.

## Auto-push to GitHub

Cursor project hooks (`.cursor/hooks.json`) run `python .cursor/hooks/auto_push.py` after agent file edits and when the agent stops. The script stages changes (honoring `.gitignore`), commits, and pushes to `origin` on the current branch. Rapid edits are debounced (~45s). Never force-pushes; `.env` stays excluded.
