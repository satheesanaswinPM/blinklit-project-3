# Blinkit Grad Project — Category Discovery Engine

Research console for the primary question:

> **Why don't Blinkit users explore new categories?**

**Primary UX:** Streamlit dashboard (`app.py`).  
**Optional advanced:** React Theme Explorer + FastAPI (see [Optional Theme Explorer](#optional-theme-explorer-advanced)).

| Doc | Purpose |
| --- | --- |
| [docs/problemstatement.md](./docs/problemstatement.md) | Problem & opportunity |
| [docs/architecture.md](./docs/architecture.md) | Phase-wise architecture |
| [docs/edge-cases.md](./docs/edge-cases.md) | Edge cases |
| [docs/LABELING_GUIDE.md](./docs/LABELING_GUIDE.md) | Gold labeling rules |
| [docs/METRICS.md](./docs/METRICS.md) | Offline metrics |

## Quick start (primary path)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env

# Reuse existing raw CSVs (no network) → full 13-stage pipeline
python main.py --skip-collect

# Research console
python -m streamlit run app.py
```

## Streamlit IA

| Page | Purpose |
| --- | --- |
| **Findings Board** | Executive summary, barriers, JTBD, unmet needs, experiments |
| **Category Opportunities** | Ranked expansion bets (e.g. snacks, home) |
| **Validation Desk** | Hypothesis triangulation + human approve/reject review |
| **Live Pipeline** | Artifact freshness board |
| **Try-it Console** | Paste a review → exploration tags |
| **Prototype Lab** | Clickable MVP mocks (snacks discovery rail · Home first-buy guarantee) |
| **Evidence Lab** | Quote / theme evidence deep-dive |
| **Methodology** | Pipeline explained |
| **Admin** | Password-gated ops (`ADMIN_DASHBOARD_PASSWORD`) |

## Layout

```
.
├── app.py                 # Primary Streamlit research console
├── main.py                # 13-stage end-to-end pipeline
├── analysis/              # themes, sentiment, segments, exploration
├── llm/                   # insights + synthesis
├── discovery_engine/      # collectors, NLP helpers, optional FastAPI
├── scripts/               # gold/eval/smoke CLIs
├── data/raw|processed/    # corpus artifacts
├── output/                # themes, sentiment, synthesis, reviews…
├── frontend/              # Optional React Theme Explorer (advanced)
└── docs/
```

## Pipeline (`main.py`) — 13 stages

1. Collect Google Play reviews  
2. Collect App Store reviews  
3. Collect Reddit posts  
4. Collect YouTube comments  
5. Merge multi-source corpus → `data/processed/merged_reviews.csv`  
6. Clean / soft-dedupe → `data/processed/preprocessed_reviews.csv`  
7. Embeddings → `data/processed/review_embeddings.npy`  
8. Themes (BERTopic) → `output/themes.csv`  
9. Sentiment → `output/sentiment.csv`  
10. Segments → `output/user_segments.csv`  
11. Exploration tagging → `output/exploration_tags.csv`  
12. Product insights → `output/insights.json`  
13. Synthesis (JTBD / experiments / category ops) → `output/synthesis.json`  

```bash
python main.py
python main.py --play-count 100 --reddit-limit 15
python main.py --skip-collect   # reuse data/raw; no live scrapes
```

NLP analysis (sentiment / segments / exploration / themes filter) aligns on the **preprocessed** corpus so dashboard KPIs stay consistent.

## Prototype Lab MVPs

Linked from synthesis when available:

| Tab | Experiment | Category / barrier |
| --- | --- | --- |
| Snacks “Also useful tonight” rail | `exp_discover_rail` | snacks · hard to discover |
| Home first-buy quality guarantee | `exp_first_buy_guarantee` | home · quality distrust |

## Smoke checks & CI

```bash
python scripts/smoke_check.py
```

GitHub Actions runs this on pull requests (see `.github/workflows/smoke.yml`).

## Experiment briefs & stakeholder analytics

```bash
# One-page Markdown brief (also downloadable in Findings Board / Prototype Lab)
python scripts/export_experiment_brief.py --mvp snacks_rail
python scripts/export_experiment_brief.py --mvp home_guarantee
```

Page views and Prototype Lab MVP usage append to `output/stakeholder_events.jsonl` (see Admin → Stakeholder analytics).

## Phase 0 — gold foundation

```bash
python -m scripts.generate_seed_gold --n 220
python -m scripts.validate_gold
# Real heuristic predictions vs gold (barrier F1) — not identity
python -m scripts.run_eval
# Optional harness sanity check (should be ~1.0 F1)
python -m scripts.run_eval --identity
```

## Optional Theme Explorer (advanced)

Secondary path for BERTopic/API deep-dives. **Stakeholders should use Streamlit.**

```bash
# API
uvicorn discovery_engine.api:app --reload

# UI (other terminal)
cd frontend
npm install
npm run dev
```

See [frontend/README.md](./frontend/README.md).

## Env

Copy `.env.example` → `.env`. Common keys: `HF_TOKEN`, `OPENAI_API_KEY` / `GROQ_API_KEY`, `REDDIT_*`, `YOUTUBE_API_KEY`, `ADMIN_DASHBOARD_PASSWORD`.
