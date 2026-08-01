# Phase-wise Architecture

AI-powered discovery insight engine for quick-commerce category exploration.

**North-star metric:** % of Monthly Active Customers (MACs) who purchase from at least one new product category each month.

**Source:** [problemstatement.md](./problemstatement.md) · **Edge cases:** [edge-cases.md](./edge-cases.md) · **Code:** [../README.md](../README.md)



---

## Architecture overview

```
Feedback sources → Ingestion → Storage → NLP & theme mining → Insight engine → Serving → Product action
```

| Layer | Role |
| --- | --- |
| Ingestion | Collect and normalize multi-channel feedback |
| Storage | Raw documents, annotations, embeddings, provenance |
| NLP & theme mining | Sentiment, clustering, barrier/theme labels |
| Insight engine | Map themes to barriers, segments, opportunities |
| Serving | Dashboard, API, alerts, exports |
| Product action | Discovery UX, merchandising, experiments |

---

## Phase 0 — Foundation (scope & evaluation)

**Goal:** Make later phases measurable and comparable.

**Implementation:** [`discovery_engine/phase0/`](../discovery_engine/phase0/) + [`taxonomies/`](../taxonomies/) + [`data/gold/`](../data/gold/). See [`README.md`](../README.md).



### What is built

| Component | Detail |
| --- | --- |
| Category taxonomy | Map feedback language → platform categories |
| Barrier taxonomy v0 | Trust, price, freshness, discovery gap, habit, past bad experience, etc. |
| Gold label set | 200–500 hand-labeled samples (theme, barrier, category, sentiment) |
| Labeling guide | Rules for annotators / LLM evaluation |
| Metric definitions | Theme precision@k, barrier F1, quote grounding rate |

### Architecture slice

```
[Manual exports / sample CSVs] → [Label store] → [Eval harness]
```

### Exit criteria

- Six research questions documented as insight types
- Gold set + labeling guide approved
- Offline metrics defined before model work starts

---

## Phase 1 — MVP insight pipeline

**Goal:** Answer the research questions with cited evidence from 2–3 feedback channels.

**Implementation:** [`discovery_engine/`](../discovery_engine/) + [`scripts/`](../scripts/) + [`frontend/`](../frontend/). See [`README.md`](../README.md).



### Scope

| In | Out (later) |
| --- | --- |
| App / Play Store reviews | Real-time streaming |
| Reddit / forums | Full social firehose |
| Product reviews (sample) | Closed-loop experiments |
| Batch NLP + clustering | Segment propensity models |
| Theme Explorer dashboard | Production warehouse / Kafka |

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         PHASE 1 MVP                             │
├──────────────┬──────────────┬──────────────┬────────────────────┤
│  Collectors  │   Storage    │  NLP batch   │      Serving       │
│  ──────────  │  ──────────  │  ──────────  │  ────────────────  │
│  Store APIs  │  Postgres    │  Clean/embed │  FastAPI           │
│  / exports   │  + pgvector  │  Cluster     │  Theme Explorer UI │
│  Reddit API  │  Raw docs    │  LLM labels  │  Evidence quotes   │
│  CSV upload  │  Annotations │  Theme cards │  Simple filters    │
└──────────────┴──────────────┴──────────────┴────────────────────┘
```

### Data contracts (minimal)

- **FeedbackDocument** — id, source, text, url, timestamps, channel metadata
- **Annotation** — sentiment, aspects, category tags, barrier labels, embedding, model version
- **ThemeCard** — cluster id, title, volume, sentiment mix, top evidence quotes

### Stack (recommended)

- Python collectors + batch jobs
- Postgres (+ pgvector) or SQLite for demo
- sentence-transformers + BERTopic / HDBSCAN + LLM batch labeling
- FastAPI + React dashboard

### Exit criteria

- Pipeline runs end-to-end on sample corpus
- Dashboard answers: habit drivers, barriers, discovery paths, info needs
- Every insight linked to source quotes (no ungrounded claims)

---

## Phase 2 — Segments & opportunity scoring

**Goal:** Turn themes into ranked, segment-aware product opportunities.

### What is added

| Component | Detail |
| --- | --- |
| Segment proxies | Language / pattern-based receptive segments |
| OpportunityBrief | barrier × segment × lever × impact × confidence |
| Ranking | Volume × severity × addressability × novelty |
| Alerts | Spike detection on emerging themes |
| Exports | Weekly opportunity pack (CSV / Notion / Slack) |

### Architecture delta

```
Phase 1 pipeline
      │
      ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Segment tagger  │ ──► │ Opportunity      │ ──► │ Opportunity     │
│                 │     │ scorer + ranker  │     │ Board + alerts  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

### Serving surfaces

1. **Theme Explorer** — clusters, trends, evidence
2. **Opportunity Board** — ranked briefs with recommended levers (search, merch, trust content, trial packs)
3. **Insight API** — `GET /themes`, `/barriers`, `/opportunities`; webhook on theme spikes

### Exit criteria

- Weekly opportunity pack usable in product rituals
- Segments and barriers filterable in UI
- Human review queue for high-impact briefs

---

## Phase 3 — Closed loop (optional / scale)

**Goal:** Connect insights to interventions and measure north-star lift.

### What is added

| Component | Detail |
| --- | --- |
| Experiment hooks | Push opportunities into discovery / merch experiments |
| Outcome feedback | New-category MAC % and experiment results flow back |
| Label refinement | Retrain / recalibrate from outcomes and human edits |
| Production ops | Orchestration (Airflow/Prefect), lake/warehouse, stronger privacy controls |

### Architecture delta

```
Opportunity Board ──► Experiment platform ──► Discovery / merch UX
                              │
                              ▼
                     Outcome metrics store
                              │
                              ▼
                     Insight engine recalibration
```

### Scale stack (when needed)

- Airflow / Prefect; optional Kafka
- Object storage + warehouse (BigQuery / Snowflake)
- Dedicated vector DB; evaluation harness in CI
- Internal portal + Slack alerts

### Exit criteria

- Measured link: insight → intervention → new-category MAC movement
- Provenance, PII hashing, and pipeline versioning enforced

---

## Cross-phase principles

1. **Grounding** — Every theme and opportunity cites source quotes with URL and date.
2. **Versioning** — Annotations carry `model_ver`; raw documents are immutable.
3. **Separation** — NLP produces insights; product systems consume them via API (no hard-coded promo logic in the model layer).
4. **Bias awareness** — Dashboard shows volume and channel mix; vocal ≠ representative.
5. **Compliance** — Prefer official APIs/exports; respect store and platform TOS.

---

## Phase summary

| Phase | Focus | Primary output |
| --- | --- | --- |
| **0** | Scope & gold set | Taxonomies, labels, eval metrics |
| **1** | MVP pipeline | Theme Explorer with cited insights |
| **2** | Opportunities | Ranked Opportunity Board + API |
| **3** | Closed loop | Experiments tied to new-category MAC % |

---

## Suggested build order

1. Phase 0 artifacts (taxonomy + gold set)
2. Phase 1 collectors → storage → NLP → Theme Explorer
3. Phase 2 opportunity scoring + segments + alerts
4. Phase 3 only if measuring product impact is in scope
