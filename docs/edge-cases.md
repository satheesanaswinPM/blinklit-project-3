# Edge Cases

Failure modes, boundary conditions, and handling rules for the discovery insight engine.

**Source:** [architecture.md](./architecture.md) · [problemstatement.md](./problemstatement.md)

---

## How to use this doc

| Severity | Meaning |
| --- | --- |
| **P0** | Must handle in Phase 1 MVP (or block the insight) |
| **P1** | Handle in Phase 2; degrade gracefully in Phase 1 |
| **P2** | Phase 3 / production hardening |

For each case: **trigger → risk → required handling**.

---

## 1. Ingestion & sources

| ID | Edge case | Severity | Handling |
| --- | --- | --- | --- |
| I-01 | Empty or whitespace-only review body | P0 | Drop from NLP; keep audit log of reject reason |
| I-02 | Duplicate same text across sources (copy-paste review) | P0 | Content-hash dedup; keep one canonical doc + source aliases |
| I-03 | Near-duplicate (same complaint, slight rewording) | P1 | Similarity dedup on embeddings before clustering |
| I-04 | Non-English / code-mixed (Hinglish) text | P0 | Language detect; route to multilingual model or “unsupported lang” bucket—do not force English labels |
| I-05 | Emoji-only or meme-only posts | P0 | Filter or low-weight; do not create themes from emoji-only clusters |
| I-06 | Very long thread / multi-comment Reddit posts | P0 | Chunk by comment; preserve parent thread id for context |
| I-07 | Deleted / 404 URL after scrape | P1 | Keep text snapshot; mark `url_status=dead`; still citable with caveat |
| I-08 | Rate limit / API quota exhausted mid-run | P0 | Checkpoint progress; resume; surface partial corpus warning in UI |
| I-09 | Source TOS forbids scraping; only export available | P0 | Disable scraper for that source; require manual/API export path |
| I-10 | Bot / spam / incentivized reviews | P0 | Heuristic spam score; exclude from theme volume by default; optional “include spam” debug filter |
| I-11 | Same user flooding many similar reviews | P1 | Cap contribution per `author_hash` per theme |
| I-12 | Timestamp missing or in future/timezone-ambiguous | P0 | Fallback to `scraped_at`; flag `time_uncertain=true` |
| I-13 | Channel metadata incomplete (no star rating, no subreddit) | P0 | Allow nullable fields; do not fail entire batch |
| I-14 | CSV upload with wrong schema / encoding (UTF-16, Latin-1) | P0 | Validate schema; try encoding detect; reject with row-level errors |
| I-15 | Extremely large single file upload | P1 | Size/row limits; async job + progress; reject above hard cap |

---

## 2. Storage & data contracts

| ID | Edge case | Severity | Handling |
| --- | --- | --- | --- |
| S-01 | Re-ingest of same document id with changed text | P0 | Never mutate raw row; write new version or new id; invalidate old annotations |
| S-02 | Annotation exists but document deleted from serving view | P0 | Cascade soft-hide; do not orphan quotes in ThemeCards |
| S-03 | Embedding dimension change after model upgrade | P0 | Version embeddings (`embedding_model_ver`); do not mix dims in one index |
| S-04 | Pipeline re-run produces conflicting labels for same doc | P0 | Annotations keyed by `(doc_id, model_ver)`; UI defaults to latest successful run |
| S-05 | PII in free text (phone, address, email) | P0 | Redact/hash before export and before dashboard display of full text options |
| S-06 | Author handle stored in plain text | P0 | Store `author_hash` only in exports and external packs |
| S-07 | Vector index out of sync with document table | P1 | Rebuild job; health check comparing counts |
| S-08 | Partial transaction (docs written, annotations failed) | P0 | Job status `failed_partial`; Theme Explorer excludes incomplete runs |

---

## 3. NLP, clustering & labeling

| ID | Edge case | Severity | Handling |
| --- | --- | --- | --- |
| N-01 | Corpus too small to cluster meaningfully (&lt; N docs) | P0 | Skip clustering; show “insufficient data”; allow keyword/LLM summary only with low confidence |
| N-02 | One giant cluster absorbs everything | P0 | Tune min cluster size / UMAP; force outlier bucket; flag low-coherence themes |
| N-03 | Many singleton clusters (noise) | P0 | Merge into “Other / long-tail”; hide from default Theme Explorer |
| N-04 | Sarcasm / irony (“Great, another rotten tomato”) | P1 | Prefer aspect+evidence over global sentiment; human review for high-rank opportunities |
| N-05 | Mixed sentiment in one review (love delivery, hate dairy) | P0 | Aspect-level sentiment; do not collapse to single polarity |
| N-06 | Feedback not about category discovery (driver tip, app crash) | P0 | Off-topic classifier; separate “ops/app” bucket—exclude from discovery opportunity board by default |
| N-07 | Mentions competitor brands / other apps | P1 | Tag `competitor_mention`; useful signal but do not map to wrong internal category |
| N-08 | Ambiguous category reference (“fresh stuff”, “that green bottle”) | P0 | Low-confidence category; omit from category-specific opportunities until resolved |
| N-09 | Multi-barrier in one utterance (price + trust) | P0 | Allow multi-label barriers; ranking should not force single barrier |
| N-10 | LLM invents a theme with no supporting quotes | P0 | **Hard gate:** reject themes with &lt; K grounded quotes; quote must be substring/retrieval hit |
| N-11 | LLM assigns barrier not in taxonomy | P0 | Constrain to taxonomy + `other`; queue `other` for taxonomy revision |
| N-12 | Label drift after prompt/model change | P0 | Compare to gold set before promoting `model_ver`; keep previous as default if F1 drops |
| N-13 | Toxic / abusive content | P0 | Safety filter; do not surface full text in exports; aggregate-only if needed |
| N-14 | Review about a category the catalog no longer sells | P1 | Map to deprecated category flag; still valid for “why users left” insights |
| N-15 | Seasonal spike misread as durable unmet need (festive demand) | P1 | Time-normalize volume; show seasonality tag on ThemeCards |

---

## 4. Insight engine, segments & opportunities (Phase 2)

| ID | Edge case | Severity | Handling |
| --- | --- | --- | --- |
| O-01 | High volume theme with low addressability (regulatory / city ops) | P1 | Score addressability low; still visible but not top of Opportunity Board |
| O-02 | High severity, tiny sample size (n=3 angry posts) | P0 | Confidence floor; require minimum evidence count for “High impact” badge |
| O-03 | Segment tagger overfits channel (all Reddit → “power users”) | P1 | Show channel mix inside segment; warn when segment ≈ single source |
| O-04 | Same opportunity generated under two theme ids | P1 | Dedup briefs by barrier+lever+category embedding similarity |
| O-05 | Contradictory themes (users want more SKUs vs fewer choices) | P0 | Surface both with evidence; do not auto-merge opposites |
| O-06 | Recommended lever not owned by product (e.g. needs logistics) | P1 | Lever taxonomy includes `owner_team`; route or mark out-of-scope |
| O-07 | Spike alert on viral one-off (meme / news event) | P1 | Require sustained elevation over W windows; allow mute/snooze |
| O-08 | Opportunity cites quotes that later fail grounding check | P0 | Revalidate on publish; demote/remove if quotes missing |
| O-09 | Human rejects brief; system re-proposes identical brief next week | P1 | Store rejection reason + fingerprint; suppress or require new evidence |
| O-10 | Segment “most receptive” empty after filters | P0 | Empty state with explanation; do not invent a segment |

---

## 5. Serving (dashboard, API, alerts)

| ID | Edge case | Severity | Handling |
| --- | --- | --- | --- |
| V-01 | User filters yield zero themes | P0 | Empty state + suggest clearing filters / widening date range |
| V-02 | Stale dashboard while batch job still running | P0 | Show `pipeline_run_status` + data-as-of timestamp |
| V-03 | Quote text truncated mid-sentence in UI | P0 | Expand/full quote view; never cut uniqueness of evidence |
| V-04 | API consumer requests deleted/deprecated `theme_id` | P0 | 404 with `gone=true` + successor id if merged |
| V-05 | Webhook storm during corpus backfill | P1 | Alert only on incremental runs; digest mode for backfills |
| V-06 | Export includes PII despite redaction bug | P0 | Pre-export scan; block export on PII hit; audit log |
| V-07 | Concurrent edits in human review queue | P1 | Optimistic locking / last-write-wins with conflict notice |
| V-08 | Very slow search over large quote corpus | P1 | Paginate; search embeddings index not full table scan |

---

## 6. Product action & closed loop (Phase 3)

| ID | Edge case | Severity | Handling |
| --- | --- | --- | --- |
| C-01 | Experiment launched from outdated opportunity | P2 | Snapshot brief version on experiment start; freeze cited evidence |
| C-02 | North-star moves for reasons unrelated to experiment (season, supply) | P2 | Require holdout / geo split; report confounders |
| C-03 | New-category definition changes mid-quarter | P2 | Version category taxonomy; recompute metric under both definitions |
| C-04 | Positive experiment but feedback themes unchanged | P2 | Expected lag; do not auto-downrank opportunity solely from short window |
| C-05 | Negative experiment wrongly used to delete a valid insight | P2 | Separate “insight validity” from “lever effectiveness” |
| C-06 | Feedback loop trains on biased experiment audience only | P2 | Keep global feedback corpus as primary; outcomes as calibration signal |
| C-07 | Orchestrator retry doubles experiment kickoff | P2 | Idempotent experiment keys tied to `opportunity_id` + version |

---

## 7. Phase 0 / evaluation edge cases

| ID | Edge case | Severity | Handling |
| --- | --- | --- | --- |
| E-01 | Annotator disagreement on barrier | P0 | Dual label + adjudication; track inter-annotator agreement |
| E-02 | Gold set skewed to one channel or category | P0 | Stratify sample; report metrics per channel |
| E-03 | Taxonomy gap discovered late (new barrier type) | P0 | Version taxonomy; remap historical labels via migration guide |
| E-04 | Metric looks good offline, themes useless to PMs | P1 | Add human usefulness rating on ThemeCards as acceptance gate |
| E-05 | Quote grounding rate high but quotes irrelevant | P0 | Relevance check: quote must support the theme claim, not just appear in cluster |

---

## 8. Cross-cutting risk matrix

| Risk | Example edge cases | Default product behavior |
| --- | --- | --- |
| Ungrounded insight | N-10, O-08 | Block publish |
| Privacy leak | S-05, S-06, V-06 | Redact + block export |
| Bias / non-representative | I-10, O-03, E-02 | Show channel mix + confidence |
| Ops / infra failure | I-08, S-08, V-02 | Partial run + explicit freshness |
| Wrong product action | O-01, C-02, C-05 | Human review + experiment design |

---

## 9. MVP (Phase 1) must-pass checklist

Before calling Phase 1 “done”, verify:

1. Empty, spam, off-topic, and non-supported-language docs do not create top themes  
2. Every ThemeCard has ≥ K retrievable quotes with working provenance fields  
3. Dedup prevents duplicate volume inflation  
4. Mixed-aspect and multi-barrier docs are not collapsed incorrectly  
5. UI shows data-as-of time and empty/filter states  
6. PII is not present in default exports  
7. Pipeline failure leaves no silently stale “success” insights  

---

## 10. Suggested test fixtures

Keep a small `fixtures/edge_cases/` corpus with labeled examples for:

- Hinglish mixed review  
- Sarcastic positive wording, negative intent  
- Duplicate across App Store + Reddit  
- App-crash rant (off-topic for discovery)  
- Multi-barrier price + freshness  
- Emoji-only  
- Competitor comparison  
- Seasonal festival demand spike  

Run these through the pipeline in CI (or a script) on every `model_ver` bump.
