# Metric Definitions (Phase 0)

Offline metrics that must be defined **before** Phase 1 model work.  
Eval harness: `python -m src.run_eval`

---

## 1. Design principles

1. **Grounding first** — Ungrounded insights count as failures even if labels look right.
2. **Multi-label aware** — Barriers and categories are sets, not single classes.
3. **Report with strata** — Overall + per channel + per category family when n allows.
4. **Do not optimize only for F1** — Add human usefulness later (Phase 1 acceptance).

---

## 2. Quote grounding rate

**Definition:** Fraction of predicted themes/opportunities whose cited quotes are valid retrievals from source documents.

```
grounding_rate = (# predictions with ≥ K valid evidence spans) / (# predictions)
```

| Parameter | Default |
| --- | --- |
| `K` | 2 for ThemeCards; 1 for single-doc labels |
| Valid span | Exact substring of document text (whitespace-normalized allowed) |
| Fail | Hallucinated quote, wrong doc id, empty evidence |

**Pass bar (Phase 0→1 gate):** `grounding_rate ≥ 0.95` on the gold-held prediction set.

---

## 3. Barrier F1

**Definition:** Multi-label F1 over barrier ids (excluding documents marked `is_spam`).

For each document, gold set `B*` and prediction `B̂`:

- Precision = `|B* ∩ B̂| / |B̂|` (0 if `B̂` empty and `B*` non-empty → 0)
- Recall = `|B* ∩ B̂| / |B*|`
- F1 = harmonic mean

Report:

| Metric | How |
| --- | --- |
| Micro-F1 | Pool TP/FP/FN across docs |
| Macro-F1 | Unweighted mean F1 per barrier id (skip ids with 0 gold support optional flag) |
| Per-class F1 | Table for each barrier |

**Pass bar (initial):** micro-F1 ≥ 0.70 on gold set for the promoted `model_ver`.

---

## 4. Category F1

Same multi-label F1 procedure over category ids.  
Treat `off_topic` and `unknown` as regular classes.

**Pass bar (initial):** micro-F1 ≥ 0.65.

---

## 5. Theme precision@k

Themes are free-text; exact match is the wrong metric.

**Procedure:**

1. Build a gold theme inventory from the gold set (unique normalized theme strings / curated theme ids).
2. Model produces ranked themes for a channel or corpus slice.
3. A predicted theme is a **hit** if embedding cosine similarity to any gold theme in that slice ≥ `τ` **or** a human judges it equivalent.
4. `precision@k = hits in top-k / k`

| Parameter | Default |
| --- | --- |
| `k` | 5 and 10 |
| `τ` | 0.75 (configurable) |

**Pass bar (initial):** precision@5 ≥ 0.60 with human spot-check on misses.

Phase 0 harness supports **exact/normalized theme match** and optional similarity if embeddings are provided; human equivalence is logged manually.

---

## 6. Sentiment accuracy

Simple accuracy and macro-F1 over `{positive, neutral, negative}`.

**Pass bar (initial):** accuracy ≥ 0.75.

---

## 7. Insight-type recall

For each gold document, predicted insight types should cover the gold set.

```
insight_type_micro_f1  (multi-label F1)
```

Used to ensure the six research questions remain coverable.

---

## 8. Inter-annotator agreement (IAA)

When dual-labeled:

| Label | Metric |
| --- | --- |
| Barriers / categories | Mean Jaccard between annotators |
| Sentiment | Cohen's κ |
| Theme | % exact or adjudicator-equivalent |

**Target:** mean barrier Jaccard ≥ 0.60 before freezing gold v0.1.

---

## 9. Stratification & reporting

Every eval run must write JSON + markdown summary including:

- `model_ver` / `run_id` / timestamp  
- n documents scored  
- grounding_rate, barrier micro/macro F1, category micro F1, sentiment accuracy  
- optional precision@k  
- counts by `source` channel  
- list of failing document ids (ungrounded or empty prediction)

---

## 10. What “approved” means for Phase 0 exit

| Artifact | Approval rule |
| --- | --- |
| Taxonomies | Reviewed; version `0.1.0` frozen |
| Labeling guide | Annotators can label 20 docs with ≤ 2 process questions |
| Metrics | This doc + harness runnable on gold set |
| Gold set | ≥ 200 valid rows, stratified, pass `validate` with 0 errors |

Predictions are not required for Phase 0 exit—only that the **eval harness can score** a predictions file in the documented schema.
