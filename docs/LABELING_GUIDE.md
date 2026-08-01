# Labeling Guide (Phase 0)

Rules for human annotators and for evaluating LLM labels against the gold set.

**Taxonomies:** `taxonomies/categories.json`, `taxonomies/barriers.json`, `taxonomies/insight_types.json`  
**Schema:** `schemas/gold_label.schema.json`

---

## 1. Unit of labeling

Label **one feedback document** (review, comment, or post body). If a Reddit thread has multiple comments, label **each comment** as its own document with a shared `thread_id` in metadata.

Do **not** invent facts that are not in the text.

---

## 2. Fields to label

| Field | Required | Rules |
| --- | --- | --- |
| `sentiment` | Yes | `positive` \| `neutral` \| `negative` — overall polarity of the doc |
| `categories` | Yes | 0+ ids from category taxonomy. Use `off_topic` if not about products/categories. Use `unknown` only if product-related but unmappable. |
| `barriers` | Yes | 1+ ids from barrier taxonomy. Multi-label allowed. Use `none` if no exploration barrier. Use `other` only when a real barrier is present but unlisted. |
| `insight_types` | Yes | 1+ insight types this doc can evidence (see mapping below) |
| `discovery_paths` | No | Only if the text mentions how they find products |
| `info_needs` | No | Only if the text states missing information before trying |
| `segment_proxies` | No | Only if clearly implied; default `unknown` or omit |
| `theme` | Yes | Short noun phrase summarizing the core claim (≤ 8 words) |
| `evidence_spans` | Yes | 1+ exact substrings from `text` that support labels (quote grounding) |
| `is_spam` | Yes | `true` if bot/incentivized/nonsensical |
| `is_off_topic_discovery` | Yes | `true` if about app crash, delivery partner tip, payment, etc., with no category-exploration signal |
| `confidence` | Yes | Annotator confidence `low` \| `medium` \| `high` |
| `notes` | No | Disambiguation notes for adjudicators |

---

## 3. Mapping text → insight types

| If the text mainly… | Add insight type |
| --- | --- |
| Explains why they stick to the same categories | `habit_drivers` |
| States a reason they won't try something new | `barrier_taxonomy` |
| Describes how they find or fail to find products | `discovery_path_map` |
| Asks for info before trying (expiry, photos, etc.) | `info_needs` |
| Signals openness or persona for exploration | `receptive_segments` |
| Suggests a fix the product could make | `opportunity_briefs` |

A document may map to **multiple** insight types.

---

## 4. Decision rules (edge cases)

1. **Mixed sentiment** — Label overall sentiment; put the negative aspect in `barriers` / `theme` (e.g. “love delivery, hate dairy freshness” → sentiment `neutral` or `negative` if dairy is the focus; barriers include `freshness_spoilage`).
2. **Sarcasm** — Prefer inferred intent over literal words; set `confidence` to `low` if unsure.
3. **Multi-barrier** — Label all clear barriers; do not force a single label.
4. **Competitor mentions** — Still map categories/barriers; note in `notes`.
5. **Seasonal demand** — Still label barriers/info needs; note `seasonal` in `notes`.
6. **Emoji-only / empty** — Mark `is_spam=true`; barriers=`none`; theme=`empty_or_spam`.
7. **Ambiguous category** — Prefer `unknown` over guessing; confidence `low`.
8. **`other` barrier** — Allowed only with a `notes` explanation proposing a future taxonomy id.

See also project root `edge-cases.md` (N-*, E-*).

---

## 5. Evidence span rules (grounding)

- Every non-spam label set must include ≥1 `evidence_spans` entry.
- Each span **must be an exact substring** of `text` (case-sensitive match preferred; validators allow normalized whitespace).
- Spans should be the shortest phrase that supports the claim (typically 3–25 words).
- Themes without grounded spans are **invalid** for the gold set.

---

## 6. Dual annotation & adjudication

For gold-set quality:

1. Two annotators label independently when possible.
2. If barriers or theme disagree → third annotator or lead adjudicates.
3. Record agreement in eval reports (see `docs/METRICS.md`).
4. Adjudicated label becomes the canonical gold row (`adjudicated=true`).

---

## 7. LLM-as-labeler evaluation

When scoring model output against gold:

- Barriers and categories: **multi-label F1** (micro and macro).
- Themes: do not require exact string match; use **precision@k** against clustered/predicted theme ids mapped by human or embedding similarity ≥ threshold (see metrics doc).
- Reject predictions that fail quote grounding even if labels match.

---

## 8. What not to do

- Do not copy taxonomy descriptions into `theme`.
- Do not label `opportunity_briefs` unless the text implies a product-actionable need or fix.
- Do not mark `habit_mental_model` solely because the user bought milk again—need language about routine/mental slotting or refusal to explore.
- Do not include author names, phones, or addresses in `notes` or exports.
