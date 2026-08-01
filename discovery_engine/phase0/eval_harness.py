from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .validate import load_jsonl, span_in_text


def _normalize_theme(t: str) -> str:
    t = t.lower().strip()
    t = re.sub(r"[^a-z0-9\s]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


@dataclass
class MultilabelScores:
    micro_precision: float
    micro_recall: float
    micro_f1: float
    macro_f1: float
    per_class_f1: dict[str, float]
    support: dict[str, int]


def multilabel_f1(
    gold: list[set[str]],
    pred: list[set[str]],
) -> MultilabelScores:
    tp = fp = fn = 0
    class_tp: dict[str, int] = defaultdict(int)
    class_fp: dict[str, int] = defaultdict(int)
    class_fn: dict[str, int] = defaultdict(int)
    support: dict[str, int] = defaultdict(int)

    for g, p in zip(gold, pred):
        for c in g:
            support[c] += 1
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
        for c in g & p:
            class_tp[c] += 1
        for c in p - g:
            class_fp[c] += 1
        for c in g - p:
            class_fn[c] += 1

    def prf(t: int, f_p: int, f_n: int) -> tuple[float, float, float]:
        prec = t / (t + f_p) if (t + f_p) else 0.0
        rec = t / (t + f_n) if (t + f_n) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        return prec, rec, f1

    micro_p, micro_r, micro_f = prf(tp, fp, fn)
    classes = sorted(set(support) | set(class_fp) | set(class_fn))
    per_f1: dict[str, float] = {}
    for c in classes:
        _, _, f1 = prf(class_tp[c], class_fp[c], class_fn[c])
        per_f1[c] = f1
    macro = sum(per_f1.values()) / len(per_f1) if per_f1 else 0.0
    return MultilabelScores(micro_p, micro_r, micro_f, macro, per_f1, dict(support))


def accuracy(gold: list[str], pred: list[str]) -> float:
    if not gold:
        return 0.0
    return sum(1 for g, p in zip(gold, pred) if g == p) / len(gold)


def grounding_rate(
    predictions: list[dict[str, Any]],
    docs_by_id: dict[str, dict[str, Any]],
    k: int = 1,
) -> tuple[float, list[str]]:
    """Return grounding rate and list of failing prediction ids."""
    if not predictions:
        return 0.0, []
    ok = 0
    fails: list[str] = []
    for pred in predictions:
        pid = str(pred.get("id", ""))
        doc_id = str(pred.get("doc_id") or pred.get("id") or "")
        doc = docs_by_id.get(doc_id)
        spans = pred.get("evidence_spans") or []
        if doc is None:
            fails.append(pid or doc_id)
            continue
        text = doc.get("text") or ""
        valid = [s for s in spans if isinstance(s, str) and span_in_text(s, text)]
        if len(valid) >= k:
            ok += 1
        else:
            fails.append(pid or doc_id)
    return ok / len(predictions), fails


def theme_precision_at_k(
    gold_themes: list[str],
    pred_themes_ranked: list[str],
    k: int = 5,
) -> float:
    if k <= 0:
        return 0.0
    gold_norm = {_normalize_theme(t) for t in gold_themes if t}
    hits = 0
    for t in pred_themes_ranked[:k]:
        if _normalize_theme(t) in gold_norm:
            hits += 1
    return hits / k


def index_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(r["id"]): r for r in rows if "id" in r}


def evaluate(
    gold_rows: list[dict[str, Any]],
    pred_rows: list[dict[str, Any]],
    *,
    grounding_k: int = 1,
    theme_k: int = 5,
) -> dict[str, Any]:
    gold_by_id = index_by_id(gold_rows)
    pred_by_id = index_by_id(pred_rows)

    paired_ids = sorted(set(gold_by_id) & set(pred_by_id))
    # Prefer evaluating on gold docs that are not spam for label metrics
    paired_ids_labels = [
        i
        for i in paired_ids
        if not gold_by_id[i].get("is_spam")
    ]

    g_barriers = [set(gold_by_id[i].get("barriers") or []) for i in paired_ids_labels]
    p_barriers = [set(pred_by_id[i].get("barriers") or []) for i in paired_ids_labels]
    g_cats = [set(gold_by_id[i].get("categories") or []) for i in paired_ids_labels]
    p_cats = [set(pred_by_id[i].get("categories") or []) for i in paired_ids_labels]
    g_insights = [set(gold_by_id[i].get("insight_types") or []) for i in paired_ids_labels]
    p_insights = [set(pred_by_id[i].get("insight_types") or []) for i in paired_ids_labels]
    g_sent = [gold_by_id[i]["sentiment"] for i in paired_ids_labels]
    p_sent = [pred_by_id[i].get("sentiment", "") for i in paired_ids_labels]

    barrier_scores = multilabel_f1(g_barriers, p_barriers)
    category_scores = multilabel_f1(g_cats, p_cats)
    insight_scores = multilabel_f1(g_insights, p_insights)

    # Grounding: use prediction evidence against gold text
    preds_for_ground = []
    for i in paired_ids:
        p = dict(pred_by_id[i])
        p["doc_id"] = i
        preds_for_ground.append(p)
    g_rate, g_fails = grounding_rate(preds_for_ground, gold_by_id, k=grounding_k)

    gold_themes = [gold_by_id[i].get("theme", "") for i in paired_ids_labels]
    pred_themes = [pred_by_id[i].get("theme", "") for i in paired_ids_labels if pred_by_id[i].get("theme")]
    # Corpus-level precision@k using unique predicted themes ranked by first appearance
    ranked: list[str] = []
    seen: set[str] = set()
    for t in pred_themes:
        nt = _normalize_theme(t)
        if nt and nt not in seen:
            seen.add(nt)
            ranked.append(t)
    p_at_k = theme_precision_at_k(gold_themes, ranked, k=theme_k)

    return {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "n_gold": len(gold_rows),
        "n_pred": len(pred_rows),
        "n_paired": len(paired_ids),
        "n_paired_non_spam": len(paired_ids_labels),
        "grounding_rate": g_rate,
        "grounding_k": grounding_k,
        "grounding_failures": g_fails[:50],
        "barrier": asdict(barrier_scores),
        "category": asdict(category_scores),
        "insight_type": asdict(insight_scores),
        "sentiment_accuracy": accuracy(g_sent, p_sent),
        f"theme_precision_at_{theme_k}": p_at_k,
        "missing_predictions": sorted(set(gold_by_id) - set(pred_by_id))[:50],
        "extra_predictions": sorted(set(pred_by_id) - set(gold_by_id))[:50],
    }


def write_report(result: dict[str, Any], out_json: Path, out_md: Path | None = None) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if out_md is None:
        return
    lines = [
        "# Eval report",
        "",
        f"- Run at: `{result['run_at']}`",
        f"- Paired non-spam docs: **{result['n_paired_non_spam']}**",
        f"- Grounding rate (k={result['grounding_k']}): **{result['grounding_rate']:.3f}**",
        f"- Barrier micro-F1: **{result['barrier']['micro_f1']:.3f}**",
        f"- Category micro-F1: **{result['category']['micro_f1']:.3f}**",
        f"- Insight-type micro-F1: **{result['insight_type']['micro_f1']:.3f}**",
        f"- Sentiment accuracy: **{result['sentiment_accuracy']:.3f}**",
        "",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")


def load_and_evaluate(gold_path: Path, pred_path: Path, **kwargs: Any) -> dict[str, Any]:
    return evaluate(load_jsonl(gold_path), load_jsonl(pred_path), **kwargs)
