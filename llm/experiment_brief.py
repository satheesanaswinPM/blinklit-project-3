"""Build a one-page experiment brief from synthesis + chosen MVP/experiment."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SYNTHESIS = ROOT / "output" / "synthesis.json"
DEFAULT_OUT_DIR = ROOT / "output" / "briefs"

MVP_PRESETS: dict[str, dict[str, str]] = {
    "snacks_rail": {
        "experiment_id": "exp_discover_rail",
        "category": "snacks",
        "mvp_title": "Grocery → Snacks discovery rail",
        "prototype": "Prototype Lab · MVP 1",
    },
    "home_guarantee": {
        "experiment_id": "exp_first_buy_guarantee",
        "category": "home",
        "mvp_title": "Home first-buy quality guarantee",
        "prototype": "Prototype Lab · MVP 2",
    },
}


def _find_experiment(synthesis: dict, experiment_id: str) -> dict[str, Any]:
    for e in synthesis.get("testable_experiments") or []:
        if str(e.get("id") or "") == experiment_id:
            return e
    return {}


def _find_category(synthesis: dict, category: str | None) -> dict[str, Any]:
    if not category:
        return {}
    needle = category.strip().lower()
    for row in synthesis.get("category_opportunities") or []:
        if str(row.get("category") or "").strip().lower() == needle:
            return row
    return {}


def _find_barrier(synthesis: dict, barrier_id: str | None) -> dict[str, Any]:
    if not barrier_id:
        return {}
    needle = barrier_id.strip().lower()
    for row in synthesis.get("barriers_ranked") or []:
        if str(row.get("barrier") or "").strip().lower() == needle:
            return row
    return {}


def _find_hypothesis(synthesis: dict, hyp_id: str | None) -> dict[str, Any]:
    if not hyp_id:
        return {}
    for h in synthesis.get("hypotheses") or []:
        if str(h.get("id") or "") == hyp_id:
            return h
    return {}


def build_experiment_brief(
    synthesis: dict[str, Any],
    *,
    experiment_id: str,
    category: str | None = None,
    mvp_title: str | None = None,
    prototype: str | None = None,
) -> str:
    """Return a one-page Markdown experiment brief."""
    exp = _find_experiment(synthesis, experiment_id)
    cat = _find_category(synthesis, category)
    barrier_id = str(exp.get("barrier") or cat.get("primary_barrier_to_attack") or "")
    barrier = _find_barrier(synthesis, barrier_id)
    hyp = _find_hypothesis(synthesis, str(exp.get("hypothesis_link") or ""))
    corpus = synthesis.get("corpus") or {}
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    title = mvp_title or exp.get("name") or experiment_id
    lines = [
        f"# Experiment brief — {title}",
        "",
        f"_Generated {now} · Category Discovery Engine_",
        "",
        "## Primary question",
        "",
        synthesis.get("primary_question")
        or "Why don't Blinkit users explore new categories?",
        "",
        "## Recommendation",
        "",
        f"- **Experiment ID:** `{experiment_id}`",
        f"- **Name:** {exp.get('name') or title}",
        f"- **Prototype:** {prototype or '—'}",
        f"- **Target category:** {category or cat.get('category') or '—'}",
        "",
        "## Why this bet",
        "",
        cat.get("why_now")
        or (synthesis.get("executive_summary") or "")[:400]
        or "See synthesis.json for corpus narrative.",
        "",
        "## Linked evidence",
        "",
        f"- **Barrier:** `{barrier_id or '—'}`"
        + (
            f" · {barrier.get('mentions', 0)} mentions · severity {barrier.get('severity', '—')}"
            if barrier
            else ""
        ),
        f"- **Hypothesis:** {hyp.get('id') or exp.get('hypothesis_link') or '—'}"
        + (f" — {hyp.get('statement')}" if hyp.get("statement") else ""),
        f"- **Evidence mentions (hypothesis):** {hyp.get('evidence_mentions', '—')}",
        f"- **Category opportunity:** "
        + (
            f"#{cat.get('rank')} {cat.get('category')} · score {cat.get('opportunity_score')} · "
            f"blocked {cat.get('blocked_mentions')}"
            if cat
            else "—"
        ),
        f"- **Corpus:** {corpus.get('total_reviews', '—')} rows · "
        f"exploration-relevant {corpus.get('exploration_relevant', '—')}",
        "",
        "## Intervention",
        "",
        exp.get("intervention") or "_Define intervention in synthesis experiments._",
        "",
        "## Success metrics",
        "",
        f"- **Primary:** {exp.get('primary_metric') or '—'}",
        f"- **Guardrail:** {exp.get('guardrail') or '—'}",
        f"- **Sample / runtime note:** {exp.get('sample_size_note') or '—'}",
        "",
        "## Decision ask",
        "",
        "Approve a 2-week instrumented A/B (or city holdout) against grocery-only cohorts, "
        "with the guardrail above as a hard stop.",
        "",
        "---",
        "",
        "_Source: `output/synthesis.json` · not a production ship brief._",
        "",
    ]
    return "\n".join(lines)


def write_experiment_brief(
    synthesis: dict[str, Any] | None = None,
    *,
    experiment_id: str,
    category: str | None = None,
    mvp_title: str | None = None,
    prototype: str | None = None,
    out_dir: Path | None = None,
) -> Path:
    if synthesis is None:
        path = DEFAULT_SYNTHESIS
        if not path.exists():
            raise FileNotFoundError(f"Missing synthesis: {path}")
        synthesis = json.loads(path.read_text(encoding="utf-8"))
    md = build_experiment_brief(
        synthesis,
        experiment_id=experiment_id,
        category=category,
        mvp_title=mvp_title,
        prototype=prototype,
    )
    target_dir = out_dir or DEFAULT_OUT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    out = target_dir / f"{experiment_id}_brief.md"
    out.write_text(md, encoding="utf-8")
    return out


def brief_from_mvp_preset(preset_key: str, synthesis: dict[str, Any] | None = None) -> tuple[str, Path]:
    preset = MVP_PRESETS.get(preset_key)
    if not preset:
        raise KeyError(f"Unknown MVP preset: {preset_key}")
    path = write_experiment_brief(
        synthesis,
        experiment_id=preset["experiment_id"],
        category=preset.get("category"),
        mvp_title=preset.get("mvp_title"),
        prototype=preset.get("prototype"),
    )
    return path.read_text(encoding="utf-8"), path
