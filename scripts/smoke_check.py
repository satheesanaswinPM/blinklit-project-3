"""Quick regression checks for discovery engine core paths."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        failures.append(msg)
        print("FAIL:", msg)
    else:
        print("PASS:", msg)


def main() -> int:
    # Artifacts exist
    for rel in [
        "output/synthesis.json",
        "output/exploration_tags.csv",
        "output/themes.csv",
        "output/sentiment.csv",
        "output/user_segments.csv",
        "data/processed/merged_reviews.csv",
    ]:
        check((ROOT / rel).exists(), f"exists {rel}")

    syn = json.loads((ROOT / "output/synthesis.json").read_text(encoding="utf-8"))
    check(bool(syn.get("category_opportunities")), "synthesis has category opportunities")
    check(bool(syn.get("testable_experiments")), "synthesis has experiments")
    check("Why don't" in syn.get("primary_question", ""), "primary question set")

    import pandas as pd
    from analysis.exploration import tag_review
    import app

    exp = pd.read_csv(ROOT / "output/exploration_tags.csv")
    check("exploration_signal" in exp.columns, "exploration has signal col")
    check(int(exp["is_relevant"].sum()) > 0, "some relevant tags")

    r = tag_review("I always reorder the same milk and bread")
    check(r["exploration_signal"] == "stuck_in_routine", "tagger routine signal")
    r2 = tag_review("Great app!")
    check(r2["is_relevant"] is False, "tagger noise filter")

    check("Prototype Lab" in app.PAGES, "Prototype Lab in nav")
    ht = app.build_hypothesis_triangulation(exp, syn)
    check(len(ht) >= 4, "hypothesis triangulation rows")

    # Unique plotly helper accepts key
    import inspect
    check("key" in inspect.signature(app._show_chart).parameters, "_show_chart supports key")

    # Corpus vs analysis size drift warning (not hard fail)
    merged = pd.read_csv(ROOT / "data/processed/merged_reviews.csv")
    sent = pd.read_csv(ROOT / "output/sentiment.csv")
    if len(merged) != len(sent):
        print(
            f"WARN: merged ({len(merged)}) != sentiment ({len(sent)}) — dashboard KPIs may disagree"
        )

    # Category opportunity experiment link smell
    snacks = next((c for c in syn["category_opportunities"] if c.get("category") == "snacks"), None)
    if snacks and snacks.get("suggested_experiment") == "exp_first_buy_guarantee":
        print(
            "WARN: snacks opportunity suggests first_buy_guarantee; discover_rail is a better MVP link"
        )

    if failures:
        print(f"\n{len(failures)} failure(s)")
        return 1
    print("\nAll critical checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
