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
    for rel in [
        "output/synthesis.json",
        "output/exploration_tags.csv",
        "output/themes.csv",
        "output/sentiment.csv",
        "output/user_segments.csv",
        "data/processed/merged_reviews.csv",
        "data/processed/preprocessed_reviews.csv",
    ]:
        check((ROOT / rel).exists(), f"exists {rel}")

    syn = json.loads((ROOT / "output/synthesis.json").read_text(encoding="utf-8"))
    check(bool(syn.get("category_opportunities")), "synthesis has category opportunities")
    check(bool(syn.get("testable_experiments")), "synthesis has experiments")
    check("Why don't" in syn.get("primary_question", ""), "primary question set")

    import pandas as pd
    from analysis.exploration import tag_review
    from analysis.themes import filter_low_information_themes, is_low_information_theme
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

    import inspect

    check("key" in inspect.signature(app._show_chart).parameters, "_show_chart supports key")

    cleaned = pd.read_csv(ROOT / "data/processed/preprocessed_reviews.csv")
    sent = pd.read_csv(ROOT / "output/sentiment.csv")
    segs = pd.read_csv(ROOT / "output/user_segments.csv")
    check(len(cleaned) == len(sent), f"preprocessed ({len(cleaned)}) == sentiment ({len(sent)})")
    check(len(cleaned) == len(segs), f"preprocessed ({len(cleaned)}) == segments ({len(segs)})")
    check(len(cleaned) == len(exp), f"preprocessed ({len(cleaned)}) == exploration ({len(exp)})")

    snacks = next((c for c in syn["category_opportunities"] if c.get("category") == "snacks"), None)
    check(snacks is not None, "snacks category opportunity present")
    if snacks:
        check(
            snacks.get("suggested_experiment") == "exp_discover_rail",
            "snacks → exp_discover_rail",
        )
    home = next((c for c in syn["category_opportunities"] if c.get("category") == "home"), None)
    if home:
        check(
            home.get("suggested_experiment") == "exp_first_buy_guarantee",
            "home → exp_first_buy_guarantee",
        )

    h1 = next((h for h in syn.get("hypotheses", []) if h.get("id") == "H1"), None)
    check(h1 is not None, "H1 hypothesis present")
    if h1:
        check(int(h1.get("evidence_mentions") or 0) > 0, "H1 evidence_mentions > 0")

    themes = pd.read_csv(ROOT / "output/themes.csv")
    noisy = themes.apply(is_low_information_theme, axis=1).sum()
    check(int(noisy) == 0, f"no low-info themes in themes.csv (found {int(noisy)})")
    filtered = filter_low_information_themes(themes)
    check(len(filtered) == len(themes), "themes already filtered")

    trust = app.build_product_rating_trust(
        keywords=["lays", "chips"],
        fallback_comments=[{"rating": 5, "text": "Demo only comment for smoke test.", "source": "demo"}],
        min_matches=9999,
    )
    check(trust["used_fallback"] is True, "trust panel falls back when matches thin")
    check(trust["top_comments"][0].get("evidence_label", "").startswith("Demo"), "demo evidence label")

    if failures:
        print(f"\n{len(failures)} failure(s)")
        return 1
    print("\nAll critical checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
