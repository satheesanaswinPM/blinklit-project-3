"""Export a one-page experiment brief (Markdown) from synthesis + MVP choice."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.experiment_brief import (  # noqa: E402
    MVP_PRESETS,
    brief_from_mvp_preset,
    write_experiment_brief,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export one-page experiment brief Markdown")
    parser.add_argument(
        "--synthesis",
        type=Path,
        default=ROOT / "output" / "synthesis.json",
    )
    parser.add_argument(
        "--mvp",
        choices=sorted(MVP_PRESETS.keys()),
        default=None,
        help="MVP preset (snacks_rail | home_guarantee)",
    )
    parser.add_argument("--experiment-id", default=None, help="Or pass experiment id directly")
    parser.add_argument("--category", default=None)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "output" / "briefs",
    )
    args = parser.parse_args()

    if not args.synthesis.exists():
        print(f"Missing synthesis: {args.synthesis}", file=sys.stderr)
        return 1

    synthesis = json.loads(args.synthesis.read_text(encoding="utf-8"))

    if args.mvp:
        md, path = brief_from_mvp_preset(args.mvp, synthesis)
        # rewrite into requested out_dir if different
        if path.parent.resolve() != args.out_dir.resolve():
            args.out_dir.mkdir(parents=True, exist_ok=True)
            path = args.out_dir / path.name
            path.write_text(md, encoding="utf-8")
    else:
        exp_id = args.experiment_id or "exp_discover_rail"
        path = write_experiment_brief(
            synthesis,
            experiment_id=exp_id,
            category=args.category,
            out_dir=args.out_dir,
        )
        md = path.read_text(encoding="utf-8")

    print(f"Wrote {path}")
    print(md[:400] + ("…" if len(md) > 400 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
