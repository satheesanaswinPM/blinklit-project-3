"""Load repo-root `.env` into process environment."""

from __future__ import annotations

import os
from pathlib import Path

from discovery_engine.config import ENV_PATH


def load_env() -> Path | None:
    """
    Load `.env` from the repository root.

    Existing OS environment variables take precedence (`override=False`).
    Values are stripped so `KEY= value` in `.env` still works.
    Syncs `HF_TOKEN` <-> `HUGGING_FACE_HUB_TOKEN` when only one is set.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return None

    if not ENV_PATH.exists():
        return None

    load_dotenv(ENV_PATH, override=False)

    # Normalize whitespace around values commonly pasted with "KEY= value"
    for key in list(os.environ.keys()):
        if key.startswith(("HF_", "HUGGING_", "OPENAI_", "REDDIT_", "PHASE1_")):
            os.environ[key] = os.environ[key].strip()

    hf = os.getenv("HF_TOKEN", "").strip()
    hub = os.getenv("HUGGING_FACE_HUB_TOKEN", "").strip()
    if hf and not hub:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf
    if hub and not hf:
        os.environ["HF_TOKEN"] = hub
    return ENV_PATH
