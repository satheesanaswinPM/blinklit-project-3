"""
Classify review sentiment with HuggingFace transformers.

Uses a 3-class sentiment pipeline (Positive / Neutral / Negative).

Output columns:
  Review
  Sentiment
  Confidence Score

Saved to:
  output/sentiment.csv

Usage (from repo root):
  python -m analysis.sentiment
  python -m analysis.sentiment --in data/raw/blinkit_play_reviews.csv --text-col Review
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from discovery_engine.env_loader import load_env  # noqa: E402
from discovery_engine.nlp.bertopic_cluster import load_reviews_from_csv  # noqa: E402

load_env()

DEFAULT_IN = ROOT / "data" / "raw" / "blinkit_play_reviews.csv"
DEFAULT_OUT = ROOT / "output" / "sentiment.csv"
LOCAL_MODEL = ROOT / "models" / "twitter-roberta-base-sentiment-latest"

# 3-way sentiment (neg / neu / pos). Override with --model if needed.
DEFAULT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
FALLBACK_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
# Used when huggingface.co is unreachable (common on some networks).
HF_MIRROR = "https://hf-mirror.com"

_LABEL_MAP = {
    "negative": "Negative",
    "neutral": "Neutral",
    "positive": "Positive",
    "label_0": "Negative",
    "label_1": "Neutral",
    "label_2": "Positive",
    "neg": "Negative",
    "neu": "Neutral",
    "pos": "Positive",
}


def _ensure_hf_hub_reachable() -> None:
    """Prefer huggingface.co; fall back to HF_ENDPOINT / public mirror if needed."""
    if os.getenv("HF_ENDPOINT"):
        return
    try:
        import urllib.request

        urllib.request.urlopen("https://huggingface.co", timeout=8)
        return
    except Exception:
        pass
    os.environ["HF_ENDPOINT"] = HF_MIRROR
    print(f"huggingface.co unreachable; using HF_ENDPOINT={HF_MIRROR}")


def _normalize_label(raw: str) -> str:
    key = (raw or "").strip().lower()
    if key in _LABEL_MAP:
        return _LABEL_MAP[key]
    if "neg" in key:
        return "Negative"
    if "pos" in key:
        return "Positive"
    if "neu" in key:
        return "Neutral"
    return raw.capitalize() if raw else "Neutral"


def _map_binary_to_ternary(label: str, score: float, neutral_band: float = 0.65) -> str:
    """Map SST-2 style POS/NEG to 3-class using a confidence band for Neutral."""
    base = _normalize_label(label)
    if score < neutral_band:
        return "Neutral"
    return base if base in {"Positive", "Negative"} else "Neutral"


def _resolve_model(model_name: str) -> str:
    """Prefer a local checkout when the default Hub id is requested."""
    if model_name == DEFAULT_MODEL and (LOCAL_MODEL / "config.json").exists():
        weights = LOCAL_MODEL / "pytorch_model.bin"
        safetensors = LOCAL_MODEL / "model.safetensors"
        if weights.exists() or safetensors.exists():
            return str(LOCAL_MODEL)
    path = Path(model_name)
    if path.exists():
        return str(path)
    return model_name


def build_pipeline(model_name: str):
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        pipeline,
    )

    resolved = _resolve_model(model_name)
    local_only = Path(resolved).is_dir()
    if not local_only:
        _ensure_hf_hub_reachable()

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or None
    load_kwargs: dict = {"local_files_only": True} if local_only else {}
    if token and not local_only:
        load_kwargs["token"] = token

    tokenizer = AutoTokenizer.from_pretrained(resolved, **load_kwargs)
    model = AutoModelForSequenceClassification.from_pretrained(resolved, **load_kwargs)
    clf = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        truncation=True,
        max_length=512,
    )
    return clf, resolved


def _try_build_pipeline(model_name: str):
    try:
        clf, resolved = build_pipeline(model_name)
        return clf, resolved, False
    except Exception as e:
        print(f"Could not load {model_name}: {e}")
        print(f"Falling back to {FALLBACK_MODEL} (binary + Neutral confidence band)")
        clf, resolved = build_pipeline(FALLBACK_MODEL)
        return clf, resolved, True


def classify_reviews(
    reviews: list[str],
    *,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 16,
    neutral_band: float = 0.65,
) -> pd.DataFrame:
    """Run HF sentiment pipeline and return Review / Sentiment / Confidence Score."""
    docs = [str(r).strip() for r in reviews if str(r).strip()]
    if not docs:
        return pd.DataFrame(columns=["Review", "Sentiment", "Confidence Score"])

    print(f"Loading sentiment model: {model_name}")
    clf, used_model, binary_fallback = _try_build_pipeline(model_name)
    if used_model != model_name:
        print(f"Using model: {used_model}")

    sentiments: list[str] = []
    scores: list[float] = []

    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        outputs = clf(batch)
        for out in outputs:
            if isinstance(out, list):
                best = max(out, key=lambda x: float(x.get("score", 0.0)))
            else:
                best = out
            score = float(best.get("score", 0.0))
            raw_label = str(best.get("label", ""))
            if binary_fallback:
                sentiment = _map_binary_to_ternary(raw_label, score, neutral_band=neutral_band)
            else:
                sentiment = _normalize_label(raw_label)
            sentiments.append(sentiment)
            scores.append(round(score, 6))

        done = min(i + batch_size, len(docs))
        print(f"  classified {done}/{len(docs)}")

    return pd.DataFrame(
        {
            "Review": docs,
            "Sentiment": sentiments,
            "Confidence Score": scores,
        }
    )


def save_sentiment(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df[["Review", "Sentiment", "Confidence Score"]].to_csv(
        path, index=False, encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="HF sentiment analysis -> output/sentiment.csv")
    parser.add_argument("--in", dest="inp", type=Path, default=DEFAULT_IN)
    parser.add_argument("--text-col", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HuggingFace model id or local path")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not args.inp.exists():
        print(f"Input not found: {args.inp}", file=sys.stderr)
        return 1

    reviews = load_reviews_from_csv(str(args.inp), text_col=args.text_col)
    print(f"Loaded {len(reviews)} rows from {args.inp}")

    df = classify_reviews(reviews, model_name=args.model, batch_size=args.batch_size)
    save_sentiment(df, args.out)

    counts = df["Sentiment"].value_counts().to_dict() if len(df) else {}
    print(f"\nSentiment counts: {counts}")
    print(f"Saved -> {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
