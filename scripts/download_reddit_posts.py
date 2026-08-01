"""
Collect Reddit posts via PRAW from r/Blinkit, r/india, and r/grocery.

Saves CSV columns: Title, Body, Score, Comments
(also Subreddit, URL, Created for provenance)

Credentials in repo `.env`:
  REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT

Usage (from repo root):
  python -m scripts.download_reddit_posts
  python -m scripts.download_reddit_posts --limit 50 --sort hot
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import praw
from praw.models import MoreComments

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from discovery_engine.env_loader import load_env  # noqa: E402

load_env()

DEFAULT_OUT = REPO_ROOT / "data" / "raw" / "reddit_posts.csv"
DEFAULT_SUBREDDITS = ("Blinkit", "india", "grocery")


def make_reddit() -> praw.Reddit:
    """Build a read-only PRAW client from `.env` credentials.

    Raises:
        RuntimeError: when REDDIT_CLIENT_ID or REDDIT_CLIENT_SECRET is missing.
    """
    client_id = os.getenv("REDDIT_CLIENT_ID", "").strip()
    client_secret = os.getenv("REDDIT_CLIENT_SECRET", "").strip()
    user_agent = os.getenv(
        "REDDIT_USER_AGENT",
        "blinkit-discovery-engine/0.1 by script",
    ).strip()

    if not client_id or not client_secret:
        raise RuntimeError(
            "Missing Reddit API credentials in .env: "
            "REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET "
            "(optional REDDIT_USER_AGENT). "
            "Create a script app at https://www.reddit.com/prefs/apps"
        )

    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )


def collect_comments(submission: praw.models.Submission, max_comments: int = 50) -> str:
    submission.comments.replace_more(limit=0)
    parts: list[str] = []
    for comment in submission.comments.list():
        if isinstance(comment, MoreComments):
            continue
        body = (comment.body or "").strip()
        if not body or body in {"[deleted]", "[removed]"}:
            continue
        parts.append(body.replace("\n", " ").strip())
        if len(parts) >= max_comments:
            break
    return " | ".join(parts)


def iter_submissions(subreddit: praw.models.Subreddit, sort: str, limit: int):
    sort = sort.lower()
    if sort == "new":
        return subreddit.new(limit=limit)
    if sort == "top":
        return subreddit.top(time_filter="month", limit=limit)
    if sort == "rising":
        return subreddit.rising(limit=limit)
    return subreddit.hot(limit=limit)


def fetch_posts(
    reddit: praw.Reddit,
    subreddits: list[str],
    *,
    limit_per_sub: int,
    sort: str,
    max_comments: int,
    query: str | None = None,
) -> list[dict]:
    rows: list[dict] = []
    seen_ids: set[str] = set()

    for name in subreddits:
        try:
            sub = reddit.subreddit(name)
            _ = sub.display_name
        except Exception as e:  # noqa: BLE001
            print(f"Skipping r/{name}: {e}", file=sys.stderr)
            continue

        print(f"Fetching r/{name} ({sort}, limit={limit_per_sub})...")
        try:
            if query:
                submissions = sub.search(
                    query,
                    sort=sort if sort != "rising" else "relevance",
                    limit=limit_per_sub,
                )
            else:
                submissions = iter_submissions(sub, sort, limit_per_sub)

            for submission in submissions:
                if submission.id in seen_ids:
                    continue
                seen_ids.add(submission.id)
                created = datetime.fromtimestamp(
                    submission.created_utc, tz=timezone.utc
                ).strftime("%Y-%m-%d %H:%M:%S UTC")
                rows.append(
                    {
                        "Title": (submission.title or "").strip(),
                        "Body": (submission.selftext or "").strip(),
                        "Score": submission.score,
                        "Comments": collect_comments(submission, max_comments=max_comments),
                        "Subreddit": str(submission.subreddit),
                        "URL": f"https://www.reddit.com{submission.permalink}",
                        "Created": created,
                    }
                )
        except Exception as e:  # noqa: BLE001
            print(f"Error while reading r/{name}: {e}", file=sys.stderr)

    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["Title", "Body", "Score", "Comments", "Subreddit", "URL", "Created"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Reddit posts for discovery insight pipeline")
    parser.add_argument("--subreddits", nargs="+", default=list(DEFAULT_SUBREDDITS))
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--sort", choices=["hot", "new", "top", "rising"], default="hot")
    parser.add_argument("--max-comments", type=int, default=30)
    parser.add_argument("--query", default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    try:
        reddit = make_reddit()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc

    reddit.read_only = True
    rows = fetch_posts(
        reddit,
        args.subreddits,
        limit_per_sub=args.limit,
        sort=args.sort,
        max_comments=args.max_comments,
        query=args.query,
    )
    write_csv(rows, args.out)
    print(f"Saved {len(rows)} posts -> {args.out.resolve()}")


if __name__ == "__main__":
    main()
