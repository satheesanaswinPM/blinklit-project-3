#!/usr/bin/env python3
"""
Auto-commit and push local changes to origin.

Triggered by Cursor hooks (stop / afterFileEdit / afterTabFileEdit).
Debounces rapid edits so a burst of writes becomes one push.

Safety:
  - Never force-pushes
  - Respects .gitignore (will not add .env / models / etc.)
  - Skips when there is nothing to commit and nothing to push
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Debounce window (seconds) between auto-pushes from edit hooks
DEBOUNCE_SECONDS = 45
STAMP_PATH = Path(".cursor") / "hooks" / ".auto_push_stamp"
REMOTE = "origin"
BRANCH = None  # resolved at runtime


def _repo_root() -> Path:
    # Hook cwd is the project root for project hooks
    return Path.cwd()


def _run(cmd: list[str], *, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=_repo_root(),
        text=True,
        capture_output=True,
        check=check,
        encoding="utf-8",
        errors="replace",
    )


def _git_env() -> dict[str, str]:
    """Ensure commit author exists without writing git config."""
    env = os.environ.copy()
    if not env.get("GIT_AUTHOR_NAME"):
        # Prefer existing git identity; else GitHub noreply style
        name = _run(["git", "config", "user.name"]).stdout.strip()
        email = _run(["git", "config", "user.email"]).stdout.strip()
        if not name:
            name = "satheesanaswinPM"
        if not email:
            email = "satheesanaswinPM@users.noreply.github.com"
        env["GIT_AUTHOR_NAME"] = name
        env["GIT_AUTHOR_EMAIL"] = email
        env["GIT_COMMITTER_NAME"] = name
        env["GIT_COMMITTER_EMAIL"] = email
    return env


def _current_branch() -> str | None:
    r = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    if r.returncode != 0:
        return None
    name = r.stdout.strip()
    return None if not name or name == "HEAD" else name


def _has_remote() -> bool:
    r = _run(["git", "remote", "get-url", REMOTE])
    return r.returncode == 0 and bool(r.stdout.strip())


def _status_porcelain() -> str:
    return _run(["git", "status", "--porcelain"]).stdout


def _commits_ahead(branch: str) -> int:
    r = _run(["git", "rev-list", "--count", f"{REMOTE}/{branch}..HEAD"])
    if r.returncode != 0:
        return 0
    try:
        return int(r.stdout.strip() or "0")
    except ValueError:
        return 0


def _debounced() -> bool:
    """Return True if we should skip because of recent push."""
    try:
        if not STAMP_PATH.exists():
            return False
        age = time.time() - STAMP_PATH.stat().st_mtime
        return age < DEBOUNCE_SECONDS
    except OSError:
        return False


def _touch_stamp() -> None:
    STAMP_PATH.parent.mkdir(parents=True, exist_ok=True)
    STAMP_PATH.write_text(str(time.time()), encoding="utf-8")


def _commit_message() -> str:
    changed = _status_porcelain().strip().splitlines()
    n = len(changed)
    sample = ", ".join(
        Path(line[3:].strip().split(" -> ")[-1]).name for line in changed[:4]
    )
    if n > 4:
        sample += ", …"
    return f"chore: auto-sync {n} change(s) — {sample}" if sample else "chore: auto-sync"


def main() -> int:
    # Consume hook JSON from stdin (required protocol)
    try:
        raw = sys.stdin.read()
        if raw.strip():
            json.loads(raw)
    except Exception:
        pass

    if not (_repo_root() / ".git").exists():
        print("auto_push: not a git repo", file=sys.stderr)
        return 0

    if not _has_remote():
        print("auto_push: no origin remote", file=sys.stderr)
        return 0

    branch = _current_branch()
    if not branch:
        print("auto_push: detached HEAD; skip", file=sys.stderr)
        return 0

    if _debounced() and not _status_porcelain().strip():
        return 0

    dirty = _status_porcelain().strip()
    env = _git_env()

    if dirty:
        # Stage everything except ignored files (.env stays out via .gitignore)
        add = subprocess.run(
            ["git", "add", "-A"],
            cwd=_repo_root(),
            env=env,
            capture_output=True,
            text=True,
        )
        if add.returncode != 0:
            print(add.stderr, file=sys.stderr)
            return 0

        # Double-check we did not stage secrets
        staged = _run(["git", "diff", "--cached", "--name-only"]).stdout.splitlines()
        blocked = [p for p in staged if p.endswith(".env") or p == ".env" or "/.env" in p]
        if blocked:
            subprocess.run(["git", "reset", "HEAD", "--"] + blocked, cwd=_repo_root())

        if not _run(["git", "diff", "--cached", "--name-only"]).stdout.strip():
            print("auto_push: nothing staged after filtering")
        else:
            msg = _commit_message()
            commit = subprocess.run(
                ["git", "commit", "-m", msg],
                cwd=_repo_root(),
                env=env,
                capture_output=True,
                text=True,
            )
            if commit.returncode != 0:
                # Likely "nothing to commit" race
                print(commit.stdout or commit.stderr, file=sys.stderr)

    # Push if ahead of remote or we just committed
    ahead = _commits_ahead(branch)
    upstream = _run(["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"])
    need_push = ahead > 0 or upstream.returncode != 0

    if not need_push:
        return 0

    if _debounced() and ahead <= 0:
        return 0

    push = subprocess.run(
        ["git", "push", "-u", REMOTE, branch],
        cwd=_repo_root(),
        env=env,
        capture_output=True,
        text=True,
    )
    if push.returncode != 0 and "fetch first" in (push.stderr or "").lower() or (
        push.returncode != 0 and "rejected" in (push.stderr or "").lower()
    ):
        # Integrate remote commits, then retry a normal (non-force) push
        rebase = subprocess.run(
            ["git", "pull", "--rebase", REMOTE, branch],
            cwd=_repo_root(),
            env=env,
            capture_output=True,
            text=True,
        )
        if rebase.returncode != 0:
            print("auto_push: pull --rebase failed", file=sys.stderr)
            print(rebase.stderr or rebase.stdout, file=sys.stderr)
            subprocess.run(["git", "rebase", "--abort"], cwd=_repo_root(), capture_output=True)
            return 0
        push = subprocess.run(
            ["git", "push", "-u", REMOTE, branch],
            cwd=_repo_root(),
            env=env,
            capture_output=True,
            text=True,
        )

    if push.returncode != 0:
        print("auto_push: push failed", file=sys.stderr)
        print(push.stderr or push.stdout, file=sys.stderr)
        return 0

    _touch_stamp()
    print(f"auto_push: pushed to {REMOTE}/{branch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
