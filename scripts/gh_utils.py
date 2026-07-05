"""Shared helpers for driving the GitHub `gh` CLI with rate-limit handling."""

import shutil
import subprocess
import time

GH = shutil.which("gh") or "gh"

MAX_RETRIES = 6
BACKOFF_BASE_SECONDS = 2.0
BACKOFF_CAP_SECONDS = 60.0


def _is_rate_limited(stderr: str) -> bool:
    """Return True if `gh`'s stderr reports a primary or secondary rate limit."""
    lowered = stderr.lower()
    return "rate limit" in lowered or "http 403" in lowered or "http 429" in lowered


def run_gh(args: list[str]) -> str:
    """Run `gh` with `args`, retrying with exponential backoff while rate-limited.

    Returns stdout on success; raises RuntimeError once retries are exhausted or
    the command fails for a non-rate-limit reason, so failures are never silently
    treated as empty results.
    """
    for attempt in range(MAX_RETRIES + 1):
        result = subprocess.run(
            [GH, *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode == 0:
            return result.stdout
        if attempt == MAX_RETRIES or not _is_rate_limited(result.stderr):
            raise RuntimeError(f"`gh {' '.join(args)}` failed: {result.stderr.strip()}")
        delay = min(BACKOFF_BASE_SECONDS * 2**attempt, BACKOFF_CAP_SECONDS)
        time.sleep(delay)
    raise RuntimeError("unreachable")
