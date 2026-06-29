#!/usr/bin/env python3
"""For each repo, count tiktoken tokens of every SKILL.md and save the per-repo average.

Reads the JSON produced by search_skill_repos.py and requires the authenticated `gh` CLI.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import tiktoken
from gh_utils import run_gh
from pydantic import BaseModel

RAW_HOST = "https://raw.githubusercontent.com"

RAW_MAX_RETRIES = 3
RAW_BACKOFF_SECONDS = 1.0


def fetch_raw_text(url: str) -> str:
    """Download `url`, retrying with backoff on transient connection errors."""
    for attempt in range(RAW_MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(url) as response:
                return response.read().decode("utf-8", errors="replace")
        except urllib.error.URLError:
            if attempt == RAW_MAX_RETRIES:
                raise
            time.sleep(RAW_BACKOFF_SECONDS * 2**attempt)
    raise RuntimeError("unreachable")


class RepoTokens(BaseModel):
    """Per-repo SKILL.md token statistics."""

    full_name: str
    skill_md_count: int
    total_tokens: int
    avg_tokens: float


def skill_md_paths(full_name: str) -> list[str]:
    """Return the paths of every SKILL.md in the repo's tree (one API call)."""
    stdout = run_gh(
        [
            "api",
            f"repos/{full_name}/git/trees/HEAD?recursive=1",
            "--jq",
            '.tree[] | select(.path | endswith("SKILL.md")) | .path',
        ]
    )
    return [line for line in stdout.splitlines() if line]


def file_token_count(full_name: str, path: str, encoding: tiktoken.Encoding) -> int:
    """Return the token count of one SKILL.md fetched from raw.githubusercontent.com.

    Raw downloads are not charged against GitHub's REST API rate limit, so the
    only API call per repo is the single tree lookup in `skill_md_paths`.
    """
    url = f"{RAW_HOST}/{full_name}/HEAD/{urllib.parse.quote(path)}"
    text = fetch_raw_text(url)
    return len(encoding.encode(text))


def count_repo(full_name: str, encoding: tiktoken.Encoding) -> RepoTokens:
    """Count tokens for every SKILL.md in a repo and return per-repo statistics."""
    paths = skill_md_paths(full_name)
    counts = [file_token_count(full_name, path, encoding) for path in paths]
    total = sum(counts)
    return RepoTokens(
        full_name=full_name,
        skill_md_count=len(counts),
        total_tokens=total,
        avg_tokens=total / len(counts) if counts else 0.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/skill_repos.json"), help="repo list JSON")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/skill_token_counts.json"),
        help="output JSON file",
    )
    parser.add_argument("--encoding", default="cl100k_base", help="tiktoken encoding name")
    parser.add_argument("--workers", type=int, default=8, help="concurrent repos")
    args = parser.parse_args()

    repos = [item["full_name"] for item in json.loads(args.input.read_text(encoding="utf-8"))]
    encoding = tiktoken.get_encoding(args.encoding)

    def safe_count(full_name: str) -> RepoTokens | None:
        try:
            return count_repo(full_name, encoding)
        except (RuntimeError, urllib.error.URLError) as error:
            print(f"warning: skipping {full_name}: {error}", file=sys.stderr)
            return None

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = [result for result in pool.map(safe_count, repos) if result is not None]

    payload = [result.model_dump() for result in results]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Counted SKILL.md tokens for {len(results)} repos -> {args.output}")


if __name__ == "__main__":
    main()
