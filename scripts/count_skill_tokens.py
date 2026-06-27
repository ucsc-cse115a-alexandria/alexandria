#!/usr/bin/env python3
"""For each repo, count tiktoken tokens of every SKILL.md and save the per-repo average.

Reads the JSON produced by search_skill_repos.py and requires the authenticated `gh` CLI.
"""

import argparse
import base64
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import tiktoken
from pydantic import BaseModel

GH = shutil.which("gh") or "gh"


class RepoTokens(BaseModel):
    """Per-repo SKILL.md token statistics."""

    full_name: str
    skill_md_count: int
    total_tokens: int
    avg_tokens: float


def skill_md_blobs(full_name: str) -> list[str]:
    """Return the git blob SHAs of every SKILL.md in the repo's tree."""
    result = subprocess.run(
        [
            GH,
            "api",
            f"repos/{full_name}/git/trees/HEAD?recursive=1",
            "--jq",
            '.tree[] | select(.path | endswith("SKILL.md")) | .sha',
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.split()


def blob_token_count(full_name: str, sha: str, encoding: tiktoken.Encoding) -> int:
    """Return the token count of a single git blob."""
    result = subprocess.run(
        [GH, "api", f"repos/{full_name}/git/blobs/{sha}", "--jq", ".content"],
        capture_output=True,
        text=True,
        check=True,
    )
    text = base64.b64decode(result.stdout).decode("utf-8", errors="replace")
    return len(encoding.encode(text))


def count_repo(full_name: str, encoding: tiktoken.Encoding) -> RepoTokens:
    """Count tokens for every SKILL.md in a repo and return per-repo statistics."""
    shas = skill_md_blobs(full_name)
    counts = [blob_token_count(full_name, sha, encoding) for sha in shas]
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
    parser.add_argument("--workers", type=int, default=16, help="concurrent repos")
    args = parser.parse_args()

    repos = [item["full_name"] for item in json.loads(args.input.read_text(encoding="utf-8"))]
    encoding = tiktoken.get_encoding(args.encoding)

    def safe_count(full_name: str) -> RepoTokens | None:
        try:
            return count_repo(full_name, encoding)
        except subprocess.CalledProcessError as error:
            print(f"warning: skipping {full_name}: {error.stderr.strip()}", file=sys.stderr)
            return None

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = [result for result in pool.map(safe_count, repos) if result is not None]

    payload = [result.model_dump() for result in results]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Counted SKILL.md tokens for {len(results)} repos -> {args.output}")


if __name__ == "__main__":
    main()
