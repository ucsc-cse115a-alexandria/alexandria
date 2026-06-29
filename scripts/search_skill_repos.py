#!/usr/bin/env python3
"""List GitHub repos matching a query, sorted by stars, that contain a root SKILL.md.

Requires the authenticated `gh` CLI on PATH.
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from gh_utils import run_gh
from pydantic import BaseModel


class Repo(BaseModel):
    """A GitHub repository from `gh search repos`."""

    full_name: str
    stars: int
    url: str
    description: str = ""


def search_repos(query: str, candidates: int) -> list[Repo]:
    """Return repos matching `query`, sorted by descending star count."""
    stdout = run_gh(
        [
            "search",
            "repos",
            query,
            "--sort",
            "stars",
            "--order",
            "desc",
            "--limit",
            str(candidates),
            "--json",
            "fullName,stargazersCount,description,url",
        ]
    )
    return [
        Repo(
            full_name=item["fullName"],
            stars=item["stargazersCount"],
            url=item["url"],
            description=item.get("description") or "",
        )
        for item in json.loads(stdout)
    ]


def has_skill_md(full_name: str) -> bool:
    """Return True if the repo contains a SKILL.md anywhere in its tree."""
    stdout = run_gh(
        [
            "api",
            f"repos/{full_name}/git/trees/HEAD?recursive=1",
            "--jq",
            'any(.tree[]?; .path | endswith("SKILL.md"))',
        ]
    )
    return stdout.strip() == "true"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="agent skills", help="GitHub search query")
    parser.add_argument("--limit", type=int, default=100, help="number of matches to list")
    parser.add_argument(
        "--candidates",
        type=int,
        default=150,
        help="star-sorted pool to scan for SKILL.md (max 1000)",
    )
    parser.add_argument("--workers", type=int, default=4, help="concurrent SKILL.md checks")
    parser.add_argument("--output", type=Path, default=Path("data/skill_repos.json"), help="output JSON file")
    args = parser.parse_args()

    repos = search_repos(args.query, args.candidates)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        flags = pool.map(has_skill_md, [repo.full_name for repo in repos])
    matches = [repo for repo, has_skill in zip(repos, flags, strict=True) if has_skill][: args.limit]

    payload = [repo.model_dump() for repo in matches]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(matches)} repos to {args.output}")


if __name__ == "__main__":
    main()
