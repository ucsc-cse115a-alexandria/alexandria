#!/usr/bin/env python3
"""Download AI-agent skill input files from repos listed in data/skill_repos.json.

Requires:
- authenticated GitHub CLI: gh auth login
- existing data/skill_repos.json from search_skill_repos.py
"""

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

GH = "gh"


def run_gh(args: list[str]) -> str:
    """Run gh and return stdout as UTF-8 text."""
    result = subprocess.run(
        [GH, *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if result.returncode != 0:
        raise RuntimeError(f"gh command failed:\ngh {' '.join(args)}\n\nSTDERR:\n{result.stderr}")

    return result.stdout


def safe_repo_dir(full_name: str) -> str:
    """Convert owner/repo into owner__repo for local storage."""
    return full_name.replace("/", "__")


def get_repo_tree(full_name: str) -> list[dict]:
    """Return recursive file tree for a GitHub repository."""
    stdout = run_gh(
        [
            "api",
            f"repos/{full_name}/git/trees/HEAD?recursive=1",
        ]
    )
    payload = json.loads(stdout)
    return payload.get("tree", [])


def is_agent_input_file(path: str) -> bool:
    """Return True for files intended as AI-agent skill inputs.

    This assignment primarily wants SKILL.md files.
    Add more filenames here only if your instructor wants them.
    """
    filename = Path(path).name
    return filename == "SKILL.md"


def download_file(full_name: str, source_path: str) -> str:
    return run_gh(
        [
            "api",
            "-H",
            "Accept: application/vnd.github.raw",
            f"repos/{full_name}/contents/{source_path}",
        ]
    )


def write_skill_file(
    output_dir: Path,
    full_name: str,
    source_path: str,
    content: str,
) -> Path:
    """Write downloaded skill file while preserving repo-relative path."""
    local_path = output_dir / safe_repo_dir(full_name) / source_path
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_text(content, encoding="utf-8")
    return local_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repos",
        type=Path,
        default=Path("data/skill_repos.json"),
        help="input JSON file from search_skill_repos.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("corpus"),
        help="directory where downloaded files will be stored",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/download_manifest.json"),
        help="output manifest JSON file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="maximum number of repositories to process",
    )
    parser.add_argument(
        "--max-files-per-repo",
        type=int,
        default=100,
        help="maximum number of SKILL.md files to download from each repository",
    )
    args = parser.parse_args()

    repos = json.loads(args.repos.read_text(encoding="utf-8"))[: args.limit]

    manifest = []
    errors = []

    for index, repo in enumerate(repos, start=1):
        full_name = repo["full_name"]
        print(f"[{index}/{len(repos)}] Scanning {full_name}")

        try:
            tree = get_repo_tree(full_name)
            all_skill_paths = [
                item["path"]
                for item in tree
                if item.get("type") == "blob" and is_agent_input_file(item.get("path", ""))
            ]

            skill_paths = all_skill_paths[: args.max_files_per_repo]

            if len(all_skill_paths) > args.max_files_per_repo:
                print(f"  Found {len(all_skill_paths)} SKILL.md files; downloading first {args.max_files_per_repo}")
            else:
                print(f"  Found {len(skill_paths)} SKILL.md file(s)")

            for source_path in skill_paths:
                try:
                    content = download_file(full_name, source_path)
                    local_path = write_skill_file(
                        args.output,
                        full_name,
                        source_path,
                        content,
                    )

                    manifest.append(
                        {
                            "repo": full_name,
                            "repo_url": repo.get("url", ""),
                            "stars": repo.get("stars"),
                            "description": repo.get("description", ""),
                            "source_path": source_path,
                            "local_path": str(local_path).replace("\\", "/"),
                            "downloaded_at": datetime.now(UTC).isoformat(),
                            "repo_skill_file_count": len(all_skill_paths),
                            "downloaded_from_repo": len(skill_paths),
                            "truncated": len(all_skill_paths) > args.max_files_per_repo,
                        }
                    )

                except Exception as exc:
                    errors.append(
                        {
                            "repo": full_name,
                            "source_path": source_path,
                            "error": str(exc),
                        }
                    )
                    print(f"  ERROR downloading {source_path}: {exc}")

        except Exception as exc:
            errors.append(
                {
                    "repo": full_name,
                    "error": str(exc),
                }
            )
            print(f"  ERROR scanning repo: {exc}")

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(
            {
                "downloaded_files": manifest,
                "errors": errors,
                "total_downloaded": len(manifest),
                "total_errors": len(errors),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(f"Downloaded {len(manifest)} file(s)")
    print(f"Encountered {len(errors)} error(s)")
    print(f"Wrote manifest to {args.manifest}")


if __name__ == "__main__":
    main()
