#!/usr/bin/env python3
"""Fetch the pinned official BABILong 8k qa1-qa5 evaluation files."""

import urllib.request
from pathlib import Path

PINNED_REVISION = "ee0d588794c7ac098062ee0d247c733d62e94fe2"
TASKS = ("qa1", "qa2", "qa3", "qa4", "qa5")
OUT_DIR = Path("data/babilong/8k")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        url = f"https://huggingface.co/datasets/RMT-team/babilong/resolve/{PINNED_REVISION}/data/{task}/8k.json"
        out_path = OUT_DIR / f"{task}.json"
        with urllib.request.urlopen(url) as response:
            out_path.write_bytes(response.read())
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
