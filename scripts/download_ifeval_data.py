#!/usr/bin/env python3
"""Fetch the official IFEval dataset and the nltk tokenizer data the vendored checkers need."""

import urllib.request
from pathlib import Path

import nltk

PINNED_COMMIT = "5b09c22d73a9d35eb6c5d2a99b95677a45053466"
DATA_URL = (
    "https://raw.githubusercontent.com/google-research/google-research/"
    f"{PINNED_COMMIT}/instruction_following_eval/data/input_data.jsonl"
)
OUT_PATH = Path("data/ifeval/input_data.jsonl")


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(DATA_URL) as response:
        OUT_PATH.write_bytes(response.read())
    print(f"wrote {OUT_PATH} ({OUT_PATH.read_text().count(chr(10))} prompts)")
    # the vendored checkers load the punkt pickle, which nltk>=3.9 backs with punkt_tab data
    nltk.download("punkt")
    nltk.download("punkt_tab")


if __name__ == "__main__":
    main()
