#!/usr/bin/env python3
"""Render a self-contained HTML review page for Alexandria proposed edits.

Reads a prompt file, runs propose(), and writes a static HTML page that can be
opened directly in a browser (no server required).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from alexandria.cli.review_html import render_review_page
from alexandria.ir.contracts import Params
from alexandria.ops import propose
from alexandria.utils.embedders import HashEmbedder, default_embedder
from alexandria.utils.merger import default_merger


class _FirstWinsMerger:
    def merge(self, first: str, second: str, feedback: str | None = None) -> str:
        del second, feedback
        return first  # identical to the first sentence, so the optimizer emits plain deletes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", type=Path, help="prompt file to optimize and review")
    parser.add_argument("-o", "--output", type=Path, default=Path("review.html"), help="output HTML path")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="use offline hash embedding and a first-wins merger instead of the OpenAI defaults",
    )
    args = parser.parse_args()

    prompt = args.prompt.read_text(encoding="utf-8")
    if args.deterministic:
        embedder, merger = HashEmbedder(), _FirstWinsMerger()
        proposal = propose(prompt, embedder, merger, params=Params(drift_budget=2.0))
    else:
        embedder, merger = default_embedder(), default_merger()
        proposal = propose(prompt, embedder, merger)
    html = render_review_page(proposal)
    args.output.write_text(html, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
