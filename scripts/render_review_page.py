#!/usr/bin/env python3
"""Render a self-contained HTML review page for Alexandria proposed edits.

Reads a prompt file, runs propose(), and writes a static HTML page that can be
opened directly in a browser (no server required).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from alexandria.cli.review_html import render_review_page
from alexandria.ops import DEFAULT_MODEL, DETERMINISTIC, build_embedder, propose


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", type=Path, help="prompt file to optimize and review")
    parser.add_argument("-o", "--output", type=Path, default=Path("review.html"), help="output HTML path")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"embedding model id, or {DETERMINISTIC!r} for deterministic hashing",
    )
    args = parser.parse_args()

    prompt = args.prompt.read_text(encoding="utf-8")
    proposal = propose(prompt, build_embedder(args.model))
    html = render_review_page(proposal)
    args.output.write_text(html, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
