#!/usr/bin/env python3
"""Download the pinned official LongBench v2 JSON dataset."""

import urllib.request
from pathlib import Path

PINNED_REVISION = "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9"
URL = f"https://huggingface.co/datasets/zai-org/LongBench-v2/resolve/{PINNED_REVISION}/data.json"
OUT_PATH = Path("data/longbench_v2/data.json")


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL) as response:
        OUT_PATH.write_bytes(response.read())
    print(f"wrote {OUT_PATH} ({OUT_PATH.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
