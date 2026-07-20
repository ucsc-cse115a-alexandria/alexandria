#!/usr/bin/env python3
"""Download the pinned official LongBench v2 JSON dataset."""

import json
import urllib.request
from pathlib import Path

PINNED_REVISION = "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9"
URL = f"https://huggingface.co/datasets/zai-org/LongBench-v2/resolve/{PINNED_REVISION}/data.json"
OUT_PATH = Path("data/longbench_v2/data.json")
SHORT_OUT_PATH = Path("data/longbench_v2/short.json")


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL) as response:
        payload = response.read()
    OUT_PATH.write_bytes(payload)
    rows = json.loads(payload)
    short_rows = [row for row in rows if row.get("length") == "short"]
    SHORT_OUT_PATH.write_text(json.dumps(short_rows, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT_PATH} ({OUT_PATH.stat().st_size:,} bytes)")
    print(f"wrote {SHORT_OUT_PATH} ({len(short_rows)} official short cases)")


if __name__ == "__main__":
    main()
