#!/usr/bin/env python3
"""Task 3 trial: confirm IFEval fits budget and degrades under redundancy inflation."""
import json
import time
from pathlib import Path
import tiktoken
from scripts.inflate_redundancy import build_generate, inflate
from alexandria.utils.embedders import default_embedder

IFEVAL_DATA = Path.home() / "ucscCode/CSE115A/ifeval-tool/google-research/instruction_following_eval/data/input_data.jsonl"
OUT_DIR = Path("trial_results")
N_PROMPTS = 5
MODEL = "gpt-5.4-mini"


def load_prompts(n):
    all_prompts = []
    with open(IFEVAL_DATA) as f:
        for line in f:
            all_prompts.append(json.loads(line))
    # pick the n longest prompts, so there's more original content to redundantly restate
    all_prompts.sort(key=lambda p: len(p["prompt"]), reverse=True)
    return all_prompts[:n]


def main():
    start = time.time()
    embedder = default_embedder()
    encoding = tiktoken.get_encoding("cl100k_base")
    generate = build_generate(MODEL)
    items = load_prompts(N_PROMPTS)
    OUT_DIR.mkdir(exist_ok=True)
    for label, factor in [("baseline", None), ("2x", 2.0), ("10x", 10.0)]:
        print(f"--- {label} ---")
        input_path = OUT_DIR / f"input_data_{label}.jsonl"
        response_path = OUT_DIR / f"responses_{label}.jsonl"
        with open(input_path, "w") as inp_f, open(response_path, "w") as resp_f:
            for i, item in enumerate(items):
                print(f"  [{label}] prompt {i + 1}/{len(items)}...")
                try:
                    prompt_text = item["prompt"] if factor is None else inflate(
                        item["prompt"], factor, generate, embedder, encoding
                    )
                except RuntimeError as e:
                    print(f"    SKIPPED (inflation failed): {e}")
                    continue
                response_text = generate(prompt_text)
                new_item = dict(item)
                new_item["prompt"] = prompt_text
                inp_f.write(json.dumps(new_item) + "\n")
                resp_f.write(json.dumps({"prompt": prompt_text, "response": response_text}) + "\n")
        print(f"wrote {input_path} and {response_path}")
    print(f"done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()