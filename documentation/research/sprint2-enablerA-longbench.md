# [LongBench](https://arxiv.org/abs/2308.14508)

## Overview
This note covers LongBench (Bai et al. 2023, ACL 2024), a bilingual benchmark that evaluates long-context understanding across a spread of real task types, not just retrieval.

## Methods
- 21 datasets across 6 categories: single-doc QA, multi-doc QA, summarization, few-shot learning, synthetic tasks, code completion
- English and Chinese, average length 6,711 words (English), 13,386 characters (Chinese)
- Only 3 tasks are fully synthetic (PassageCount, PassageRetrieval-en, PassageRetrieval-zh), the rest are adapted from real QA/summarization/code datasets with distractor content added to force long-context reliance
- All tasks scored automatically (F1, ROUGE-L, Edit Sim, classification accuracy), no human or LLM judge
- LongBench-E is a companion subset with a deliberately even length distribution (0-4k, 4k-8k, 8k+), built to isolate the effect of length from task difficulty
- [Code and data](https://github.com/THUDM/LongBench)

## How it is validated
- 8 models evaluated, zero-shot except few-shot tasks
- Score per model is macro-averaged over the 6 major categories, not a flat average over all 21 datasets, since the paper found synthetic tasks alone can dominate and skew a flat average
- LongBench-E's length-bucketed scores directly test length sensitivity: GPT-3.5-Turbo-16k drops 17% from 0-4k to 8k+ despite strong overall performance, ChatGLM2-6B-32k and LongChat-32k are more robust (4% and 7% drop)
- Retrieval-based compression tracks model strength: -2% for GPT-3.5-Turbo-16k, +21% for the weakest model (Llama2-7B-chat-4k), -5% for ChatGLM2-6B-32k, so it compensates for weak long-context models rather than beating strong ones
- Summarization-based compression follows a different pattern: it only helps on one dataset (VCSUM) regardless of model, because VCSUM's source documents are longer than the benchmark's other summarization datasets

## Relevance to our project
LongBench-E's design, same tasks, deliberately varied length, isolating length as the variable, is close to exactly what Enabler A's must-have criterion asks for: does the score degrade as input grows. The compression experiments are also directly relevant, they're testing whether compressing the context hurts accuracy, our whole premise, just with generic context compression instead of instruction-specific redundancy removal.

Gap: none of the 21 tasks are instruction-following tasks. The closest is PassageCount, which asks the model to determine the number of unique passages among a set of duplicates, structurally similar to redundancy detection, but the task is counting unique passages, not verifying instruction compliance. We'd still need Enabler B's own redundant-instruction dataset; LongBench doesn't give us that task type directly.

## Related papers
- [Zhou et al. — IFEval (2023)](https://arxiv.org/abs/2311.07911)
- [Hsieh et al. — RULER (2024)](https://arxiv.org/abs/2404.06654)
- [Liu et al. — Lost in the Middle: How Language Models Use Long Contexts (2023)](https://arxiv.org/abs/2307.03172)