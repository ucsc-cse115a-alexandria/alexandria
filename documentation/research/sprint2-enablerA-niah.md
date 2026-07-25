# [Needle In A Haystack (NIAH)](https://github.com/gkamradt/needle-in-a-haystack)

## Overview
This note covers Needle In A Haystack (Kamradt, 2023), a tool that sweeps context length and needle depth to test whether a model can retrieve a planted fact, with no accompanying paper, the repo itself is the primary source.

## Methods
- Sweeps a grid of (context length × depth) cells, one result row per cell, written to JSONL
- Built-in tasks: `single` (one fact), `multi` (N facts, fractional scoring), `uuid` (fresh UUID, exact repeat), `uuid_chain` (linked facts, model must discover and follow the chain unprompted)
- Tasks, haystack sources, and scorers are each a small Protocol behind a registry, so adding a new one is a new file, no edit to the runner
- Ships a `FakeProvider` so the full pipeline runs with no API keys, plus OpenAI, Anthropic, and Cohere for real runs
- v2 fixed a v1 bug where multi-needle depth was computed after earlier needles had already inflated the token count, so reported depths were off

## How it is validated
- No fixed baseline or accuracy threshold ships with the tool itself, unlike RULER's Llama2-7B cutoff
- Original runs backed a public GPT-4/Claude analysis, visualized as a length × depth heatmap; those raw results are kept in `original_results/` but don't load against the current schema
- Haystack source is Paul Graham's essays, since adopted, this exact haystack is reused directly inside RULER's own NIAH tasks (its NIAH-s2 and NIAH-MV variants) for consistency across benchmarks
- No paper, so no citation count, clears the must-have bar on GitHub stars alone (2.3k)

## Relevance to our project
The Protocol-plus-registry structure (tasks, haystack sources, scorers each a small pluggable piece) mirrors how our own score/optimize/select phases are organized, and the recipe-based JSONL output is a workable pattern for Enabler B: store what generated a trial's prompt, not the full rendered text, and reconstruct on demand.

Gap: same as RULER, this tests single or chained fact retrieval in a haystack of unrelated filler text, not instruction-following in a redundant instruction set. It's an even narrower match to our failure mode than RULER, and with no paper behind it, "established" rests entirely on GitHub adoption rather than any peer-reviewed validation.

## Related papers
- [Hsieh et al. — RULER: What's the Real Context Size of Your Long-Context Language Models? (2024)](https://arxiv.org/abs/2404.06654)
- [Zhou et al. — IFEval (2023)](https://arxiv.org/abs/2311.07911)