# [RULER](https://arxiv.org/abs/2404.06654)

## Overview
This note covers RULER (Hsieh et al. 2024), a synthetic benchmark that tests whether models actually use their full claimed context, not just retrieve one obvious fact.

## Methods
- 13 tasks across four categories: retrieval, multi-hop tracing, aggregation, question answering
- Retrieval: 8 needle-in-a-haystack variants (single/multi-key/multi-value/multi-query)
- Multi-hop tracing: variable tracking, trace a chain of assignments
- Aggregation: common/frequent word extraction
- QA: SQuAD and HotpotQA with distractor paragraphs added
- All tasks synthetic, auto-generated for any target sequence length
- [Code and data](https://github.com/hsiehjackson/RULER)

## How it is validated
- 17 models evaluated at context lengths 4K to 128K
- Score per length = average accuracy across all 13 tasks
- Effective context length = max length staying above Llama2-7B's 4K score (85.6%), compared to claimed length
- Finding: near-perfect specifically on passkey retrieval and vanilla NIAH (2 of 13 tasks), but sharp drops across the other 11 as length increases, including other NIAH variants

## Relevance to our project
Matches the must-have length-sensitivity criterion directly, RULER already reports accuracy by length.

Gap: tests retrieval/tracing in distractor text, not instruction-following in near-duplicate instructions. Doesn't exercise our redundancy scorer's target failure mode, Enabler B would still need to build that on top.

## Related papers
- [Zhou et al. — IFEval (2023)](https://arxiv.org/abs/2311.07911)
- [Bai et al. — LongBench (2023)](https://arxiv.org/abs/2308.14508)