# [LongLLMLingua](https://arxiv.org/abs/2310.06839)

## Overview
This paper addresses three core problems with LLMs in long-context settings: high computational cost, performance degradation from irrelevant information and position bias. It proposes LongLLMLingua, a question-aware prompt compression system that compresses long prompts while preserving the information most relevant to the query.

## Methods
- **Question-aware coarse-to-fine compression:** scores documents by how much they help predict the question, rather than by generic information entropy
- **Contrastive perplexity:** measures how much a token's perplexity shifts when the question is added, tokens strongly tied to the question are kept
- **Document reordering:** reorders retained documents by relevance score to push key info to the beginning/end, mitigating lost-in-the-middle
- **Dynamic compression ratios:** allocates more budget to more relevant documents
- **Subsequence recovery:** post-processes LLM output to restore entity names that got garbled during compression

## How it is validated
**Benchmarks:**
- NaturalQuestions (multi-doc QA)
- LongBench
- ZeroSCROLLS
- MuSiQue (multi-hop QA)
- LooGLE

**Baselines compared against:**

- Retrieval: BM25, SBERT, OpenAI embeddings
- Compression: Selective Context, LLMLingua

**Key results:**

- 21.4% performance boost on NaturalQuestions with 4x fewer tokens
- 94% cost reduction on LooGLE
- 1.4-2.6x latency speedup

## Relevance to our project
- LongLLMLingua's contrastive perplexity is a query-aware scoring signal, whereas our redundancy scorer is query-agnostic, highlighting a design tradeoff worth knowing
- Their coarse-to-fine compression pipeline mirrors our Represent -> Score -> Optimize -> Select flow, validating the general architecture
- Their finding that removing redundant/irrelevant content can actually improve LLM performance (not just reduce cost) supports our core premise
- The `cos_sim_diff_budget` in our Select phase serves the same role as their compression ratio constraint, both acting as a meaning-preservation stop condition

## Related papers
- [Pan et al. - LLMLingua-2 (2024)](https://arxiv.org/abs/2403.12968) - task-agnostic compression via token classification trained on GPT-4 distilled data

- [Liu et al. - Lost in the Middle (2023)](https://arxiv.org/abs/2307.03172) - the position bias finding that motivates LongLLMLingua's document reordering strategy

