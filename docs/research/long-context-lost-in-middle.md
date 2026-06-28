# [Lost in the Middle](https://arxiv.org/abs/2307.03172)

## Overview
This paper investigates how LLMs use information across long input contexts. It finds that performance follows a U-shaped curve based on where relevant information is positioned: models perform best when key information appears at the beginning or end of the context, and significantly worse when it appears in the middle. This "lost in the middle" effect was observed consistently across six model families and multiple tasks, revealing a fundamental limitation in how LLMs attend to long contexts.

## Methods
- **Multi-document QA:** tests whether models can identify a relevant document among many irrelevant ones, with the position of the relevant document systematically varied
- **Key-value retrieval:** a synthetic task where models must retrieve a value given a key from a long list, with the target pair placed at different positions
- **Position sweeping:** both tasks vary the position of the relevant information across all possible positions to measure how position affects accuracy
- **Models tested:** GPT-3.5-Turbo, GPT-4, Claude 1.3, LongChat-13B, MPT-30B, Cohere Command

## How it is validated
**Benchmarks:**
- Multi-document QA (NaturalQuestions dataset, 20 documents)
- Key-value retrieval (synthetic task)

**Metrics:**
- Exact match accuracy for QA
- Retrieval accuracy for key-value task

**Key results:**
- Performance degrades by over 30% when relevant information is in the middle vs the beginning/end
- Effect holds across all six model families tested
- More documents in context makes the effect worse

## Relevance to our project
- The lost-in-the-middle effect directly motivates why we compress prompts rather than just appending instructions, long instruction-heavy prompts suffer the same positional degradation
- Our drift_budget in Select ensures we don't compress so aggressively that load-bearing instructions get pushed into vulnerable middle positions
- The U-shaped attention pattern suggests that our redundancy scorer should prioritize removing instructions from the middle of a prompt over the edges, a future enhancement worth exploring
- This paper's findings have been confirmed in newer models by Du et al. (2025), so the effect is still relevant despite the older model baselines

## Related papers
- [Hsieh et al. - Found in the Middle (2024)](https://arxiv.org/abs/2406.16008) - proposes a calibration mechanism to fix the lost-in-the-middle problem by disentangling attention bias from relevance
