# [Context Length Alone Hurts](https://arxiv.org/abs/2510.05381)

## Overview
This paper challenges the common assumption that LLM performance on long-context tasks degrades because models fail to retrieve relevant information. Through systematic experiments, it shows that even when models can perfectly retrieve all relevant evidence, performance still degrades substantially as context length increases. The length of the input itself hurts performance, independent of retrieval quality and without any distraction, revealing a previously unrecognized limitation in how LLMs handle long contexts.

## Methods
- **Controlled length experiments:** inserts padding tokens (whitespace, irrelevant text) to increase context length while keeping the relevant content identical, isolating length as the variable
- **Perfect retrieval setting:** forces models to attend only to relevant tokens by masking irrelevant ones, confirming degradation persists even without distraction
- **Tasks:** math (GSM8K), question answering (MMLU), and code generation (HumanEval)
- **Models tested:** 5 open and closed-source LLMs including GPT-4o
- **Mitigation strategy:** prompts the model to recite the relevant evidence before answering, transforming a long-context task into a short-context one

## How it is validated
**Benchmarks:**
- GSM8K (math reasoning)
- MMLU (question answering)
- HumanEval (code generation)
- RULER (long-context evaluation)

**Metrics:**
- Accuracy on each benchmark task

**Key results:**
- Performance degrades substantially as context length increases even with perfect retrieval
- Effect holds across all 5 models tested
- Mitigation strategy (reciting evidence before answering) improves GPT-4o by up to 4% on RULER

## Relevance to our project
- Provides the strongest justification for our core premise: even without redundant instructions, long prompts hurt performance, so compression is beneficial beyond just cost savings
- Suggests our `cos_sim_diff_budget` threshold may need to be more aggressive than 1% since even small amounts of added length degrade performance
- The mitigation strategy of reciting evidence before answering is analogous to our Select phase keeping only the most load-bearing instructions at the top of the reduced prompt
- Confirms that Lost in the Middle's findings hold in newer models, validating our use of that paper as foundational

## Related papers
- [An et al. - Make Your LLM Fully Utilize the Context (2024)](https://arxiv.org/abs/2404.16811) - proposes IN2 training to fix lost-in-the-middle by training on data where answers can appear at any position
