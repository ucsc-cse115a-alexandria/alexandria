# [IFEval](https://arxiv.org/abs/2311.07911)

## Overview
Note covers IFEval (Zhou et al. 2023), a benchmark that checks instruction compliance using code verifiers instead of human or LLM judges.

## Methods
- 541 prompts, each built from a base request plus one to three sampled instructions
- Instructions come from 25 verifiable types across nine groups: keywords, language, length constraints, detectable content, detectable format, combination, change cases, start/end phrasing, punctuation
- Each instruction has its own paired verifier function, so no model is needed to grade responses
- [Code and data](https://github.com/google-research/google-research/tree/master/instruction_following_eval): verifier library, evaluation runner, and the 541-prompt dataset as jsonl

## How it is validated
- Strict accuracy: exact pass/fail on raw response
- Loose accuracy: tolerates minor formatting noise (8 total transformations: markdown strip, intro/outro line removal, and their combinations)
- Both scored at prompt level and instruction level
- Baselines: GPT-4 and PaLM 2 Small

## Relevance to our project
Maps onto `Document` sentences after Represent, each a checkable directive. Gives us a way to test G2's accuracy claim: run an agent on the original vs. reduced prompt, check whether verifiers still pass.

Gap: prompts only carry one to three instructions with no redundancy, so the benchmark doesn't test degradation from near-duplicate restatements. For Enabler B to add.

## Related papers
- [Bai et al. — LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding (2023)](https://arxiv.org/abs/2308.14508)