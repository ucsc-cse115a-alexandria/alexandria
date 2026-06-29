# [Guidelines to Prompt LLMs for Code Generation](https://arxiv.org/abs/2601.13118)

## Overview
This paper derives and evaluates 10 empirically grounded guidelines for writing better prompts for LLM-based code generation. Starting from code generation benchmarks where LLMs consistently failed, the authors used an automated iterative process to optimize the prompts until tests passed, then manually analyzed what changed to extract the improvement patterns. The question it answers is: what specific elements should a developer include in a prompt to maximize the chance of getting correct code from an LLM?

## Methods
- **Automated prompt optimization:** for tasks where LLMs always failed, the system iteratively fed test failure messages back to the LLM to generate improved code, then asked the LLM to revise the original prompt based on what made the code pass
- **Manual analysis:** authors inspected pairs of original and optimized prompts to identify what changed, deriving a taxonomy of improvement patterns
- **Practitioner survey:** 50 developers rated their usage frequency and perceived usefulness of each guideline
- **Models tested:** GPT-4o mini, Llama 3.3 70B, Qwen2.5 72B, DeepSeek Coder V2
- **Benchmarks:** BigCodeBench, HumanEval+, MBPP+

## How it is validated
**Benchmarks:**
- BigCodeBench (1,140 Python tasks)
- HumanEval+ (163 tasks)
- MBPP+ (375 tasks)

**Metrics:**
- Test pass rate for validating optimized prompts
- 5-level usage frequency scale from practitioner survey
- 5-level Likert scale for perceived usefulness from practitioner survey

**Key results:**
- 10 guidelines extracted across 627 successfully optimized prompt pairs
- I/O format and pre/post conditions rated most useful and most used by practitioners
- Adding examples rated highly useful despite being underused, suggesting a gap in awareness

## Relevance to our project
- The guideline to use assertive language ("must" not "should") directly applies to how we evaluate instruction quality in our redundancy scorer. Two instructions that say the same thing but one uses weaker language may score as redundant when they aren't
- The I/O format and pre/post conditions guidelines suggest that well-written instructions have a specific structure. Our Scorer could use this as a signal for instruction completeness, not just redundancy
- The algorithmic details guideline supports our merge optimizer. When collapsing redundant instructions via a Replace edit, the merged instruction should preserve the most specific and assertive version, not just the shortest
- The paper focuses on code generation prompts but the guidelines are language-agnostic and apply directly to the instruction-heavy system prompts we target

## Related papers
- [Fagadau et al. - Analyzing Prompt Influence on Automated Method Generation (2024)](https://arxiv.org/abs/2402.08430) - empirically shows that structural cues like I/O examples and summaries strongly correlate with higher quality code generation from Copilot
