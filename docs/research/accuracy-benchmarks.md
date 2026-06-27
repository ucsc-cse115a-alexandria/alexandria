# Accuracy Benchmarks With Published Prompts and Ground Truth

## Overview

This note answers a narrow question for Alexandria: which public benchmark
publishes its EXACT evaluation prompts AND ground-truth answers, so we can run
the same prompt before and after compression and score the result
automatically? We need this to validate that compressing an `InstructionSet`
preserves task accuracy. The primary recommendation is **IFEval**
(Instruction-Following Eval), whose prompts are the literal model inputs and
whose "ground truth" is a set of deterministic, program-checkable constraints.
The runner-up is **GSM8K**, which pairs each question with a single numeric gold
answer and a documented prompt template in the EleutherAI lm-evaluation-harness.

## Methods

### Primary candidate — IFEval (Zhou et al., 2023)

- **What it tests:** Whether a model obeys explicit, verifiable instructions
  embedded in a prompt (for example, "write at least 300 words", "respond in
  JSON", "include the keyword X at least 3 times", "do not use any commas").
  The scoring targets *form*, not content, so it needs no human or LLM judge.
- **Size:** 541 prompts, single `train` split, 25 distinct instruction types,
  English only (BCP-47 `en`). Apache 2.0 license.
- **Exact-prompt availability:** The `prompt` field IS the literal string sent
  to the model — there is no hidden template wrapping it. This is the key reason
  IFEval fits our A/B: the published artifact is the exact eval prompt itself.
- **Ground-truth format:** Each row carries
  `key`, `prompt`, `instruction_id_list` (the constraint types that must hold),
  and `kwargs` (per-constraint parameters such as `num_words: 300`,
  `relation: "at least"`, `num_highlights: 3`). The "answer key" is therefore a
  machine-checkable specification of constraints, not a single gold string.
- **Scoring:** A deterministic checker verifies each instruction against the
  model output. The official Google Research code
  (`instruction_following_eval`) reports four metrics: prompt-level and
  instruction-level accuracy, each in a `strict` and a `loose` variant. Strict
  requires exact compliance; loose strips preambles/outros and formatting
  modifiers to reduce false negatives.

### Runner-up — GSM8K (Cobbe et al., 2021)

- **What it tests:** Multi-step grade-school math word problems requiring
  arithmetic reasoning.
- **Size:** 8.5K problems (7,473 train / 1,319 test), two plain string fields
  `question` and `answer`. The `answer` ends with a `#### <final_number>` line.
- **Exact-prompt availability:** The literal eval prompt is not in the dataset
  itself but is published in the lm-evaluation-harness task YAML:
  `doc_to_text: "Question: {{question}}\nAnswer:"` with `doc_to_target:
  "{{answer}}"`. The default task is 5-shot; a `gsm8k_cot` variant is 8-shot.
- **Ground-truth format:** A single numeric gold answer, extracted from the
  `#### <number>` suffix of the `answer` field.
- **Scoring:** `exact_match` on the extracted number, via two filters —
  `strict-match` (requires the `#### <number>` format) and `flexible-extract`
  (takes the last number in the generation).

IFEval is preferred because its dataset row *is* the prompt, making the
before/after comparison unambiguous, and its deterministic per-constraint
checker gives fine-grained signal (instruction-level accuracy) that is sensitive
to small wording changes from compression.

## How it is validated

- **IFEval:** The model is given each `prompt` verbatim and produces a free-form
  response. The checker evaluates every constraint named in
  `instruction_id_list` using the parameters in `kwargs`. A prompt counts as
  "followed" only if all its constraints pass (prompt-level); each constraint is
  also scored individually (instruction-level). Accuracy =
  (followed) / (total), reported separately under strict and loose criteria. No
  judge model is involved, so scores are fully reproducible.
- **GSM8K:** Standardized through the EleutherAI lm-evaluation-harness. The
  harness renders the documented prompt template, runs greedy decoding
  (`temperature 0.0`), extracts the final number with the `strict-match` /
  `flexible-extract` regex filters, normalizes it (ignoring commas, `$`, and the
  `#### ` prefix), and compares against the gold number using `exact_match`.
- **Common harness:** Both benchmarks are first-class tasks in the
  lm-evaluation-harness, so a single CLI can drive prompt rendering, generation,
  answer extraction, and metric aggregation reproducibly.

## Relevance to our project

Alexandria compresses an `InstructionSet` by redundancy scoring and greedy
pairwise pruning. To prove compression preserves accuracy, we run an A/B with a
published prompt + ground-truth benchmark:

1. **Baseline (before):** Take each IFEval `prompt` as the uncompressed
   `InstructionSet` input, send it to the target model, and run the official
   strict/loose checker. Record prompt-level and instruction-level accuracy.
2. **Compressed (after):** Pass the same prompt through Alexandria's
   represent -> score -> greedy-pairwise pipeline to produce a compressed prompt,
   send it to the same model under identical decoding settings, and run the same
   checker against the same `instruction_id_list` / `kwargs`. Crucially, the
   ground truth is tied to the constraints, not to the prompt wording, so it
   stays valid even after we rewrite/shorten the prompt — exactly what we need to
   measure whether compression drops constraints.
3. **Compare:** Accuracy delta (compressed minus baseline) measures fidelity;
   token-count delta measures savings. We report both, plus per-instruction-type
   regressions to see which constraint categories Alexandria's pruning tends to
   lose. IFEval's deterministic, judge-free scoring makes this delta trustworthy.
4. **Cross-check on reasoning:** Repeat with GSM8K to confirm compression of a
   reasoning-style prompt (the few-shot exemplars + question) preserves the final
   numeric answer under `exact_match`.

Practical note: IFEval is the better first integration because its dataset row is
the exact prompt, so wiring it to `InstructionSet` is a direct map with no
template reconstruction. GSM8K requires reproducing the harness template, but
gives us a second, content-correctness axis beyond format-following.

## Related papers

- [Zhou et al. — Instruction-Following Evaluation for Large Language Models
  (2023)] — defines IFEval's 541 verifiable-instruction prompts and the
  strict/loose, prompt/instruction-level metrics. arXiv:2311.07911. Dataset:
  https://huggingface.co/datasets/google/IFEval — Code:
  https://github.com/google-research/google-research/tree/master/instruction_following_eval
- [Cobbe et al. — Training Verifiers to Solve Math Word Problems (2021)] —
  introduces GSM8K, 8.5K grade-school math problems with `#### <answer>` gold
  format. arXiv:2110.14168. Dataset: https://huggingface.co/datasets/openai/gsm8k
- [EleutherAI — lm-evaluation-harness] — standardized task definitions
  (`doc_to_text` prompt templates, `doc_to_target` answer keys, metric/filter
  configs) for both IFEval and GSM8K. Repo:
  https://github.com/EleutherAI/lm-evaluation-harness — GSM8K task:
  https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/gsm8k.yaml
