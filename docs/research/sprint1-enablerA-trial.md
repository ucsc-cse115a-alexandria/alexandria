# Enabler A, Task 3: IFEval Trial and Benchmark Rationale

## Trial setup

- Benchmark: IFEval (Zhou et al. 2023), installed locally, scored with the original
  google-research implementation
- Model under test: gpt-5.4-mini, used both to generate the redundant-inflation
  restatements (via scripts/inflate_redundancy.py) and to generate the responses
  IFEval scores
- Two trial runs, both on 5 prompts pulled from input_data.jsonl:
  - Trial 1: the first 5 prompts in the file (short, single-instruction prompts)
  - Trial 2: the 5 longest prompts in the file (longer, more content per prompt)
- Each run scored baseline, 2x inflation, and 10x inflation through evaluation_main

## Results

| Metric | Trial 1 (short prompts) | Trial 2 (long prompts) |
|---|---|---|
| Baseline accuracy | 100% (5/5) | 100% (5/5) |
| 2x accuracy | 100% (2/5 survived inflation) | 100% (4/5 survived inflation) |
| 10x | 0/5 survived, run completed in full | killed after ~8 min on prompt 1/5, no convergence |
| Cost | $0.16 | $0.72 |
| Time | 339s (~5.6 min) | not fully timed (10x killed early) |

## Budget criterion

A subset of 5 prompts completes well within the 10 minute / $1 must-have bar. Trial 1
ran end to end in under 6 minutes for $0.16. Trial 2's baseline and 2x passes together
also stayed well inside the budget; only the 10x attempt threatened to exceed it, and
was stopped manually rather than let run past the ceiling.

## Length-sensitivity criterion

Not demonstrated at this scale. Both trials show gpt-5.4-mini holding 100% accuracy
at 2x redundancy inflation, with no measurable drop from baseline. Two independent
prompt sets, two consistent null results.

10x inflation itself is not achievable in practice for IFEval-style prompts:
- Short prompts: every one of 5 prompts failed scripts/inflate_redundancy.py's
  similarity check within its built-in retry limit. There is little room to
  redundantly restate a single short instruction ten times over without either
  repeating verbatim or diverging from the original meaning.
- Longer prompts: after starting the first prompt of the 10x pass, the script
  produced no further output for over 8 minutes, at which point it was killed
  manually. The cause was not confirmed; the run simply did not complete or
  produce a result in that time.

This is a property of the redundancy-inflation method applied to IFEval's prompt
style (short, single- or few-instruction), not necessarily evidence that IFEval
itself is insensitive to redundancy. Alexandria's real target prompts (AGENT.md /
CLAUDE.md files) are longer and carry many more instructions per document, which
may behave differently under compression than a single short IFEval prompt does
under inflation.

## Rationale

IFEval remains the strongest candidate of the four surveyed (see
sprint1-enablerA-rating.md): it is the only one that tests per-instruction
compliance rather than long-context retrieval, which is the actual failure mode
Alexandria targets. This trial confirms it clears the time and cost budget on a
small subset.

It does not confirm the length-sensitivity must-have criterion at the single-prompt
scale tested here. That is a real gap against the acceptance bar, and worth flagging
to the team rather than treating as resolved. Two paths forward for Sprint 3:

- Test length-sensitivity on Alexandria's actual target artifact (a full AGENT.md /
  CLAUDE.md), inflating a whole document rather than a single short IFEval prompt,
  where there is more content to redundantly restate before hitting a similarity
  or convergence wall
- Treat 2x/10x prompt-level inflation as the wrong proxy for this benchmark, and
  instead test degradation by dropping a real instruction from a multi-instruction
  prompt (closer to Alexandria's actual failure mode) rather than inflating length

Given the budget criterion is met and IFEval is still the best-fit candidate by a
clear margin on the other criteria, recommend proceeding with IFEval as the base
benchmark, with the length-sensitivity gap tracked as an open risk for Sprint 3
rather than blocking selection.
