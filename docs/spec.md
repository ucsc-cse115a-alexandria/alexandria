# Alexandria — Design Spec

**Product:** Alexandria · **Team:** Alexandria Team · **Revision:** 1 (2026-06-24)

Label-free prompt optimization: compress instruction-heavy prompts while preserving
their meaning, using sentence embeddings.

## Goal

Given a long, redundant prompt, surface which instructions overlap (a **redundancy
score**) and remove the redundant ones — without labels, training, or a target output.

## Architecture — one intermediate representation, composable stages

Following the Unix philosophy, each stage does one thing and is a pure function over a
shared **intermediate representation (IR)**. Stages are independently testable and can be
swapped or composed in any order.

```
prompt ─▶ segmentation ─▶ embedding ─▶ scoring ─▶ reduction ─▶ benchmark ─▶ cli
          (str→[str])     (→ IR)       (→ scores) (→ IR')      (→ report)
```

| Stage | Module | Signature (intent) | Milestone |
|---|---|---|---|
| Segmentation | `segmentation` | `segment_instructions(str) -> list[str]` | done |
| Embedding | `embedding` | `embed(list[str]) -> InstructionSet` | Sprint 1 |
| Scoring | `scoring` | `redundancy_scores(InstructionSet) -> list[float]` | Sprint 1 |
| Reduction | `reduction` | `reduce(InstructionSet, scores, threshold) -> InstructionSet` | Sprint 2 |
| CLI | `__main__` / `cli` | stdin → stdout (JSON), pipes stages | Sprint 3 |
| Benchmark | `benchmark` | `evaluate(prompts) -> Report` | Sprint 3 |

## Intermediate representation

The IR is the spine: it manages the embeddings and keeps them aligned to their segments.
Parse-don't-construct (validate on build) and make impossible states unrepresentable
(one object, no parallel lists drifting out of sync).

```python
@dataclass(frozen=True)
class InstructionSet:
    """Segments and their embeddings, row-aligned. The shared IR for all stages."""
    segments: list[str]
    embeddings: NDArray  # shape (n, dim); row i corresponds to segments[i]
    # invariant (checked on construction): len(segments) == embeddings.shape[0]
```

Every downstream stage takes an `InstructionSet` and returns scores or a new
`InstructionSet`. Nothing depends on how the embeddings were produced — so a hosted
embedding API or a different model can be dropped in later without touching scoring or
reduction.

## Scoring — redundancy (Sprint 1, prototype completion point)

Redundancy score for instruction `i` = its similarity to the *most* similar other
instruction (cosine similarity over embeddings). High score ⇒ near-duplicate ⇒ a
reduction candidate. This is fully label-free.

```
sim = cosine_similarity_matrix(embeddings)   # (n, n)
score[i] = max(sim[i][j] for j != i)
```

Why this is the proto milestone: once scoring exists, any well-known prompt (Anthropic's
system prompts, popular Agent Skills) can be scored as-is — no reduction or evaluation
needed yet — which de-risks the rest of the project.

## Reduction (Sprint 2)

Take the scored IR and produce a smaller one. Start with the simplest policy and grow:
- **drop**: remove instructions whose redundancy score ≥ threshold (keep one of each
  near-duplicate cluster).
- **merge** (stretch): combine a redundant cluster into one representative instruction.

Meaning preservation is the constraint: reduction must never drop a *unique* instruction.

## Benchmark (Sprint 3)

Run the full pipeline over a fixed corpus of known prompts and report:
- token reduction (via `tiktoken`, stated as a model-independent proxy),
- redundancy score distribution before/after.

Results are tabulated with Polars.

## CLI (Sprint 3)

`alexandria` reads a prompt on stdin and writes the reduced prompt (or a JSON report) on
stdout, so stages compose with normal shell pipes. **The full pipeline runs end to end as
a command by the end of Sprint 3**, alongside the benchmark. Sprint 4 then hardens it with
benchmark-driven improvements and a `--report` mode → Release 1.0.

## Non-goals (YAGNI)

- No fine-tuning or labeled optimization.
- No hosted embedding API in v1 (the IR keeps that swappable for later).
- No prompt *rewriting* beyond merge; reduction is selection-first.
