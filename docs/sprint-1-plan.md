# Sprint 1 Plan

**Product:** Alexandria · **Team:** Alexandria Team · **Sprint completion:** 2026-06-30
**Revision:** 1 (2026-06-24)

## Goal

Stand up the composable pipeline through **redundancy scoring**: a prompt is segmented,
embedded into the shared IR, and each instruction gets a redundancy score that is displayed
concisely. In parallel, run a **literature survey** that grounds our scoring and benchmark
design in prior work. This is the prototype completion point — any known prompt can then be
scored as-is.

## Tasks by user story

Estimates are ideal hours (each ≤ 6). Module contracts are defined in [`spec.md`](spec.md).

> **`ALX-1.1` [P1] — Masa Ishihara** As a developer, I want segmented instructions embedded
> into a typed intermediate representation, so that every stage shares one aligned data
> structure. **[5]**
> - Add `sentence-transformers` + `numpy` deps via `uv` (1h)
> - Define the `InstructionSet` IR dataclass with its row-alignment invariant (2h)
> - Implement `embed(segments)` using `all-MiniLM-L6-v2` (3h)
> - Unit tests: IR invariant + embedding shape/alignment (2h)
> - **Total: 8h**

> **`ALX-1.2` [P1] — Masa Ishihara** As a prompt author, I want a concise per-instruction
> redundancy score displayed, so I can see which instructions overlap. **[5]**
> - Cosine-similarity matrix from the IR's embeddings (2h)
> - `redundancy_scores()` = max similarity to any other instruction (3h)
> - Unit tests with near-duplicate fixtures (e.g. repeated "answer in English") (2h)
> - `__main__` display: print each instruction with its score, sorted high→low (1h)
> - **Total: 8h**

> **`ALX-1.5` [P2]** As a team, I want a literature survey of prompt optimization, token /
> context-length effects on LLM accuracy, and LLM/agent/prompt evaluation benchmarks, so our
> scoring and benchmark design is grounded in prior work. **[5]**
> Deliverable: short annotated notes (2–3 key papers per topic) committed under `docs/research/`.
> - Prompt optimization & compression — e.g. APE, OPRO, LLMLingua (3h)
> - Token / context-length effects on accuracy — e.g. "Lost in the Middle", long-context
>   degradation, instruction interference (3h)
> - LLM / agent / prompt evaluation benchmarks — e.g. HELM, MT-Bench, AgentBench (3h)
> - Synthesize 1-page implications for Alexandria's scoring & benchmark (3h)
> - **Total: 12h** (paper names are starting points — verify and extend during the survey)

> **`ALX-1.3` [P2]** As a team, I want dependencies, CI, and quality gates wired up, so we
> can develop reliably. **[3]**
> - Confirm CI runs ruff + pyright (strict) + pytest on the new modules (2h)
> - README: document the pipeline stages and how to run the demo (2h)
> - **Total: 4h**

> **`ALX-1.4` [P3]** As a developer, I want segmentation hardened against varied prompt
> formats, so the pipeline's input is robust. **[2]**
> - Edge-case tests for `segment_instructions` (nested lists, numbered, blank) (2h)
> - Fix any gaps found (2h)
> - **Total: 4h**

**Sprint 1 total: ~36 ideal hours across the team (20 story points).**

## Definition of Done (Sprint 1)

- `prompt → segments → InstructionSet → redundancy scores` runs end to end via `__main__`,
  displaying each instruction with its score.
- New code passes ruff, pyright (strict), and pytest in CI.
- Each public function has a one-line doc comment describing its contract.
- `ALX-1.5` survey notes committed under `docs/research/` with a 1-page implications summary.

## Assignments

- **Masa Ishihara:** `ALX-1.1` (intermediate representation) and `ALX-1.2` (concise score
  display).
- Remaining stories (`ALX-1.5`, `ALX-1.3`, `ALX-1.4`) are assigned at the sprint planning
  meeting.

## To be set at the sprint planning meeting

Remaining task assignments, team roles, the initial burnup chart, the scrum board, and
scrum meeting days/times (incl. the TA visit slot) are filled in live at sprint planning —
intentionally left open here.
