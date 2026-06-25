# Release Plan

**Product:** Alexandria · **Team:** Alexandria Team · **Release:** 1.0
**Release date:** 2026-07-21 (end of Sprint 4) · **Revision:** 1 (2026-06-24)

## High-level goals

1. A **label-free prompt-compression pipeline** that segments a prompt, embeds it into a
   shared intermediate representation, and scores each instruction's redundancy.
2. **Reduction** that removes redundant instructions while preserving meaning.
3. A **benchmark** that reports token reduction over well-known prompts (e.g. Anthropic
   system prompts, popular Agent Skills).
4. A **composable CLI** (`alexandria`) delivering the full pipeline as a Unix-style tool.

Architecture and module contracts: see [`spec.md`](spec.md). Stack: see
[`tech-stack.md`](tech-stack.md).

## Sprint mapping

Four one-week sprints (per `schedule.md`). Each sprint adds one composable stage on top
of the shared IR, so each is independently demoable. **A working CLI ships at the end of
Sprint 3**; Sprint 4 is then purely benchmark-driven improvement and release hardening.

- **Sprint 1** (Jun 24–30) — Redundancy scoring (prototype)
- **Sprint 2** (Jul 1–7) — Reduction
- **Sprint 3** (Jul 8–14) — Benchmark & **working CLI** (full pipeline runs as a command)
- **Sprint 4** (Jul 15–21) — Benchmark-driven improvement & release hardening → Release 1.0

## User stories (priority: P1 highest)

### Sprint 1 — Scoring prototype & literature survey
- `ALX-1.1` **[P1]** As a developer, I want segmented instructions embedded into a typed
  intermediate representation, so that every stage shares one aligned data structure. **[5]**
- `ALX-1.2` **[P1]** As a prompt author, I want a concise per-instruction redundancy score
  displayed, so I can see which instructions overlap. **[5]**
- `ALX-1.5` **[P2]** As a team, I want a literature survey of (a) prompt optimization,
  (b) how token count / context length affects LLM accuracy, and (c) LLM / agent / prompt
  evaluation benchmarks, so our scoring and benchmark design is grounded in prior work. **[5]**
- `ALX-1.3` **[P2]** As a team, I want dependencies, CI, and quality gates wired up, so we
  can develop reliably. **[3]**
- `ALX-1.4` **[P3]** As a developer, I want segmentation hardened against varied prompt
  formats, so the pipeline's input is robust. **[2]**

### Sprint 2 — Reduction
- `ALX-2.1` **[P1]** As a prompt author, I want redundant instructions dropped above a
  threshold, so my prompt gets shorter while keeping one of each near-duplicate. **[5]**
- `ALX-2.2` **[P2]** As a prompt author, I want a redundant cluster merged into one
  representative instruction, so meaning is kept with fewer words. **[5]**
- `ALX-2.3` **[P3]** As a prompt author, I want a guardrail that never drops a unique
  instruction, so reduction is safe. **[3]**

### Sprint 3 — Benchmark & working CLI
- `ALX-3.1` **[P1]** As a user, I want a CLI that runs the full pipeline (segment → embed
  → score → reduce) from stdin to stdout, so I can compress a prompt with one command. **[5]**
- `ALX-3.2` **[P1]** As a maintainer, I want a benchmark harness over a corpus of known
  prompts, so improvements are measurable. **[5]**
- `ALX-3.3` **[P1]** As a maintainer, I want a token-reduction metric and before/after
  results tabulated, so the effect is easy to read. **[3]**

### Sprint 4 — Improvement & release hardening
- `ALX-4.1` **[P1]** As a maintainer, I want improvements driven by Sprint 3 benchmark
  findings, so the release reduces tokens without losing meaning. **[5]**
- `ALX-4.2` **[P2]** As a user, I want a `--report` mode and clearer CLI output, so I can
  see what was removed and why. **[3]**
- `ALX-4.3` **[P2]** As a user, I want a README, install steps, and examples, so I can
  adopt the tool. **[3]**
- `ALX-4.4` **[P3]** As a prompt author, I want merge/threshold tuning from the benchmark,
  so reduction quality improves before release. **[3]**

## Capacity sanity check

- Team capacity: 4 members × one-week sprints. Roughly 1 story point ≈ 3 ideal hours.
- Per-sprint load: S1 = 20, S2 = 13, S3 = 13, S4 = 14 points (**total 60**).
- Sprint 1 is the fullest (20) because it carries the `ALX-1.5` literature-survey spike
  alongside the prototype; the survey runs in parallel with coding and overlaps the
  individual RAC paper reading, and `ALX-1.4` is a P3 stretch that may slip.
- **Jul 3 (Fri) is a holiday** during Sprint 2 → that sprint is deliberately kept lighter
  (13) and front-loads its P1 work.
- Other course load (RACs, two tests, TSRs) runs in parallel; sprints stay near the lower
  end of capacity rather than maxed out.

## Product backlog (not in Release 1.0)

- Hosted embedding API backend (swappable via the IR).
- Per-model exact tokenizers (Claude / GPT) instead of the proxy counter.
- Semantic *rewriting* of instructions beyond drop/merge.
- Clustering-based reduction (group then summarize), vs. the threshold-first v1.
- Web UI / hosted service.
- Configurable redundancy metric (mean vs. max similarity, learned thresholds).

The plan and backlog are revisited and updated after each sprint.
