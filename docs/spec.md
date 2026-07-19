# Alexandria — Design Spec

Label-free prompt optimization: shorten instruction-heavy prompts while preserving their
meaning, using sentence embeddings.

## Goal

Given a long, redundant prompt, find which instructions overlap (a **redundancy score**) and
drop the redundant ones — with no labels, no training, and no target output to compare against.

## The four phases

Alexandria optimizes a prompt in **four phases over one intermediate representation (IR)**.
The prompt becomes a `Document`, the `Document` is scored, the scores drive a `Plan` of ranked edit
candidates, and a selector folds the `Plan` into a smaller `Document`. That is the whole system:

```
                 ┌──────────────────────── reduce ────────────────────────┐
prompt ─────────▶│  Represent   ──▶   Score   ──▶   Optimize   ──▶  Select │──▶ reduced prompt
                 │  prompt→Document   →Scores       →Plan            →Document
                 └────────────────────────────────────────────────────────┘
```

| Phase | What it does | Contract |
|---|---|---|
| **1. Represent** | Turn the prompt into the IR: split into instructions, count tokens, embed each one. | `represent(prompt: str, embedder: Embedder \| None = None) -> Document` |
| **2. Score** | Rate every instruction. The first scorer rates **redundancy**. | `Scorer = (Document) -> list[float]` |
| **3. Optimize** | Propose and rank the edits the scores justify. | `Optimizer = (Document, Scores, Embedder, SentenceMerger, Params, ReductionReporter) -> Plan` |
| **4. Select** | Apply candidates in ascending `cos_sim_diff` order while the prompt stays within the `cos_sim_diff` budget. | `Selector = (Document, Plan, Embedder, Params) -> Selection` |

Each phase is a pure function over the IR (Represent, Optimize, and Select take the impure `Embedder`
— and Optimize also the `SentenceMerger` — as injected dependencies), so each is tested on its own and
the four compose in order. The split between **Optimize** and **Select** is deliberate: Optimize
*proposes and ranks* candidates — probing each edit's whole-document `cos_sim_diff` only to feed the merge model
back a retry (a per-edit quality mechanism, not the stop condition) — and Select is the controller that
*chooses and applies* under the cumulative `cos_sim_diff` budget, keeping the "which edits, in what order, and
when to stop" decision in one place.

**Score** and **Optimize** are *open sets*, not single steps: many scorers and many optimizers coexist
behind a registry, an optimizer **declares which scorer(s) it consumes**, and the scores it receives travel
as a name-keyed **bundle** (`Scores = dict[str, dict[SentenceId, float]]`) so one optimizer can read
several scorers at once. An optimizer does not return a `Document`; it returns a **`Plan`** — an ordered
**stack** of edit candidates, each carrying the `confidence` that ranked it — so several optimizers can run
at once and their stacks **concatenate** into one series.

**Select** is an open set too: a *selector* reads the `Plan` and folds it into a `Selection` (the smaller
`Document` plus the candidates it applied). The default `least_cos_sim_diff` selector applies candidates in
ascending whole-document `cos_sim_diff` order and keeps each only while the reduced prompt's
`cos_sim_diff` from the original stays within `cos_sim_diff_budget` — so unattended `reduce` compresses as
far as the meaning budget allows. A future interactive selector (the `review` verb) walks the same stack
and asks a human to accept or reject each edit. Both call one
pure primitive, `Document.apply`, folding the stack candidate by candidate into the smaller `Document` —
each edit applying to the result of the one before it, the way `sed` streams commands — the only place the tree is rewritten.
Adding a scorer, optimizer, or selector is one new file, never an edit to an existing one, and a third
party can ship one from its own package (see [Extending a phase](#extending-a-phase-scorers-and-optimizers)).

Two things sit **outside** the three phases on purpose:

- The **CLI** is how you *invoke* the phases — a thin wrapper that moves text in and out.
- The **benchmark** is how you *measure* the phases — a separate concern (see
  [Evaluation](#evaluation--a-separate-concern)). The design stays easy to benchmark, but
  benchmarking is not part of the optimization.

## Design principles

A few rules the whole codebase obeys. Everything below applies one of them; when a later
decision and a principle disagree, the principle wins.

**Modularity through contracts, not implementations.** The system talks to itself across exactly
two stable contracts: the **IR** (`Document` / `Section` / `Sentence`) and the **phase signatures**
(`represent`, `Scorer`, `Optimizer`, `Selector`). A caller depends on the contract and nothing behind it. Any
scorer, optimizer, selector, tokenizer, or embedding backend can be swapped without a neighbor noticing,
because only the data shape and the signature cross the boundary.

**Every Score and Optimize strategy is a plugin.** Scorers and optimizers are the same kind of
thing — a named strategy behind a `Protocol`, kept one-per-file in its phase's package, and
registered by name. The folder layout and the `Protocol` make "add a new strategy" mechanical, so
anyone who adds one writes it the same way. The design assumes *many* coexist: redundancy is the
first scorer, but centrality, cost, and others drop in without touching the IR or each other. When
several run, their outputs travel as a name-keyed **score bundle**, and each optimizer declares the
scorer(s) it needs (`requires=(...)`) so the pipeline runs exactly those — selection is data, not a
hardcoded pairing. A strategy registers by name with `@register` as an import side effect: importing its
module runs the decorator that adds it to the registry, so the built-in strategies are live as soon as
their `features` module is imported. (A future extension could wire the same registry to Python entry
points for out-of-tree packages; that discovery path is not implemented today.) Optimizers return a **`Plan`** rather than a
`Document`, so the stacks of several optimizers concatenate into one ordered series that a human (or the
pipeline) folds, deciding what to apply.

**Library first; the CLI is a thin wrapper.** All behavior lives in importable, pure functions
over the IR. The CLI only parses arguments, moves text in and out, and calls the library. Anything
you can do from the terminal you can do from `import alexandria`, and the CLI adds no capability
the library lacks.

**Testable without mocks — by design.** Phases are pure functions over plain data: feed a known
`Document` in, get deterministic scores, a `Plan`, or a new `Document` out, assert on the real value. Tests
build small `Document` fixtures directly — a handful of sentences with tiny hand-written embeddings
— so Score is tested end to end with no model in the loop. Represent and Select take the model as an
**injected `Embedder` Protocol**, and Optimize additionally takes an injected `SentenceMerger`; both run
under deterministic fakes in tests. These two — the embedder and the merge LLM — are the system's *only*
impure dependencies: the *functional core, imperative shell* split keeps both at the very edge (the
`utils` shell), and nothing else ever constructs a model client.

**Correctness by construction.** Where this spec states an invariant, the code makes violating it
impossible or makes the build fail — it is never left to reviewer vigilance. The IR round-trip is a
property test, layering is an `import-linter` contract, score vectors are length-checked against their
`Document`, the registry rejects duplicate or unresolved names at startup, and `apply` refuses any edit
that would empty a `Section` or the `Document`. The mechanisms are collected in [Invariants & enforcement](#invariants--enforcement).

**Unix philosophy, readable code.** Small single-purpose verbs, text in and text out. Prefer the
simplest design that reads clearly over a clever one; each piece does one thing well and composes
with the rest.

## Project layout

The folder structure is part of the design: every layer is its own folder, so the dependency order is
visible in the tree — `cli` wraps `ops`, the `ops` pipe chains the standalone `features`, and everything
builds on the imperative shell in `utils` and the stable contract in `ir/`. It separates that contract
from the swappable strategies and makes the plugin convention visible.

```
src/
  alexandria/
    __init__.py         # public API: represent, score, optimize, select, compare, diffs, propose, reduce, score_report, optimization_report, Document, …
    __main__.py         # `python -m alexandria` → the CLI
    cli/                # layer 1 — thin wrapper; verbs represent / score / optimize / select / compare / reduce
      main.py           #   click commands; parse args, move text in/out, call the library
      envelope.py       #   the JSON wire envelopes between piped verbs (schema_version=1)
      interactive.py    #   reduce --interactive: ReviewState machine + render + the getchar review loop
    ops/                # layer 2 — the library body
      pipe.py           #   chain the phases end to end — reduce / propose / score_report + ReduceResult / Proposal
      report.py         #   benchmark reporting: OptimizationReport / ReportComparison + optimization_report / compare_reports
      features/         #   the standalone features; each imports ir, plus utils only for the default embedder/merger
        represent.py    #     phase 1 — prompt → Document (split + tiktoken + injected Embedder)
        score.py        #     phase 2 — @register_scorer("redundancy"); score(...) -> Scores, score_rows
        optimize.py     #     phase 3 — @register_optimizer("merge_rewrite", requires=("redundancy",))
        select.py       #     phase 4 — @register_selector("least_cos_sim_diff"); fold a Plan → Selection
        target.py       #     hard-target merge search: merge_to_target (used by reduce when a max-tokens target is required)
        compare.py      #     compare(original, edited) → CompareResult (similarity + token reduction)
        diff.py         #     diffs(document, plan) → displayable Diffs, one per candidate, confidence order
    utils/              # layer 3 — the imperative shell
      embedders.py      #   Embedder implementations; the only place an embedding model is built (default_embedder)
      merger.py         #   SentenceMerger implementations; the only place the merge LLM is built (default_merger)
    ir/                 # layer 4 — the contract; depends on nothing else in alexandria
      document.py       #   Document / Section / Sentence + validation + Document.apply (the only rewrite)
      contracts.py      #   Scorer/Optimizer/Selector/Embedder/SentenceMerger Protocols + Scores/Params/SentenceId/Edit/Candidate/Plan/Selection/Diff
      registry.py       #   @register_* (rejects dup names) + lookup + requires-check
      similarity.py     #   cosine helpers shared by score, optimize, and select
pyproject.toml          # import-linter layering contracts
tests/                  # end-to-end pipeline tests; unit tests are co-located as *_test.py
```

## The intermediate representation

The IR is the spine every phase shares: a `Document` → `Section` → `Sentence` tree. Every level
exposes the same trio — `text`, `token_count`, `embedding` — so the same unit is reused at each
granularity, and all three are **stored** because each `embedding` is a real model embedding (see
below). A sentence's group is self-evident from where it sits in the tree. We parse, don't construct
(validate on build) and make impossible states unrepresentable: nesting replaces parallel lists, so
columns cannot fall out of sync.

```python
type Embedding = Annotated[NDArray[np.float32], PlainValidator(_as_vector)]   # a 1-D vector
type TokenCount = Annotated[int, Field(ge=0)]
SentenceId = NewType("SentenceId", str)   # content-addressed id (sha256 of the sentence text); how an Edit names a row across edits

class SectionKind(StrEnum):
    """Where a Section came from. The Document is the root, so every Section has one of these."""
    MARKDOWN = "markdown"   # a markdown header
    XML = "xml"             # an XML tag block
    PLAIN = "plain"         # body text under no header or tag (a preamble or a structureless prompt)

class Encoded(BaseModel):
    """Shared base: text plus the encodings computed from it, stored at every level."""
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    text: str                # round-trips: concatenating children reproduces it exactly
    token_count: TokenCount  # sum of the sentence counts beneath
    embedding: Embedding     # embed(text) — a real embedding of the reconstructed text

class Sentence(Encoded):
    """The atomic instruction and keep/drop unit; split no further."""
    node: Literal["sentence"] = "sentence"   # discriminates a child Sentence from a child Section
    id: SentenceId            # content-hash handle an Edit addresses; identical text keeps its id across re-represent and Parquet

class Section(Encoded):
    """A markdown header / XML block / plain run. Membership is the nesting itself; sections nest."""
    node: Literal["section"] = "section"     # discriminator (mirrors Edit's `op`)
    kind: SectionKind                            # markdown / xml / plain
    header: str                                  # markdown: header text, xml: tag name, plain: ""
    children: tuple[Node, ...] = Field(min_length=1)   # Sentences and sub-Sections, interleaved

    @property
    def sentences(self) -> tuple[Sentence, ...]:
        """Every descendant Sentence, recursively, in document order."""

type Node = Annotated[Sentence | Section, Field(discriminator="node")]

class Document(Encoded):
    """The whole prompt — the apex of the IR; its sections are the top level (it is the root)."""
    embedding_model: str                         # id of the model that produced every embedding
    sections: tuple[Section, ...] = Field(min_length=1)

    @property
    def sentences(self) -> tuple[Sentence, ...]:
        """Every section's descendants flattened in document order — the row axis for scores."""
        return tuple(s for section in self.sections for s in section.sentences)
```

**The trio, every level.** `text` is the **lossless** concatenation of child texts — sentences keep
their original separators, so `"".join` rebuilds the section and joining sections rebuilds the whole
prompt verbatim (encode/decode round-trip). `token_count` is the **sum** of the sentence counts
beneath; it is additive by definition, while BPE is not, so this is the summed-leaf count, not a
re-tokenization of the whole text. `embedding` is a **fresh embedding of this node's reconstructed
`text`** — not a mean of child vectors — so a `Section` / `Document` embedding is the model's real
representation of that span. Because re-embedding needs a model call, the trio is computed once in
Represent and stored at every level.

**Validated on build.** Constructing the tree checks that each node's `text` equals the concatenation
of its children and its `token_count` equals their sum (so stored values can never diverge from the
children), that every embedding shares one dimension, that token counts are non-negative
(`TokenCount`), that no `Section` or `Document` is empty (`min_length=1`), and that every `Sentence`
`id` is unique within the `Document` (an `Edit` must name exactly one row, never two).

**Content-addressed ids.** A `Sentence` id is `s` + the first 12 hex chars of the sha256 of its exact
`text`, so identical text always maps to the same id — re-representing a lightly edited prompt keeps
every unchanged line's id, and only edited lines get new ids (a whitespace change is a text change, so
it shifts the id too). When the same text appears more than once, occurrences after the first take a
`-2`, `-3`, … suffix in document order, keeping ids unique and deterministic.

**One model per Document.** A `Document` records the `embedding_model` that produced its vectors.
`redundancy` and the optimizers refuse to cosine-compare vectors across model ids — an embedder whose
`model_id` differs from the `Document`'s is rejected — so the impure embedding input can never silently
mix two incompatible embedding spaces. The id round-trips through Parquet, so a loaded `Document` is
reproducible.

**Row alignment.** Scorers rate `Document.sentences` — the flattened, document-order tuple — so the
`list[float]` a scorer returns is row-aligned to it, and `score()` validates its length against
`len(document.sentences)` before keying it by sentence `id`. An optimizer therefore receives its scores
as a **name-keyed, id-keyed bundle**, `Scores = dict[str, dict[SentenceId, float]]` — scorer name → each
sentence's `id` → its score — and looks each score up by `id`, so an edit stays valid as earlier edits
reshape the document. A short or misaligned vector is a construction error at scoring time, never a
silent off-by-one downstream. Nothing depends on how the embeddings or counts were produced, so a
different embedding model or tokenizer can be dropped in later without touching Score or Optimize.

## Phase 1 — Represent

`represent(prompt) -> Document` builds the IR in two internal steps:

1. **`split`** groups the prompt's lines into a **nested section tree**: a markdown header opens a
   section that nests by `#` depth, a standalone-line XML tag opens a section that nests by tag
   stack (the block is a hard boundary), and text under no header or tag becomes a `plain` section
   (a preamble, or a structureless prompt as one section). The header line and the XML tags stay as
   `Sentence`s, so the split is **lossless**: concatenating every leaf `Sentence`'s text in document
   order reproduces the exact input.
2. **`encode`** tokenizes each sentence with `tiktoken` and embeds it, then embeds every `Section`'s
   (including nested) and the `Document`'s reconstructed `text`, and assembles the validated tree.

These are sub-steps, not phases — you always run them together to get a `Document`, and only the
`Document` crosses into Score.

**Injected embedder.** `encode` never hard-wires a model: it calls an `Embedder` Protocol passed into
`represent` — and when no embedder is passed, `represent` builds the cached default (OpenAI
`text-embedding-3-small`, which needs an API key) so `reduce(prompt)` works with no wiring. Tests pass a
deterministic fake (`HashEmbedder`). The embedder is one of the system's two impure boundaries — `ir`,
the `features`, and `pipe` never import `openai` directly; the model backend lives in the `utils` shell,
and `ops` reaches it only through the default embedder (the layering linter forbids direct imports; see
[Invariants & enforcement](#invariants--enforcement)). The model's id is stamped onto the `Document` as
`embedding_model`.

**Batched encoding.** Embedding dominates cost, so `encode` batches: one embed call for every leaf
sentence, then one more for every section text plus the document text. (Cross-`Document` incremental
reuse — carrying unchanged nodes' embeddings forward from a prior `Document` — is not implemented; each
`represent` embeds the prompt from scratch.)

## Phase 2 — Score

A scorer rates every sentence, returning a `list[float]` row-aligned to `Document.sentences`.
High score ⇒ stronger optimization candidate.

**The first scorer — `redundancy`.** An instruction's redundancy is its similarity to the *most*
similar other instruction (cosine similarity over embeddings). A high score means a near-duplicate
exists. This is fully label-free.

```
emb = stack(s.embedding for s in document.sentences)  # (n, dim), document order
sim = cosine_similarity_matrix(emb)                    # (n, n)
score[i] = max(sim[i][j] for j != i)
```

**Max, not mean.** Redundancy is about duplication: an instruction that duplicates exactly one
other is redundant even if unlike everything else, so we take the max over other instructions, not
the mean. A mean-similarity score measures *genericness / centrality* — a real signal, but a
**different** scorer.

**Scope.** By default `redundancy` compares each sentence against every other in the document.
Because the IR makes group membership explicit, a scorer can also take `scope="within_section"` to
confine the comparison to sibling sentences — turning a `Section` into a scoring boundary — with no
change to the IR. Document-wide is the v1 default.

**Many scorers, one bundle.** The registry holds any number of scorers; `redundancy` is only the
first. When several run, their outputs travel together as a `Scores` bundle —
`dict[str, dict[SentenceId, float]]` keyed by scorer name then by sentence `id`, each scorer's
`list[float]` length-validated against `Document.sentences` before it is keyed. A scorer never knows
about the others; the bundle is just how the next phase receives whichever ones it asked for.

**Why scoring is built first.** It stands alone: any well-known prompt (Anthropic's system prompts,
popular Agent Skills) can be scored as-is, with no optimizer or benchmark yet. That de-risks the
rest of the project.

## Phase 3 — Optimize

An optimizer reads a scored `Document` and **proposes and ranks edits**:
`(Document, Scores, Embedder, SentenceMerger, Params, ReductionReporter) -> Plan`, never mutating the
input and never reconstructing the surviving `Document`. It may re-embed — but only to *probe* a
candidate's whole-document `cos_sim_diff` so it can feed the merge model a retry (a per-edit quality check); the
cumulative `cos_sim_diff` budget that decides how many edits survive is still Select's job. Each proposal is a
pure, serializable **`Edit`** wrapped with the `confidence` that ranked it and a short reason:

```python
class Delete(BaseModel):
    """Drop the addressed sentences."""
    model_config = ConfigDict(frozen=True)
    op: Literal["delete"] = "delete"
    targets: tuple[SentenceId, ...]    # ids of the sentences this edit removes (the cluster)

class Replace(BaseModel):
    """Rewrite the first target and remove any remaining merged targets."""
    model_config = ConfigDict(frozen=True)
    op: Literal["replace"] = "replace"
    targets: tuple[SentenceId, ...]    # ids of the cluster being collapsed (kept at the first)
    replacement: Encoded               # merged text + its token count and embedding, precomputed at plan time

type Edit = Annotated[Delete | Replace, Field(discriminator="op")]   # no nullable field to keep in sync

class Candidate(BaseModel):
    """A proposed Edit plus the ranking metadata Select orders by and review shows."""
    edit: Edit
    confidence: float   # pair similarity that ranked it; review shows it, Select orders by measured cos_sim_diff
    source: str         # the optimizer that proposed it
    reason: str         # human-readable rationale, shown during review

type Plan = tuple[Candidate, ...]   # an ordered stack Select folds, each edit on the prior result
```

Returning a `Plan` of edits instead of a `Document` is what lets several optimizers run at once and
keeps a human in the loop: a stack is data you can **concatenate, rank, review, and serialize**, where
a reconstructed `Document` is not. The single pure primitive `Document.apply(candidate) -> Document | None`
(a method on the IR) rebuilds a smaller `Document` for one candidate, and a selector **folds the stack** by
calling it once per candidate — each edit applying to the `Document` the previous edit produced, the way
`sed` streams commands; it is the *only* place the tree is rewritten. Because every edit addresses its
targets by stable `id` rather than by row position, an edit stays valid as earlier edits reshape the
document — there is no index-shift hazard even though the stack applies in sequence, and an edit whose
targets an earlier edit already removed is simply skipped (`apply` returns the document unchanged).
`diffs(document, plan)` renders the stack as displayable per-candidate diffs for review and audit.

An optimizer **declares the scorer(s) it needs** when it registers (`requires=(...)`); the pipeline
runs exactly those, assembles the `Scores` bundle, and hands it over. So
`reduce --optimizer merge_rewrite` runs `redundancy` on its own — the optimizer names its
dependency instead of the caller wiring it up.

**The first optimizer — `merge_rewrite` (over `redundancy`).** Registered as
`@register_optimizer("merge_rewrite", requires=("redundancy",))`, it ranks near-duplicate pairs by their
cosine similarity and, for each pair, asks the injected `SentenceMerger` (an LLM) to **rewrite both
sentences into one**, kept at the first occurrence:

```
for (a, b) in pairs with similarity ≥ threshold, sorted by similarity desc, both still present:
    for up to MAX_MERGE_ATTEMPTS:
        merged = merger.merge(a.text, b.text, feedback)          # LLM rewrite; feedback on retry
        if merged ≈ a (cosine ≥ 0.99): candidate = Delete(b)     # a already covers both
        elif tokens(merged) < tokens(a)+tokens(b): candidate = Replace((a,b), merged)
        else: feedback = "make it shorter"; continue
        cos_sim_diff = compute_cos_sim_diff(embed(apply(candidate).text), document.embedding)  # probe whole-doc cos_sim_diff
        if cos_sim_diff ≤ cos_sim_diff_budget: emit candidate; break
        feedback = "cos_sim_diff exceeded the budget; preserve more meaning"  # retry with feedback
```

An exact-duplicate pair is deleted without a model call. A `present` set keeps the proposals coherent —
after a `Delete` the unchanged first sentence stays pairable (so a triple duplicate collapses fully),
while a rewritten pair consumes both ids. The per-edit `cos_sim_diff ≤ cos_sim_diff_budget` probe is a **quality
gate on each rewrite** (it feeds the merger a retry), *not* the compression stop condition — Select
still decides, cumulatively, how much of the ranked stack survives.

Collapsing a cluster into one representative instruction is exactly the `Replace` op — a single edit
over the cluster's ids that the `delete` op alone cannot express. The `Edit` algebra is what gives the
merge a home without a second mutation path: it is still an edit that stacks and reviews like any other.

**Running several at once.** Every optimizer reads the same original `Document` and returns a stack, so
`reduce --optimizer A,B` **concatenates** both stacks into one series; **Select** then orders the whole by
each candidate's **measured whole-document `cos_sim_diff`** and folds it. No dedup step is needed: two candidates
that target the same cluster apply in that order, and the second — finding its targets already removed —
is skipped. Concatenating once and folding in order stays sound because edits address stable ids and
`apply` re-checks each edit against the *current* document as it folds, so the ranking only *orders* the
stack while correctness is enforced step by step.

**Meaning preservation — the hard constraint.** It is enforced at two levels. (1) Only sentences marked
`optimizable` and paired above the similarity `threshold` are ever proposed; a unique, load-bearing
instruction never reaches the stack. (2) **Optimize** probes each rewrite's whole-document `cos_sim_diff` and
retries the merge model when `cos_sim_diff` exceeds `cos_sim_diff_budget` (a per-edit quality gate), and **Select**
re-embeds the reduced prompt after each accepted edit and keeps it only while its `cos_sim_diff` from
the original prompt stays within `cos_sim_diff_budget` — the cumulative stop condition (see
[Phase 4 — Select](#phase-4--select)). As it folds, `apply` additionally rejects any edit that would empty a
`Section` or the `Document` — the structural half of the guarantee, re-checked at each step against the
current document rather than a stale snapshot.

## Phase 4 — Select

A *selector* turns a ranked `Plan` into a `Selection` (the reduced `Document` plus the candidates it
applied): `(Document, Plan, Embedder, Params) -> Selection`.
Optimize decides *what could change*; Select decides *which changes
actually happen* and *when to stop*. Keeping that controller in one phase is why the cumulative `cos_sim_diff`
budget lives in one place instead of being threaded through every optimizer.

**The default selector — `least_cos_sim_diff`.** It ranks candidates by the whole-document `cos_sim_diff`
each would cause on its own, then applies them in **ascending `cos_sim_diff` order**, accepting each only
while the reduced prompt stays within the cumulative `cos_sim_diff` budget:

```
base = Document.embedding                       # the real embedding of the original prompt
trials = [(c, t) for c in plan if (t := document.apply(c)) is not None and t is not document]
ranked = sort trials by compute_cos_sim_diff(embed(t.text), base)   # one batched embed call
current = Document
for candidate in ranked:                        # ascending cos_sim_diff
    trial = current.apply(candidate)            # None/unchanged if targets already gone → skip
    if trial is None or trial is current: continue
    if compute_cos_sim_diff(embed(trial.text), base) > params.cos_sim_diff_budget: continue   # cumulative
    current = trial
    if params.max_tokens is not None and current.token_count ≤ params.max_tokens: break
return Selection(document=current, applied=…)
```

The `cos_sim_diff` is measured against `base` after *every* accepted edit, so the budget is **cumulative**:
the loop compresses as far as `cos_sim_diff_budget` allows and then stops. Applying candidates in ascending
`cos_sim_diff` order is self-regulating — once one half of a redundant pair is dropped, removing its peer would
delete genuinely
unique content, exceed the `cos_sim_diff` budget, and be skipped — so `least_cos_sim_diff` keeps one copy of a cluster
without any explicit "don't delete both" rule. `Params` carries both phases' knobs (`threshold` for
Optimize's eligibility, `cos_sim_diff_budget` for both the per-edit probe and the cumulative stop, and an
optional `max_tokens` target); the defaults live in one frozen model (`cos_sim_diff_budget` defaults to `0.5`).

Because `least_cos_sim_diff` re-embeds the reduced prompt, the budget reflects the embedding model's real
representation of the shortened text. With a non-semantic embedder (the deterministic `HashEmbedder`
test backend, which hashes the whole string) any edit re-embeds to an unrelated vector, so a faithful
test sets a generous budget to exercise deletion and a tight one to exercise the stop condition.

**A future selector — `review`.** The same ranked stack, folded interactively: `review` walks candidates
and asks a human to accept or reject each, then returns the reduced
`Document`. `least_cos_sim_diff` and `review` are the same phase with different stop logic — automatic budget
versus a person — which is exactly why Select is its own pluggable phase rather than a branch inside `reduce`.

**The fold primitive — `Document.apply` (a method on the IR).** `document.apply(candidate)` returns a
smaller rebuilt `Document`, returns the document unchanged when the candidate's targets are already gone,
and returns `None` when the edit would empty a `Section` or the `Document` — so a selector folds a plan by
calling it once per candidate and skips (rather than aborts) on `None`. It addresses targets by stable
`id`, so an edit whose targets an earlier edit already removed is simply skipped. This is the only place
the IR tree is rewritten.

## Extending a phase: scorers, optimizers, and selectors

Score, Optimize, and Select are open sets. Adding a strategy follows one convention for all three, so
every addition looks the same:

1. **Add its function** to that feature's module (`ops/features/score.py`, `optimize.py`, or `select.py`).
2. **Implement the phase `Protocol`.** All live in `ir/contracts.py`, alongside the data they
   exchange:

   ```python
   type Scores = dict[str, dict[SentenceId, float]]   # scorer name → sentence id → length-validated score
   type Plan   = tuple[Candidate, ...]                # an ordered stack of edit candidates

   class Scorer(Protocol):
       def __call__(self, document: Document) -> list[float]: ...

   class Optimizer(Protocol):
       def __call__(self, document: Document, scores: Scores, embedder: Embedder,
                    merger: SentenceMerger, params: Params, reporter: ReductionReporter) -> Plan: ...

   class Selector(Protocol):
       def __call__(self, document: Document, plan: Plan, embedder: Embedder, params: Params) -> Selection: ...
   ```

3. **Self-register by name** with `@register_scorer` / `@register_optimizer` / `@register_selector`.
   Registration rejects a duplicate name at import time, and an optimizer additionally declares the
   scorer(s) it consumes — validated against the registry at startup, so an unknown `requires` is a clear
   error before any work runs, not a `KeyError` mid-pipeline:

   ```python
   # ops/features/score.py
   @register_scorer("centrality")
   def centrality(document: Document) -> list[float]:
       ...

   # ops/features/optimize.py
   @register_optimizer("merge_rewrite", requires=("redundancy",))
   def merge_rewrite(document: Document, scores: Scores, embedder: Embedder,
                     merger: SentenceMerger, params: Params, reporter: ReductionReporter) -> Plan:
       ...   # return a tuple of Candidate, never a Document

   # ops/features/select.py
   @register_selector("least_cos_sim_diff")
   def least_cos_sim_diff(document: Document, plan: Plan, embedder: Embedder, params: Params) -> Selection:
       ...   # fold the plan within params.cos_sim_diff_budget; return the Selection
   ```

4. **Done.** The CLI resolves `--scorer NAME` / `--optimizer NAME` against the registry; for an
   optimizer the pipeline runs its `requires` scorers and builds the `Scores` bundle automatically.
   No existing file is edited, the IR is untouched, and the new strategy is testable in isolation by
   feeding a small `Document` fixture and asserting on the returned scores or `Plan` — no model, no
   mocks.

**Third-party strategies (future extension — not implemented).** Registration today is a pure import
side effect: importing a strategy's module runs its `@register_*` decorator, and the built-ins are
imported when their `ops.features` module loads. There is **no** entry-point discovery — no
`load_plugins()`, no `[project.entry-points."alexandria.*"]` in `pyproject.toml`. A future extension
could wire the registry to Python entry points so an out-of-tree package's strategy is discovered at
startup without editing Alexandria; it would look like:

```toml
# a future third-party package's pyproject.toml — NOT read by Alexandria today
[project.entry-points."alexandria.optimizers"]
my_optimizer = "my_package.my_optimizer:my_optimizer"
```

Because the shape never varies, a reviewer's bar for "clean" is mechanical: one file, one
`Protocol` implementation, one `@register`.

## Invariants & enforcement

The design principles are only as strong as their enforcement. Each invariant below is wired to a
tool, so breaking it fails a test or the build — none of it relies on review catching it.

**Round-trip law (property-tested).** At every level, decode∘encode is the identity: `"".join` over a
node's children reproduces its `text` verbatim, and `save` / `load` round-trips a `Document`
exactly, embeddings included. These are **Hypothesis** properties over generated `Document`s, not a
handful of fixtures — `tests/strategies.py` ships the generators (valid trees, plus deliberately
malformed ones for negative tests), so the laws are checked against thousands of shapes.

**Layering (`import-linter`).** The dependency rule the layout describes is enforced by five
`import-linter` contracts run in CI: (1) *Layered architecture* — `ir` is the shared contract every
layer may import; among `cli`, `ops`, and `utils`, imports are adjacent-only (`cli` → `ops` → `utils`),
and the layers type forbids the reverse so `ir` imports nothing else in `alexandria`; (2) *CLI reaches
utils only through ops* — a **forbidden** contract closing the one gap the layers type cannot express,
barring `cli` from importing `utils` directly while indirect `cli` → `ops` → `utils` chains stay legal;
(3) *Pipe composes features* — within `ops`, `pipe` may import the standalone `features` to chain them,
never the reverse; (4) *Features are independent* — each feature (`represent`, `score`, `optimize`,
`select`, `target`, `compare`, `diff`) is a standalone capability and may not import a sibling, so shared
types live in `ir.contracts`; and (5) *OpenAI client
isolated to the shell* — a **forbidden** contract barring `ir`, `ops`, and `cli` from importing
`openai`, so only the `utils` shell (`embedders.py` and `merger.py`) builds a model client. A cycle or a
layering violation fails the build — *modularity through contracts* and *library first* cannot quietly rot.

**Functional core, imperative shell.** The embedder is injected into Represent and Select, and the
`SentenceMerger` into Optimize, each behind a `Protocol`; the concrete model clients live only in the
shell. Tests pass deterministic fakes, making the phases — and therefore the whole pipeline — pure
functions with no mocks.

**Model identity.** A `Document` records the `embedding_model` that produced it, and `redundancy` and the
optimizers refuse to compare vectors from a different model (an embedder whose `model_id` differs is
rejected), so the impure embedding input cannot silently poison a similarity score; the id round-trips
through Parquet for reproducibility.

**Row-aligned scores.** A scorer's `list[float]` output must have length `len(document.sentences)` or
`score()` raises before keying it by sentence `id` into the `Scores` bundle. A misaligned or short vector
is a construction error, not a silent off-by-one.

**Fail-fast registry.** `@register` rejects a duplicate name at import time, and resolving an
optimizer validates its `requires=(...)` against the registered scorers at startup — an unknown
scorer is a clear error before any work runs.

**Edit laws (property-tested).** Folding a stack never increases the `Document`'s token count; an edit
addressing an already-removed `id` leaves the document unchanged, not misapplied; and `apply` refuses any
edit that would empty a `Section` or the `Document`. Because edits address stable ids, **non-overlapping
edits are order-independent** — folding them in any order yields the same `Document`. These are Hypothesis
properties over generated documents and stacks.

## Persistence — Parquet

**Status.** The wire and on-disk format today is **JSON** (pydantic native, zero extra dependencies):
a `Document` serializes with `model_dump_json` — every `embedding` becomes a float list — and the phase
verbs pass `DocumentEnvelope` / `ScoredEnvelope` / `PlanEnvelope`, each stamped `schema_version=1`, over
stdin/stdout. The float32 → JSON double → float32 round-trip is exact. The Parquet table below is the
**planned** format for when data volume demands it (dense-vector density and Arrow interop); it adds a
`polars` dependency and is deferred under YAGNI until then.

The tree is the in-memory view; on disk it flattens to a single **Parquet** table, **one row per
node** (columns `level` — `document` / `section` / `sentence` — plus `text`, `token_count`,
`embedding`, a `parent` id and `order` (which together encode the nesting at any depth), the
section-level `kind` and `header`, the sentence-level `id`, and the document-level `embedding_model`).
Every level's `embedding` is stored, because each is a real model embedding that a mean cannot
reconstruct; `load` rebuilds the tree by linking each node to its `parent` in `order`, attaches each
stored `Section` / `Document` embedding, and restores `embedding_model`, so `load(path) -> Document`
needs no model and `save(Document, path)` / `load(path)` round-trip exactly. Because a `Plan` is plain
serializable data and every `Edit` addresses sentences by their stable `id`, it can be saved beside the
`Document` and replayed against the reloaded tree — its ids still match — without re-running an optimizer.

Parquet beats JSON/JSONL here: embeddings are dense `float32` vectors that JSON bloats ~3–5× and
rounds lossily, whereas Parquet stores them as typed fixed-size lists, compresses them, and is read
and written natively by Polars (already our table tool) and any Arrow-aware reader in other
languages. Human-debuggable text output (no embeddings) stays a CLI concern (`--json`), not the
storage format.

## CLI

A few single-purpose verbs, built with [`click`](https://github.com/pallets/click). Each reads from a
`FILE` argument or stdin and writes its result to stdout; diagnostics go to stderr; exit `0` on success,
`1` on a known boundary error (bad envelope, unknown strategy name, empty prompt). Each of the four
phases is its own verb, and they **compose over a Unix pipe**: every phase emits a self-contained JSON
**envelope** carrying the `Document` (and, downstream, the `Scores` or `Plan`) the next phase needs.

- `alexandria represent [FILE] [--model MODEL] [--out PATH]` — raw prompt in, a **`DocumentEnvelope`** (JSON) out.
- `alexandria score [FILE] [--scorer NAME[,NAME...]] [--table] [--out PATH]` — `DocumentEnvelope` in, a
  **`ScoredEnvelope`** out; `--table` prints a human-readable per-instruction report instead. Run
  several scorers and their columns are the `Scores` bundle. No embedder is needed — it scores the
  Document it is handed.
- `alexandria optimize [FILE] [--optimizer NAME[,NAME...]] [--threshold T] [--out PATH]` — `ScoredEnvelope` in, a
  **`PlanEnvelope`** out. Pass several optimizers and their stacks concatenate into one series.
- `alexandria select [FILE] [--cos-sim-diff-budget D] [--json]` — `PlanEnvelope` in, the
  **reduced prompt** out (or a JSON reduction summary with `--json`). The `least_cos_sim_diff` selector folds the
  `Plan` in ascending whole-document `cos_sim_diff` order while the prompt stays within `--cos-sim-diff-budget` (default
  `0.5`) of the original.
- `alexandria reduce [FILE] [--optimizer NAME[,NAME...]] [--selector NAME] [--threshold T] [--cos-sim-diff-budget D] [--model MODEL] [--json]`
  — prompt in, **reduced prompt out**, running all four phases in one process. This is the headline
  path and the fast in-process route; `--json` emits the same reduction summary as `select --json`
  (`text`, `applied`, `source_tokens`, `reduced_tokens`). There is no `--scorer` flag because each
  chosen optimizer declares the scorer(s) it needs.

`--out PATH` saves the JSON envelope while preserving stdout for a pipe; the next compatible phase can
read that saved file as its `FILE` argument. See the [CLI guide](cli.md) for user-facing examples and
the complete workflow reference.

So `alexandria represent < p.txt | alexandria score | alexandria optimize | alexandria select`
reproduces `alexandria reduce < p.txt`, and any prefix of that pipe is a useful stop
(`... | alexandria score --table` to inspect redundancy). A future interactive `review` selector walks
the same `Plan` and asks a human to accept or reject each drop.

**On composing via pipes.** Phase composition is first a *library* property — pure functions over the
in-memory `Document` — and the envelopes make it a *process* property too. The envelope is
self-contained: it always carries the `Document` a downstream phase needs, so no phase re-derives it,
and `score` needs no model at all (`optimize` and `select` build the default embedder — and `optimize`
the merger — since they measure `cos_sim_diff` and rewrite). Envelopes are JSON today (`schema_version=1`, pydantic
native, zero extra dependencies); a Parquet path is added later when data volume asks for it (see
[Persistence](#persistence--parquet)). The `reduce` verb stays the fast route that skips serialization
between phases.

## Evaluation — a separate concern

Benchmarking measures how well the three phases do their job; it is not one of them. It runs the
full pipeline over a fixed corpus of known prompts and reports:

- token reduction (via `tiktoken`, stated as a model-independent proxy),
- redundancy score distribution before and after.

Results are tabulated with Polars. The phases are designed to make this easy — they are pure
functions over the IR, so the benchmark just calls them — but the benchmark lives in its own
package and never sits in the optimization path.

## Non-goals (YAGNI)

- No fine-tuning or labeled optimization.
- No entry-point plugin discovery yet — strategies register by import side effect (see
  [Extending a phase](#extending-a-phase-scorers-optimizers-and-selectors)); out-of-tree discovery is a
  future extension.
- No cross-`Document` incremental re-embedding — each `represent` embeds from scratch.
