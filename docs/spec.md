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
| **1. Represent** | Turn the prompt into the IR: split into instructions, count tokens, embed each one. | `represent(str, *, embedder: Embedder = ..., reuse: Document \| None = None) -> Document` |
| **2. Score** | Rate every instruction. The first scorer rates **redundancy**. | `Scorer = (Document) -> list[float]` |
| **3. Optimize** | Propose and rank the edits the scores justify. | `Optimizer = (Document, Scores, Params) -> Plan` |
| **4. Select** | Apply candidates highest-confidence-first while the prompt stays within the drift budget. | `Selector = (Document, Plan, Embedder, Params) -> Document` |

Each phase is a pure function over the IR, so each is tested on its own and the four compose in
order. The split between **Optimize** and **Select** is deliberate: Optimize only *proposes and ranks*
candidates (it never decides how many survive or re-embeds anything), and Select is the controller that
*chooses and applies* — keeping the "which edits, in what order, and when to stop" decision in one place.

**Score** and **Optimize** are *open sets*, not single steps: many scorers and many optimizers coexist
behind a registry, an optimizer **declares which scorer(s) it consumes**, and the scores it receives travel
as a name-keyed **bundle** (`Scores = dict[str, ScoreVector]`) so one optimizer can read several scorers at
once. An optimizer does not return a `Document`; it returns a **`Plan`** — an ordered **stack** of edit
candidates, each carrying the `confidence` that ranked it — so several optimizers can run at once and their
stacks **concatenate** into one series.

**Select** is an open set too: a *selector* reads the `Plan` and folds it into a smaller `Document`. The
default `auto` selector sorts candidates by `confidence` (highest first) and applies each only while the
reduced prompt stays within `drift_budget` cosine distance of the original prompt embedding — so an
unattended `reduce` compresses as far as the meaning budget allows. A future interactive selector (the
`review` verb) walks the same ranked stack and asks a human to accept or reject each edit. Both call one
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
hardcoded pairing. A strategy registers by name with `@register`, and the same registry is wired to
Python **entry points**, so a third party can ship a scorer or optimizer from its own package and the
registry discovers it with no edit to Alexandria. Optimizers return a **`Plan`** rather than a
`Document`, so the stacks of several optimizers concatenate into one ordered series that a human (or the
pipeline) folds, deciding what to apply.

**Library first; the CLI is a thin wrapper.** All behavior lives in importable, pure functions
over the IR. The CLI only parses arguments, moves text in and out, and calls the library. Anything
you can do from the terminal you can do from `import alexandria`, and the CLI adds no capability
the library lacks.

**Testable without mocks — by design.** Phases are pure functions over plain data: feed a known
`Document` in, get deterministic scores, a `Plan`, or a new `Document` out, assert on the real value. Tests
build small `Document` fixtures directly — a handful of sentences with tiny hand-written embeddings
— so Score and Optimize are tested end to end with no model in the loop. Represent takes the model as
an **injected `Embedder` Protocol**, so even it runs under a deterministic fake: the *functional core,
imperative shell* split keeps the one impure dependency at the very edge, and nothing else ever
imports a model.

**Correctness by construction.** Where this spec states an invariant, the code makes violating it
impossible or makes the build fail — it is never left to reviewer vigilance. The IR round-trip is a
property test, layering is an `import-linter` contract, score vectors are length-checked against their
`Document`, the registry rejects duplicate or unresolved names at startup, and every `Edit` carries
its own inverse. The mechanisms are collected in [Invariants & enforcement](#invariants--enforcement).

**Unix philosophy, readable code.** Small single-purpose verbs, text in and text out. Prefer the
simplest design that reads clearly over a clever one; each piece does one thing well and composes
with the rest.

## Project layout

The folder structure is part of the design: every layer is its own folder, so the dependency order is
visible in the tree — `cli` wraps `ops`, the `ops` pipe drives the four `features`, and everything
builds on the imperative shell in `utils` and the stable contract in `ir/`. It separates that contract
from the swappable strategies and makes the plugin convention visible.

```
src/
  alexandria/
    __init__.py         # public API: represent, score, optimize, select, reduce, score_report, Document
    __main__.py         # `python -m alexandria` → the CLI
    cli/                # layer 1 — thin wrapper; verbs represent / score / optimize / select / reduce
      main.py           #   click commands; parse args, move text in/out, call the library
      envelope.py       #   the JSON wire envelopes between piped verbs (schema_version=1)
    ops/                # layer 2 — the library body
      pipe.py           #   compose the four features end to end — reduce / score_report
      features/         #   the four pluggable features; each imports only ir
        represent.py    #     phase 1 — prompt → Document (split + tiktoken + injected Embedder)
        score.py        #     phase 2 — @register_scorer("redundancy"); score(...) -> Scores, score_rows
        optimize.py     #     phase 3 — @register_optimizer("greedy_pairwise", requires=("redundancy",))
        select.py       #     phase 4 — @register_selector("auto"); fold a Plan → reduced Document
    utils/              # layer 3 — the imperative shell
      embedders.py      #   Embedder implementations; the only place a model is built (default_embedder)
    ir/                 # layer 4 — the contract; depends on nothing else in alexandria
      document.py       #   Document / Section / Sentence + validation + Document.apply (the only rewrite)
      contracts.py      #   Scorer/Optimizer/Selector/Embedder Protocols + Scores/Params/SentenceId/Edit/Candidate/Plan
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
columns cannot drift out of sync.

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
    id: SentenceId            # content-hash handle an Edit addresses; identical text keeps its id across re-represent, reuse=, and Parquet

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
of its children and its `token_count` equals their sum (so stored values can never drift from the
children), that every embedding shares one dimension, that token counts are non-negative
(`TokenCount`), that no `Section` or `Document` is empty (`min_length=1`), and that every `Sentence`
`id` is unique within the `Document` (an `Edit` must name exactly one row, never two).

**Content-addressed ids.** A `Sentence` id is `s` + the first 12 hex chars of the sha256 of its exact
`text`, so identical text always maps to the same id — re-representing a lightly edited prompt keeps
every unchanged line's id, and only edited lines get new ids (a whitespace change is a text change, so
it shifts the id too). When the same text appears more than once, occurrences after the first take a
`-2`, `-3`, … suffix in document order, keeping ids unique and deterministic.

**One model per Document.** A `Document` records the `embedding_model` that produced its vectors.
`reuse=` carries embeddings forward only when the model id matches, and `redundancy` refuses to
cosine-compare vectors across model ids — so the single impure input can never silently mix two
incompatible embedding spaces. The id round-trips through Parquet, so a loaded `Document` is
reproducible.

**Row alignment.** Scorers rate `Document.sentences` — the flattened, document-order tuple — so a
`list[float]` of scores is row-aligned to it; an optimizer reads that vector positionally but **names the
rows it edits by their stable `id`**, so an edit stays valid as earlier edits reshape the document. The pipeline parses each
scorer's output into a **`ScoreVector`** whose length is validated against `len(document.sentences)`,
and an optimizer receives one or more of them as a **score bundle**,
`Scores = dict[str, ScoreVector]` keyed by scorer name, every vector validated against the same
`Document` — a short or misaligned vector is a construction error, never a silent off-by-one
downstream. Nothing depends on how the embeddings or counts were produced, so a hosted embedding API,
a different model, or another tokenizer can be dropped in later without touching Score or Optimize.

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
`represent` — and when no embedder is passed, `represent` builds the cached default (`all-MiniLM-L6-v2`)
so `reduce(prompt)` works with no setup. Production wires in `sentence-transformers`; tests pass a
deterministic fake. This is the system's only impure boundary — `ir`, the `features`, and `pipe` never
import `sentence_transformers` directly; the model backend lives in the `utils` shell, and `ops` reaches
it only through the default embedder (the layering linter forbids direct imports; see
[Invariants & enforcement](#invariants--enforcement)). The model's id is stamped onto the `Document` as
`embedding_model`.

**Incremental encoding.** Embedding dominates cost, so `encode` is content-addressed at every level.
Given a prior `Document` via `reuse`, any node — `Sentence`, `Section`, or `Document` — whose `text`
is unchanged keeps its existing `token_count` and `embedding`; only new or edited texts are
re-tokenized and re-embedded. Re-running on a lightly edited prompt recomputes only what changed.
`reuse` applies only when the prior `Document`'s `embedding_model` matches the current embedder;
a model change forces a full re-embed.

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
`dict[str, ScoreVector]` keyed by scorer name, each vector validated row-aligned to
`Document.sentences`. A scorer never knows about the others; the bundle is just how the next phase
receives whichever ones it asked for.

**Why scoring is built first.** It stands alone: any well-known prompt (Anthropic's system prompts,
popular Agent Skills) can be scored as-is, with no optimizer or benchmark yet. That de-risks the
rest of the project.

## Phase 3 — Optimize

An optimizer reads a scored `Document` and **proposes and ranks edits**:
`(Document, Scores, Params) -> Plan`, never mutating the input, never reconstructing the `Document`, and
never re-embedding (that is Select's job). Each proposal is a pure, serializable **`Edit`** wrapped with the
`confidence` that ranked it and a short reason:

```python
class Delete(BaseModel):
    """Drop the addressed sentences."""
    model_config = ConfigDict(frozen=True)
    op: Literal["delete"] = "delete"
    targets: tuple[SentenceId, ...]    # ids of the sentences this edit removes (the cluster)

class Replace(BaseModel):
    """Collapse the addressed sentences into one new sentence."""
    model_config = ConfigDict(frozen=True)
    op: Literal["replace"] = "replace"
    targets: tuple[SentenceId, ...]    # ids of the cluster being collapsed
    replacement: str                   # text of the single sentence that replaces them

type Edit = Annotated[Delete | Replace, Field(discriminator="op")]   # no nullable field to keep in sync

class Candidate(BaseModel):
    """A proposed Edit plus the ranking metadata Select orders by and review shows."""
    edit: Edit
    confidence: float   # ranking key — high means a stronger candidate (Select applies high → low)
    source: str         # the optimizer that proposed it
    reason: str         # human-readable rationale, shown during review

type Plan = tuple[Candidate, ...]   # an ordered stack Select folds top-to-bottom, each edit on the prior result
```

Returning a `Plan` of edits instead of a `Document` is what lets several optimizers run at once and
keeps a human in the loop: a stack is data you can **concatenate, rank, review, serialize, and undo**, where
a reconstructed `Document` is not. The single pure primitive
`apply(document, edits) -> (Document, inverse)` (in the `ir` layer) **folds the stack left** — each edit applies to
the `Document` the previous edit produced, the way `sed` streams commands — and returns the **inverse stack**
alongside it; it is the *only* place the tree is rewritten. Because every edit addresses its targets by stable
`id` rather than by row position, an edit stays valid as earlier edits reshape the document — there is no
index-shift hazard even though the stack applies in sequence, and an edit whose targets an earlier edit
already removed is simply skipped. Because each `Edit` carries an inverse — a `delete`'s inverse restores the
removed `Sentence`s verbatim (their stored embeddings and ids included), a `replace`'s inverse puts the
original cluster back — **undo is just folding the inverse stack** (`apply(reduced, inverse)`), and
`render(document, plan)` shows the stack as a unified diff for review and audit.

An optimizer **declares the scorer(s) it needs** when it registers (`requires=(...)`); the pipeline
runs exactly those, assembles the `Scores` bundle, and hands it over. So
`reduce --optimizer greedy_pairwise` runs `redundancy` on its own — the optimizer names its
dependency instead of the caller wiring it up.

**The first optimizer — `greedy_pairwise` (over `redundancy`).** Registered as
`@register_optimizer("greedy_pairwise", requires=("redundancy",))`, it reads `scores["redundancy"]`.
For each near-duplicate pair it proposes dropping the **more redundant** of the two — the one whose
redundancy score is higher, i.e. the one most duplicated elsewhere — and ranks the candidate by the
pair's similarity (its `confidence`):

```
for (a, b) in pairs with similarity ≥ threshold, sorted by similarity desc:
    if redundancy[a] ≥ threshold and redundancy[b] ≥ threshold and both still un-proposed:
        drop = argmax(redundancy[a], redundancy[b]); keep the other
        emit Candidate(Delete(drop), confidence=similarity)
```

A per-pass `present` set keeps the proposals coherent — for three identical sentences it proposes two
deletes, never all three — so at least one copy of every cluster survives the proposal stage. The
optimizer does **not** re-embed and does **not** decide how far to compress; it hands Select a ranked,
non-conflicting stack and lets the drift budget decide how much of it to apply.

**`merge` (stretch).** Collapse a redundant cluster into one representative instruction — a single
`replace` `Edit` over the cluster's ids, which the `delete` op alone cannot express. The `Edit`
algebra is what gives `merge` a home without a second mutation path: it is still an edit that stacks,
reviews, and undoes like any other.

**Running several at once.** Every optimizer reads the same original `Document` and returns a stack, so
`reduce --optimizer A,B` **concatenates** both stacks into one series; **Select** then orders the whole by
`confidence` and folds it. No dedup step is needed: two candidates that target the same cluster apply in
confidence order, and the second — finding its targets already removed — is skipped. Concatenating once and
folding in order stays sound because edits address stable ids and `apply` re-checks each edit against the
*current* document as it folds, so `confidence` only *ranks* the stack while correctness is enforced step by
step.

**Meaning preservation — the hard constraint.** It is enforced at two points. (1) *Unique* sentences (no
peer above the redundancy threshold) are never proposed for deletion, so no optimizer can put a
load-bearing instruction on the stack. (2) Select re-embeds the reduced prompt after each tentative edit
and keeps the edit only while the prompt stays within `drift_budget` cosine distance of the original — so
even among redundant candidates, compression stops before meaning drifts too far (see
[Phase 4 — Select](#phase-4--select)). As it folds, `apply` additionally rejects any edit that would empty a
`Section` or the `Document` — the structural half of the guarantee, re-checked at each step against the
current document rather than a stale snapshot.

## Phase 4 — Select

A *selector* turns a ranked `Plan` into a reduced `Document`: `(Document, Plan, Embedder, Params) -> Document`.
Optimize decides *what could change* and *how confident* each change is; Select decides *which changes
actually happen* and *when to stop*. Keeping that controller in one phase is why `confidence` is finally
read here, and why the drift budget lives in one place instead of being threaded through every optimizer.

**The default selector — `auto`.** It applies candidates greedily, highest-confidence-first, accepting each
only while the reduced prompt stays within the drift budget:

```
base = Document.embedding                       # the real embedding of the original prompt
current = Document
for candidate in plan sorted by confidence desc:
    trial = current.apply(candidate)            # None if it would empty a Section/Document → skip
    if trial is None: continue
    drift = cosine_distance(embed(trial.text), base)   # re-embed the reduced prompt
    if drift ≤ params.drift_budget:             # default 0.01 — within 1% of the original
        current = trial                         # accept; the budget is cumulative from here
return current
```

The drift is measured against `base` after *every* accepted edit, so the budget is **cumulative**: the
loop compresses as far as 1% of meaning allows and then stops. This is self-regulating — once one half of
a redundant pair is dropped, removing its peer would delete genuinely unique content, drift past the
budget, and be skipped — so the `auto` selector keeps one copy of a cluster without any explicit
"don't delete both" rule. `Params` carries both phases' knobs (`threshold` for Optimize's eligibility,
`drift_budget` for Select's stop condition); `OptimizerParams`-style defaults live in one frozen model.

Because `auto` re-embeds the reduced prompt, the budget reflects the embedding model's real
representation of the shortened text. With a non-semantic embedder (the deterministic test backend, which
hashes the whole string) any edit re-embeds to an unrelated vector, so a faithful test sets a generous
budget to exercise deletion and the default 1% to exercise the stop condition.

**A future selector — `review`.** The same ranked stack, folded interactively: `review` walks candidates
from highest confidence down and asks a human to accept or reject each, then returns the reduced
`Document`. `auto` and `review` are the same phase with different stop logic — automatic budget versus a
person — which is exactly why Select is its own pluggable phase rather than a branch inside `reduce`.

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
   type Scores = dict[str, ScoreVector]   # scorer name → row-aligned, length-validated scores
   type Plan   = tuple[Candidate, ...]    # an ordered stack of edit candidates, folded top-to-bottom

   class Scorer(Protocol):
       def __call__(self, document: Document) -> list[float]: ...

   class Optimizer(Protocol):
       def __call__(self, document: Document, scores: Scores, params: Params) -> Plan: ...

   class Selector(Protocol):
       def __call__(self, document: Document, plan: Plan, embedder: Embedder, params: Params) -> Document: ...
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
   @register_optimizer("greedy_pairwise", requires=("redundancy",))
   def greedy_pairwise(document: Document, scores: Scores, params: Params) -> Plan:
       redundancy = scores["redundancy"]
       ...   # return a tuple of Candidate, never a Document

   # ops/features/select.py
   @register_selector("auto")
   def auto(document: Document, plan: Plan, embedder: Embedder, params: Params) -> Document:
       ...   # fold the plan within params.drift_budget; return the reduced Document
   ```

4. **Done.** The CLI resolves `--scorer NAME` / `--optimizer NAME` against the registry; for an
   optimizer the pipeline runs its `requires` scorers and builds the `Scores` bundle automatically.
   No existing file is edited, the IR is untouched, and the new strategy is testable in isolation by
   feeding a small `Document` fixture and asserting on the returned scores or `Plan` — no model, no
   mocks.

**Third-party strategies.** The registry is also wired to Python **entry points**, so a strategy
need not live in this repository at all. An external package implements the same `Protocol`, declares
the entry point in its own `pyproject.toml`, and `load_plugins()` discovers it by name at startup —
Alexandria itself is never edited:

```toml
# in a third-party package's pyproject.toml
[project.entry-points."alexandria.optimizers"]
my_optimizer = "my_package.my_optimizer:my_optimizer"
```

Built-in strategies are registered through the same entry points (declared in this repo's
`pyproject.toml`), so there is exactly one discovery path, internal and external alike.

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

**Layering (`import-linter`).** The dependency rule the layout describes is enforced by four
`import-linter` contracts run in CI: (1) *Layered architecture* — `cli` over `ops` over `utils` over
`ir`, higher layers importing lower ones and never the reverse, so `ir` imports nothing else in
`alexandria`; (2) *Pipe composes features* — within `ops`, `pipe` may import the four `features` to
compose them, never the reverse; (3) *Features are independent* — the four features (`represent`,
`score`, `optimize`, `select`) may not import each other, so shared types live in `ir.contracts`; and
(4) *Embedding backend isolated to the shell* — a **forbidden** contract barring `ir`, `ops`, and `cli`
from importing `sentence_transformers`, so only the `utils.embedders` shell builds a model. A cycle or a
layering violation fails the build — *modularity through contracts* and *library first* cannot quietly rot.

**Functional core, imperative shell.** The embedder is injected into Represent behind an `Embedder`
Protocol; the concrete model lives only in the shell. Tests pass a deterministic fake, making
Represent — and therefore the whole pipeline — a pure function with no mocks.

**Model identity.** A `Document` records the `embedding_model` that produced it. `reuse=` refuses to
carry embeddings across a model change and `redundancy` refuses to compare vectors from different
models, so the one impure input cannot silently poison a similarity score; the id round-trips through
Parquet for reproducibility.

**Row-aligned scores.** A scorer's output is parsed into a `ScoreVector` whose length must equal
`len(document.sentences)`, and a `Scores` bundle validates every member against the same `Document`.
A misaligned or short vector is a construction error, not a silent off-by-one.

**Fail-fast registry.** `@register` rejects a duplicate name at import time, and resolving an
optimizer validates its `requires=(...)` against the registered scorers at startup — an unknown
scorer is a clear error before any work runs.

**Edit laws (property-tested).** Folding a stack and then its inverse restores the original exactly
(`apply(reduced, inverse) == original`); folding never increases the `Document`'s token count; no edit
targets a unique sentence; an edit addressing an already-removed `id` is skipped, not misapplied; and
`apply` refuses any edit that would empty a `Section` or the `Document`. Because edits address stable ids,
**non-overlapping edits are order-independent** — folding them in any order yields the same `Document`. All
are Hypothesis properties over generated documents and stacks.

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
`Document` and replayed against the reloaded tree — its ids still match — or inverted to undo, without
re-running an optimizer.

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

- `alexandria represent [FILE] [--model MODEL]` — raw prompt in, a **`DocumentEnvelope`** (JSON) out.
- `alexandria score [FILE] [--scorer NAME[,NAME...]] [--table]` — `DocumentEnvelope` in, a
  **`ScoredEnvelope`** out; `--table` prints a human-readable per-instruction report instead. Run
  several scorers and their columns are the `Scores` bundle. No embedder is needed — it scores the
  Document it is handed.
- `alexandria optimize [FILE] [--optimizer NAME[,NAME...]] [--threshold T]` — `ScoredEnvelope` in, a
  **`PlanEnvelope`** out. Pass several optimizers and their stacks concatenate into one series.
- `alexandria select [FILE] [--model MODEL] [--drift-budget D] [--json]` — `PlanEnvelope` in, the
  **reduced prompt** out (or a JSON reduction summary with `--json`). The `auto` selector folds the
  ranked `Plan` highest-confidence-first while the prompt stays within `--drift-budget` (default `0.01`,
  i.e. 1%) of the original.
- `alexandria reduce [FILE] [--optimizer NAME[,NAME...]] [--selector NAME] [--threshold T] [--drift-budget D] [--model MODEL] [--json]`
  — prompt in, **reduced prompt out**, running all four phases in one process. This is the headline
  path and the fast in-process route; `--json` emits the same reduction summary as `select --json`
  (`text`, `applied`, `source_tokens`, `reduced_tokens`). There is no `--scorer` flag because each
  chosen optimizer declares the scorer(s) it needs.

So `alexandria represent < p.txt | alexandria score | alexandria optimize | alexandria select`
reproduces `alexandria reduce < p.txt`, and any prefix of that pipe is a useful stop
(`... | alexandria score --table` to inspect redundancy). A future interactive `review` selector walks
the same `Plan` and asks a human to accept or reject each drop.

**On composing via pipes.** Phase composition is first a *library* property — pure functions over the
in-memory `Document` — and the envelopes make it a *process* property too. The envelope is
self-contained: it always carries the `Document` a downstream phase needs, so no phase re-derives it,
and `score`/`optimize` need no model at all. Envelopes are JSON today (`schema_version=1`, pydantic
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
- No hosted embedding API in v1 (the IR keeps that swappable for later).
- No prompt *rewriting* beyond `merge`; optimization is selection-first.
