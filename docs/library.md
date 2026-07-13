# Library guide

Alexandria's CLI calls the same importable functions that are available to Python applications.

## Reduce a prompt

`reduce` runs the complete pipeline. Without an explicit embedder, it downloads and builds the default
`all-MiniLM-L6-v2` model on first use.

```python
import alexandria

result = alexandria.reduce("Be concise.\nBe concise.\nUse examples.\n")
print(result.text)
```

`result` contains the reduced `Document`, the original source document, and the candidates that were
applied. See `examples/reduce_prompt.py` for a runnable version.

## Use a deterministic embedder

Pass `HashEmbedder` when tests or CI must run without downloading a model. It is deterministic but not
semantic, so use an appropriate drift budget when testing reductions.

```python
import alexandria
from alexandria.ir.contracts import Params
from alexandria.utils.embedders import HashEmbedder

result = alexandria.reduce(
    "Be concise.\nBe concise.\nUse examples.\n",
    HashEmbedder(),
    params=Params(drift_budget=2.0),
)
print(result.text)
```

## Compose the phases

Each phase is independently callable. This is the in-memory equivalent of the CLI pipeline and avoids
serializing JSON between phases.

```python
from alexandria import optimize, represent, score, select
from alexandria.utils.embedders import default_embedder

embedder = default_embedder()
document = represent("Be concise.\nBe concise.\nUse examples.\n", embedder)
scores = score(document)
plan = optimize(document, scores)
selection = select(document, plan, embedder)
print(selection.document.text)
```

`represent` converts raw text into a `Document`; `score` returns redundancy scores; `optimize` creates a
plan of candidate edits; and `select` applies acceptable edits. For their contracts, extension points,
and internal data model, see [the design specification](spec.md).
