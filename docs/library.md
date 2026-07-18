# Library guide

Alexandria's CLI calls the same importable functions that are available to Python applications.

## Reduce a prompt

`reduce` runs the complete pipeline. Its signature is:

```python
def reduce(
    prompt: str,
    embedder: Embedder | None = None,
    merger: SentenceMerger | None = None,
    *,
    api_key: str | None = None,
    params: Params | None = None,
) -> ReduceResult: ...
```

Without an explicit `embedder` and `merger`, it builds the OpenAI defaults
(`text-embedding-3-small` for embeddings, `gpt-5.6-luna` for merging), so a key must be resolvable.
Resolution order is the `api_key` argument, then the `OPENAI_API_KEY` environment variable, then the
config file written by `alexandria config set openai-api-key`.

```python
import alexandria

result = alexandria.reduce("Be concise.\nBe concise.\nUse examples.\n")
print(result.text)
```

`result` contains the reduced `Document`, the original source document, and the candidates that were
applied. `examples/reduce_prompt.py` is a runnable version that loads the key from a `.env` file — copy
`.env.example` to `.env`, add your key, then run `uv run python examples/reduce_prompt.py`.

## Run offline by injecting an embedder and merger

For tests or CI that must run without network access, pass your own `Embedder` and `SentenceMerger`
instead of the OpenAI defaults. `HashEmbedder` is reproducible but not semantic (it only scores exact
duplicates as redundant and re-embeds edited text to an unrelated vector), so use a generous drift
budget. A minimal first-wins merger returns the first sentence unchanged:

```python
import alexandria
from alexandria.ir.contracts import Params
from alexandria.utils.embedders import HashEmbedder


class FirstWinsMerger:
    def merge(self, first: str, second: str, feedback: str | None = None) -> str:
        del second, feedback
        return first


result = alexandria.reduce(
    "repeat me\nrepeat me\nunique line\n",
    HashEmbedder(),
    FirstWinsMerger(),
    params=Params(drift_budget=2.0),
)
print(result.text)
```

See `tests/pipeline_e2e_test.py` for the same pattern used as a test.

## Compose the phases

Each phase is independently callable. This is the in-memory equivalent of the CLI pipeline and avoids
serializing JSON between phases. Pass the same `embedder` and `merger` you would give `reduce` — the
defaults below require an API key.

```python
from alexandria import optimize, represent, score, select
from alexandria.utils.embedders import default_embedder
from alexandria.utils.merger import default_merger

embedder = default_embedder()
merger = default_merger()
document = represent("Be concise.\nBe concise.\nUse examples.\n", embedder)
scores = score(document)
plan = optimize(document, scores, embedder, merger)
selection = select(document, plan, embedder)
print(selection.document.text)
```

`represent` converts raw text into a `Document`; `score` returns redundancy scores; `optimize` merges
near-duplicate pairs into LLM-rewritten candidate edits; and `select` applies acceptable edits
least-drift-first. For their contracts, extension points, and internal data model, see
[the design specification](spec.md).
