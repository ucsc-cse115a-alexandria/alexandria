# Library guide

The public Python API exposes end-to-end reduction, reviewable proposals, individual phases,
comparison and report helpers, result models, and the `Document` IR.

## Reduce

```python
import alexandria

result = alexandria.reduce("Be concise.\nBe concise.\nUse examples.\n")
print(result.text)
print(result.source_tokens, result.reduced_tokens)
```

Without an injected embedder and merger, `reduce` builds the OpenAI defaults. The API key resolution
order is `api_key=`, `OPENAI_API_KEY`, then the config written by
`alexandria config set openai-api-key`.

`ReduceResult` contains the source and reduced `Document` values, applied edits, token convenience
properties, and merger-call, retry, pruning, embedding, and elapsed-time metrics.

Use `Params` for semantic and token controls:

```python
import alexandria
from alexandria.ir.contracts import Params

result = alexandria.reduce(
    "# Rules\nKeep responses concise.\nDo not repeat information.\nInclude a useful example.\n",
    params=Params(
        cos_sim_diff_budget=0.01,
        max_tokens=12,
        require_target=True,
    ),
)
```

With `require_target=True`, `max_tokens` is mandatory and the hard-target path replaces the normal
Optimize/Select path. It preserves structural lines, prioritizes the token ceiling, and reports
semantic-budget compliance separately. It can raise `alexandria.InfeasibleTargetError` when protected
structure cannot fit or `alexandria.TargetMergeError` when generation cannot make progress.

A custom merger used for hard targets must also implement
`merge_candidates_to_target(prompt, max_tokens)`. The normal best-effort path only requires
`merge(first, second, feedback)`.

## Run offline

Inject an `Embedder` and `SentenceMerger` to avoid network calls. `HashEmbedder` is deterministic but
not semantic; it is intended for tests.

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
    params=Params(cos_sim_diff_budget=2.0),
)
print(result.text)
```

See [`tests/pipeline_e2e_test.py`](../tests/pipeline_e2e_test.py) for a tested offline composition.

## Compose the normal phases

This example uses the OpenAI defaults and requires a configured API key.

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

This is the explicit normal pipeline. Exact duplicates may produce a `Delete` without calling the
merger; other eligible pairs may produce `Replace` edits. A hard target is available only through the
end-to-end `reduce` path.

See the [design specification](spec.md) for phase contracts, IR invariants, and registration
boundaries.
