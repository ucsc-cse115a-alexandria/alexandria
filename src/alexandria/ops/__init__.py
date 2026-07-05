"""The library body: the four features and the pipe that composes them."""

from alexandria.ops.features.optimize import optimize
from alexandria.ops.features.represent import represent
from alexandria.ops.features.score import score, score_rows
from alexandria.ops.features.select import select
from alexandria.ops.pipe import ReduceResult, reduce, score_report

# ops re-exports embedder construction so the CLI never imports utils directly.
from alexandria.utils.embedders import DEFAULT_MODEL, DETERMINISTIC, build_embedder

__all__ = [
    "DEFAULT_MODEL",
    "DETERMINISTIC",
    "ReduceResult",
    "build_embedder",
    "optimize",
    "reduce",
    "represent",
    "score",
    "score_report",
    "score_rows",
    "select",
]
