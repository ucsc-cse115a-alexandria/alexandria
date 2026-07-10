"""Alexandria — label-free prompt optimization."""

from alexandria.ir.document import Document
from alexandria.ops import (
    CompareResult,
    Diff,
    DiffSpan,
    ReduceResult,
    compare,
    diffs,
    optimize,
    reduce,
    represent,
    score,
    score_report,
    select,
)

__all__ = [
    "CompareResult",
    "Diff",
    "DiffSpan",
    "Document",
    "ReduceResult",
    "compare",
    "diffs",
    "optimize",
    "reduce",
    "represent",
    "score",
    "score_report",
    "select",
]
