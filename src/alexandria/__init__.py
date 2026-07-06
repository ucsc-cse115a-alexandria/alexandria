"""Alexandria — label-free prompt optimization."""

from alexandria.ir.document import Document
from alexandria.ops import (
    CompareResult,
    ReduceResult,
    compare,
    optimize,
    reduce,
    represent,
    score,
    score_report,
    select,
)

__all__ = [
    "CompareResult",
    "Document",
    "ReduceResult",
    "compare",
    "optimize",
    "reduce",
    "represent",
    "score",
    "score_report",
    "select",
]
