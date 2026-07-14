"""Alexandria — label-free prompt optimization."""

from alexandria.ir.document import Document
from alexandria.ops import (
    CompareResult,
    Diff,
    DiffSpan,
    Proposal,
    ReduceResult,
    compare,
    diffs,
    optimize,
    propose,
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
    "Proposal",
    "ReduceResult",
    "compare",
    "diffs",
    "optimize",
    "propose",
    "reduce",
    "represent",
    "score",
    "score_report",
    "select",
]
