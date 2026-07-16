"""Alexandria — label-free prompt optimization."""

from alexandria.ir.document import Document
from alexandria.ops import (
    CompareResult,
    Diff,
    DiffSpan,
    OptimizationReport,
    Proposal,
    ReduceResult,
    compare,
    diffs,
    optimization_report,
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
    "OptimizationReport",
    "Proposal",
    "ReduceResult",
    "compare",
    "diffs",
    "optimization_report",
    "optimize",
    "propose",
    "reduce",
    "represent",
    "score",
    "score_report",
    "select",
]
