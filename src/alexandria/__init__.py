"""Alexandria — label-free prompt optimization."""

from alexandria.ir.document import Document
from alexandria.ops import (
    OptimizationReport,
    ReduceResult,
    optimization_report,
    optimize,
    reduce,
    represent,
    score,
    score_report,
    select,
)

__all__ = [
    "Document",
    "OptimizationReport",
    "ReduceResult",
    "optimization_report",
    "optimize",
    "reduce",
    "represent",
    "score",
    "score_report",
    "select",
]
