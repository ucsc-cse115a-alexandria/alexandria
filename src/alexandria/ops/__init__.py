"""The library body: the standalone features and the pipe that chains them."""

from alexandria.ir.contracts import Diff, DiffSpan
from alexandria.ops.features.compare import CompareResult, compare
from alexandria.ops.features.diff import diffs
from alexandria.ops.features.optimize import optimize
from alexandria.ops.features.represent import represent
from alexandria.ops.features.score import score, score_rows
from alexandria.ops.features.select import select
from alexandria.ops.pipe import (
    OptimizationReport,
    Proposal,
    ReduceResult,
    ReportComparison,
    compare_reports,
    optimization_report,
    propose,
    reduce,
    score_report,
)

# ops re-exports embedder construction and config writes so the CLI never imports utils directly.
from alexandria.utils.config import write_config_key
from alexandria.utils.embedders import DEFAULT_MODEL, DETERMINISTIC, build_embedder

__all__ = [
    "DEFAULT_MODEL",
    "DETERMINISTIC",
    "CompareResult",
    "Diff",
    "DiffSpan",
    "OptimizationReport",
    "Proposal",
    "ReduceResult",
    "ReportComparison",
    "build_embedder",
    "compare",
    "compare_reports",
    "diffs",
    "optimization_report",
    "optimize",
    "propose",
    "reduce",
    "represent",
    "score",
    "score_report",
    "score_rows",
    "select",
    "write_config_key",
]
