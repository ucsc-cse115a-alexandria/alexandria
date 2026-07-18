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

# ops re-exports embedder/merger construction, token counting, and config writes so the CLI never
# imports utils directly.
from alexandria.utils.config import write_config_key
from alexandria.utils.embedders import HashEmbedder, default_embedder
from alexandria.utils.merger import default_merger
from alexandria.utils.tokens import count_tokens

__all__ = [
    "CompareResult",
    "Diff",
    "DiffSpan",
    "HashEmbedder",
    "OptimizationReport",
    "Proposal",
    "ReduceResult",
    "ReportComparison",
    "compare",
    "compare_reports",
    "count_tokens",
    "default_embedder",
    "default_merger",
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
