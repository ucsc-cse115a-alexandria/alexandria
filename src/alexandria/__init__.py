"""Alexandria — label-free prompt optimization."""

from alexandria.ir.document import Document
from alexandria.ops import ReduceResult, optimize, reduce, represent, score, score_report, select

__all__ = ["Document", "ReduceResult", "optimize", "reduce", "represent", "score", "score_report", "select"]
