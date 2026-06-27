"""Alexandria — label-free prompt optimization."""

from alexandria.core import Document
from alexandria.phases.optimize import optimize
from alexandria.phases.represent import represent
from alexandria.phases.score import score
from alexandria.phases.select import select
from alexandria.runtime.pipeline import reduce, score_report

__all__ = ["Document", "optimize", "reduce", "represent", "score", "score_report", "select"]
