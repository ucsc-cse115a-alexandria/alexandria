"""Shared minimal-pair catalog for the fidelity probes.

Both probes -- the sentence-embedding one and the spec-v2 behavioral one -- score the *same*
labeled pairs so their separation can be compared directly. The pairs live in `fidelity_pairs.json`
next to this module; the data and its provenance are documented inline there (arXiv ids in `source`).
"""

import json
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel

DEFAULT_PAIRS_PATH = Path(__file__).with_name("fidelity_pairs.json")


class Label(StrEnum):
    """Ground-truth effect of the edit on the set of acceptable outputs."""

    MEANING_CHANGING = "MC"
    MEANING_PRESERVING = "MP"
    UNRELATED = "unrelated"


class Pair(BaseModel):
    """A minimal pair: an original instruction and one edited variant, with its provenance."""

    label: Label
    kind: str
    source: str
    original: str
    edited: str


def load_catalogs(path: Path = DEFAULT_PAIRS_PATH) -> dict[str, list[Pair]]:
    """Load every named catalog of minimal pairs, validating each entry through `Pair`."""
    raw: dict[str, list[dict[str, str]]] = json.loads(path.read_text())
    return {name: [Pair(**entry) for entry in entries] for name, entries in raw.items()}
