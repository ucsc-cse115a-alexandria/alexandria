import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
import tiktoken
from generate_inflate_dataset import (
    FACTORS,
    RowStatus,
    build_dataset,
    factor_dir,
    select_skills,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from alexandria.ir.contracts import Embedder

_ENCODING = tiktoken.get_encoding("cl100k_base")

_LARGE = "This is the larger skill document with more words in it. " * 20
_SMALL = "short skill"
_BOB = "bob's skill document lives here"


class ConstantEmbedder:
    model_id = "constant"

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        return [np.ones(8, dtype=np.float32) for _ in texts]


def _identity(prompt: str) -> str:
    return prompt


def fake_inflate(
    prompt: str,
    factor: float,
    generate: Callable[[str], str],
    embedder: Embedder,
    encoding: tiktoken.Encoding,
    max_attempts: int = 2,
) -> str:
    """Deterministic stand-in: excludes factor 10 (never clears the gate), otherwise restates once."""
    tokens = len(encoding.encode(prompt))
    if factor >= 10.0:
        raise RuntimeError(f"inflated text stayed below the gate after {max_attempts} attempts")
    return f"{generate(prompt)}\n\nrestated {factor:g}x ({tokens} tok, {embedder.model_id})"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """A minimal skill-corpus checkout: two repos (alice has two SKILL.md, bob one)."""
    _write(
        tmp_path / "data" / "skill_repos.json",
        json.dumps(
            [
                {"full_name": "bob/small", "stars": 50, "url": "u", "description": "d"},
                {"full_name": "alice/big", "stars": 100, "url": "u", "description": "d"},
            ]
        ),
    )
    _write(
        tmp_path / "data" / "download_manifest.json",
        json.dumps(
            {
                "downloaded_files": [
                    {"repo": "alice/big", "local_path": "corpus/alice__big/skills/a/SKILL.md"},
                    {"repo": "alice/big", "local_path": "corpus/alice__big/skills/b/SKILL.md"},
                    {"repo": "bob/small", "local_path": "corpus/bob__small/SKILL.md"},
                ],
                "errors": [],
            }
        ),
    )
    _write(tmp_path / "corpus/alice__big/skills/a/SKILL.md", _SMALL)
    _write(tmp_path / "corpus/alice__big/skills/b/SKILL.md", _LARGE)
    _write(tmp_path / "corpus/bob__small/SKILL.md", _BOB)
    return tmp_path


def test_factor_dir_names() -> None:
    assert factor_dir(1.2) == "1_2"
    assert factor_dir(1.5) == "1_5"
    assert factor_dir(2.0) == "2"
    assert factor_dir(10.0) == "10"


def test_select_ranks_by_stars_and_picks_largest_file(corpus: Path) -> None:
    selections = select_skills(corpus, limit=20, encoding=_ENCODING)
    assert [s.full_name for s in selections] == ["alice/big", "bob/small"]  # stars descending
    assert selections[0].dir_name == "alice__big"
    assert selections[0].source_path == "corpus/alice__big/skills/b/SKILL.md"  # the larger of alice's two
    assert selections[0].text == _LARGE


def test_select_honors_limit(corpus: Path) -> None:
    selections = select_skills(corpus, limit=1, encoding=_ENCODING)
    assert [s.full_name for s in selections] == ["alice/big"]


def test_select_caps_source_tokens_and_falls_back_to_smaller_file(corpus: Path) -> None:
    small = len(_ENCODING.encode(_SMALL))
    large = len(_ENCODING.encode(_LARGE))
    selections = select_skills(corpus, limit=20, encoding=_ENCODING, max_source_tokens=(small + large) // 2)
    alice = next(s for s in selections if s.full_name == "alice/big")
    assert alice.source_path == "corpus/alice__big/skills/a/SKILL.md"  # the larger file is over the cap
    assert alice.text == _SMALL


def test_build_dataset_writes_copy_variants_and_manifest(corpus: Path) -> None:
    selections = select_skills(corpus, limit=20, encoding=_ENCODING)
    manifest = build_dataset(
        corpus,
        selections,
        _identity,
        ConstantEmbedder(),
        _ENCODING,
        model_id="fake-model",
        inflate_fn=fake_inflate,
    )
    root = corpus / "inflate"

    assert (root / "1" / "alice__big.md").read_text() == _LARGE  # verbatim copy, no LLM
    assert (root / "1" / "bob__small.md").read_text() == _BOB
    assert (root / "1_2" / "alice__big.md").read_text().startswith(_LARGE)
    assert (root / "2" / "alice__big.md").exists()
    assert not (root / "10" / "alice__big.md").exists()  # factor 10 raised -> not written

    rows = {(row.full_name, row.kind): row for row in manifest.rows}
    excluded = rows[("alice/big", "inflate-10x")]
    assert excluded.status is RowStatus.EXCLUDED
    assert excluded.similarity is None
    assert excluded.tokens_edited is None

    included = rows[("alice/big", "inflate-1.2x")]
    assert included.status is RowStatus.INCLUDED
    assert included.similarity == pytest.approx(1.0)  # ConstantEmbedder -> identical embeddings
    assert included.tokens_original == len(_ENCODING.encode(_LARGE))

    assert manifest.model_id == "fake-model"
    assert manifest.prompt_version  # imported from inflate_redundancy
    assert manifest.factors == list(FACTORS)


def test_build_dataset_skips_existing_files(corpus: Path) -> None:
    selections = select_skills(corpus, limit=20, encoding=_ENCODING)
    seeded = corpus / "inflate" / "1_2" / "alice__big.md"
    _write(seeded, "SENTINEL")
    build_dataset(
        corpus,
        selections,
        _identity,
        ConstantEmbedder(),
        _ENCODING,
        model_id="fake-model",
        inflate_fn=fake_inflate,
    )
    assert seeded.read_text() == "SENTINEL"  # resumable: existing output is not regenerated
