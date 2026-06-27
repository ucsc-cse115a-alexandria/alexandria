from __future__ import annotations

import numpy as np

from alexandria.runtime.embedding import HashEmbedder, build_embedder


def test_identical_text_yields_identical_vector() -> None:
    embedder = HashEmbedder()
    assert np.array_equal(embedder.embed(["hello"])[0], embedder.embed(["hello"])[0])


def test_different_text_yields_different_vector() -> None:
    embedder = HashEmbedder()
    assert not np.array_equal(embedder.embed(["hello"])[0], embedder.embed(["world"])[0])


def test_model_id_and_build_deterministic() -> None:
    assert HashEmbedder().model_id == "hash-64"
    assert build_embedder("deterministic").model_id == "hash-64"
