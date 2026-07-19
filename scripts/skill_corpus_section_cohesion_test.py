from __future__ import annotations

import csv
from typing import TYPE_CHECKING

import numpy as np

from scripts.skill_corpus_section_cohesion import (
    _benjamini_hochberg,
    _cohesion,
    discover_skill_paths,
    run_analysis,
)

if TYPE_CHECKING:
    from pathlib import Path


class KeywordEmbedder:
    model_id = "keyword-2"

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "cat" in lowered or "kitten" in lowered:
                vector = (1.0, 0.0)
            elif "rocket" in lowered or "orbit" in lowered:
                vector = (0.0, 1.0)
            else:
                vector = (0.7, 0.7)
            vectors.append(np.asarray(vector, dtype=np.float32))
        return vectors


def _write_skill(path: Path, noun: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"# Animals\n{noun} cat behavior is calm. The kitten stays near the cat.\n\n"
        "## Space\nA rocket enters orbit. The rocket remains in orbit.\n",
        encoding="utf-8",
    )


def test_cohesion_is_one_for_identical_vectors() -> None:
    vectors = np.asarray([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    leave_one_out, pairwise = _cohesion(vectors)
    assert leave_one_out == 1.0
    assert pairwise == 1.0


def test_benjamini_hochberg_is_monotone_in_p_value() -> None:
    adjusted = _benjamini_hochberg({"a": 0.001, "b": 0.02, "c": 0.5})
    assert adjusted["a"] <= adjusted["b"] <= adjusted["c"]
    assert all(0 <= value <= 1 for value in adjusted.values())


def test_run_analysis_writes_repo_outputs(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    for repo in ("alice__one", "bob__two"):
        for index in range(3):
            _write_skill(corpus / repo / "skills" / f"skill-{index}" / "SKILL.md", repo)

    selected = discover_skill_paths(corpus, max_skills_per_repo=2, seed=42)
    assert len(selected) == 4

    output = tmp_path / "output"
    result = run_analysis(
        corpus,
        output,
        tmp_path / "cache",
        KeywordEmbedder(),
        api_batch_size=4,
        group_sentence_limit=8,
        bootstrap_samples=20,
        min_skills_for_test=2,
    )

    assert result["eligible_repo_count"] == 2
    assert (output / "sections.csv.gz").is_file()
    assert (output / "skills.csv").is_file()
    assert (output / "repos.png").stat().st_size > 0
    with (output / "repos.csv").open(encoding="utf-8") as repo_file:
        rows = list(csv.DictReader(repo_file))
    assert {row["repo"] for row in rows} == {"alice__one", "bob__two"}
