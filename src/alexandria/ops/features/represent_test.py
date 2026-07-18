from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from alexandria.ir.document import Section, SectionKind
from alexandria.ops.features.represent import RawSection, RawSentence, represent, split
from alexandria.utils.embedders import HashEmbedder

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from alexandria.ir.document import Node


def _leaf_text(sections: tuple[RawSection, ...]) -> str:
    parts: list[str] = []

    def walk(section: RawSection) -> None:
        for child in section.children:
            if isinstance(child, RawSentence):
                parts.append(child.text)
            else:
                walk(child)

    for section in sections:
        walk(section)
    return "".join(parts)


def test_blank_prompt_yields_no_sections() -> None:
    assert split("   \n\n") == ()


def test_lossless_with_leading_newlines_and_no_trailing_newline() -> None:
    prompt = "\n\n# Goal\nDo Z."
    assert _leaf_text(split(prompt)) == prompt


def test_structureless_prompt_is_one_plain_section() -> None:
    sections = split("Do X.\nDo Y.\n")
    assert len(sections) == 1
    assert sections[0].kind is SectionKind.PLAIN
    assert sections[0].header == ""
    assert sections[0].children == (RawSentence("Do X.\n"), RawSentence("Do Y.\n"))


def test_lossless_across_headers_and_tags() -> None:
    prompt = "Intro.\n\n# Setup\nInstall deps.\n<note>\nbe careful\n</note>\n"
    assert _leaf_text(split(prompt)) == prompt


def test_markdown_header_starts_a_section() -> None:
    sections = split("# Goal\nDo Z.\n")
    assert len(sections) == 1
    section = sections[0]
    assert section.kind is SectionKind.MARKDOWN
    assert section.header == "Goal"
    assert section.children == (RawSentence("# Goal\n", optimizable=False), RawSentence("Do Z.\n"))


def test_deeper_header_nests_under_shallower() -> None:
    sections = split("# Setup\nInstall deps.\n## Database\nRun migrations.\n")
    assert len(sections) == 1
    setup = sections[0]
    assert setup.header == "Setup"
    assert setup.children[0] == RawSentence("# Setup\n", optimizable=False)
    assert setup.children[1] == RawSentence("Install deps.\n")
    database = setup.children[2]
    assert isinstance(database, RawSection)
    assert database.kind is SectionKind.MARKDOWN
    assert database.header == "Database"
    assert database.children == (
        RawSentence("## Database\n", optimizable=False),
        RawSentence("Run migrations.\n"),
    )


def test_same_depth_header_is_a_sibling() -> None:
    sections = split("# A\nx\n# B\ny\n")
    assert [s.header for s in sections] == ["A", "B"]


def test_xml_tag_starts_a_section() -> None:
    sections = split("<instructions>\nDo X.\n</instructions>\n")
    assert len(sections) == 1
    section = sections[0]
    assert section.kind is SectionKind.XML
    assert section.header == "instructions"
    assert section.children == (
        RawSentence("<instructions>\n", optimizable=False),
        RawSentence("Do X.\n"),
        RawSentence("</instructions>\n", optimizable=False),
    )


def test_xml_block_is_a_hard_boundary_for_markdown() -> None:
    # The '## Steps' header nests inside the xml block and is closed by '</task>'.
    sections = split("<task>\n## Steps\ndo it\n</task>\nafter\n")
    task = sections[0]
    assert task.kind is SectionKind.XML
    steps = task.children[1]
    assert isinstance(steps, RawSection)
    assert steps.kind is SectionKind.MARKDOWN
    assert steps.header == "Steps"
    assert task.children[-1] == RawSentence("</task>\n", optimizable=False)
    # 'after' falls back to a top-level plain section, outside the xml block.
    assert sections[1].kind is SectionKind.PLAIN
    assert sections[1].children == (RawSentence("after\n"),)


def test_preamble_before_first_header_is_plain() -> None:
    sections = split("Read carefully.\n# Goal\nDo Z.\n")
    assert [s.kind for s in sections] == [SectionKind.PLAIN, SectionKind.MARKDOWN]
    assert sections[0].children == (RawSentence("Read carefully.\n"),)


def test_represent_builds_a_document() -> None:
    document = represent("first\nsecond\n", HashEmbedder())
    assert document.embedding_model == "hash-64"
    ids = [s.id for s in document.sentences]
    assert len(set(ids)) == 2
    assert all(sid.startswith("s") for sid in ids)
    assert document.text == "first\nsecond\n"
    assert document.token_count > 0


def test_represent_marks_markup_boundaries_as_not_optimizable() -> None:
    document = represent("# Goal\n<example>\nDo X.\n</example>\n", HashEmbedder())

    by_text = {sentence.text.strip(): sentence.optimizable for sentence in document.sentences}
    assert by_text == {
        "# Goal": False,
        "<example>": False,
        "Do X.": True,
        "</example>": False,
    }


def test_represent_rejects_an_empty_prompt() -> None:
    with pytest.raises(ValueError):
        represent("   \n", HashEmbedder())


def test_ids_are_stable_across_re_represent() -> None:
    prompt = "Do X.\nDo Y.\nDo Z.\n"
    first = [s.id for s in represent(prompt, HashEmbedder()).sentences]
    second = [s.id for s in represent(prompt, HashEmbedder()).sentences]
    assert first == second


def test_editing_one_line_leaves_other_ids_unchanged() -> None:
    before = {s.text: s.id for s in represent("Do X.\nDo Y.\nDo Z.\n", HashEmbedder()).sentences}
    after = {s.text: s.id for s in represent("Do X.\nEDITED.\nDo Z.\n", HashEmbedder()).sentences}
    assert after["Do X.\n"] == before["Do X.\n"]
    assert after["Do Z.\n"] == before["Do Z.\n"]
    assert after["EDITED.\n"] != before["Do Y.\n"]


def test_identical_lines_get_unique_deterministic_ids() -> None:
    ids = [s.id for s in represent("same\nsame\nsame\n", HashEmbedder()).sentences]
    assert len(set(ids)) == 3
    assert ids == [ids[0], f"{ids[0]}-2", f"{ids[0]}-3"]


class _CallRecordingEmbedder:
    def __init__(self) -> None:
        self._inner = HashEmbedder()
        self.calls: list[list[str]] = []

    @property
    def model_id(self) -> str:
        return self._inner.model_id

    def embed(self, texts: list[str]) -> list[NDArray[np.float32]]:
        self.calls.append(list(texts))
        return self._inner.embed(texts)


def _count_sections(nodes: tuple[Node, ...]) -> int:
    return sum(1 + _count_sections(node.children) for node in nodes if isinstance(node, Section))


def test_represent_embeds_in_two_batched_calls() -> None:
    embedder = _CallRecordingEmbedder()

    document = represent("# A\nline one\n## B\nline two\n<x>\nline three\n</x>\n", embedder)

    assert len(embedder.calls) == 2
    assert embedder.calls[1][-1] == document.text  # the second batch ends with the document text
    assert len(embedder.calls[1]) == _count_sections(document.sections) + 1
