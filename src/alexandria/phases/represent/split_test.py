from __future__ import annotations

from alexandria.core.ir import SectionKind
from alexandria.phases.represent.split import RawSection, RawSentence, split


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
    assert section.children == (RawSentence("# Goal\n"), RawSentence("Do Z.\n"))


def test_deeper_header_nests_under_shallower() -> None:
    sections = split("# Setup\nInstall deps.\n## Database\nRun migrations.\n")
    assert len(sections) == 1
    setup = sections[0]
    assert setup.header == "Setup"
    assert setup.children[0] == RawSentence("# Setup\n")
    assert setup.children[1] == RawSentence("Install deps.\n")
    database = setup.children[2]
    assert isinstance(database, RawSection)
    assert database.kind is SectionKind.MARKDOWN
    assert database.header == "Database"
    assert database.children == (RawSentence("## Database\n"), RawSentence("Run migrations.\n"))


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
        RawSentence("<instructions>\n"),
        RawSentence("Do X.\n"),
        RawSentence("</instructions>\n"),
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
    assert task.children[-1] == RawSentence("</task>\n")
    # 'after' falls back to a top-level plain section, outside the xml block.
    assert sections[1].kind is SectionKind.PLAIN
    assert sections[1].children == (RawSentence("after\n"),)


def test_preamble_before_first_header_is_plain() -> None:
    sections = split("Read carefully.\n# Goal\nDo Z.\n")
    assert [s.kind for s in sections] == [SectionKind.PLAIN, SectionKind.MARKDOWN]
    assert sections[0].children == (RawSentence("Read carefully.\n"),)
