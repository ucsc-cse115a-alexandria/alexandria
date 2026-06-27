"""Split a prompt into a tree of sections grouped under markdown headers and XML tag blocks."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from alexandria.core.ir import SectionKind

_SEPARATOR = re.compile(r"\n+")
_MARKDOWN_HEADER = re.compile(r"(#{1,6})\s+\S")
_XML_OPEN = re.compile(r"<([A-Za-z][\w.-]*)>")
_XML_CLOSE = re.compile(r"</([A-Za-z][\w.-]*)>")


@dataclass(frozen=True)
class RawSentence:
    """A leaf text segment, with its original separators, before tokenizing and embedding."""

    text: str


@dataclass(frozen=True)
class RawSection:
    """A section before encoding."""

    kind: SectionKind
    header: str
    children: tuple[RawSentence | RawSection, ...]


def _new_children() -> list[_Open | RawSentence]:
    return []


@dataclass
class _Open:
    """A section still being built while parsing; frozen into a RawSection at the end."""

    kind: SectionKind
    header: str
    depth: int | None  # markdown header depth (number of '#'); None for xml/plain
    tag: str | None  # xml tag name to match its closing tag; None for markdown/plain
    children: list[_Open | RawSentence] = field(default_factory=_new_children)


def split(prompt: str) -> tuple[RawSection, ...]:
    """Group a prompt's lines into a nested section tree.

    Markdown headers nest by '#' depth and XML blocks nest by tag stack; text under no header or
    tag becomes a ``plain`` section. The split is lossless: concatenating every leaf sentence's
    text in document order reproduces ``prompt`` exactly. A blank prompt yields ``()``.
    """
    if not prompt.strip():
        return ()

    roots: list[_Open] = []
    stack: list[_Open] = []

    def open_section(section: _Open) -> None:
        (stack[-1].children if stack else roots).append(section)
        stack.append(section)

    def drop_open_plain() -> None:
        if stack and stack[-1].kind is SectionKind.PLAIN:
            stack.pop()

    for segment in _segments(prompt):
        line = segment.strip()
        if header := _MARKDOWN_HEADER.match(line):
            depth = len(header.group(1))
            drop_open_plain()
            while stack and stack[-1].kind is SectionKind.MARKDOWN and (stack[-1].depth or 0) >= depth:
                stack.pop()
            open_section(_Open(SectionKind.MARKDOWN, line.lstrip("#").strip(), depth, None))
            stack[-1].children.append(RawSentence(segment))
        elif opening := _XML_OPEN.fullmatch(line):
            drop_open_plain()
            open_section(_Open(SectionKind.XML, opening.group(1), None, opening.group(1)))
            stack[-1].children.append(RawSentence(segment))
        elif (closing := _XML_CLOSE.fullmatch(line)) and _has_open_tag(stack, closing.group(1)):
            tag = closing.group(1)
            while not (stack[-1].kind is SectionKind.XML and stack[-1].tag == tag):
                stack.pop()
            stack[-1].children.append(RawSentence(segment))
            stack.pop()
        else:
            if not stack:
                open_section(_Open(SectionKind.PLAIN, "", None, None))
            stack[-1].children.append(RawSentence(segment))

    return tuple(_freeze(section) for section in roots)


def _has_open_tag(stack: list[_Open], tag: str) -> bool:
    return any(section.kind is SectionKind.XML and section.tag == tag for section in stack)


def _freeze(node: _Open) -> RawSection:
    children = tuple(child if isinstance(child, RawSentence) else _freeze(child) for child in node.children)
    return RawSection(kind=node.kind, header=node.header, children=children)


def _segments(prompt: str) -> list[str]:
    """One content line plus its trailing newline run per segment; leading newlines join the first."""
    segments: list[str] = []
    pending = ""
    pos = 0
    for match in _SEPARATOR.finditer(prompt):
        content = prompt[pos : match.start()]
        separator = match.group()
        if content:
            segments.append(pending + content + separator)
            pending = ""
        else:
            pending += separator  # leading newlines, before any content — they join the first segment
        pos = match.end()
    tail = prompt[pos:]
    if tail:
        segments.append(pending + tail)
    return segments
