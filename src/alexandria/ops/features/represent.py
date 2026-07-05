from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import tiktoken

from alexandria.ir.document import Document, Node, Section, SectionKind, Sentence, SentenceId, rollup

if TYPE_CHECKING:
    from collections.abc import Iterator

    from alexandria.ir.contracts import Embedder

_SEPARATOR = re.compile(r"\n+")
_MARKDOWN_HEADER = re.compile(r"(#{1,6})\s+\S")
_XML_OPEN = re.compile(r"<([A-Za-z][\w.-]*)>")
_XML_CLOSE = re.compile(r"</([A-Za-z][\w.-]*)>")
_ENCODING = tiktoken.get_encoding("cl100k_base")


@dataclass(frozen=True)
class RawSentence:
    """A leaf text segment, with its original separators, before tokenizing and embedding."""

    text: str


@dataclass(frozen=True)
class RawSection:
    kind: SectionKind
    header: str
    children: tuple[RawSentence | RawSection, ...]


def _new_children() -> list[_Open | RawSentence]:
    return []


def _new_sections() -> list[_Open]:
    return []


@dataclass
class _Open:
    """A section still being built while parsing; frozen into a RawSection at the end."""

    kind: SectionKind
    header: str
    depth: int | None  # markdown header depth (number of '#'); read only by the markdown rule
    tag: str | None  # xml tag to match its closing tag; read only by the xml rules
    children: list[_Open | RawSentence] = field(default_factory=_new_children)


@dataclass
class _Builder:
    """The parse state a rule manipulates: the section stack and the completed roots."""

    roots: list[_Open] = field(default_factory=_new_sections)
    stack: list[_Open] = field(default_factory=_new_sections)

    def open(self, section: _Open) -> None:
        (self.stack[-1].children if self.stack else self.roots).append(section)
        self.stack.append(section)

    def append(self, segment: str) -> None:
        self.stack[-1].children.append(RawSentence(segment))

    def drop_open_plain(self) -> None:
        if self.stack and self.stack[-1].kind is SectionKind.PLAIN:
            self.stack.pop()

    def has_open_tag(self, tag: str) -> bool:
        return any(section.kind is SectionKind.XML and section.tag == tag for section in self.stack)

    def append_plain(self, segment: str) -> None:
        """The fallback when no rule claims the line: body text under no header or tag."""
        if not self.stack:
            self.open(_Open(SectionKind.PLAIN, "", None, None))
        self.append(segment)


# A rule recognizes one line shape, performs its stack action, and reports whether it claimed the
# line. Adding a SectionKind means adding one rule to _RULES, not editing the split() engine.
BlockRule = Callable[[str, str, _Builder], bool]


def _markdown_rule(line: str, segment: str, builder: _Builder) -> bool:
    header = _MARKDOWN_HEADER.match(line)
    if not header:
        return False
    depth = len(header.group(1))
    builder.drop_open_plain()
    while builder.stack and builder.stack[-1].kind is SectionKind.MARKDOWN and (builder.stack[-1].depth or 0) >= depth:
        builder.stack.pop()
    builder.open(_Open(SectionKind.MARKDOWN, line.lstrip("#").strip(), depth, None))
    builder.append(segment)
    return True


def _xml_open_rule(line: str, segment: str, builder: _Builder) -> bool:
    opening = _XML_OPEN.fullmatch(line)
    if not opening:
        return False
    builder.drop_open_plain()
    builder.open(_Open(SectionKind.XML, opening.group(1), None, opening.group(1)))
    builder.append(segment)
    return True


def _xml_close_rule(line: str, segment: str, builder: _Builder) -> bool:
    closing = _XML_CLOSE.fullmatch(line)
    if not (closing and builder.has_open_tag(closing.group(1))):
        return False
    tag = closing.group(1)
    while not (builder.stack[-1].kind is SectionKind.XML and builder.stack[-1].tag == tag):
        builder.stack.pop()
    builder.append(segment)
    builder.stack.pop()
    return True


_RULES: tuple[BlockRule, ...] = (_markdown_rule, _xml_open_rule, _xml_close_rule)


def split(prompt: str) -> tuple[RawSection, ...]:
    """Group a prompt's lines into a nested section tree.

    Markdown headers nest by '#' depth and XML blocks nest by tag stack; text under no header or
    tag becomes a ``plain`` section. The split is lossless: concatenating every leaf sentence's
    text in document order reproduces ``prompt`` exactly. A blank prompt yields ``()``.
    """
    if not prompt.strip():
        return ()

    builder = _Builder()
    for segment in _segments(prompt):
        line = segment.strip()
        for rule in _RULES:
            if rule(line, segment, builder):
                break
        else:
            builder.append_plain(segment)

    return tuple(_freeze(section) for section in builder.roots)


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


def _assign_ids(texts: list[str]) -> list[SentenceId]:
    """Content-addressed ids: 's' + the first 12 hex chars of the text's sha256, so the same text
    always gets the same id. A '-N' suffix (N from 2) disambiguates the Nth identical text in
    document order, keeping ids unique and their assignment deterministic."""
    seen: Counter[str] = Counter()
    ids: list[SentenceId] = []
    for text in texts:
        base = "s" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        seen[base] += 1
        suffix = "" if seen[base] == 1 else f"-{seen[base]}"
        ids.append(SentenceId(base + suffix))
    return ids


def encode(sections: tuple[RawSection, ...], embedder: Embedder) -> Document:
    leaves = _leaf_sentences(sections)
    vectors = embedder.embed([leaf.text for leaf in leaves])
    ids = _assign_ids([leaf.text for leaf in leaves])
    built_sentences = iter(
        Sentence(id=sid, text=leaf.text, token_count=len(_ENCODING.encode(leaf.text)), embedding=vector)
        for sid, leaf, vector in zip(ids, leaves, vectors, strict=True)
    )
    built = tuple(_build_section(section, embedder, built_sentences) for section in sections)
    text, token_count = rollup(built)
    return Document(
        embedding_model=embedder.model_id,
        sections=built,
        text=text,
        token_count=token_count,
        embedding=embedder.embed([text])[0],
    )


def _build_section(raw: RawSection, embedder: Embedder, sentences: Iterator[Sentence]) -> Section:
    children: list[Node] = [
        next(sentences) if isinstance(child, RawSentence) else _build_section(child, embedder, sentences)
        for child in raw.children
    ]
    kept = tuple(children)
    text, token_count = rollup(kept)
    return Section(
        kind=raw.kind,
        header=raw.header,
        children=kept,
        text=text,
        token_count=token_count,
        embedding=embedder.embed([text])[0],
    )


def _leaf_sentences(sections: tuple[RawSection, ...]) -> list[RawSentence]:
    """Every leaf sentence in document (pre-order) order — the order _build_section consumes them."""
    leaves: list[RawSentence] = []

    def walk(section: RawSection) -> None:
        for child in section.children:
            if isinstance(child, RawSentence):
                leaves.append(child)
            else:
                walk(child)

    for section in sections:
        walk(section)
    return leaves


def represent(prompt: str, embedder: Embedder) -> Document:
    """Build the Document IR: split losslessly, then tokenize and embed every node."""
    segments = split(prompt)
    if not segments:
        raise ValueError("cannot represent an empty prompt")
    return encode(segments, embedder)


__all__ = ["represent"]
