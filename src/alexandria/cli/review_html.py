"""Self-contained HTML review page for proposed prompt edits."""

from __future__ import annotations

import html
import json
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict

from alexandria.cli.hunks import hunk_lines

if TYPE_CHECKING:
    from alexandria.ir.contracts import Diff
    from alexandria.ir.document import Document, SentenceId
    from alexandria.ops.pipe import Proposal

SCHEMA_VERSION = 1

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Alexandria — Review Edits</title>
<style>
  :root {
    color-scheme: light;
    --bg: #f7f8fa;
    --card: #ffffff;
    --border: #e7e9ee;
    --ink: #1b2430;
    --muted: #6b7585;
    --red: #dc2626;
    --red-bg: #fef2f2;
    --green: #16a34a;
    --green-bg: #f0fdf4;
    --mono: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
    --shadow: 0 1px 2px rgba(16, 24, 40, 0.04), 0 8px 24px rgba(16, 24, 40, 0.06);
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 0;
    font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
    background: var(--bg); color: var(--ink); min-height: 100vh;
    -webkit-font-smoothing: antialiased;
  }
  .sticky-top {
    position: sticky; top: 0; z-index: 10;
    background: var(--bg); border-bottom: 1px solid var(--border);
    padding: 16px 24px; box-shadow: var(--shadow);
  }
  .header-row {
    display: flex; flex-wrap: wrap; align-items: center; gap: 12px 20px;
    max-width: 960px; margin: 0 auto;
  }
  .header-row h1 { font-size: 1.1rem; font-weight: 650; margin: 0; letter-spacing: -0.01em; }
  .summary { color: var(--muted); font-size: 0.9rem; font-variant-numeric: tabular-nums; }
  .shortcuts { margin-left: auto; display: flex; gap: 8px; }
  .shortcuts button {
    font: inherit; font-size: 0.82rem; padding: 6px 12px;
    border: 1px solid var(--border); border-radius: 8px;
    background: var(--card); cursor: pointer; color: var(--ink);
  }
  .shortcuts button:hover { background: var(--bg); }
  .cumulative {
    max-width: 960px; margin: 12px auto 0;
    background: var(--card); border: 1px solid var(--border);
    border-radius: 12px; overflow: hidden;
  }
  .cumulative-title {
    padding: 8px 14px; font-size: 0.75rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted);
    border-bottom: 1px solid var(--border); background: var(--bg);
  }
  .cumulative-body {
    max-height: 220px; overflow-y: auto;
    font-family: var(--mono); font-size: 0.82rem; line-height: 1.5;
    padding: 10px 14px; white-space: pre-wrap; word-break: break-word;
  }
  .wrap { max-width: 960px; margin: 0 auto; padding: 24px; }
  .empty { color: var(--muted); text-align: center; padding: 48px 24px; }
  .edit-block {
    background: var(--card); border: 2px solid var(--border);
    border-radius: 14px; margin-bottom: 16px; overflow: hidden;
    box-shadow: var(--shadow); transition: border-color 0.15s;
  }
  .edit-block.accepted { border-color: var(--green); }
  .edit-block.rejected { border-color: var(--border); }
  .edit-meta { padding: 14px 16px 10px; border-bottom: 1px solid var(--border); }
  .edit-meta .loc { font-weight: 600; font-size: 0.95rem; margin-bottom: 4px; }
  .edit-meta .detail { color: var(--muted); font-size: 0.85rem; line-height: 1.45; }
  .hunk {
    font-family: var(--mono); font-size: 0.82rem; line-height: 1.5;
    padding: 10px 14px; white-space: pre-wrap; word-break: break-word;
    background: #fafbfc;
  }
  .line.context { color: var(--ink); }
  .line.removed { color: var(--red); background: var(--red-bg); margin: 0 -14px; padding: 0 14px; }
  .line.added { color: var(--green); background: var(--green-bg); margin: 0 -14px; padding: 0 14px; }
  .line.gap { color: var(--muted); font-style: italic; }
  .line.empty-msg { color: var(--muted); font-style: italic; }
  .controls {
    display: flex; gap: 10px; padding: 12px 16px;
    border-top: 1px solid var(--border); background: var(--bg);
  }
  .controls button {
    font: inherit; font-size: 0.88rem; font-weight: 550;
    padding: 8px 18px; border-radius: 8px; cursor: pointer;
    border: 1px solid var(--border); background: var(--card);
  }
  .controls button.accept.active {
    background: var(--green-bg); border-color: var(--green); color: var(--green);
  }
  .controls button.reject.active {
    background: var(--red-bg); border-color: var(--red); color: var(--red);
  }
  .controls button:hover:not(.active) { background: #eef0f4; }
</style>
</head>
<body>
<div class="sticky-top">
  <div class="header-row">
    <h1>Alexandria — Review Edits</h1>
    <span class="summary" id="summary"></span>
    __SHORTCUTS__
  </div>
  <div class="cumulative">
    <div class="cumulative-title">Cumulative preview (original → selection)</div>
    <div class="cumulative-body" id="cumulative"></div>
  </div>
</div>
<div class="wrap" id="edits">
__EDIT_BLOCKS__
</div>
<script type="application/json" id="payload">__PAYLOAD__</script>
<script type="application/json" id="selection">__SELECTION__</script>
<script>
(function () {
  const DATA = JSON.parse(document.getElementById("payload").textContent);
  const accepted = new Set(DATA.initial_selection);

  function targetsFor(index) {
    return DATA.diffs[index].targets;
  }

  function removedIds() {
    const ids = new Set();
    for (const index of accepted) {
      for (const id of targetsFor(index)) ids.add(id);
    }
    return ids;
  }

  function reducedTokens() {
    const removed = removedIds();
    let saved = 0;
    for (const sentence of DATA.document.sentences) {
      if (removed.has(sentence.id)) saved += sentence.token_count;
    }
    let added = 0;
    for (const index of accepted) {
      added += DATA.diffs[index].replacement_token_count;
    }
    return DATA.document.token_count - saved + added;
  }

  function renderHunk(removed, limitTo) {
    const sentences = DATA.document.sentences;
    const targetSet = limitTo ? new Set(limitTo) : removed;
    const shown = new Set();
    for (let position = 0; position < sentences.length; position++) {
      if (targetSet.has(sentences[position].id)) {
        for (const index of [position - 1, position, position + 1]) {
          if (index >= 0 && index < sentences.length) shown.add(index);
        }
      }
    }
    if (!shown.size) {
      return '<div class="line empty-msg">(no edits accepted — output equals the original)</div>';
    }
    const parts = [];
    let previous = null;
    for (const index of [...shown].sort((a, b) => a - b)) {
      if (previous !== null && index > previous + 1) {
        parts.push('<div class="line gap">···</div>');
      }
      const sentence = sentences[index];
      const lines = sentence.text.split("\\n");
      if (!lines.length) lines.push("");
      for (const line of lines) {
        if (removed.has(sentence.id)) {
          parts.push('<div class="line removed">-' + esc(line) + '</div>');
        } else {
          parts.push('<div class="line context"> ' + esc(line) + '</div>');
        }
      }
      previous = index;
    }
    return parts.join("");
  }

  function esc(text) {
    return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  function writeSelection() {
    const indices = [...accepted].sort((a, b) => a - b);
    const blob = {
      schema_version: DATA.schema_version,
      accepted_indices: indices,
      accepted_count: indices.length,
      total_count: DATA.diffs.length,
    };
    document.getElementById("selection").textContent = JSON.stringify(blob);
  }

  function updateSummary() {
    const n = accepted.size;
    const total = DATA.diffs.length;
    const src = DATA.document.token_count;
    const dst = reducedTokens();
    document.getElementById("summary").textContent =
      n + "/" + total + " accepted · " + src + " → " + dst + " tokens";
  }

  function updateCumulative() {
    document.getElementById("cumulative").innerHTML = renderHunk(removedIds(), null);
  }

  function updateBlock(index) {
    const block = document.querySelector('.edit-block[data-index="' + index + '"]');
    if (!block) return;
    block.classList.remove("accepted", "rejected");
    if (accepted.has(index)) block.classList.add("accepted");
    else block.classList.add("rejected");
    const acceptBtn = block.querySelector(".accept");
    const rejectBtn = block.querySelector(".reject");
    acceptBtn.classList.toggle("active", accepted.has(index));
    rejectBtn.classList.toggle("active", !accepted.has(index));
  }

  function refresh() {
    updateSummary();
    updateCumulative();
    writeSelection();
    for (let i = 0; i < DATA.diffs.length; i++) updateBlock(i);
  }

  function setAccepted(index, value) {
    if (value) accepted.add(index);
    else accepted.delete(index);
    refresh();
  }

  document.getElementById("edits").addEventListener("click", (event) => {
    const button = event.target.closest("button");
    if (!button) return;
    const block = button.closest(".edit-block");
    if (!block) return;
    const index = Number(block.dataset.index);
    if (button.classList.contains("accept")) setAccepted(index, true);
    else if (button.classList.contains("reject")) setAccepted(index, false);
  });

  const acceptAll = document.getElementById("accept-all");
  const rejectAll = document.getElementById("reject-all");
  if (acceptAll) {
    acceptAll.addEventListener("click", () => {
      accepted.clear();
      for (let i = 0; i < DATA.diffs.length; i++) accepted.add(i);
      refresh();
    });
  }
  if (rejectAll) {
    rejectAll.addEventListener("click", () => {
      accepted.clear();
      refresh();
    });
  }

  refresh();
})();
</script>
</body>
</html>
"""


class SentencePreview(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str
    text: str
    token_count: int


class DocumentPreview(BaseModel):
    model_config = ConfigDict(frozen=True)
    token_count: int
    sentences: tuple[SentencePreview, ...]


class DiffPreview(BaseModel):
    """The per-diff data the page's JS needs: which sentences the edit removes and, for a
    Replace, how many tokens its replacement text adds back (0 for a Delete)."""

    model_config = ConfigDict(frozen=True)
    targets: tuple[str, ...]
    replacement_token_count: int


class ReviewPayload(BaseModel):
    model_config = ConfigDict(frozen=True)
    schema_version: Literal[1] = SCHEMA_VERSION
    document: DocumentPreview
    diffs: tuple[DiffPreview, ...]
    initial_selection: tuple[int, ...] = ()


def _location(diff: Diff) -> str:
    path = diff.spans[0].section_path
    return " > ".join(part for part in path if part) or "(top level)"


def _hunk_html(document: Document, removed: frozenset[SentenceId]) -> str:
    parts: list[str] = []
    for kind, text in hunk_lines(document, removed):
        escaped = html.escape(text)
        if kind == "removed":
            parts.append(f'<div class="line removed">-{escaped}</div>')
        elif kind == "added":
            parts.append(f'<div class="line added">+{escaped}</div>')
        elif kind == "context":
            parts.append(f'<div class="line context"> {escaped}</div>')
        elif kind == "gap":
            parts.append(f'<div class="line gap">{escaped}</div>')
        else:
            parts.append(f'<div class="line empty-msg">{escaped}</div>')
    return "\n".join(parts)


def _edit_block_html(document: Document, index: int, diff: Diff) -> str:
    candidate = diff.candidate
    targets = frozenset(candidate.edit.targets)
    hunk = _hunk_html(document, targets)
    if diff.replacement:
        # Appended after the whole hunk (not interleaved via hunk_lines' `added`) to keep the
        # replacement line at the block's end, where this per-edit view has always shown it.
        hunk += f'\n<div class="line added">+{html.escape(diff.replacement)}</div>'
    confidence = f"{candidate.confidence:.3f}"
    return f"""<article class="edit-block rejected" data-index="{index}">
  <div class="edit-meta">
    <div class="loc">{html.escape(_location(diff))}</div>
    <div class="detail">confidence: {confidence} · optimizer: {html.escape(candidate.source)}<br>
    reason: {html.escape(candidate.reason)}</div>
  </div>
  <div class="hunk">{hunk}</div>
  <div class="controls">
    <button type="button" class="accept">Accept</button>
    <button type="button" class="reject active">Reject</button>
  </div>
</article>"""


def _diff_preview(diff: Diff) -> DiffPreview:
    edit = diff.candidate.edit
    replacement_token_count = edit.replacement.token_count if edit.op == "replace" else 0
    return DiffPreview(targets=edit.targets, replacement_token_count=replacement_token_count)


def _document_preview(document: Document) -> DocumentPreview:
    return DocumentPreview(
        token_count=document.token_count,
        sentences=tuple(
            SentencePreview(id=sentence.id, text=sentence.text, token_count=sentence.token_count)
            for sentence in document.sentences
        ),
    )


def _initial_selection(total: int) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "accepted_indices": [],
        "accepted_count": 0,
        "total_count": total,
    }


def render_review_page(proposal: Proposal) -> str:
    """Render a self-contained HTML page for reviewing proposed edits."""
    payload = ReviewPayload(
        document=_document_preview(proposal.document),
        diffs=tuple(_diff_preview(diff) for diff in proposal.diffs),
    )
    payload_json = json.dumps(payload.model_dump(mode="json"))
    selection_json = json.dumps(_initial_selection(len(proposal.diffs)))

    if not proposal.diffs:
        edit_blocks = '<p class="empty">no proposed edits; the prompt is unchanged</p>'
        shortcuts = ""
    else:
        edit_blocks = "\n".join(
            _edit_block_html(proposal.document, index, diff) for index, diff in enumerate(proposal.diffs)
        )
        shortcuts = """<div class="shortcuts">
      <button type="button" id="accept-all">Accept all</button>
      <button type="button" id="reject-all">Reject all</button>
    </div>"""

    return (
        TEMPLATE.replace("__PAYLOAD__", payload_json)
        .replace("__SELECTION__", selection_json)
        .replace("__EDIT_BLOCKS__", edit_blocks)
        .replace("__SHORTCUTS__", shortcuts)
    )
