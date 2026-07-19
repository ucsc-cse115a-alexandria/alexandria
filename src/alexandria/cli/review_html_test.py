from __future__ import annotations

import json
import re
from typing import Any

from alexandria.cli.review_html import render_review_page
from alexandria.ir.contracts import Candidate, Delete, Diff, DiffSpan, Params, Replace
from alexandria.ir.document import Encoded
from alexandria.ops import HashEmbedder, Proposal, diffs, optimize, represent, score

_REDUNDANT = "# Alpha\nrepeat me\nrepeat me\n# Beta\necho twice\necho twice\n"


class _FirstWinsMerger:
    """Offline merger: the first sentence wins, so an exact-duplicate pair becomes a delete."""

    def merge(self, first: str, second: str, feedback: str | None = None) -> str:
        del second, feedback
        return first.strip()


def _reviewable_proposal() -> Proposal:
    embedder = HashEmbedder()
    document = represent(_REDUNDANT, embedder)
    plan = optimize(
        document, score(document, names=("redundancy",)), embedder, _FirstWinsMerger(), params=Params(drift_budget=2.0)
    )
    return Proposal(document=document, diffs=diffs(document, plan))


def _empty_proposal() -> Proposal:
    document = represent("only one unique line\n", HashEmbedder())
    return Proposal(document=document, diffs=())


def _parse_selection(html: str) -> dict[str, Any]:
    match = re.search(r'<script type="application/json" id="selection">(.*?)</script>', html, re.DOTALL)
    assert match is not None
    return json.loads(match.group(1))


def _parse_payload(html: str) -> dict[str, Any]:
    match = re.search(r'<script type="application/json" id="payload">(.*?)</script>', html, re.DOTALL)
    assert match is not None
    return json.loads(match.group(1))


def test_render_includes_one_block_per_diff() -> None:
    proposal = _reviewable_proposal()
    html = render_review_page(proposal)

    assert html.count('class="edit-block') == len(proposal.diffs)


def test_render_embeds_diff_metadata() -> None:
    proposal = _reviewable_proposal()
    diff = proposal.diffs[0]
    html = render_review_page(proposal)

    assert diff.candidate.reason in html
    assert diff.candidate.source in html
    assert f"{diff.candidate.confidence:.3f}" in html
    assert diff.spans[0].original.strip() in html


def test_render_escapes_html_in_metadata() -> None:
    document = represent("<unsafe>\n", HashEmbedder())
    sentence_id = document.sentences[0].id
    diff = Diff(
        candidate=Candidate(
            edit=Delete(targets=(sentence_id,)),
            confidence=0.5,
            source="<script>",
            reason='say "hello" & <goodbye>',
        ),
        spans=(
            DiffSpan(
                sentence_id=sentence_id,
                section_path=("<top>",),
                original=document.sentences[0].text,
            ),
        ),
        replacement="",
    )
    proposal = Proposal(document=document, diffs=(diff,))
    html = render_review_page(proposal)

    assert "&lt;script&gt;" in html
    assert 'class="line removed">-&lt;unsafe&gt;' in html
    assert "&amp;" in html
    assert "optimizer: &lt;script&gt;" in html
    assert "&lt;goodbye&gt;" in html


def test_render_initial_selection_empty() -> None:
    proposal = _reviewable_proposal()
    selection = _parse_selection(render_review_page(proposal))

    assert selection == {
        "schema_version": 1,
        "accepted_indices": [],
        "accepted_count": 0,
        "total_count": len(proposal.diffs),
    }


def test_render_includes_document_token_count() -> None:
    proposal = _reviewable_proposal()
    html = render_review_page(proposal)

    assert str(proposal.document.token_count) in html
    payload = _parse_payload(html)
    assert payload["document"]["token_count"] == proposal.document.token_count


def test_render_empty_proposal_shows_message() -> None:
    html = render_review_page(_empty_proposal())

    assert "no proposed edits" in html
    assert 'class="edit-block' not in html
    selection = _parse_selection(html)
    assert selection["total_count"] == 0


def test_render_payload_round_trips() -> None:
    proposal = _reviewable_proposal()
    payload = _parse_payload(render_review_page(proposal))

    assert payload["schema_version"] == 1
    assert len(payload["diffs"]) == len(proposal.diffs)
    assert len(payload["document"]["sentences"]) == len(proposal.document.sentences)
    assert payload["initial_selection"] == []


def test_render_shows_the_replacement_line_for_a_replace_diff() -> None:
    document = represent(_REDUNDANT, HashEmbedder())
    ids = [s.id for s in document.sentences]
    replacement = Encoded(text="merged line\n", token_count=2, embedding=document.sentences[0].embedding)
    candidate = Candidate(
        edit=Replace(targets=(ids[1], ids[2]), replacement=replacement), confidence=0.9, source="t", reason="r"
    )
    proposal = Proposal(document=document, diffs=diffs(document, (candidate,)))

    page = render_review_page(proposal)

    assert '<div class="line added">+merged line' in page


def test_payload_diff_carries_targets_and_replacement_token_count() -> None:
    document = represent(_REDUNDANT, HashEmbedder())
    ids = [s.id for s in document.sentences]
    replacement = Encoded(text="merged line\n", token_count=2, embedding=document.sentences[0].embedding)
    replace = Candidate(
        edit=Replace(targets=(ids[1], ids[2]), replacement=replacement), confidence=0.9, source="t", reason="r"
    )
    delete = Candidate(edit=Delete(targets=(ids[4],)), confidence=0.8, source="t", reason="r")
    proposal = Proposal(document=document, diffs=diffs(document, (replace, delete)))

    payload = _parse_payload(render_review_page(proposal))

    replace_entry, delete_entry = payload["diffs"]
    assert replace_entry["targets"] == [ids[1], ids[2]]
    assert replace_entry["replacement_token_count"] == 2
    assert delete_entry["targets"] == [ids[4]]
    assert delete_entry["replacement_token_count"] == 0
