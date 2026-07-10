from __future__ import annotations

import re

from alexandria.cli.interactive import ReviewState, apply_candidates, render, review
from alexandria.ir.contracts import Candidate, Delete, Diff, DiffSpan
from alexandria.ir.document import Document, SentenceId
from alexandria.ops import DETERMINISTIC, build_embedder, diffs, optimize, represent, score

_REDUNDANT = "# Alpha\nrepeat me\nrepeat me\n# Beta\necho twice\necho twice\n"


def _reviewable() -> tuple[Document, tuple[Diff, ...]]:
    document = represent(_REDUNDANT, build_embedder(DETERMINISTIC))
    plan = optimize(document, score(document, names=("redundancy",)))
    return document, diffs(document, plan)


def _diff(tag: str, confidence: float) -> Diff:
    sentence_id = SentenceId(f"s-{tag}")
    return Diff(
        candidate=Candidate(edit=Delete(targets=(sentence_id,)), confidence=confidence, source="t", reason="r"),
        spans=(DiffSpan(sentence_id=sentence_id, section_path=("S",), original=f"{tag}\n"),),
        replacement="",
    )


def test_move_clamps_cursor_at_both_ends() -> None:
    state = ReviewState(diffs=(_diff("a", 0.9), _diff("b", 0.8)), cursor=0, accepted=frozenset())

    assert state.move(-1).cursor == 0
    assert state.move(1).cursor == 1
    assert state.move(1).move(1).cursor == 1


def test_toggle_flips_only_the_cursor_row() -> None:
    state = ReviewState(diffs=(_diff("a", 0.9), _diff("b", 0.8)), cursor=1, accepted=frozenset())

    toggled = state.toggle()
    assert toggled.accepted == frozenset({1})
    assert toggled.toggle().accepted == frozenset()


def test_toggle_all_flips_between_everything_and_nothing() -> None:
    state = ReviewState(diffs=(_diff("a", 0.9), _diff("b", 0.8)), cursor=0, accepted=frozenset({1}))

    everything = state.toggle_all()
    assert everything.accepted == frozenset({0, 1})
    assert everything.toggle_all().accepted == frozenset()


def test_accepted_candidates_keep_list_order() -> None:
    diffs = (_diff("a", 0.9), _diff("b", 0.8), _diff("c", 0.7))
    state = ReviewState(diffs=diffs, cursor=0, accepted=frozenset({2, 0}))

    assert state.accepted_candidates() == (diffs[0].candidate, diffs[2].candidate)


def test_apply_candidates_keeps_accepted_and_drops_rejected() -> None:
    document, proposed = _reviewable()
    assert len(proposed) == 2
    accepted_diff = next(d for d in proposed if d.spans[0].original == "repeat me\n")

    reduced = apply_candidates(document, (accepted_diff.candidate,))

    assert reduced.text.count("repeat me") == 1
    assert reduced.text.count("echo twice") == 2


def _scripted_review(document: Document, proposed: tuple[Diff, ...], keys: str) -> tuple[Candidate, ...] | None:
    feed = iter(keys)
    frames: list[str] = []
    return review(document, proposed, read_key=lambda: next(feed), write=frames.append)


def test_review_with_scripted_keys_keeps_accepted_and_drops_rejected() -> None:
    document, proposed = _reviewable()

    accepted = _scripted_review(document, proposed, "\rjd")  # toggle first, move down, done

    assert accepted == (proposed[0].candidate,)


def test_review_quit_returns_none() -> None:
    document, proposed = _reviewable()

    assert _scripted_review(document, proposed, "\raq") is None


def test_toggle_detail_round_trips() -> None:
    state = ReviewState(diffs=(_diff("a", 0.9),), cursor=0, accepted=frozenset())

    assert state.detail is False
    assert state.toggle_detail().detail is True
    assert state.toggle_detail().toggle_detail().detail is False


def test_render_preview_builds_hunks_from_the_removed_sentences() -> None:
    # The preview must come from the known removed sentences, not a text diff: on large
    # documents difflib's junk heuristics misalign a pure one-line deletion into huge
    # phantom "+" blocks (reproduced on skill-corpus inflate/10, 872 "+" lines).
    filler_a = "".join(f"alpha paragraph {i} says something unique.\n\n" for i in range(80))
    filler_b = "".join(f"beta paragraph {i} says something else entirely.\n\n" for i in range(80))
    prompt = f"repeat me\nrepeat me\n{filler_a}{filler_b}echo twice\necho twice\n"
    document = represent(prompt, build_embedder(DETERMINISTIC))
    plan = optimize(document, score(document, names=("redundancy",)))
    proposed = diffs(document, plan)
    state = ReviewState(diffs=proposed, cursor=0, accepted=frozenset(range(len(proposed))))

    frame = render(state, document, (120, 40))

    plain = re.sub(r"\x1b\[[0-9;]*m", "", frame)
    preview = plain.split("─ diff (original → selection) ")[1]
    assert "-repeat me" in preview
    assert "-echo twice" in preview
    assert "···" in preview  # the two removals are far apart: one gap marker between hunks
    assert not any(line.startswith("+") for line in preview.splitlines())


def test_render_detail_shows_the_full_edit_for_the_cursor_row() -> None:
    document, proposed = _reviewable()
    state = ReviewState(diffs=proposed, cursor=0, accepted=frozenset(), detail=True)

    frame = render(state, document, (80, 24))

    assert proposed[0].candidate.reason in frame
    assert proposed[0].candidate.source in frame
    assert "edit detail" in frame
    plain = re.sub(r"\x1b\[[0-9;]*m", "", frame)
    assert "-repeat me" in plain  # what gets removed, with its surrounding context
    assert " repeat me" in plain  # the kept twin right next to it


def test_review_s_shows_detail_and_d_confirms() -> None:
    document, proposed = _reviewable()
    feed = iter("s\rd")  # open detail, toggle the first edit, done
    frames: list[str] = []

    accepted = review(document, proposed, read_key=lambda: next(feed), write=frames.append)

    assert accepted == (proposed[0].candidate,)
    assert any("edit detail" in frame for frame in frames)


def test_render_marks_cursor_checkboxes_and_previews_only_accepted() -> None:
    document, proposed = _reviewable()
    state = ReviewState(diffs=proposed, cursor=0, accepted=frozenset({0}))

    frame = render(state, document, (80, 24))

    assert "1/2 accepted" in frame
    assert "▸ [x]" in frame  # cursor row is the accepted one
    assert "[ ]" in frame
    assert "-repeat me" in frame  # the accepted edit shows as a removal in the preview
    assert "-echo twice" not in frame  # the rejected edit does not


def test_render_marks_hidden_rows_when_the_list_overflows() -> None:
    document, _ = _reviewable()
    many = tuple(_diff(f"t{i}", 0.99 - i * 0.01) for i in range(10))

    top = render(ReviewState(diffs=many, cursor=0, accepted=frozenset()), document, (80, 14))
    bottom = render(ReviewState(diffs=many, cursor=9, accepted=frozenset()), document, (80, 14))

    assert "↓ 9 more" in top
    assert "↑ 9 more" in bottom
