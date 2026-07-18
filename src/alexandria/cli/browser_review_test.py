from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

import click
import pytest

from alexandria.cli.browser_review import (
    ReviewServer,
    accepted_candidates,
    inject_bridge,
    parse_selection,
    reserve_port,
    run_browser_review,
    validate_selection_payload,
)
from alexandria.cli.review_html import render_review_page
from alexandria.ir.contracts import Params
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


def _post_json(url: str, payload: dict[str, Any]) -> int:
    request = urllib.request.Request(  # noqa: S310
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request) as response:  # noqa: S310
        return response.status


def test_parse_selection_round_trips_rendered_html() -> None:
    proposal = _reviewable_proposal()
    selection = parse_selection(render_review_page(proposal))

    assert selection == {
        "schema_version": 1,
        "accepted_indices": [],
        "accepted_count": 0,
        "total_count": len(proposal.diffs),
    }


def test_inject_bridge_adds_apply_cancel_and_port() -> None:
    proposal = _reviewable_proposal()
    html = inject_bridge(render_review_page(proposal), port=4242)

    assert "Apply selection" in html
    assert "Cancel" in html
    assert "http://127.0.0.1:4242/selection" in html


def test_validate_selection_payload_rejects_bad_schema_version() -> None:
    with pytest.raises(click.ClickException, match="schema_version"):
        validate_selection_payload({"schema_version": 2, "accepted_indices": [], "total_count": 1}, total_count=1)


def test_validate_selection_payload_rejects_out_of_range_index() -> None:
    with pytest.raises(click.ClickException, match="out of range"):
        validate_selection_payload(
            {"schema_version": 1, "accepted_indices": [99], "total_count": 2},
            total_count=2,
        )


def test_accepted_candidates_preserves_confidence_order() -> None:
    proposal = _reviewable_proposal()
    candidates = accepted_candidates(proposal, (1, 0))

    assert len(candidates) == 2
    assert candidates[0] is proposal.diffs[0].candidate
    assert candidates[1] is proposal.diffs[1].candidate


def test_review_server_returns_accepted_candidates_on_done() -> None:
    proposal = _reviewable_proposal()
    port = reserve_port()
    html = inject_bridge(render_review_page(proposal), port=port)
    server = ReviewServer(html, port=port)
    server.start()

    try:
        status = _post_json(
            f"http://127.0.0.1:{port}/selection",
            {
                "status": "done",
                "schema_version": 1,
                "accepted_indices": [0],
                "accepted_count": 1,
                "total_count": len(proposal.diffs),
            },
        )
        assert status == 204

        result = server.wait_for_selection()
        indices = validate_selection_payload(
            {
                "schema_version": 1,
                "accepted_indices": list(result.accepted_indices),
                "accepted_count": len(result.accepted_indices),
                "total_count": len(proposal.diffs),
            },
            total_count=len(proposal.diffs),
        )
        assert accepted_candidates(proposal, indices) == (proposal.diffs[0].candidate,)
    finally:
        server.shutdown()


def test_review_server_returns_none_on_abort() -> None:
    proposal = _reviewable_proposal()
    port = reserve_port()
    html = inject_bridge(render_review_page(proposal), port=port)
    server = ReviewServer(html, port=port)
    server.start()

    try:
        status = _post_json(f"http://127.0.0.1:{port}/selection", {"status": "aborted"})
        assert status == 204
        assert server.wait_for_selection().status == "aborted"
    finally:
        server.shutdown()


def test_review_server_rejects_invalid_done_payload() -> None:
    proposal = _reviewable_proposal()
    port = reserve_port()
    html = inject_bridge(render_review_page(proposal), port=port)
    server = ReviewServer(html, port=port)
    server.start()

    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/selection",
            data=json.dumps({"status": "done", "accepted_indices": ["nope"]}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)  # noqa: S310
        assert error.value.code == 400
    finally:
        server.shutdown()


def test_run_browser_review_returns_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    proposal = _reviewable_proposal()
    opened: list[str] = []
    monkeypatch.setattr("alexandria.cli.browser_review.webbrowser.open", opened.append)

    def fake_wait(_self: ReviewServer) -> Any:
        return type("Result", (), {"status": "done", "accepted_indices": (0,)})()

    monkeypatch.setattr(ReviewServer, "wait_for_selection", fake_wait)

    chosen = run_browser_review(proposal, open_browser=True)

    assert chosen == (proposal.diffs[0].candidate,)
    assert opened and opened[0].startswith("http://127.0.0.1:")


def test_run_browser_review_abort_keeps_temp_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    proposal = _reviewable_proposal()
    kept_dirs: list[str] = []

    def fake_mkdtemp(*_args: object, **_kwargs: object) -> str:
        path = tmp_path / "alexandria-review-test"
        path.mkdir()
        kept_dirs.append(str(path))
        return str(path)

    def aborted_wait(_self: ReviewServer) -> Any:
        return type("R", (), {"status": "aborted", "accepted_indices": ()})()

    monkeypatch.setattr("alexandria.cli.browser_review.tempfile.mkdtemp", fake_mkdtemp)
    monkeypatch.setattr(ReviewServer, "wait_for_selection", aborted_wait)

    assert run_browser_review(proposal, open_browser=False) is None
    assert kept_dirs
    assert (tmp_path / "alexandria-review-test" / "review.html").exists()


def test_run_browser_review_success_cleans_temp_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    proposal = _reviewable_proposal()
    temp_dir = tmp_path / "alexandria-review-success"
    temp_dir.mkdir()

    def done_wait(_self: ReviewServer) -> Any:
        return type("R", (), {"status": "done", "accepted_indices": ()})()

    def fake_mkdtemp_success(*_args: object, **_kwargs: object) -> str:
        return str(temp_dir)

    monkeypatch.setattr("alexandria.cli.browser_review.tempfile.mkdtemp", fake_mkdtemp_success)
    monkeypatch.setattr(ReviewServer, "wait_for_selection", done_wait)

    assert run_browser_review(proposal, open_browser=False) == ()
    assert not temp_dir.exists()


def test_run_browser_review_error_keeps_temp_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    proposal = _reviewable_proposal()
    temp_dir = tmp_path / "alexandria-review-error"
    temp_dir.mkdir()

    def fail_start(_self: ReviewServer) -> None:
        raise RuntimeError("server failed to start")

    def fake_mkdtemp_success(*_args: object, **_kwargs: object) -> str:
        return str(temp_dir)

    monkeypatch.setattr("alexandria.cli.browser_review.tempfile.mkdtemp", fake_mkdtemp_success)
    monkeypatch.setattr(ReviewServer, "start", fail_start)

    with pytest.raises(RuntimeError, match="server failed to start"):
        run_browser_review(proposal, open_browser=False)

    assert temp_dir.exists()
