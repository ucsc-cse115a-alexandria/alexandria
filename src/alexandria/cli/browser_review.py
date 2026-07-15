"""Localhost bridge for reviewing proposed edits in a browser."""

from __future__ import annotations

import json
import re
import shutil
import socket
import tempfile
import threading
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import click

from alexandria.cli.review_html import SCHEMA_VERSION, render_review_page

if TYPE_CHECKING:
    from alexandria.ir.contracts import Candidate
    from alexandria.ops.pipe import Proposal

_SELECTION_RE = re.compile(
    r'<script type="application/json" id="selection">(.*?)</script>',
    re.DOTALL,
)


@dataclass(frozen=True)
class SelectionResult:
    status: Literal["done", "aborted"]
    accepted_indices: tuple[int, ...] = ()


def parse_selection(html: str) -> dict[str, Any]:
    """Extract the embedded #selection JSON blob from a review page."""
    match = _SELECTION_RE.search(html)
    if match is None:
        raise click.ClickException("review page is missing a #selection block")
    return json.loads(match.group(1))


def _validate_selection_payload(payload: dict[str, Any], *, total_count: int) -> tuple[int, ...]:
    schema_version = payload.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise click.ClickException(f"unsupported selection schema_version: {schema_version!r}")

    raw_indices = payload.get("accepted_indices")
    if not isinstance(raw_indices, list) or not all(isinstance(index, int) for index in raw_indices):
        raise click.ClickException("accepted_indices must be a list of integers")

    total = payload.get("total_count")
    if total != total_count:
        raise click.ClickException(f"selection total_count {total!r} does not match {total_count} proposed edits")

    accepted = tuple(raw_indices)
    if len(accepted) != len(set(accepted)):
        raise click.ClickException("accepted_indices contains duplicates")

    for index in accepted:
        if index < 0 or index >= total_count:
            raise click.ClickException(f"accepted index {index} is out of range for {total_count} proposed edits")

    return accepted


def accepted_candidates(proposal: Proposal, indices: tuple[int, ...]) -> tuple[Candidate, ...]:
    """Map accepted diff indices to candidates in list (confidence) order."""
    accepted = frozenset(indices)
    return tuple(diff.candidate for index, diff in enumerate(proposal.diffs) if index in accepted)


def inject_bridge(html: str, port: int) -> str:
    """Inject Apply/Cancel controls and a localhost POST bridge before </body>."""
    bridge = f"""
<div id="alexandria-bridge" style="
  position: fixed; bottom: 0; left: 0; right: 0; z-index: 20;
  display: flex; justify-content: center; gap: 12px; padding: 14px 20px;
  background: rgba(247, 248, 250, 0.96); border-top: 1px solid #e7e9ee;
  box-shadow: 0 -4px 16px rgba(16, 24, 40, 0.08);
  font-family: ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif;
">
  <button type="button" id="alexandria-apply" style="
    font: inherit; font-size: 0.95rem; font-weight: 600; padding: 10px 22px;
    border-radius: 8px; cursor: pointer; border: 1px solid #16a34a;
    background: #f0fdf4; color: #16a34a;
  ">Apply selection</button>
  <button type="button" id="alexandria-cancel" style="
    font: inherit; font-size: 0.95rem; font-weight: 600; padding: 10px 22px;
    border-radius: 8px; cursor: pointer; border: 1px solid #dc2626;
    background: #fef2f2; color: #dc2626;
  ">Cancel</button>
</div>
<script>
(function () {{
  const endpoint = "http://127.0.0.1:{port}/selection";

  function currentSelection() {{
    return JSON.parse(document.getElementById("selection").textContent);
  }}

  function post(body) {{
    return fetch(endpoint, {{
      method: "POST",
      headers: {{"Content-Type": "application/json"}},
      body: JSON.stringify(body),
    }});
  }}

  function disableButtons() {{
    for (const button of document.querySelectorAll("#alexandria-bridge button")) {{
      button.disabled = true;
      button.style.opacity = "0.6";
      button.style.cursor = "default";
    }}
  }}

  document.getElementById("alexandria-apply").addEventListener("click", () => {{
    disableButtons();
    post(Object.assign({{status: "done"}}, currentSelection()));
  }});

  document.getElementById("alexandria-cancel").addEventListener("click", () => {{
    disableButtons();
    post({{status: "aborted"}});
  }});
}})();
</script>
"""
    marker = "</body>"
    if marker not in html:
        raise click.ClickException("review page is missing a </body> tag")
    return html.replace(marker, bridge + marker, 1)


def _reserve_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class ReviewServer:
    """Serve the injected review page and collect Apply/Cancel POSTs."""

    def __init__(self, html: str, *, port: int) -> None:
        self._html = html.encode("utf-8")
        self._result: SelectionResult | None = None
        self._event = threading.Event()
        self._httpd = self._build_server(port)
        self.port = self._httpd.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}/"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def _build_server(self, port: int) -> ThreadingHTTPServer:
        parent = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, _format: str, *_args: object) -> None:
                return

            def do_GET(self) -> None:
                if self.path != "/":
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(parent._html)))
                self.end_headers()
                self.wfile.write(parent._html)

            def do_POST(self) -> None:
                if self.path != "/selection":
                    self.send_error(404)
                    return
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length)
                try:
                    payload = json.loads(body)
                except json.JSONDecodeError as error:
                    self.send_error(400, explain=str(error))
                    return

                status = payload.get("status")
                if status == "aborted":
                    parent._result = SelectionResult(status="aborted")
                elif status == "done":
                    accepted = payload.get("accepted_indices", [])
                    if not isinstance(accepted, list) or not all(isinstance(index, int) for index in accepted):
                        self.send_error(400, explain="accepted_indices must be a list of integers")
                        return
                    parent._result = SelectionResult(status="done", accepted_indices=tuple(accepted))
                else:
                    self.send_error(400, explain="status must be 'done' or 'aborted'")
                    return

                parent._event.set()
                self.send_response(204)
                self.end_headers()

        return ThreadingHTTPServer(("127.0.0.1", port), Handler)

    def start(self) -> None:
        self._thread.start()

    def wait_for_selection(self) -> SelectionResult:
        try:
            self._event.wait()
        except KeyboardInterrupt:
            return SelectionResult(status="aborted")
        if self._result is None:
            raise click.ClickException("review ended without a selection")
        return self._result

    def shutdown(self) -> None:
        self._httpd.shutdown()
        self._thread.join(timeout=5)


def run_browser_review(proposal: Proposal, *, open_browser: bool = True) -> tuple[Candidate, ...] | None:
    """Generate the #54 page, serve it locally, and return accepted candidates or None on abort."""
    temp_dir = Path(tempfile.mkdtemp(prefix="alexandria-review-"))
    cleanup = False
    server: ReviewServer | None = None
    server_started = False
    try:
        port = _reserve_port()
        html = inject_bridge(render_review_page(proposal), port)
        (temp_dir / "review.html").write_text(html, encoding="utf-8")

        server = ReviewServer(html, port=port)
        server.start()
        server_started = True

        click.echo(f"review page: {server.url}", err=True)
        if open_browser:
            webbrowser.open(server.url)

        result = server.wait_for_selection()
        if result.status == "aborted":
            return None

        indices = _validate_selection_payload(
            {
                "schema_version": SCHEMA_VERSION,
                "accepted_indices": list(result.accepted_indices),
                "accepted_count": len(result.accepted_indices),
                "total_count": len(proposal.diffs),
            },
            total_count=len(proposal.diffs),
        )
        cleanup = True
        return accepted_candidates(proposal, indices)
    finally:
        if server is not None and server_started:
            server.shutdown()
        if cleanup:
            shutil.rmtree(temp_dir, ignore_errors=True)
