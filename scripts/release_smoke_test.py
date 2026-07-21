"""Build the release wheel and verify its installed CLI and Python API."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

PROMPT = "repeat me\nrepeat me\nunique line\n"


def _run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        rendered = " ".join(command)
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {rendered}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result


def _venv_executable(venv: Path, name: str) -> Path:
    scripts_dir = "Scripts" if os.name == "nt" else "bin"
    suffix = ".exe" if os.name == "nt" else ""
    return venv / scripts_dir / f"{name}{suffix}"


def _hash_embedding(text: str, dimensions: int = 32) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values = [((digest[index % len(digest)] / 255.0) * 2.0) - 1.0 for index in range(dimensions)]
    norm = math.sqrt(sum(value * value for value in values))
    return [value / norm for value in values]


class _OpenAIMockHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length))
        if self.path != "/v1/embeddings":
            self._send_json(500, {"error": {"message": f"unexpected smoke-test endpoint: {self.path}"}})
            return

        raw_inputs = request.get("input", [])
        inputs = [raw_inputs] if isinstance(raw_inputs, str) else raw_inputs
        response = {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": index,
                    "embedding": _hash_embedding(str(text)),
                }
                for index, text in enumerate(inputs)
            ],
            "model": request.get("model", "text-embedding-3-small"),
            "usage": {
                "prompt_tokens": sum(max(1, len(str(text).split())) for text in inputs),
                "total_tokens": sum(max(1, len(str(text).split())) for text in inputs),
            },
        }
        self._send_json(200, response)

    def log_message(self, format: str, *args: Any) -> None:
        del format, args

    def _send_json(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@contextmanager
def _openai_mock_server() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _OpenAIMockHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _runtime_env(base_url: str) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.update(
        {
            "NO_PROXY": "127.0.0.1,localhost",
            "OPENAI_API_KEY": "release-smoke-test-key",
            "OPENAI_BASE_URL": base_url,
            "PYTHONNOUSERSITE": "1",
        }
    )
    return env


def _assert_cli_reduction(result: subprocess.CompletedProcess[str]) -> None:
    payload = json.loads(result.stdout)
    if payload["reduced_tokens"] >= payload["source_tokens"]:
        raise RuntimeError(f"Installed CLI did not reduce the fixture: {result.stdout}")
    if payload["text"] != "repeat me\nunique line\n":
        raise RuntimeError(f"Installed CLI returned unexpected text: {result.stdout}")


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required to run the release installation smoke test")

    with tempfile.TemporaryDirectory(prefix="alexandria-release-smoke-") as temp_dir:
        workspace = Path(temp_dir)
        source = workspace / "source"
        dist_dir = workspace / "dist"
        venv = workspace / ".venv"

        source.mkdir()
        for filename in ("LICENSE", "README.md", "pyproject.toml"):
            shutil.copy2(repository / filename, source / filename)
        shutil.copytree(repository / "src", source / "src")

        _run([uv, "venv", "--python", "3.14", str(venv)], cwd=workspace)
        python = _venv_executable(venv, "python")
        alexandria = _venv_executable(venv, "alexandria")
        _run(
            [uv, "build", "--python", str(python), "--out-dir", str(dist_dir)],
            cwd=source,
        )
        wheels = list(dist_dir.glob("*.whl"))
        source_distributions = list(dist_dir.glob("*.tar.gz"))
        if len(wheels) != 1 or len(source_distributions) != 1:
            raise RuntimeError(
                "Expected one wheel and one source distribution, found "
                f"{len(wheels)} wheel(s) and {len(source_distributions)} source distribution(s) in {dist_dir}"
            )
        _run([uv, "pip", "install", "--python", str(python), str(wheels[0])], cwd=workspace)

        prompt_path = workspace / "prompt.txt"
        prompt_path.write_text(PROMPT, encoding="utf-8")
        with _openai_mock_server() as base_url:
            cli_result = _run(
                [
                    str(alexandria),
                    "reduce",
                    str(prompt_path),
                    "--cos-sim-diff-budget",
                    "2.0",
                    "--json",
                ],
                cwd=workspace,
                env=_runtime_env(base_url),
            )
        _assert_cli_reduction(cli_result)

        api_program = """
import importlib.metadata
import json
from pathlib import Path

import alexandria
from alexandria.ir.contracts import Params
from alexandria.ops import HashEmbedder


class FirstWinsMerger:
    def merge(self, first: str, second: str, feedback: str | None = None) -> str:
        del second, feedback
        return first


result = alexandria.reduce(
    "repeat me\\nrepeat me\\nunique line\\n",
    HashEmbedder(),
    FirstWinsMerger(),
    params=Params(cos_sim_diff_budget=2.0),
)
if result.reduced_tokens >= result.source_tokens:
    raise SystemExit("installed Python API did not reduce the fixture")
if result.text != "repeat me\\nunique line\\n":
    raise SystemExit("installed Python API returned unexpected text")
print(json.dumps({
    "module": str(Path(alexandria.__file__).resolve()),
    "version": importlib.metadata.version("alexandria-prompt"),
    "source_tokens": result.source_tokens,
    "reduced_tokens": result.reduced_tokens,
}))
"""
        api_result = _run([str(python), "-c", api_program], cwd=workspace, env=_runtime_env("http://127.0.0.1"))
        api_payload = json.loads(api_result.stdout)
        installed_module = Path(api_payload["module"])
        if venv.resolve() not in installed_module.parents:
            raise RuntimeError(
                f"Python smoke test did not import from the clean virtual environment: {installed_module}"
            )

        print("Release installation smoke test passed.")
        print(f"Wheel: {wheels[0].name}")
        print(f"Source distribution: {source_distributions[0].name}")
        print(f"Installed module: {installed_module}")
        print(f"Token reduction: {api_payload['source_tokens']} -> {api_payload['reduced_tokens']}")


if __name__ == "__main__":
    main()
