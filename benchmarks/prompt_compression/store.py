from __future__ import annotations

import gzip
import hashlib
import json
from typing import TYPE_CHECKING

from benchmarks.prompt_compression.contracts import ConditionRecord

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from pathlib import Path


class RunStore:
    """Append-only case checkpoints plus compressed exact-prompt artifacts."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.records_path = output_dir / "records.jsonl"
        self.prompts_path = output_dir / "prompts.jsonl.gz"
        self.summary_path = output_dir / "summary.json"
        self.report_path = output_dir / "report.md"
        output_dir.mkdir(parents=True, exist_ok=True)

    def load_records(self) -> tuple[ConditionRecord, ...]:
        if not self.records_path.exists():
            return ()
        return tuple(
            ConditionRecord.model_validate_json(line)
            for line in self.records_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )

    def completed_keys(self) -> frozenset[tuple[str, str]]:
        return frozenset((record.case_key, record.condition) for record in self.load_records())

    def append(self, record: ConditionRecord, prompt: str) -> None:
        expected_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if expected_hash != record.prompt_sha256:
            raise ValueError("prompt hash does not match condition record")
        with self.records_path.open("a", encoding="utf-8") as output:
            output.write(record.model_dump_json() + "\n")
            output.flush()
        prompt_payload = {"case_key": record.case_key, "condition": record.condition, "prompt": prompt}
        with gzip.open(self.prompts_path, "at", encoding="utf-8") as output:
            output.write(json.dumps(prompt_payload, ensure_ascii=False) + "\n")

    def write_manifest(self, manifest: Mapping[str, object]) -> None:
        path = self.output_dir / "manifest.json"
        rendered = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            candidate = json.loads(rendered)
            existing.pop("command", None)
            candidate.pop("command", None)
            if existing != candidate:
                raise ValueError(
                    f"{path} already describes a different run; resume with identical options or choose a new --out"
                )
            return
        path.write_text(rendered, encoding="utf-8")

    def write_summary(self, summary: Mapping[str, object], report: str) -> None:
        temporary = self.summary_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(self.summary_path)
        self.report_path.write_text(report.rstrip() + "\n", encoding="utf-8")

    def append_errors(self, errors: Iterable[Mapping[str, object]]) -> None:
        path = self.output_dir / "errors.jsonl"
        with path.open("a", encoding="utf-8") as output:
            for error in errors:
                output.write(json.dumps(error, ensure_ascii=False) + "\n")
