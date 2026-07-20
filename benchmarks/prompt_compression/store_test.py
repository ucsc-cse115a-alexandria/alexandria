from __future__ import annotations

import gzip
import hashlib
import json
from typing import TYPE_CHECKING

import pytest

from benchmarks.prompt_compression.contracts import BenchmarkVerdict, ConditionRecord
from benchmarks.prompt_compression.store import RunStore

if TYPE_CHECKING:
    from pathlib import Path


def _record(prompt: str) -> ConditionRecord:
    return ConditionRecord(
        case_key="case:1",
        benchmark="fixture",
        task="qa",
        condition="original",
        reduction_percent=0,
        source_tokens=3,
        target_tokens=3,
        sent_tokens=3,
        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        response="A",
        response_model="stub",
        verdict=BenchmarkVerdict(score=1.0, correct=True, parsed_answer="A"),
    )


def test_run_store_checkpoints_records_and_exact_prompts(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")
    record = _record("full prompt")
    store.append(record, "full prompt")
    assert store.load_records() == (record,)
    assert store.completed_keys() == frozenset({("case:1", "original")})
    with gzip.open(store.prompts_path, "rt", encoding="utf-8") as source:
        assert json.loads(source.readline())["prompt"] == "full prompt"


def test_run_store_drops_legacy_provider_response_id(tmp_path: Path) -> None:
    prompt = "full prompt"
    legacy_payload = _record(prompt).model_dump(mode="json")
    legacy_payload["response_id"] = "resp_private"
    record = ConditionRecord.model_validate(legacy_payload)

    store = RunStore(tmp_path / "run")
    store.append(record, prompt)

    assert "response_id" not in json.loads(store.records_path.read_text())


def test_run_store_rejects_wrong_prompt_hash(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="hash"):
        RunStore(tmp_path).append(_record("one"), "two")


def test_run_store_refuses_to_mix_different_manifests(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    store.write_manifest({"seed": 42, "command": "first"})
    store.write_manifest({"seed": 42, "command": "equivalent rerun"})
    assert json.loads((tmp_path / "manifest.json").read_text())["command"] == "first"
    with pytest.raises(ValueError, match="different run"):
        store.write_manifest({"seed": 7})


def test_run_store_flushes_api_events_as_jsonl(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    store.append_api_event({"case_key": "case:1", "status": "completed", "estimated_cost_usd": 0.25})

    assert json.loads(store.api_events_path.read_text()) == {
        "case_key": "case:1",
        "status": "completed",
        "estimated_cost_usd": 0.25,
    }
    assert store.api_event_cost() == 0.25


def test_run_store_loads_persisted_errors(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    store.append_errors(({"condition": "budget0p02", "terminal": True},))

    assert store.load_errors() == ({"condition": "budget0p02", "terminal": True},)
