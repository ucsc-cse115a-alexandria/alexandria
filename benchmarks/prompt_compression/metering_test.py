from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pytest
from openai.resources.responses.responses import Responses

from benchmarks.prompt_compression.contracts import UsageRecord
from benchmarks.prompt_compression.metering import OpenAIUsageMeter, estimate_cost, pricing_for_model

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch


@dataclass
class _InputDetails:
    cached_tokens: int = 3


@dataclass
class _Usage:
    input_tokens: int = 10
    output_tokens: int = 2
    total_tokens: int = 12
    input_tokens_details: _InputDetails = field(default_factory=_InputDetails)


@dataclass
class _Response:
    model: str = "answer-model"
    usage: _Usage = field(default_factory=_Usage)


def test_usage_meter_records_category_tokens_and_estimated_cost(monkeypatch: MonkeyPatch) -> None:
    def fake_create(_resource: object, *args: Any, **kwargs: Any) -> _Response:  # noqa: ARG001
        return _Response()

    monkeypatch.setattr(Responses, "create", fake_create)
    meter = OpenAIUsageMeter()
    with meter, meter.category("answer"):
        Responses.create(None)  # type: ignore[arg-type]

    assert Responses.create is fake_create
    assert len(meter.records) == 1
    record = meter.records[0]
    assert record.category == "answer"
    assert record.cached_input_tokens == 3
    assert estimate_cost((record,)) == pytest.approx((7 * 1.00 + 3 * 0.10 + 2 * 6.00) / 1_000_000)


def test_nano_usage_uses_its_lower_short_context_price() -> None:
    record = UsageRecord(
        category="answer",
        model="gpt-5.4-nano-2026-07-01",
        input_tokens=1_000_000,
        cached_input_tokens=100_000,
        output_tokens=100_000,
        total_tokens=1_100_000,
        elapsed_seconds=1.0,
    )

    assert pricing_for_model(record.model)["input"] == 0.20
    assert estimate_cost((record,)) == pytest.approx(0.9 * 0.20 + 0.1 * 0.02 + 0.1 * 1.25)
