from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pytest
from openai.resources.responses.responses import Responses

from benchmarks.prompt_compression.metering import OpenAIUsageMeter, estimate_cost

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
