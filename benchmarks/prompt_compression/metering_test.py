from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pytest
from openai.resources.responses.responses import Responses

from benchmarks.prompt_compression.contracts import UsageRecord
from benchmarks.prompt_compression.metering import (
    OpenAIUsageMeter,
    UsageLimitExceeded,
    estimate_cost,
    pricing_for_model,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

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
    id: str = "resp_123"
    model: str = "answer-model"
    usage: _Usage = field(default_factory=_Usage)


def _fake_create(_resource: object, *args: Any, **kwargs: Any) -> _Response:  # noqa: ARG001
    return _Response()


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


def test_usage_meter_flushes_scoped_event_immediately(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(Responses, "create", _fake_create)
    events: list[dict[str, object]] = []

    def capture(event: Mapping[str, object]) -> None:
        events.append(dict(event))

    with (
        OpenAIUsageMeter(capture) as meter,
        meter.scope(case_key="case:1", condition="budget0p02"),
        meter.category("answer"),
    ):
        Responses.create(None)  # type: ignore[arg-type]

    assert events[0]["case_key"] == "case:1"
    assert events[0]["condition"] == "budget0p02"
    assert events[0]["response_id"] == "resp_123"
    assert events[0]["status"] == "completed"
    cost = events[0]["estimated_cost_usd"]
    assert isinstance(cost, int | float)
    assert cost > 0.0


def test_usage_meter_stops_before_exceeding_generation_call_limit(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(Responses, "create", _fake_create)
    events: list[dict[str, object]] = []

    def capture(event: Mapping[str, object]) -> None:
        events.append(dict(event))

    with (
        OpenAIUsageMeter(capture) as meter,
        meter.scope(case_key="case:1", condition="budget0p02", max_generation_calls=1),
    ):
        Responses.create(None)  # type: ignore[arg-type]
        with pytest.raises(UsageLimitExceeded, match="generation-call limit"):
            Responses.create(None)  # type: ignore[arg-type]

    assert [event["status"] for event in events] == ["completed", "limit"]


def test_usage_meter_honors_cost_already_persisted_on_resume(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(Responses, "create", _fake_create)
    events: list[dict[str, object]] = []

    def capture(event: Mapping[str, object]) -> None:
        events.append(dict(event))

    with (
        OpenAIUsageMeter(
            capture,
            max_estimated_cost_usd=1.0,
            initial_estimated_cost_usd=1.0,
        ) as meter,
        meter.scope(case_key="case:1", condition="budget0p02"),
        pytest.raises(UsageLimitExceeded, match="estimated API cost limit"),
    ):
        Responses.create(None)  # type: ignore[arg-type]

    assert events[0]["status"] == "limit"
