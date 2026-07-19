from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

from openai.resources.embeddings import Embeddings
from openai.resources.responses.responses import Responses

from benchmarks.prompt_compression.contracts import UsageRecord

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Mapping
    from contextlib import ExitStack

_DEFAULT_GENERATION_PRICING = {
    "input": 1.00,
    "cached_input": 0.10,
    "cache_write": 1.25,
    "output": 6.00,
}
_GENERATION_PRICING = {
    "gpt-5.6-luna": _DEFAULT_GENERATION_PRICING,
    "gpt-5.4-nano": {
        "input": 0.20,
        "cached_input": 0.02,
        "cache_write": 0.0,
        "output": 1.25,
    },
}
EMBEDDING_INPUT_USD_PER_MILLION = 0.02


class UsageLimitExceeded(RuntimeError):
    """A benchmark condition reached its predeclared request or elapsed-time limit."""


def pricing_for_model(model: str) -> dict[str, float]:
    """Return standard short-context pricing, accepting version-suffixed model IDs."""
    for prefix, pricing in _GENERATION_PRICING.items():
        if model == prefix or model.startswith(f"{prefix}-"):
            return pricing.copy()
    return _DEFAULT_GENERATION_PRICING.copy()


class OpenAIUsageMeter:
    """Process-local metering for answer, merge, and embedding requests."""

    def __init__(
        self,
        event_sink: Callable[[Mapping[str, object]], None] | None = None,
        *,
        max_estimated_cost_usd: float | None = None,
        initial_estimated_cost_usd: float = 0.0,
    ) -> None:
        self.records: list[UsageRecord] = []
        self._category = "unclassified"
        self._event_sink = event_sink
        self._event_sequence = 0
        self._scope: dict[str, object] = {}
        self._scope_generation_calls = 0
        self._max_estimated_cost_usd = max_estimated_cost_usd
        self.estimated_cost_usd = initial_estimated_cost_usd
        self._lock = threading.Lock()
        self._stack: ExitStack | None = None

    @contextmanager
    def category(self, category: str) -> Generator[None]:
        previous = self._category
        self._category = category
        try:
            yield
        finally:
            self._category = previous

    @contextmanager
    def scope(
        self,
        *,
        case_key: str,
        condition: str,
        max_generation_calls: int | None = None,
        max_elapsed_seconds: float | None = None,
    ) -> Generator[None]:
        """Attach resumable case metadata and optional safety limits to subsequent requests."""
        previous_scope = self._scope
        previous_calls = self._scope_generation_calls
        self._scope = {
            "case_key": case_key,
            "condition": condition,
            "started_monotonic": time.monotonic(),
            "max_generation_calls": max_generation_calls,
            "max_elapsed_seconds": max_elapsed_seconds,
        }
        self._scope_generation_calls = 0
        try:
            yield
        finally:
            self._scope = previous_scope
            self._scope_generation_calls = previous_calls

    def __enter__(self) -> OpenAIUsageMeter:
        from contextlib import ExitStack

        original_create = Responses.create
        original_parse = Responses.parse
        original_embedding = Embeddings.create

        def response_create(resource: Responses, *args: Any, **kwargs: Any) -> Any:
            started = time.monotonic()
            try:
                self._before_request("generation")
                response = original_create(resource, *args, **kwargs)
            except Exception as error:
                self._record_error("generation", kwargs, time.monotonic() - started, error)
                raise
            self._record_response(response, time.monotonic() - started, request_kind="generation")
            return response

        def response_parse(resource: Responses, *args: Any, **kwargs: Any) -> Any:
            started = time.monotonic()
            try:
                self._before_request("generation")
                response = original_parse(resource, *args, **kwargs)
            except Exception as error:
                self._record_error("generation", kwargs, time.monotonic() - started, error)
                raise
            self._record_response(response, time.monotonic() - started, request_kind="generation")
            return response

        def embedding_create(resource: Embeddings, *args: Any, **kwargs: Any) -> Any:
            started = time.monotonic()
            try:
                self._before_request("embedding")
                response = original_embedding(resource, *args, **kwargs)
            except Exception as error:
                self._record_error("embedding", kwargs, time.monotonic() - started, error)
                raise
            usage = response.usage
            self._append(
                UsageRecord(
                    category="embedding",
                    model=str(kwargs.get("model", "embedding")),
                    input_tokens=int(usage.prompt_tokens),
                    cached_input_tokens=0,
                    output_tokens=0,
                    total_tokens=int(usage.total_tokens),
                    elapsed_seconds=time.monotonic() - started,
                ),
                request_kind="embedding",
                response_id=str(getattr(response, "_request_id", "") or "") or None,
            )
            return response

        stack = ExitStack()
        stack.enter_context(patch.object(Responses, "create", response_create))
        stack.enter_context(patch.object(Responses, "parse", response_parse))
        stack.enter_context(patch.object(Embeddings, "create", embedding_create))
        self._stack = stack
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        if self._stack is not None:
            self._stack.close()

    def _before_request(self, request_kind: str) -> None:
        if self._max_estimated_cost_usd is not None and self.estimated_cost_usd >= self._max_estimated_cost_usd:
            raise UsageLimitExceeded(f"run reached ${self._max_estimated_cost_usd:.2f} estimated API cost limit")
        started = self._scope.get("started_monotonic")
        elapsed_limit = self._scope.get("max_elapsed_seconds")
        if (
            isinstance(started, float)
            and isinstance(elapsed_limit, int | float)
            and time.monotonic() - started >= float(elapsed_limit)
        ):
            raise UsageLimitExceeded(f"condition exceeded {float(elapsed_limit):g}s elapsed-time limit")
        if request_kind != "generation":
            return
        generation_limit = self._scope.get("max_generation_calls")
        if isinstance(generation_limit, int) and self._scope_generation_calls >= generation_limit:
            raise UsageLimitExceeded(f"condition reached {generation_limit} generation-call limit")
        self._scope_generation_calls += 1

    def _append(self, record: UsageRecord, *, request_kind: str, response_id: str | None) -> None:
        with self._lock:
            self.records.append(record)
            self.estimated_cost_usd += estimate_cost((record,))
            self._event_sequence += 1
            event = self._event_payload(
                request_kind=request_kind,
                status="completed",
                elapsed_seconds=record.elapsed_seconds,
                model=record.model,
                response_id=response_id,
                usage=record,
            )
        if self._event_sink is not None:
            self._event_sink(event)

    def _record_response(self, response: Any, elapsed_seconds: float, *, request_kind: str) -> None:
        usage = response.usage
        if usage is None:
            return
        details = getattr(usage, "input_tokens_details", None)
        self._append(
            UsageRecord(
                category=self._category,
                model=str(response.model),
                input_tokens=int(usage.input_tokens),
                cached_input_tokens=int(getattr(details, "cached_tokens", 0) or 0),
                output_tokens=int(usage.output_tokens),
                total_tokens=int(usage.total_tokens),
                elapsed_seconds=elapsed_seconds,
            ),
            request_kind=request_kind,
            response_id=str(getattr(response, "id", "") or "") or None,
        )

    def _record_error(
        self, request_kind: str, kwargs: Mapping[str, object], elapsed_seconds: float, error: Exception
    ) -> None:
        with self._lock:
            self._event_sequence += 1
            event = self._event_payload(
                request_kind=request_kind,
                status="limit" if isinstance(error, UsageLimitExceeded) else "error",
                elapsed_seconds=elapsed_seconds,
                model=str(kwargs.get("model", request_kind)),
                response_id=None,
                error=error,
            )
        if self._event_sink is not None:
            self._event_sink(event)

    def _event_payload(
        self,
        *,
        request_kind: str,
        status: str,
        elapsed_seconds: float,
        model: str,
        response_id: str | None,
        usage: UsageRecord | None = None,
        error: Exception | None = None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": 1,
            "sequence": self._event_sequence,
            "recorded_at": datetime.now(UTC).isoformat(),
            "request_kind": request_kind,
            "category": self._category,
            "status": status,
            "model": model,
            "response_id": response_id,
            "elapsed_seconds": elapsed_seconds,
            "case_key": self._scope.get("case_key"),
            "condition": self._scope.get("condition"),
        }
        if usage is not None:
            payload["usage"] = usage.model_dump(mode="json")
            payload["estimated_cost_usd"] = estimate_cost((usage,))
        if error is not None:
            payload["error_type"] = type(error).__name__
            payload["error_message"] = str(error)
        return payload


def estimate_cost(records: tuple[UsageRecord, ...]) -> float:
    """Estimate cost using the recorded pricing assumptions in the CLI manifest."""
    total = 0.0
    for record in records:
        if record.category == "embedding":
            total += record.input_tokens * EMBEDDING_INPUT_USD_PER_MILLION / 1_000_000
            continue
        pricing = pricing_for_model(record.model)
        uncached = record.input_tokens - record.cached_input_tokens
        total += (
            uncached * pricing["input"]
            + record.cached_input_tokens * pricing["cached_input"]
            + record.output_tokens * pricing["output"]
        ) / 1_000_000
    return total
