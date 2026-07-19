from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

from openai.resources.embeddings import Embeddings
from openai.resources.responses.responses import Responses

from benchmarks.prompt_compression.contracts import UsageRecord

if TYPE_CHECKING:
    from collections.abc import Generator
    from contextlib import ExitStack


class OpenAIUsageMeter:
    """Process-local metering for answer, merge, and embedding requests."""

    def __init__(self) -> None:
        self.records: list[UsageRecord] = []
        self._category = "unclassified"
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

    def __enter__(self) -> OpenAIUsageMeter:
        from contextlib import ExitStack

        original_create = Responses.create
        original_parse = Responses.parse
        original_embedding = Embeddings.create

        def response_create(resource: Responses, *args: Any, **kwargs: Any) -> Any:
            started = time.monotonic()
            response = original_create(resource, *args, **kwargs)
            self._record_response(response, time.monotonic() - started)
            return response

        def response_parse(resource: Responses, *args: Any, **kwargs: Any) -> Any:
            started = time.monotonic()
            response = original_parse(resource, *args, **kwargs)
            self._record_response(response, time.monotonic() - started)
            return response

        def embedding_create(resource: Embeddings, *args: Any, **kwargs: Any) -> Any:
            started = time.monotonic()
            response = original_embedding(resource, *args, **kwargs)
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
                )
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

    def _append(self, record: UsageRecord) -> None:
        with self._lock:
            self.records.append(record)

    def _record_response(self, response: Any, elapsed_seconds: float) -> None:
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
            )
        )


def estimate_cost(records: tuple[UsageRecord, ...]) -> float:
    """Estimate cost using the recorded pricing assumptions in the CLI manifest."""
    total = 0.0
    for record in records:
        if record.category == "embedding":
            total += record.input_tokens * 0.02 / 1_000_000
            continue
        uncached = record.input_tokens - record.cached_input_tokens
        total += (uncached * 1.00 + record.cached_input_tokens * 0.10 + record.output_tokens * 6.00) / 1_000_000
    return total
