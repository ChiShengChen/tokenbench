"""Opt-in model-call instrumentation for trusted Python targets."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter_ns

from tokenbench.config import TargetSpec
from tokenbench.pricing import ModelPrice, ProviderUsage, normalize_usage
from tokenbench.storage import CallStatus, Ledger


@dataclass
class ModelCall:
    _reported: ProviderUsage | None = field(init=False, default=None)
    _estimate: tuple[int, int] | None = field(init=False, default=None)

    def report_usage(self, usage: ProviderUsage) -> None:
        """Supply normalized counts or USD cost from a provider response."""
        self._reported = usage

    def estimate_usage(self, input_tokens: int, output_tokens: int) -> None:
        """Supply counts from a local tokenizer or heuristic."""
        ProviderUsage(input_tokens=input_tokens, output_tokens=output_tokens)
        self._estimate = input_tokens, output_tokens


@dataclass(frozen=True)
class TraceSink:
    _ledger: Ledger
    _attempt_id: int
    _target: TargetSpec
    _prices: dict[tuple[str, str], ModelPrice]

    @contextmanager
    def model_call(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
    ) -> Iterator[ModelCall]:
        """Record one call even when the underlying call raises an exception."""
        provider_id = provider or self._target.provider
        model_id = model or self._target.model
        call_id = self._ledger.start_call(self._attempt_id, provider_id, model_id)
        call = ModelCall()
        started = perf_counter_ns()
        status: CallStatus = "completed"
        error_type = None
        try:
            yield call
        except BaseException as exc:
            status = "error" if isinstance(exc, Exception) else "interrupted"
            error_type = type(exc).__name__
            raise
        finally:
            price = self._prices.get((provider_id, model_id))
            usage = normalize_usage(call._reported, call._estimate, price)
            self._ledger.finish_call(call_id, status, started, usage, error_type)
