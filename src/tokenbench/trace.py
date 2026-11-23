"""Opt-in model-call instrumentation for trusted Python targets."""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from decimal import Decimal
from time import perf_counter_ns

from tokenbench.budget import BudgetStopped, ExperimentBudget
from tokenbench.config import TargetSpec
from tokenbench.policies.models import ContextHook, ContextSegment
from tokenbench.policies.runtime import PolicySession
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
    _policies: PolicySession | None = None
    _budget: ExperimentBudget | None = None

    def apply_context(
        self,
        hook: ContextHook,
        segments: Sequence[ContextSegment],
    ) -> tuple[ContextSegment, ...]:
        """Pass a ranked, chronological, or tool-provided context through an opted-in hook."""
        if self._budget is not None:
            self._budget.remaining_seconds()
        if self._policies is None:
            return tuple(segments)
        return self._policies.apply_context(hook, segments)

    def step(self) -> None:
        """Call before work begins on each agent step."""
        if self._budget is not None:
            self._budget.remaining_seconds()
        if self._policies is not None:
            self._policies.step()

    @contextmanager
    def model_call(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        max_cost_usd: Decimal | int | None = None,
    ) -> Iterator[ModelCall]:
        """Record one call even when the underlying call raises an exception."""
        if self._policies is not None:
            self._policies.check()
        provider_id = provider or self._target.provider
        model_id = model or self._target.model
        reservation = self._budget.reserve(max_cost_usd) if self._budget is not None else None
        call_id = self._ledger.start_call(
            self._attempt_id,
            provider_id,
            model_id,
            str(reservation) if reservation is not None else None,
        )
        call = ModelCall()
        started = perf_counter_ns()
        status: CallStatus = "completed"
        error_type = None
        try:
            yield call
        except BaseException as exc:
            status = (
                "error"
                if isinstance(exc, Exception) and not isinstance(exc, BudgetStopped)
                else "interrupted"
            )
            error_type = type(exc).__name__
            raise
        finally:
            price = self._prices.get((provider_id, model_id))
            usage = normalize_usage(call._reported, call._estimate, price)
            self._ledger.finish_call(call_id, status, started, usage, error_type)
            if self._budget is not None:
                self._budget.settle(reservation, usage)
