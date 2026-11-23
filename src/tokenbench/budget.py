"""Shared time and pre-call spend accounting for policy experiments."""

import signal
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from decimal import Decimal
from time import monotonic
from types import FrameType
from typing import Annotated, Literal, Never, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from tokenbench.policies.models import PositiveInt
from tokenbench.pricing import NonNegativeUSD, UsageRecord

StopReason: TypeAlias = Literal[
    "max_trials",
    "max_seconds",
    "max_usd",
    "missing_cost_bound",
    "cost_bound_exceeded",
]


class BudgetLimits(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_trials: PositiveInt | None = None
    max_seconds: Annotated[float, Field(gt=0, strict=True, allow_inf_nan=False)] | None = None
    max_usd: NonNegativeUSD | None = None


class BudgetStopped(RuntimeError):
    def __init__(self, reason: StopReason) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass
class ExperimentBudget:
    limits: BudgetLimits
    started: float = field(default_factory=monotonic, init=False)
    known_cost: Decimal = field(default=Decimal(0), init=False)
    settled_upper_bound: Decimal = field(default=Decimal(0), init=False)
    reserved_cost: Decimal = field(default=Decimal(0), init=False)
    unpriced_calls: int = field(default=0, init=False)
    stop_reason: StopReason | None = field(default=None, init=False)

    @property
    def committed_cost(self) -> Decimal:
        return self.settled_upper_bound + self.reserved_cost

    @property
    def elapsed_seconds(self) -> float:
        return monotonic() - self.started

    def stop(self, reason: StopReason) -> Never:
        if self.stop_reason is None:
            self.stop_reason = reason
        raise BudgetStopped(self.stop_reason)

    def remaining_seconds(self) -> float | None:
        if self.stop_reason is not None:
            raise BudgetStopped(self.stop_reason)
        if self.limits.max_seconds is None:
            return None
        remaining = self.limits.max_seconds - self.elapsed_seconds
        if remaining <= 0:
            self.stop("max_seconds")
        return remaining

    @contextmanager
    def interrupt_sync_call(self) -> Iterator[None]:
        seconds = self.remaining_seconds()
        if (
            seconds is None
            or threading.current_thread() is not threading.main_thread()
            or not hasattr(signal, "setitimer")
            or signal.getsignal(signal.SIGALRM) != signal.SIG_DFL
            or signal.getitimer(signal.ITIMER_REAL) != (0, 0)
        ):
            yield
            return

        def expire(_signum: int, _frame: FrameType | None) -> None:
            self.stop("max_seconds")

        signal.signal(signal.SIGALRM, expire)
        try:
            try:
                signal.setitimer(signal.ITIMER_REAL, seconds)
            except (OSError, OverflowError):
                # A platform timing limit still allows checks at the target and hook boundaries.
                pass
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, signal.SIG_DFL)

    def reserve(self, max_cost_usd: Decimal | int | None) -> Decimal | None:
        self.remaining_seconds()
        if self.limits.max_usd is None:
            return None
        bound = Decimal(max_cost_usd) if max_cost_usd is not None else None
        if bound is None or not bound.is_finite() or bound < 0:
            self.stop("missing_cost_bound")
        if self.committed_cost + bound > self.limits.max_usd:
            self.stop("max_usd")
        self.reserved_cost += bound
        return bound

    def settle(self, reservation: Decimal | None, usage: UsageRecord) -> None:
        cost = usage.cost_usd
        if cost is None:
            self.unpriced_calls += 1
        else:
            self.known_cost += cost

        if reservation is not None:
            self.reserved_cost -= reservation
            if cost is not None and (usage.cost_source == "provider" or cost > reservation):
                self.settled_upper_bound += cost
                if cost > reservation:
                    self.stop_reason = "cost_bound_exceeded"
            else:
                self.settled_upper_bound += reservation
