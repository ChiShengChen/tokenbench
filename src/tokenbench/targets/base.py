"""Shared contract for targets executed by the trial runner."""

from typing import Protocol

from pydantic import JsonValue

from tokenbench.budget import ExperimentBudget
from tokenbench.trace import TraceSink


class PermanentTargetError(RuntimeError):
    """The target rejected the request or returned an unusable completion."""


class TargetAdapter(Protocol):
    def source_hash(self) -> str | None: ...

    async def run(
        self,
        task_input: dict[str, JsonValue],
        trace: TraceSink,
        budget: ExperimentBudget | None = None,
    ) -> JsonValue: ...
