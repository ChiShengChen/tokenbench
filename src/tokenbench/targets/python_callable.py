"""Import an existing Python callable and expose it to the baseline runner."""

import inspect
from collections.abc import Awaitable, Callable
from contextlib import nullcontext
from dataclasses import dataclass
from hashlib import sha256
from importlib import import_module
from pathlib import Path
from typing import TypeAlias

from pydantic import JsonValue, TypeAdapter

from tokenbench.budget import ExperimentBudget
from tokenbench.config import ConfigurationError, PythonTargetSpec
from tokenbench.trace import TraceSink

TargetCallable: TypeAlias = Callable[
    [dict[str, JsonValue], TraceSink], JsonValue | Awaitable[JsonValue]
]
_JSON_VALUE = TypeAdapter[JsonValue](JsonValue)


@dataclass(frozen=True)
class PythonCallableTarget:
    function: TargetCallable

    @classmethod
    def from_spec(cls, spec: PythonTargetSpec) -> "PythonCallableTarget":
        module_name, _, function_name = spec.entrypoint.partition(":")
        try:
            module = import_module(module_name)
            function = getattr(module, function_name)
        except Exception as exc:
            raise ConfigurationError(
                f"Cannot import target {spec.entrypoint} ({type(exc).__name__})"
            ) from exc
        try:
            if inspect.isclass(function):
                raise TypeError
            inspect.signature(function).bind({}, object())
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(
                f"Target {spec.entrypoint} must be callable with two positional arguments (input, trace)"
            ) from exc
        return cls(function)

    def source_hash(self) -> str | None:
        try:
            source = inspect.getsourcefile(self.function)
            if source is not None:
                return sha256(Path(source).read_bytes()).hexdigest()
        except (OSError, TypeError):
            return None
        return None

    async def run(
        self,
        task_input: dict[str, JsonValue],
        trace: TraceSink,
        budget: ExperimentBudget | None = None,
    ) -> JsonValue:
        with budget.interrupt_sync_call() if budget is not None else nullcontext():
            output = self.function(task_input, trace)
        if inspect.isawaitable(output):
            output = await output
        return _JSON_VALUE.validate_python(output, strict=True)
