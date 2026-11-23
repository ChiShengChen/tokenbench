"""Resolve and preflight the target used by either CLI entry point."""

from collections.abc import Iterable

from pydantic import ValidationError

from tokenbench.config import ConfigurationError, TargetSpec, describe_validation
from tokenbench.targets.base import TargetAdapter
from tokenbench.targets.openai_compatible import ChatInput, OpenAICompatibleTarget
from tokenbench.targets.python_callable import PythonCallableTarget
from tokenbench.tasks import TaskCase


def require_cost_bounds(spec: TargetSpec) -> None:
    if not spec.budgeted_model_calls:
        requirement = (
            "target.max_cost_usd"
            if spec.adapter == "openai_compatible"
            else ("target.budgeted_model_calls and per-call cost bounds")
        )
        raise ConfigurationError(f"A USD budget requires {requirement}")


def load_target(spec: TargetSpec, cases: Iterable[TaskCase]) -> TargetAdapter:
    if spec.adapter == "python":
        return PythonCallableTarget.from_spec(spec)
    for case in cases:
        try:
            ChatInput.model_validate(case.input)
        except ValidationError as exc:
            raise ConfigurationError(
                f"Invalid Chat Completions input for task {case.task_id!r}: {describe_validation(exc)}"
            ) from exc
    return OpenAICompatibleTarget.from_spec(spec)
