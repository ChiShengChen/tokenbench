from collections.abc import Sequence
from dataclasses import dataclass

from tokenbench.policies.models import HookName, PolicySpec

_ORDER = {
    "filter.schema_projection": 0,
    "filter.boilerplate": 0,
    "filter.log_tail": 0,
    "filter.deduplicate": 1,
    "rag.topk": 1,
    "history.window": 2,
    "agent.step_budget": 0,
}


class PolicyConfigurationError(ValueError):
    """An enabled policy cannot be compiled for a target."""


@dataclass(frozen=True)
class CompiledPolicies:
    policies: tuple[PolicySpec, ...] = ()

    def for_hook(self, hook: HookName) -> tuple[PolicySpec, ...]:
        return tuple(policy for policy in self.policies if policy.hook == hook)


def compile_policies(
    specs: Sequence[PolicySpec],
    hooks: Sequence[HookName],
) -> CompiledPolicies:
    enabled = tuple(policy for policy in specs if policy.enabled)
    seen: set[tuple[str, HookName]] = set()
    stages: dict[HookName, int] = {}
    for policy in enabled:
        label = (policy.id, policy.hook)
        if policy.hook not in hooks:
            raise PolicyConfigurationError(f"{policy.id} requires the {policy.hook} hook")
        if label in seen:
            raise PolicyConfigurationError(f"Duplicate policy {policy.id} on {policy.hook}")
        rank = _ORDER[policy.id]
        if rank < stages.get(policy.hook, 0):
            raise PolicyConfigurationError(f"{policy.id} is out of order on {policy.hook}")
        seen.add(label)
        stages[policy.hook] = rank
    return CompiledPolicies(enabled)
