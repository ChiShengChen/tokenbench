from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from time import perf_counter_ns

from tokenbench.policies.compiler import CompiledPolicies
from tokenbench.policies.deterministic import Context, PolicyError, transform
from tokenbench.policies.models import ContextHook, ContextSegment, PolicyAudit, StepBudget
from tokenbench.tasks import canonical_json


class StepBudgetExceeded(PolicyError):
    """The target entered another step after spending its allowance."""


def estimate_tokens(context: Context) -> int:
    return sum(
        len(
            (
                segment.content
                if isinstance(segment.content, str)
                else canonical_json(segment.content)
            ).split()
        )
        for segment in context
    )


def references(context: Context) -> tuple[str, ...]:
    result: dict[str, None] = {}
    for segment in context:
        result.update(dict.fromkeys(segment.source_ids))
        if segment.format == "log" and isinstance(segment.content, dict):
            artifact = segment.content.get("artifact_ref")
            if isinstance(artifact, str):
                result[artifact] = None
    return tuple(result)


def check_boundaries(before: Context, after: Context) -> dict[str, ContextSegment]:
    by_id = {segment.segment_id: segment for segment in after}
    expected_ids = tuple(segment.segment_id for segment in before if segment.segment_id in by_id)
    if expected_ids != tuple(segment.segment_id for segment in after):
        raise PolicyError("unexpected_context_order")
    for original in before:
        updated = by_id.get(original.segment_id)
        if updated is None:
            if not original.droppable:
                raise PolicyError("protected_segment_removed")
        elif (
            not original.editable
            and updated is not original
            and updated.model_dump_json() != original.model_dump_json()
        ):
            raise PolicyError("protected_segment_changed")
        elif not set(original.source_ids) <= set(updated.source_ids):
            raise PolicyError("source_reference_changed")
        elif original.required_keys and isinstance(original.content, dict):
            if not isinstance(updated.content, dict) or any(
                key not in updated.content
                or canonical_json(updated.content[key]) != canonical_json(original.content[key])
                for key in original.required_keys
            ):
                raise PolicyError("required_field_changed")
    return by_id


@dataclass
class PolicySession:
    compiled: CompiledPolicies
    emit: Callable[[PolicyAudit, int], None]
    _steps: int = field(init=False, default=0)
    _failure: PolicyError | None = field(init=False, default=None)

    def check(self) -> None:
        if self._failure is not None:
            raise self._failure

    def apply_context(self, hook: ContextHook, segments: Sequence[ContextSegment]) -> Context:
        self.check()
        context = tuple(segments)
        policies = self.compiled.for_hook(hook)
        if not policies:
            return context

        for policy in policies:
            started = perf_counter_ns()
            before_tokens = None
            prior_references = references(context)
            try:
                if len({segment.segment_id for segment in context}) != len(context):
                    raise PolicyError("duplicate_segment_id")
                before_tokens = estimate_tokens(context)
                result = transform(context, policy)
                updated = check_boundaries(context, result)
                after_tokens = estimate_tokens(result)
            except ValueError as exc:
                reason = str(exc) if isinstance(exc, PolicyError) else "invalid_context_content"
                self._failure = PolicyError(reason)
                self.emit(
                    PolicyAudit(
                        policy_id=policy.id,
                        hook=hook,
                        status="failed",
                        reason=reason,
                        token_source="estimated",
                        before_tokens=before_tokens,
                        after_tokens=before_tokens,
                        preserved_references=prior_references,
                    ),
                    started,
                )
                raise self._failure from exc

            result_references = set(references(result))
            modified = tuple(
                segment.segment_id
                for segment in context
                if updated.get(segment.segment_id) != segment
            )
            self.emit(
                PolicyAudit(
                    policy_id=policy.id,
                    hook=hook,
                    status="applied" if modified else "unchanged",
                    before_tokens=before_tokens,
                    after_tokens=after_tokens,
                    token_source="estimated",
                    segments_modified=modified,
                    preserved_references=tuple(
                        ref for ref in prior_references if ref in result_references
                    ),
                    removed_references=tuple(
                        ref for ref in prior_references if ref not in result_references
                    ),
                ),
                started,
            )
            context = result
        return context

    def step(self) -> None:
        self.check()
        for policy in self.compiled.for_hook("step"):
            assert isinstance(policy, StepBudget)
            started = perf_counter_ns()
            self._steps += 1
            exceeded = self._steps > policy.parameters.max_steps
            self.emit(
                PolicyAudit(
                    policy_id=policy.id,
                    hook="step",
                    status="blocked" if exceeded else "allowed",
                    token_source="not_applicable",
                    step=self._steps,
                    reason="step_budget_exceeded" if exceeded else None,
                ),
                started,
            )
            if exceeded:
                self._failure = StepBudgetExceeded("step_budget_exceeded")
                raise self._failure
