"""Paired percentile intervals over complete task clusters."""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from random import Random
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from tokenbench.report import AttemptSummary, BaselineSummary, final_outcomes


class BootstrapSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    resamples: Annotated[int, Field(strict=True, ge=2)] = 2_000
    seed: Annotated[int, Field(strict=True)] = 42


class Interval(BaseModel):
    estimate: Decimal
    lower: Decimal
    upper: Decimal


class PairedIntervals(BaseModel):
    method: Literal["paired_task_percentile"] = "paired_task_percentile"
    confidence_percent: Literal[95] = 95
    status: Literal["computed", "insufficient_tasks", "incomplete_comparison"]
    task_pairs: int
    trial_pairs: int
    tasks_with_token_counts: int
    tasks_with_costs: int
    success_rate_delta: Interval | None = None
    tokens_per_trial_delta: Interval | None = None
    cost_usd_per_trial_delta: Interval | None = None


@dataclass(frozen=True)
class TaskDifference:
    trials: int
    successes: int
    tokens: int | None
    cost_usd: Decimal | None


def task_usage(attempts: Sequence[AttemptSummary]) -> tuple[int | None, Decimal | None]:
    if not attempts or any(not attempt.model_calls for attempt in attempts):
        return None, None
    calls = [call for attempt in attempts for call in attempt.model_calls]
    tokens = None
    if all(call.input_tokens is not None and call.output_tokens is not None for call in calls):
        tokens = sum((call.input_tokens or 0) + (call.output_tokens or 0) for call in calls)
    costs = [call.cost_usd for call in calls if call.cost_usd is not None]
    return tokens, sum(costs, Decimal(0)) if len(costs) == len(calls) else None


def paired_tasks(
    baseline: BaselineSummary, candidate: BaselineSummary
) -> tuple[TaskDifference, ...]:
    before = final_outcomes(baseline.attempts, baseline.manifest.max_retries)
    after = final_outcomes(candidate.attempts, candidate.manifest.max_retries)
    keys = before.keys() & after.keys()
    pairs: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for key in sorted(keys):
        pairs[key[0]].append(key)
    baseline_attempts: dict[str, list[AttemptSummary]] = defaultdict(list)
    candidate_attempts: dict[str, list[AttemptSummary]] = defaultdict(list)
    for summary, groups in ((baseline, baseline_attempts), (candidate, candidate_attempts)):
        for attempt in summary.attempts:
            if (attempt.task_id, attempt.repetition) in keys:
                groups[attempt.task_id].append(attempt)

    differences = []
    for task_id, repetitions in pairs.items():
        tokens_a, cost_a = task_usage(baseline_attempts[task_id])
        tokens_b, cost_b = task_usage(candidate_attempts[task_id])
        differences.append(
            TaskDifference(
                trials=len(repetitions),
                successes=sum(after[key] - before[key] for key in repetitions),
                tokens=tokens_b - tokens_a
                if tokens_a is not None and tokens_b is not None
                else None,
                cost_usd=cost_b - cost_a if cost_a is not None and cost_b is not None else None,
            )
        )
    return tuple(differences)


def percentile(ordered: list[Decimal], probability: Decimal) -> Decimal:
    position = probability * (len(ordered) - 1)
    index = int(position)
    fraction = position - index
    return (
        ordered[index] + fraction * (ordered[index + 1] - ordered[index])
        if fraction
        else ordered[index]
    )


def interval(distribution: list[Decimal], estimate: Decimal) -> Interval:
    ordered = sorted(distribution)
    return Interval(
        estimate=estimate,
        lower=percentile(ordered, Decimal("0.025")),
        upper=percentile(ordered, Decimal("0.975")),
    )


def bootstrap(
    tasks: tuple[TaskDifference, ...],
    settings: BootstrapSettings,
    *,
    complete: bool,
) -> PairedIntervals:
    count = len(tasks)
    trials = sum(task.trials for task in tasks)
    tokens = [task.tokens for task in tasks if task.tokens is not None]
    costs = [task.cost_usd for task in tasks if task.cost_usd is not None]
    result = PairedIntervals(
        task_pairs=count,
        trial_pairs=trials,
        tasks_with_token_counts=len(tokens),
        tasks_with_costs=len(costs),
        status="incomplete_comparison"
        if not complete
        else "insufficient_tasks"
        if count < 2
        else "computed",
    )
    if result.status != "computed":
        return result

    random = Random(settings.seed)
    successes, token_samples, cost_samples = [], [], []
    for _ in range(settings.resamples):
        # Reuse each sampled task index for both sides and every available metric.
        indices = random.choices(range(count), k=count)
        denominator = sum(tasks[index].trials for index in indices)
        successes.append(Decimal(sum(tasks[index].successes for index in indices)) / denominator)
        if len(tokens) == count:
            token_samples.append(Decimal(sum(tokens[index] for index in indices)) / denominator)
        if len(costs) == count:
            cost_samples.append(sum((costs[index] for index in indices), Decimal(0)) / denominator)

    result.success_rate_delta = interval(
        successes, Decimal(sum(task.successes for task in tasks)) / trials
    )
    if token_samples:
        result.tokens_per_trial_delta = interval(token_samples, Decimal(sum(tokens)) / trials)
    if cost_samples:
        result.cost_usd_per_trial_delta = interval(cost_samples, sum(costs, Decimal(0)) / trials)
    return result
