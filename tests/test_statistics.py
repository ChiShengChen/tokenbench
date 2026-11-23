from decimal import Decimal

from tokenbench.report import AttemptSummary, CallSummary
from tokenbench.statistics import BootstrapSettings, TaskDifference, bootstrap, task_usage


def test_bootstrap_resamples_whole_tasks_and_is_reproducible() -> None:
    tasks = (
        TaskDifference(trials=10, successes=10, tokens=100, cost_usd=Decimal("1")),
        TaskDifference(trials=10, successes=-10, tokens=-100, cost_usd=Decimal("-1")),
    )
    settings = BootstrapSettings(resamples=200, seed=42)
    report = bootstrap(tasks, settings, complete=True)
    assert report == bootstrap(tasks, settings, complete=True)
    assert report.task_pairs == 2 and report.trial_pairs == 20
    rate = report.success_rate_delta
    tokens = report.tokens_per_trial_delta
    usd = report.cost_usd_per_trial_delta
    assert rate is not None and (rate.estimate, rate.lower, rate.upper) == (
        Decimal(0),
        Decimal(-1),
        Decimal(1),
    )
    assert tokens is not None and (tokens.estimate, tokens.lower, tokens.upper) == (
        Decimal(0),
        Decimal(-10),
        Decimal(10),
    )
    assert usd is not None and (usd.estimate, usd.lower, usd.upper) == (
        Decimal(0),
        Decimal("-0.1"),
        Decimal("0.1"),
    )


def test_intervals_withhold_unobserved_metrics_or_incomplete_comparisons() -> None:
    settings = BootstrapSettings(resamples=100)
    tasks = (
        TaskDifference(trials=2, successes=0, tokens=-8, cost_usd=None),
        TaskDifference(trials=2, successes=-1, tokens=-4, cost_usd=Decimal("0")),
    )
    report = bootstrap(tasks, settings, complete=True)
    assert report.tasks_with_costs == 1 and report.cost_usd_per_trial_delta is None
    assert report.success_rate_delta is not None and report.tokens_per_trial_delta is not None
    insufficient = bootstrap(tasks[:1], settings, complete=True)
    assert insufficient.status == "insufficient_tasks" and insufficient.success_rate_delta is None
    incomplete = bootstrap(tasks, settings, complete=False)
    assert (
        incomplete.status == "incomplete_comparison" and incomplete.tokens_per_trial_delta is None
    )


def test_task_usage_includes_failed_attempts() -> None:
    usage = CallSummary(
        id=1,
        provider="fixture",
        model="trial-model",
        status="error",
        error_type="RuntimeError",
        input_tokens=2,
        output_tokens=1,
        cost_usd=Decimal("0.01"),
        usage_source="provider",
        cost_source="provider",
    )
    retry = usage.model_copy(
        update={
            "id": 2,
            "status": "completed",
            "error_type": None,
            "input_tokens": 6,
            "cost_usd": Decimal("0.04"),
        }
    )
    initial = AttemptSummary(
        task_id="retry",
        repetition=1,
        attempt=1,
        status="error",
        passed=None,
        error_type="RuntimeError",
        model_calls=[usage],
        policy_events=[],
    )
    recovered = initial.model_copy(
        update={
            "attempt": 2,
            "status": "passed",
            "passed": True,
            "error_type": None,
            "model_calls": [retry],
        }
    )
    assert task_usage([initial, recovered]) == (10, Decimal("0.05"))
    untraced = initial.model_copy(update={"model_calls": []})
    assert task_usage([untraced, recovered]) == (None, None)
