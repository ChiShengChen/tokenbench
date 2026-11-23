"""Run target trials with scoped policy hooks and exact grading."""

import asyncio
from contextlib import closing
from copy import deepcopy
from functools import partial
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from time import perf_counter_ns

from pydantic import JsonValue

from tokenbench.budget import BudgetStopped, ExperimentBudget, StopReason
from tokenbench.config import LoadedSuite
from tokenbench.policies.compiler import CompiledPolicies
from tokenbench.policies.deterministic import PolicyError
from tokenbench.policies.runtime import PolicySession
from tokenbench.pricing import ModelPrice
from tokenbench.report import BaselineSummary, PolicyManifest, RunManifest, summarize, write_reports
from tokenbench.storage import Ledger
from tokenbench.targets.base import PermanentTargetError, TargetAdapter
from tokenbench.tasks import TaskCase
from tokenbench.trace import TraceSink


async def run_baseline(
    suite: LoadedSuite,
    target: TargetAdapter,
    output: Path,
    budget: ExperimentBudget | None = None,
) -> BaselineSummary:
    return await run_suite(suite, target, output, CompiledPolicies(), budget)


def create_output(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output}")


async def run_suite(
    suite: LoadedSuite,
    target: TargetAdapter,
    output: Path,
    compiled: CompiledPolicies,
    budget: ExperimentBudget | None = None,
) -> BaselineSummary:
    spec = suite.spec
    fingerprints = dict(suite.fingerprints)
    source_hash = target.source_hash()
    if source_hash is not None:
        fingerprints["target_module"] = source_hash
    manifest = RunManifest(
        tokenbench_version=version("tokenbench"),
        suite_id=spec.suite_id,
        execution=spec.execution,
        target=spec.target,
        split=spec.split,
        case_count=len(suite.cases),
        repetitions=spec.repetitions,
        max_retries=spec.max_retries,
        price_snapshot_id=suite.prices.snapshot_id if suite.prices is not None else None,
        fingerprints=fingerprints,
        policies=tuple(
            PolicyManifest(
                id=policy.id,
                hook=policy.hook,
                config_sha256=sha256(policy.model_dump_json().encode()).hexdigest(),
            )
            for policy in compiled.policies
        ),
    )
    prices = (
        {(price.provider, price.model): price for price in suite.prices.prices}
        if suite.prices
        else {}
    )

    create_output(output)
    database = "candidate.sqlite3" if compiled.policies else "baseline.sqlite3"
    with closing(Ledger(output / database, manifest.model_dump_json())) as ledger:
        stopped: StopReason | None = None
        try:
            for task in suite.cases:
                for repetition in range(1, spec.repetitions + 1):
                    await run_trial(
                        suite, target, compiled, ledger, task, repetition, prices, budget
                    )
        except BudgetStopped as exc:
            stopped = exc.reason
        ledger.finish_run(stopped)
        summary = summarize(ledger, manifest, stopped)
        write_reports(output, summary)
    return summary


async def run_trial(
    suite: LoadedSuite,
    target: TargetAdapter,
    compiled: CompiledPolicies,
    ledger: Ledger,
    task: TaskCase,
    repetition: int,
    prices: dict[tuple[str, str], ModelPrice],
    budget: ExperimentBudget | None,
) -> None:
    for attempt in range(1, suite.spec.max_retries + 2):
        if budget is not None:
            budget.remaining_seconds()
        attempt_id = ledger.start_attempt(task.task_id, repetition, attempt)
        policies = (
            PolicySession(compiled, partial(ledger.record_policy, attempt_id))
            if compiled.policies
            else None
        )
        trace = TraceSink(ledger, attempt_id, suite.spec.target, prices, policies, budget)
        started = perf_counter_ns()
        try:
            answer = await evaluate(target, deepcopy(task.input), trace, budget)
            if policies is not None:
                policies.check()
            passed = task.grader_spec.passes(answer)
        except BudgetStopped as exc:
            ledger.finish_attempt(attempt_id, "interrupted", started, exc.reason)
            raise
        except (PolicyError, PermanentTargetError) as exc:
            ledger.finish_attempt(attempt_id, "failed", started, type(exc).__name__)
            return
        except Exception as exc:
            ledger.finish_attempt(attempt_id, "error", started, type(exc).__name__)
        except BaseException as exc:
            ledger.finish_attempt(attempt_id, "interrupted", started, type(exc).__name__)
            raise
        else:
            ledger.finish_attempt(attempt_id, "passed" if passed else "failed", started)
            return


async def evaluate(
    target: TargetAdapter,
    task_input: dict[str, JsonValue],
    trace: TraceSink,
    budget: ExperimentBudget | None,
) -> JsonValue:
    if budget is None:
        return await target.run(task_input, trace)
    timer = asyncio.timeout(budget.remaining_seconds())
    try:
        async with timer:
            output = await target.run(task_input, trace, budget)
    except TimeoutError:
        if timer.expired():
            budget.stop("max_seconds")
        budget.remaining_seconds()
        raise
    except Exception:
        budget.remaining_seconds()
        raise
    budget.remaining_seconds()
    return output
