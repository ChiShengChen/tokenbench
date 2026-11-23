"""Run Python target trials with scoped policy hooks and exact grading."""

from contextlib import closing
from copy import deepcopy
from functools import partial
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from time import perf_counter_ns

from tokenbench.config import LoadedSuite
from tokenbench.policies.compiler import CompiledPolicies
from tokenbench.policies.deterministic import PolicyError
from tokenbench.policies.runtime import PolicySession
from tokenbench.report import BaselineSummary, PolicyManifest, RunManifest, summarize, write_reports
from tokenbench.storage import Ledger
from tokenbench.targets.python_callable import PythonCallableTarget
from tokenbench.trace import TraceSink


async def run_baseline(
    suite: LoadedSuite,
    target: PythonCallableTarget,
    output: Path,
) -> BaselineSummary:
    return await run_suite(suite, target, output, CompiledPolicies())


def create_output(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output}")


async def run_suite(
    suite: LoadedSuite,
    target: PythonCallableTarget,
    output: Path,
    compiled: CompiledPolicies,
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
        for task in suite.cases:
            for repetition in range(1, spec.repetitions + 1):
                for attempt in range(1, spec.max_retries + 2):
                    attempt_id = ledger.start_attempt(task.task_id, repetition, attempt)
                    policies = (
                        PolicySession(compiled, partial(ledger.record_policy, attempt_id))
                        if compiled.policies
                        else None
                    )
                    trace = TraceSink(ledger, attempt_id, spec.target, prices, policies)
                    started = perf_counter_ns()
                    try:
                        answer = await target.run(deepcopy(task.input), trace)
                        if policies is not None:
                            policies.check()
                        passed = task.grader_spec.passes(answer)
                    except PolicyError as exc:
                        ledger.finish_attempt(attempt_id, "failed", started, type(exc).__name__)
                        break
                    except Exception as exc:
                        ledger.finish_attempt(attempt_id, "error", started, type(exc).__name__)
                    except BaseException as exc:
                        ledger.finish_attempt(
                            attempt_id, "interrupted", started, type(exc).__name__
                        )
                        raise
                    else:
                        ledger.finish_attempt(attempt_id, "passed" if passed else "failed", started)
                        break
        ledger.finish_run()
        summary = summarize(ledger, manifest)
        write_reports(output, summary)
    return summary
