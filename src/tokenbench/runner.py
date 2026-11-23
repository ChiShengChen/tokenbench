"""Run unmodified Python targets and grade each completed baseline trial."""

from contextlib import closing
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path
from time import perf_counter_ns

from tokenbench.config import LoadedSuite
from tokenbench.report import BaselineSummary, RunManifest, summarize, write_reports
from tokenbench.storage import Ledger
from tokenbench.targets.python_callable import PythonCallableTarget
from tokenbench.trace import TraceSink


async def run_baseline(
    suite: LoadedSuite,
    target: PythonCallableTarget,
    output: Path,
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
    )
    prices = (
        {(price.provider, price.model): price for price in suite.prices.prices}
        if suite.prices
        else {}
    )

    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output}")
    with closing(Ledger(output / "baseline.sqlite3", manifest.model_dump_json())) as ledger:
        for task in suite.cases:
            for repetition in range(1, spec.repetitions + 1):
                for attempt in range(1, spec.max_retries + 2):
                    attempt_id = ledger.start_attempt(task.task_id, repetition, attempt)
                    trace = TraceSink(ledger, attempt_id, spec.target, prices)
                    started = perf_counter_ns()
                    try:
                        answer = await target.run(deepcopy(task.input), trace)
                        passed = task.grader_spec.passes(answer)
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
