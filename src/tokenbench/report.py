"""Render deterministic baseline summaries from the committed ledger."""

import json
from collections import Counter, defaultdict
from decimal import Decimal
from html import escape
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from tokenbench.config import TargetSpec
from tokenbench.pricing import UsageRecord
from tokenbench.storage import AttemptStatus, CallStatus, Ledger
from tokenbench.tasks import Split


class RunManifest(BaseModel):
    schema_version: Literal[1] = 1
    tokenbench_version: str
    suite_id: str
    execution: Literal["python_callable", "local_fixture"]
    target: TargetSpec
    split: Split | None
    case_count: int
    repetitions: int
    max_retries: int
    grader: Literal["exact-json-v1"] = "exact-json-v1"
    price_snapshot_id: str | None
    fingerprints: dict[str, str]


class CallSummary(UsageRecord):
    id: int
    provider: str
    model: str
    status: CallStatus
    error_type: str | None


class AttemptSummary(BaseModel):
    task_id: str
    repetition: int
    attempt: int
    status: AttemptStatus
    passed: bool | None
    error_type: str | None
    model_calls: list[CallSummary]


class Totals(BaseModel):
    trials: int
    successful_trials: int
    success_rate: float
    attempts: int
    retries: int
    attempt_errors: int
    untraced_attempts: int
    model_calls: int
    usage_sources: dict[str, int]
    cost_sources: dict[str, int]
    calls_with_missing_token_counts: int
    known_input_tokens: int
    known_output_tokens: int
    total_tokens: int | None
    known_cost_usd: Decimal | None
    total_cost_usd: Decimal | None


class BaselineSummary(BaseModel):
    manifest: RunManifest
    totals: Totals
    attempts: list[AttemptSummary]


def summarize(ledger: Ledger, manifest: RunManifest) -> BaselineSummary:
    calls_by_attempt: dict[int, list[CallSummary]] = defaultdict(list)
    calls: list[CallSummary] = []
    for row in ledger.connection.execute("SELECT * FROM model_calls ORDER BY id"):
        data = dict(row)
        attempt_id = data.pop("attempt_id")
        del data["started_at"], data["duration_ms"]
        call = CallSummary.model_validate(data)
        calls_by_attempt[attempt_id].append(call)
        calls.append(call)

    attempts = [
        AttemptSummary.model_validate({**dict(row), "model_calls": calls_by_attempt[row["id"]]})
        for row in ledger.connection.execute("SELECT * FROM attempts ORDER BY id")
    ]
    final_attempts = {(attempt.task_id, attempt.repetition): attempt for attempt in attempts}
    successful = sum(attempt.passed is True for attempt in final_attempts.values())
    missing_counts = sum(call.input_tokens is None or call.output_tokens is None for call in calls)
    input_tokens = sum(call.input_tokens or 0 for call in calls)
    output_tokens = sum(call.output_tokens or 0 for call in calls)
    untraced = sum(not attempt.model_calls for attempt in attempts)
    costs = [call.cost_usd for call in calls if call.cost_usd is not None]
    known_cost = sum(costs, Decimal(0)) if costs else None
    totals = Totals(
        trials=len(final_attempts),
        successful_trials=successful,
        success_rate=successful / len(final_attempts),
        attempts=len(attempts),
        retries=sum(attempt.attempt > 1 for attempt in attempts),
        attempt_errors=sum(attempt.status == "error" for attempt in attempts),
        untraced_attempts=untraced,
        model_calls=len(calls),
        usage_sources=dict(Counter(call.usage_source for call in calls)),
        cost_sources=dict(Counter(call.cost_source for call in calls)),
        calls_with_missing_token_counts=missing_counts,
        known_input_tokens=input_tokens,
        known_output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens if not (missing_counts or untraced) else None,
        known_cost_usd=known_cost,
        total_cost_usd=known_cost if not untraced and len(costs) == len(calls) else None,
    )
    return BaselineSummary(manifest=manifest, totals=totals, attempts=attempts)


def table(headers: tuple[str, ...], rows: list[tuple[object, ...]]) -> str:
    head = "".join(f'<th scope="col">{escape(header)}</th>' for header in headers)
    body = "\n".join(
        "<tr>" + "".join(f"<td>{escape(str(cell))}</td>" for cell in row) + "</tr>" for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def render_html(summary: BaselineSummary) -> str:
    manifest = summary.manifest
    totals = summary.totals
    metadata = table(
        ("Setting", "Value"),
        [
            ("Execution", manifest.execution),
            ("Target", manifest.target.entrypoint),
            ("Model", f"{manifest.target.provider}/{manifest.target.model}"),
            ("Grader", manifest.grader),
            ("Price snapshot", manifest.price_snapshot_id or "None"),
            ("Split", manifest.split or "All"),
            ("Cases", manifest.case_count),
            ("Repetitions", manifest.repetitions),
        ],
    )
    metrics = table(
        ("Metric", "Value"),
        [
            ("Successful trials", f"{totals.successful_trials}/{totals.trials}"),
            ("Success rate", f"{totals.success_rate:.1%}"),
            ("Attempts", totals.attempts),
            ("Retries", totals.retries),
            ("Attempt errors", totals.attempt_errors),
            ("Attempts without model traces", totals.untraced_attempts),
            ("Instrumented model calls", totals.model_calls),
            (
                "Usage sources",
                ", ".join(f"{key} ({value})" for key, value in sorted(totals.usage_sources.items()))
                or "None",
            ),
            (
                "Cost sources",
                ", ".join(f"{key} ({value})" for key, value in sorted(totals.cost_sources.items()))
                or "None",
            ),
            ("Calls with missing token counts", totals.calls_with_missing_token_counts),
            ("Known input tokens", totals.known_input_tokens),
            ("Known output tokens", totals.known_output_tokens),
            ("Total tokens", totals.total_tokens if totals.total_tokens is not None else "Unknown"),
            (
                "Known cost (USD)",
                totals.known_cost_usd if totals.known_cost_usd is not None else "Unknown",
            ),
            (
                "Total cost (USD)",
                totals.total_cost_usd if totals.total_cost_usd is not None else "Unknown",
            ),
        ],
    )
    attempts = table(
        ("Task", "Repetition", "Attempt", "Status", "Error type", "Model calls"),
        [
            (a.task_id, a.repetition, a.attempt, a.status, a.error_type or "", len(a.model_calls))
            for a in summary.attempts
        ],
    )
    return f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(manifest.suite_id)} baseline</title>
<style>
body {{ font: 1rem system-ui, sans-serif; max-width: 68rem; padding: 1rem; margin: auto; }}
table {{ border-collapse: collapse; margin-bottom: 1rem; width: 100%; text-align: left; }}
th, td {{ padding: .5rem; border-bottom: 1px solid #bbb; overflow-wrap: anywhere; }}
</style>
<main>
<h1>{escape(manifest.suite_id)} baseline</h1>
{metadata}
<h2>Results</h2>
{metrics}
<h2>Attempts</h2>
{attempts}
</main>
</html>
"""


def write_reports(output: Path, summary: BaselineSummary) -> None:
    files = {
        "summary.json": json.dumps(summary.model_dump(mode="json"), indent=2, sort_keys=True)
        + "\n",
        "summary.html": render_html(summary),
    }
    for name, content in files.items():
        temporary = output / f".{name}.tmp"
        temporary.write_text(content, encoding="utf-8", newline="\n")
        temporary.replace(output / name)
