import asyncio
import json
import signal
import sqlite3
import sys
from collections.abc import Awaitable, Callable
from contextlib import closing
from decimal import Decimal
from pathlib import Path
from time import sleep
from types import ModuleType

import pytest
from pydantic import JsonValue
from typer.testing import CliRunner

from tokenbench.budget import BudgetLimits, BudgetStopped, ExperimentBudget
from tokenbench.cli import app
from tokenbench.pareto import ParetoPoint, frontier
from tokenbench.pricing import ModelPrice, ProviderUsage, normalize_usage
from tokenbench.trace import TraceSink

cli = CliRunner()


def fixture_experiment(
    path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: Callable[[dict[str, JsonValue], TraceSink], JsonValue | Awaitable[JsonValue]],
    budget: str,
) -> Path:
    module = ModuleType("budget_fixture")
    module.__dict__["run"] = target
    monkeypatch.setitem(sys.modules, module.__name__, module)
    (path / "suite.yaml").write_text(
        """schema_version: 1
suite_id: budget-test
target:
  adapter: python
  entrypoint: budget_fixture:run
  provider: local
  model: budget-test
  budgeted_model_calls: true
cases: cases.jsonl
""",
        encoding="utf-8",
    )
    (path / "cases.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "task_id": task,
                    "input": {"task": task},
                    "task_type": "cost-test",
                    "grader_spec": {"kind": "exact", "expected": 1},
                }
            )
            for task in ("first-task", "second-task")
        ),
        encoding="utf-8",
    )
    experiment = path / "experiment.yaml"
    experiment.write_text(
        f"""schema_version: 1
experiment_id: budget-test
suite: suite.yaml
budget: {budget}
candidates:
  - {{name: first, policies: []}}
  - {{name: second, policies: []}}
""",
        encoding="utf-8",
    )
    return experiment


def test_cost_cap_blocks_the_next_call_and_retains_partial_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executed = []

    def target(task: dict[str, JsonValue], trace: TraceSink) -> JsonValue:
        with trace.model_call(max_cost_usd=Decimal("0.06")) as call:
            executed.append(task["task"])
            call.report_usage(
                ProviderUsage(input_tokens=3, output_tokens=1, cost_usd=Decimal("0.04"))
            )
        return 1

    experiment = fixture_experiment(tmp_path, monkeypatch, target, "{max_usd: '0.14'}")
    output = tmp_path / "out"
    result = cli.invoke(app, ["run", "--experiment", str(experiment), "--out", str(output)])
    assert result.exit_code == 0, (result.output, result.exception)
    summary = json.loads((output / "comparison.json").read_text())
    assert executed == ["first-task", "second-task", "first-task"]
    assert (summary["status"], summary["stop_reason"]) == ("partial", "max_usd")
    assert Decimal(summary["budget"]["known_cost_usd"]) == Decimal("0.12")
    assert Decimal(summary["budget"]["committed_cost_upper_bound_usd"]) == Decimal("0.12")
    assert summary["baseline"]["successful_trials"] == 2
    (candidate,) = summary["candidates"]
    assert candidate["status"] == "partial" and candidate["totals"]["successful_trials"] == 1
    assert candidate["total_cost_delta_usd"] is None
    assert not (output / "second").exists()
    with closing(sqlite3.connect(output / "first" / "baseline.sqlite3")) as ledger:
        assert ledger.execute("SELECT status, error_type FROM attempts ORDER BY id").fetchall() == [
            ("passed", None),
            ("interrupted", "max_usd"),
        ]
        assert ledger.execute("SELECT cost_upper_bound_usd FROM model_calls").fetchall() == [
            ("0.06",)
        ]
    observed = json.loads((output / "pareto.json").read_text())["phases"][0]
    assert observed["frontier"] == ["baseline"]
    assert observed["points"][0]["p95_latency_ms"] > 0


def test_validation_shares_spend_limit_and_cannot_confirm_a_partial_finalist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executed = []

    def target(task: dict[str, JsonValue], trace: TraceSink) -> JsonValue:
        with trace.model_call(max_cost_usd=Decimal("0.06")) as call:
            executed.append(task["task"])
            call.report_usage(
                ProviderUsage(input_tokens=3, output_tokens=1, cost_usd=Decimal("0.04"))
            )
        return 1

    experiment = fixture_experiment(
        tmp_path, monkeypatch, target, "{max_trials: 1, max_usd: '0.18'}"
    )
    with experiment.open("a") as handle:
        handle.write("quality: {min_success_rate: '1'}\n")
    with (tmp_path / "cases.jsonl").open("a") as handle:
        handle.write(
            "\n"
            + json.dumps(
                {
                    "task_id": "validation-task",
                    "input": {"task": "validation-task"},
                    "task_type": "cost-test",
                    "split": "validation",
                    "grader_spec": {"kind": "exact", "expected": 1},
                }
            )
        )
    output = tmp_path / "out"
    result = cli.invoke(app, ["run", "--experiment", str(experiment), "--out", str(output)])
    assert result.exit_code == 0, (result.output, result.exception)
    assert executed == ["first-task", "second-task"] * 2
    summary = json.loads((output / "comparison.json").read_text())
    assert summary["stop_reason"] == "max_usd"
    assert Decimal(summary["budget"]["known_cost_usd"]) == Decimal("0.16")
    assert summary["validation"]["selected"] == ["first"]
    assert summary["validation"]["status"] == "partial"
    assert summary["validation"]["baseline"]["trials"] == 0
    assert summary["validation"]["confirmed"] == []


def test_unknown_and_estimated_costs_keep_full_reservations() -> None:
    budget = ExperimentBudget(BudgetLimits(max_usd=Decimal("1")))
    price = ModelPrice(
        provider="local",
        model="test",
        input_usd_per_million=Decimal("1"),
        output_usd_per_million=Decimal("0"),
    )
    budget.settle(budget.reserve(Decimal("0.6")), normalize_usage(None, (200_000, 0), price))
    budget.settle(budget.reserve(Decimal("0.25")), normalize_usage(None, None, None))
    assert budget.known_cost == Decimal("0.2")
    assert budget.committed_cost == Decimal("0.85")
    assert budget.unpriced_calls == 1
    with pytest.raises(BudgetStopped, match="max_usd"):
        budget.reserve(Decimal("0.2"))

    missing = ExperimentBudget(BudgetLimits(max_usd=Decimal("1")))
    with pytest.raises(BudgetStopped, match="missing_cost_bound"):
        missing.reserve(None)

    violated = ExperimentBudget(BudgetLimits(max_usd=Decimal("1")))
    actual = normalize_usage(ProviderUsage(cost_usd=Decimal("1.5")), None, None)
    violated.settle(violated.reserve(Decimal("0.25")), actual)
    with pytest.raises(BudgetStopped, match="cost_bound_exceeded"):
        violated.reserve(0)
    assert violated.committed_cost == Decimal("1.5")


@pytest.mark.parametrize("asynchronous", [True, False])
def test_deadline_keeps_the_finished_trial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, asynchronous: bool
) -> None:
    if not asynchronous and (
        not hasattr(signal, "setitimer") or signal.getsignal(signal.SIGALRM) != signal.SIG_DFL
    ):
        pytest.skip("sync deadline uses an available POSIX timer")

    async def async_target(task: dict[str, JsonValue], trace: TraceSink) -> JsonValue:
        if task["task"] == "second-task":
            await asyncio.sleep(3)
        return 1

    def sync_target(task: dict[str, JsonValue], trace: TraceSink) -> JsonValue:
        if task["task"] == "second-task":
            sleep(3)
        return 1

    config = fixture_experiment(
        tmp_path, monkeypatch, async_target if asynchronous else sync_target, "{max_seconds: 0.5}"
    )
    output = tmp_path / "out"
    result = cli.invoke(app, ["run", "--experiment", str(config), "--out", str(output)])
    assert result.exit_code == 0, (result.output, result.exception)
    summary = json.loads((output / "comparison.json").read_text())
    assert summary["baseline_stop_reason"] == summary["stop_reason"] == "max_seconds"
    assert summary["baseline"]["successful_trials"] == 1
    assert summary["candidates"] == []
    assert json.loads((output / "pareto.json").read_text())["elapsed_seconds"] < 2.5


def test_pareto_dominance_keeps_tradeoffs_and_excludes_unusable_points() -> None:
    points = [
        ParetoPoint(
            name="accurate",
            status="complete",
            success_rate=1,
            total_tokens=100,
            total_cost_usd=Decimal("1"),
            p95_latency_ms=3,
        ),
        ParetoPoint(
            name="lean",
            status="complete",
            success_rate=0.9,
            total_tokens=50,
            total_cost_usd=Decimal("0.5"),
            p95_latency_ms=2,
        ),
        ParetoPoint(
            name="dominated",
            status="complete",
            success_rate=0.9,
            total_tokens=51,
            total_cost_usd=Decimal("0.6"),
            p95_latency_ms=2,
        ),
        ParetoPoint(
            name="unknown-cost",
            status="complete",
            success_rate=1,
            total_tokens=1,
            total_cost_usd=None,
            p95_latency_ms=1,
        ),
        ParetoPoint(
            name="partial",
            status="partial",
            success_rate=1,
            total_tokens=1,
            total_cost_usd=Decimal(0),
            p95_latency_ms=1,
        ),
        ParetoPoint(
            name="rejected",
            status="complete",
            quality_passed=False,
            success_rate=1,
            total_tokens=1,
            total_cost_usd=Decimal(0),
            p95_latency_ms=1,
        ),
    ]
    assert frontier("dev", points).frontier == ["accurate", "lean"]
