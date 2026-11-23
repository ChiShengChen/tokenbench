import asyncio
import json
import sqlite3
import sys
from contextlib import closing
from decimal import Decimal
from pathlib import Path
from types import ModuleType

import pytest
from pydantic import JsonValue
from typer.testing import CliRunner

from tokenbench.cli import app
from tokenbench.config import load_suite
from tokenbench.pricing import ProviderUsage
from tokenbench.runner import run_baseline
from tokenbench.targets.python_callable import PythonCallableTarget
from tokenbench.trace import TraceSink

cli = CliRunner()


def test_help_and_doctor() -> None:
    for command in ([], ["init"], ["doctor"], ["baseline"], ["run"]):
        result = cli.invoke(app, [*command, "--help"])
        assert result.exit_code == 0, result.output
        assert "Usage:" in result.output
    result = cli.invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output
    assert "Python" in result.output and "SQLite" in result.output


def test_demo_runs_and_reruns_with_identical_reports(tmp_path: Path) -> None:
    demo = tmp_path / "demo"
    initialized = cli.invoke(app, ["init", "--path", str(demo)])
    assert initialized.exit_code == 0, initialized.output
    assert sorted(path.name for path in demo.iterdir()) == [
        "cases.jsonl",
        "experiment.yaml",
        "prices.json",
        "quality-screening.yaml",
        "screening-cases.jsonl",
        "screening-suite.yaml",
        "screening.yaml",
        "search.yaml",
        "suite.yaml",
    ]
    assert cli.invoke(app, ["init", "--path", str(demo)]).exit_code == 1

    baseline = ["baseline", "--suite", str(demo / "suite.yaml")]
    dry_run = cli.invoke(app, [*baseline, "--dry-run", "--out", str(tmp_path / "dry")])
    assert dry_run.exit_code == 0, dry_run.output
    assert "36 cases" in dry_run.output
    assert not (tmp_path / "dry").exists()

    first, second = tmp_path / "first", tmp_path / "second"
    for output in (first, second):
        result = cli.invoke(app, [*baseline, "--out", str(output)])
        assert result.exit_code == 0, (result.output, result.exception)
        assert "36/36 trials passed (36 attempts)" in result.output
    assert cli.invoke(app, [*baseline, "--out", str(first)]).exit_code == 1

    for filename in ("summary.json", "summary.html"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()

    summary = json.loads((first / "summary.json").read_text(encoding="utf-8"))
    assert summary["manifest"]["execution"] == "local_fixture"
    assert summary["totals"]["usage_sources"] == {"estimated": 36}
    assert summary["totals"]["cost_sources"] == {"price_snapshot": 36}
    assert summary["totals"]["total_tokens"] > 0
    assert Decimal(summary["totals"]["total_cost_usd"]) == 0
    with closing(sqlite3.connect(first / "baseline.sqlite3")) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM attempts WHERE status = 'passed'"
        ).fetchone() == (36,)
        assert connection.execute(
            "SELECT COUNT(*) FROM model_calls WHERE duration_ms >= 0"
        ).fetchone() == (36,)


def make_suite(path: Path, *, entrypoint: str = "test_target:run") -> Path:
    suite = path / "suite.yaml"
    suite.write_text(
        f"""schema_version: 1
suite_id: '<script>suite</script>'
target:
  adapter: python
  entrypoint: {entrypoint}
  provider: test
  model: baseline-v1
cases: cases.jsonl
split: dev
max_retries: 1
""",
        encoding="utf-8",
    )
    cases = [
        {
            "task_id": kind,
            "input": {"kind": kind, "data": {"value": "private prompt content"}},
            "task_type": "lookup",
            "grader_spec": {"kind": "exact", "expected": "private expected answer"},
            "split": split,
        }
        for kind, split in (
            ("recoverable", "dev"),
            ("incorrect", "dev"),
            ("untraced", "dev"),
            ("skip", "test"),
        )
    ]
    (path / "cases.jsonl").write_text("\n".join(map(json.dumps, cases)), encoding="utf-8")
    return suite


def test_dry_run_checks_config_and_import_without_calling_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def run(task_input: dict[str, JsonValue], trace: TraceSink) -> JsonValue:
        calls.append(task_input)
        return None

    module = ModuleType("test_target")
    module.__dict__["run"] = run
    monkeypatch.setitem(sys.modules, module.__name__, module)
    suite = make_suite(tmp_path)
    result = cli.invoke(app, ["baseline", "--suite", str(suite), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "3 cases" in result.output
    assert calls == []

    make_suite(tmp_path, entrypoint="test_target:missing")
    result = cli.invoke(app, ["baseline", "--suite", str(suite), "--dry-run"])
    assert result.exit_code == 1
    assert "Cannot import target test_target:missing" in result.output

    original = make_suite(tmp_path).read_text(encoding="utf-8")
    suite.write_text(
        original.replace("schema_version: 1", "schema_version: true"), encoding="utf-8"
    )
    result = cli.invoke(app, ["baseline", "--suite", str(suite), "--dry-run"])
    assert result.exit_code == 1
    assert "schema_version" in result.output


def test_async_custom_target_records_errors_retries_and_missing_usage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failed_once = False

    async def run(task_input: dict[str, JsonValue], trace: TraceSink) -> JsonValue:
        nonlocal failed_once
        assert "grader_spec" not in task_input
        assert task_input.pop("data") == {"value": "private prompt content"}
        if task_input["kind"] == "untraced":
            raise ValueError("private exception details")
        if task_input["kind"] == "incorrect":
            with trace.model_call() as call:
                call.estimate_usage(1, 2)
            return "private response content"
        if task_input["kind"] != "recoverable":
            pytest.fail("Unexpected case or missing split filter")
        with trace.model_call() as call:
            call.report_usage(
                ProviderUsage(input_tokens=4, output_tokens=2, cost_usd=Decimal("0.01"))
            )
            if not failed_once:
                failed_once = True
                raise RuntimeError("private exception details")
        with trace.model_call(model="auxiliary-v1") as call:
            call.report_usage(
                ProviderUsage(input_tokens=3, output_tokens=1, cost_usd=Decimal("0.02"))
            )
        return "private expected answer"

    module = ModuleType("test_target")
    module.__dict__["run"] = run
    monkeypatch.setitem(sys.modules, module.__name__, module)
    suite = make_suite(tmp_path)
    output = tmp_path / "output"
    result = cli.invoke(app, ["baseline", "--suite", str(suite), "--out", str(output)])
    assert result.exit_code == 0, (result.output, result.exception)
    assert "1/3 trials passed (5 attempts)" in result.output

    raw_summary = (output / "summary.json").read_text(encoding="utf-8")
    summary = json.loads(raw_summary)
    totals = summary["totals"]
    assert totals["retries"] == 2
    assert totals["attempt_errors"] == 3
    assert totals["untraced_attempts"] == 2
    assert totals["usage_sources"] == {"estimated": 1, "provider": 3}
    assert totals["cost_sources"] == {"provider": 3, "unknown": 1}
    assert totals["known_input_tokens"] == 12
    assert totals["known_output_tokens"] == 7
    assert totals["total_tokens"] is None
    assert totals["known_cost_usd"] == "0.04"
    assert totals["total_cost_usd"] is None
    assert [attempt["status"] for attempt in summary["attempts"]] == [
        "error",
        "passed",
        "failed",
        "error",
        "error",
    ]
    assert summary["attempts"][0]["model_calls"][0]["error_type"] == "RuntimeError"

    database = output / "baseline.sqlite3"
    with closing(sqlite3.connect(database)) as connection:
        rows = connection.execute(
            "SELECT attempt_id, model, status FROM model_calls ORDER BY id"
        ).fetchall()
        assert rows == [
            (1, "baseline-v1", "error"),
            (2, "baseline-v1", "completed"),
            (2, "auxiliary-v1", "completed"),
            (3, "baseline-v1", "completed"),
        ]
    html = (output / "summary.html").read_text(encoding="utf-8")
    assert "&lt;script&gt;suite&lt;/script&gt;" in html
    assert "<script>" not in html
    for payload in (raw_summary.encode(), html.encode(), database.read_bytes()):
        assert b"private " not in payload


def test_interrupted_call_and_attempt_remain_in_ledger(tmp_path: Path) -> None:
    async def run(task_input: dict[str, JsonValue], trace: TraceSink) -> JsonValue:
        with trace.model_call() as call:
            call.estimate_usage(1, 0)
            raise asyncio.CancelledError

    suite = load_suite(make_suite(tmp_path))
    output = tmp_path / "output"
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run_baseline(suite, PythonCallableTarget(run), output))
    with closing(sqlite3.connect(output / "baseline.sqlite3")) as connection:
        assert connection.execute("SELECT finished_at FROM run_manifest").fetchone() == (None,)
        assert connection.execute("SELECT status, error_type FROM attempts").fetchall() == [
            ("interrupted", "CancelledError"),
        ]
        assert connection.execute("SELECT status, input_tokens FROM model_calls").fetchall() == [
            ("interrupted", 1),
        ]
