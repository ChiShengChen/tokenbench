import json
import sqlite3
import sys
from contextlib import closing
from pathlib import Path
from types import ModuleType

import pytest
from pydantic import JsonValue
from typer.testing import CliRunner

from tokenbench.cli import app
from tokenbench.policies.models import ContextSegment
from tokenbench.policies.runtime import StepBudgetExceeded
from tokenbench.trace import TraceSink

cli = CliRunner()


def test_offline_experiment_repeats_and_disabled_policies_match_baseline(tmp_path: Path) -> None:
    demo = tmp_path / "demo"
    assert cli.invoke(app, ["init", "--path", str(demo)]).exit_code == 0
    experiment = demo / "experiment.yaml"
    with experiment.open("a", encoding="utf-8") as config:
        config.write("""  - name: disabled
    policies:
      - id: rag.topk
        enabled: false
        parameters: {k: 1}
""")
    first, second = tmp_path / "first", tmp_path / "second"
    argv = ["run", "--experiment", str(experiment)]
    validation = cli.invoke(app, [*argv, "--dry-run", "--out", str(first)])
    assert validation.exit_code == 0, validation.output
    assert "3 candidates" in validation.output
    assert not first.exists()
    for output in (first, second):
        result = cli.invoke(app, [*argv, "--out", str(output)])
        assert result.exit_code == 0, (result.output, result.exception)

    for extension in ("json", "html"):
        assert (first / f"comparison.{extension}").read_bytes() == (
            second / f"comparison.{extension}"
        ).read_bytes()
        assert (first / "topk-1" / f"summary.{extension}").read_bytes() == (
            second / "topk-1" / f"summary.{extension}"
        ).read_bytes()
        assert (first / "disabled" / f"summary.{extension}").read_bytes() == (
            first / "baseline" / f"summary.{extension}"
        ).read_bytes()

    comparison = json.loads((first / "comparison.json").read_text(encoding="utf-8"))
    assert comparison["baseline"]["successful_trials"] == 36
    top2, top1, disabled = comparison["candidates"]
    assert top1["total_token_delta"] < top2["total_token_delta"] < 0
    assert top1["totals"]["successful_trials"] == 36
    assert top1["regressed_trials"] == []
    assert disabled["total_token_delta"] == 0
    assert disabled["coverage"] == []
    with closing(sqlite3.connect(first / "topk-1" / "candidate.sqlite3")) as ledger:
        assert (
            ledger.execute("""SELECT COUNT(*) FROM policy_events WHERE policy_id = 'rag.topk'
                              AND before_tokens > after_tokens AND runtime_ms >= 0""").fetchone()
            == (36,)
        )


def test_paired_experiment_records_regression_budget_and_unobserved_hooks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def target(task_input: dict[str, JsonValue], trace: TraceSink) -> JsonValue:
        trace.step()
        context = trace.apply_context(
            "retrieval",
            (
                ContextSegment(
                    segment_id="distractor",
                    kind="retrieved_doc",
                    content="other text",
                    mutable=True,
                ),
                ContextSegment(
                    segment_id="answer", kind="retrieved_doc", content="answer 27", mutable=True
                ),
            ),
        )
        with trace.model_call() as call:
            call.estimate_usage(len(context) * 2, 1)
        expected = 27 if any(seg.content == "answer 27" for seg in context) else None
        try:
            trace.step()
        except StepBudgetExceeded:
            return expected
        with trace.model_call() as call:
            call.estimate_usage(1, 1)
        return expected

    module = ModuleType("paired_fixture")
    module.__dict__["run"] = target
    monkeypatch.setitem(sys.modules, module.__name__, module)
    (tmp_path / "suite.yaml").write_text(
        """schema_version: 1
suite_id: paired
target:
  adapter: python
  entrypoint: paired_fixture:run
  provider: local
  model: paired-fixture
  hooks: [retrieval, step, history]
cases: cases.jsonl
max_retries: 2
repetitions: 2
""",
        encoding="utf-8",
    )
    (tmp_path / "cases.jsonl").write_text(
        json.dumps(
            {
                "task_id": "answer-on-second-rank",
                "input": {},
                "task_type": "retrieval",
                "grader_spec": {"kind": "exact", "expected": 27},
            }
        ),
        encoding="utf-8",
    )
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(
        """schema_version: 1
experiment_id: paired
suite: suite.yaml
candidates:
  - name: topk-1
    policies:
      - id: rag.topk
        parameters: {k: 1}
  - name: step-1
    policies:
      - id: agent.step_budget
        parameters: {max_steps: 1}
  - name: history-unused
    policies:
      - id: history.window
        parameters: {max_turns: 0}
""",
        encoding="utf-8",
    )
    output = tmp_path / "out"
    result = cli.invoke(app, ["run", "--experiment", str(experiment), "--out", str(output)])
    assert result.exit_code == 0, (result.output, result.exception)
    report = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    assert report["baseline"]["successful_trials"] == 2
    retrieval, budget, unused = report["candidates"]
    assert retrieval["total_token_delta"] < 0
    assert (
        retrieval["regressed_trials"]
        == budget["regressed_trials"]
        == [{"task_id": "answer-on-second-rank", "repetition": repetition} for repetition in (1, 2)]
    )
    assert budget["totals"]["attempts"] == 2 and budget["totals"]["model_calls"] == 2
    assert budget["coverage"][0]["events"] == {"allowed": 2, "blocked": 2}
    assert unused["status"] == "unsupported"
    assert unused["total_token_delta"] is None


def test_experiment_validation_rejects_unsupported_enabled_policy(tmp_path: Path) -> None:
    assert cli.invoke(app, ["init", "--path", str(tmp_path)]).exit_code == 0
    config = tmp_path / "experiment.yaml"
    config.write_text(
        """schema_version: 1
experiment_id: missing-hook
suite: suite.yaml
candidates:
  - name: history
    policies:
      - id: history.window
        parameters: {max_turns: 2}
""",
        encoding="utf-8",
    )
    result = cli.invoke(app, ["run", "--experiment", str(config), "--dry-run"])
    assert result.exit_code == 1
    assert "history.window requires the history hook" in result.output
