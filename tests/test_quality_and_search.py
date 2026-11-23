import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tokenbench.cli import app

cli = CliRunner()


def run_demo(demo: Path, filename: str, output: Path) -> None:
    result = cli.invoke(app, ["run", "--experiment", str(demo / filename), "--out", str(output)])
    assert result.exit_code == 0, (result.output, result.exception)


def test_quality_gate_validates_dev_finalist_when_search_trial_budget_stops(tmp_path: Path) -> None:
    demo = tmp_path / "demo"
    assert cli.invoke(app, ["init", "--path", str(demo)]).exit_code == 0
    config = demo / "quality-screening.yaml"
    config.write_text(config.read_text().replace("max_trials: 6", "max_trials: 2"))
    output = tmp_path / "out"
    run_demo(demo, config.name, output)
    result = json.loads((output / "comparison.json").read_text())
    assert result["stop_reason"] == "max_trials"
    assert result["budget"]["search_trials_started"] == 2
    assert result["budget"]["validation_trials_started"] == 1
    assert result["budget"]["committed_cost_upper_bound_usd"] == "0"
    assert [candidate["name"] for candidate in result["candidates"]] == ["project", "boilerplate"]
    assert set(result["candidates"][0]["quality"]["reasons"]) == {
        "min_success_rate",
        "max_absolute_drop",
        "critical_regression",
    }
    assert result["screening"]["pairs"][0]["status"] == "not_run"
    assert result["validation"]["selected"] == result["validation"]["confirmed"] == ["boilerplate"]
    assert result["validation"]["status"] == "complete"
    assert not (output / "validation" / "project").exists()
    validation = json.loads((output / "validation" / "boilerplate" / "summary.json").read_text())
    assert validation["manifest"]["split"] == "validation"
    validation_tasks = {attempt["task_id"] for attempt in validation["attempts"]}
    assert len(validation_tasks) == 8
    assert "west-validation-unit" in validation_tasks
    assert all("-validation-" in task for task in validation_tasks)
    pareto = json.loads((output / "pareto.json").read_text())
    assert pareto["status"] == "partial"
    dev, val = pareto["phases"]
    assert "project" not in dev["frontier"]
    assert {point["name"] for point in val["points"]} == {"baseline", "boilerplate"}


def test_dev_finalist_can_fail_independent_critical_case(tmp_path: Path) -> None:
    demo = tmp_path / "demo"
    assert cli.invoke(app, ["init", "--path", str(demo)]).exit_code == 0
    cases = demo / "screening-cases.jsonl"
    records = [json.loads(line) for line in cases.read_text().splitlines()]
    for record in records:
        if record["task_id"] == "west-validation-unit":
            record["input"]["field"] = "debug"
            record["grader_spec"]["expected"] = (
                "sample calibration notes kept by the local instrument collector"
            )
    cases.write_text("\n".join(map(json.dumps, records)))
    (demo / "critical.yaml").write_text("""schema_version: 1
experiment_id: independent-test
suite: screening-suite.yaml
quality:
  min_success_rate: '0'
  max_absolute_drop: '1'
  critical_task_ids: [west-validation-unit]
candidates:
  - name: project
    policies:
      - {id: filter.schema_projection, parameters: {allow_keys: [value, unit]}}
""")
    output = tmp_path / "out"
    run_demo(demo, "critical.yaml", output)
    result = json.loads((output / "comparison.json").read_text())
    assert result["candidates"][0]["quality"]["passed"]
    assert result["validation"]["selected"] == ["project"]
    assert result["validation"]["baseline"]["successful_trials"] == 8
    (candidate,) = result["validation"]["candidates"]
    assert candidate["quality"]["reasons"] == ["critical_regression"]
    assert candidate["quality"]["critical_regressions"] == ["west-validation-unit"]
    assert result["validation"]["confirmed"] == []
    assert result["validation"]["replay_configs"] == {}
    assert "central-test-value" not in (output / "comparison.json").read_text()


def test_optuna_search_runs_unique_trials_with_the_same_budget_and_gate(tmp_path: Path) -> None:
    pytest.importorskip("optuna")
    demo = tmp_path / "demo"
    assert cli.invoke(app, ["init", "--path", str(demo)]).exit_code == 0
    output = tmp_path / "search"
    run_demo(demo, "search.yaml", output)
    result = json.loads((output / "comparison.json").read_text())
    assert result["search"] == {"sampler": "optuna-nsgaii", "seed": 42, "space_size": 11}
    assert result["stop_reason"] == "max_trials"
    choices = [
        tuple(sorted(candidate["search_choices"].items())) for candidate in result["candidates"]
    ]
    assert len(set(choices)) == len(choices) == result["budget"]["search_trials_started"] == 6
    assert all(any(value for _, value in choice) for choice in choices)
    assert result["validation"]["confirmed"] == result["validation"]["selected"]
    assert len(result["validation"]["confirmed"]) == 1
