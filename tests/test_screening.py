import json
from decimal import Decimal
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tokenbench.cli import app

cli = CliRunner()


@pytest.mark.parametrize("priced", [True, False])
def test_screening_pairs_regressions_and_dev_only_execution(tmp_path: Path, priced: bool) -> None:
    demo = tmp_path / "demo"
    assert cli.invoke(app, ["init", "--path", str(demo)]).exit_code == 0
    suite = demo / "screening-suite.yaml"
    config = suite.read_text(encoding="utf-8").replace("split: dev\n", "")
    if priced:
        prices = demo / "prices.json"
        snapshot = json.loads(prices.read_text(encoding="utf-8"))
        fixture_price = next(row for row in snapshot["prices"] if row["model"] == "fixture-tool-v1")
        fixture_price["input_usd_per_million"] = "1.25"
        prices.write_text(json.dumps(snapshot), encoding="utf-8")
    else:
        config = config.replace("price_snapshot: prices.json\n", "")
    suite.write_text(config, encoding="utf-8")

    output = tmp_path / "out"
    argv = ["run", "--experiment", str(demo / "screening.yaml"), "--out", str(output)]
    validation = cli.invoke(app, [*argv, "--dry-run"])
    assert validation.exit_code == 0, validation.output
    assert (
        "6 candidates" in validation.output and "4 cases, 0 incompatible pairs" in validation.output
    )
    assert not output.exists()
    run = cli.invoke(app, argv)
    assert run.exit_code == 0, (run.output, run.exception)

    comparison = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    assert comparison["baseline"]["successful_trials"] == 4
    assert comparison["screening"]["split"] == "dev"
    assert comparison["screening"]["singles"] == ["project", "boilerplate", "dedup"]
    candidates = {row["name"]: row for row in comparison["candidates"]}
    assert list(candidates) == [
        "project",
        "boilerplate",
        "dedup",
        "project--boilerplate",
        "project--dedup",
        "boilerplate--dedup",
    ]
    pair = candidates["project--dedup"]
    assert pair["total_token_delta"] == -128
    assert pair["regressed_trials"] == [
        {"task_id": "south-unit", "repetition": 1},
        {"task_id": "north-verified-unit", "repetition": 1},
    ]
    interactions = {row["name"]: row for row in comparison["screening"]["pairs"]}
    overlap = interactions["project--dedup"]
    assert overlap["status"] == "complete"
    assert overlap["token_interaction"] == 48
    assert overlap["successful_trial_interaction"] == 1
    assert interactions["project--boilerplate"]["token_interaction"] == 0
    if priced:
        assert Decimal(pair["total_cost_delta_usd"]) == Decimal("-0.00016")
        assert Decimal(overlap["cost_interaction_usd"]) == Decimal("0.00006")
    else:
        assert pair["total_cost_delta_usd"] is None
        assert overlap["cost_interaction_usd"] is None

    for name in ["baseline", *candidates]:
        summary = json.loads((output / name / "summary.json").read_text(encoding="utf-8"))
        assert summary["manifest"]["split"] == "dev"
        assert {row["task_id"] for row in summary["attempts"]} == {
            "north-value",
            "south-unit",
            "east-value",
            "north-verified-unit",
        }
    report = (output / "comparison.html").read_text(encoding="utf-8")
    assert "Pairwise interactions on dev" in report
    assert "north-verified-unit" in report


@pytest.mark.parametrize("split", ["validation", "test"])
def test_screening_rejects_explicit_non_dev_suites(tmp_path: Path, split: str) -> None:
    assert cli.invoke(app, ["init", "--path", str(tmp_path)]).exit_code == 0
    suite = tmp_path / "screening-suite.yaml"
    suite.write_text(
        suite.read_text(encoding="utf-8").replace("split: dev", f"split: {split}"), encoding="utf-8"
    )
    output = tmp_path / "out"
    run = cli.invoke(
        app, ["run", "--experiment", str(tmp_path / "screening.yaml"), "--out", str(output)]
    )
    assert run.exit_code == 1
    assert f"selects '{split}' but screening requires 'dev'" in run.output
    assert not output.exists()


def test_incompatible_pairs_are_reported_without_execution(tmp_path: Path) -> None:
    assert cli.invoke(app, ["init", "--path", str(tmp_path)]).exit_code == 0
    config = tmp_path / "topk-variants.yaml"
    config.write_text(
        """schema_version: 1
experiment_id: topk-variants
suite: suite.yaml
screening:
  - name: first
    policy: {id: rag.topk, parameters: {k: 1}}
  - name: second
    policy: {id: rag.topk, parameters: {k: 2}}
""",
        encoding="utf-8",
    )
    output = tmp_path / "out"
    run = cli.invoke(app, ["run", "--experiment", str(config), "--out", str(output)])
    assert run.exit_code == 0, (run.output, run.exception)
    report = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    assert [candidate["name"] for candidate in report["candidates"]] == ["first", "second"]
    (pair,) = report["screening"]["pairs"]
    assert pair["status"] == "incompatible"
    assert "Duplicate policy rag.topk" in pair["reason"]
    assert pair["token_interaction"] is None
    assert not (output / "first--second").exists()
