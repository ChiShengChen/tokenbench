import json
from decimal import Decimal
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from tokenbench.cli import app

cli = CliRunner()


@pytest.fixture
def confirmed_run(tmp_path: Path) -> tuple[Path, Path]:
    demo = tmp_path / "demo"
    assert cli.invoke(app, ["init", "--path", str(demo)]).exit_code == 0
    suite = demo / "screening-suite.yaml"
    with suite.open("a", encoding="utf-8") as file:
        file.write("repetitions: 2\n")
    output = tmp_path / "original"
    result = cli.invoke(
        app, ["run", "--experiment", str(demo / "quality-screening.yaml"), "--out", str(output)]
    )
    assert result.exit_code == 0, (result.output, result.exception)
    return demo, output


def test_confirmed_yaml_reruns_the_same_comparison(
    confirmed_run: tuple[Path, Path], tmp_path: Path
) -> None:
    _, output = confirmed_run
    original = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    assert original["validation"]["confirmed"] == ["boilerplate"]
    exports = original["validation"]["replay_configs"]
    assert list(exports) == ["boilerplate"]
    yaml_path = output / exports["boilerplate"]
    document = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert document["replay"]["source_config_sha256"] == original["config_sha256"]
    assert document["replay"]["source_candidate"] == "boilerplate"
    assert document["replay"]["target_module_sha256"]
    assert document["candidates"][0]["policies"][0]["id"] == "filter.boilerplate"
    measured = original["validation"]["candidates"][0]
    intervals = measured["intervals"]
    assert intervals["task_pairs"] == 8 and intervals["trial_pairs"] == 16
    tokens = intervals["tokens_per_trial_delta"]
    assert Decimal(tokens["lower"]) < Decimal(tokens["estimate"]) < Decimal(tokens["upper"])
    assert Decimal(tokens["estimate"]) == Decimal("-12.5")
    assert Decimal(intervals["success_rate_delta"]["estimate"]) == 0
    assert Decimal(intervals["cost_usd_per_trial_delta"]["estimate"]) == 0
    assert "Validation paired 95% confidence intervals" in (output / "comparison.html").read_text(
        encoding="utf-8"
    )

    args = ["run", "--experiment", str(yaml_path)]
    dry_run = cli.invoke(app, [*args, "--dry-run"])
    assert dry_run.exit_code == 0, dry_run.output
    replay = tmp_path / "replayed"
    rerun = cli.invoke(app, [*args, "--out", str(replay)])
    assert rerun.exit_code == 0, (rerun.output, rerun.exception)
    result = json.loads((replay / "comparison.json").read_text(encoding="utf-8"))
    assert result["validation"]["candidates"] == [measured]
    assert result["validation"]["replay_configs"] == {}


def test_replay_rejects_changed_settings_and_assets(confirmed_run: tuple[Path, Path]) -> None:
    demo, output = confirmed_run
    path = output / "validation" / "boilerplate" / "replay.yaml"
    untouched = path.read_text(encoding="utf-8")
    cases = (
        ("settings", "Replay budget, quality, or bootstrap settings changed"),
        ("policies", "Replay policy configuration or order changed"),
        ("target", "Replay target module fingerprint changed"),
    )
    for modification, expected in cases:
        document = yaml.safe_load(untouched)
        if modification == "settings":
            document["quality"]["max_absolute_drop"] = "0.05"
        elif modification == "policies":
            document["candidates"][0]["policies"][0]["parameters"]["strip_ansi"] = False
        else:
            document["replay"]["target_module_sha256"] = "0" * 64
        modified = path.with_name(f"{modification}.yaml")
        modified.write_text(yaml.safe_dump(document), encoding="utf-8")
        result = cli.invoke(app, ["run", "--experiment", str(modified), "--dry-run"])
        assert result.exit_code == 1 and expected in result.output

    with (demo / "screening-cases.jsonl").open("a", encoding="utf-8") as file:
        file.write("\n")
    result = cli.invoke(app, ["run", "--experiment", str(path), "--dry-run"])
    assert (
        result.exit_code == 1
        and "Replay suite, cases, or pricing fingerprint changed" in result.output
    )
