"""Exercise the installed CLI, a capped search, and its exported replay."""

import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from tokenbench.experiment import ExperimentSummary
from tokenbench.report import BaselineSummary


def tokenbench(*args: str | Path) -> None:
    subprocess.run(["tokenbench", *(str(arg) for arg in args)], check=True)


def main() -> None:
    with TemporaryDirectory(prefix="tokenbench-ci-") as directory:
        work = Path(directory)
        demo = work / "demo"
        tokenbench("doctor")
        tokenbench("init", "--path", demo)
        tokenbench("baseline", "--suite", demo / "suite.yaml", "--out", work / "baseline")
        baseline = BaselineSummary.model_validate_json(
            (work / "baseline" / "summary.json").read_bytes()
        )
        assert baseline.totals.successful_trials == baseline.totals.trials == 36

        search_dir = work / "search"
        tokenbench("run", "--experiment", demo / "search.yaml", "--out", search_dir)
        search = ExperimentSummary.model_validate_json(
            (search_dir / "comparison.json").read_bytes()
        )
        assert search.status == "partial" and search.stop_reason == "max_trials"
        assert search.budget is not None and search.budget.limits.max_trials is not None
        assert len(search.candidates) == search.budget.limits.max_trials
        assert (search_dir / "pareto.html").is_file()
        assert search.validation is not None and search.validation.status == "complete"
        (confirmed,) = search.validation.confirmed
        original = next(
            candidate for candidate in search.validation.candidates if candidate.name == confirmed
        )
        assert original.totals.successful_trials == original.totals.trials == 8
        assert original.intervals is not None and original.intervals.task_pairs == 8

        replay_yaml = search_dir / search.validation.replay_configs[confirmed]
        replay_dir = work / "replay"
        tokenbench("run", "--experiment", replay_yaml, "--dry-run")
        tokenbench("run", "--experiment", replay_yaml, "--out", replay_dir)
        replay = ExperimentSummary.model_validate_json(
            (replay_dir / "comparison.json").read_bytes()
        )
        assert replay.status == "complete" and replay.validation is not None
        assert replay.validation.confirmed == [confirmed]
        (rerun,) = replay.validation.candidates
        assert rerun.totals == original.totals
        assert rerun.quality == original.quality
        assert rerun.intervals == original.intervals

    print("Offline baseline, capped search, and confirmed replay passed.")


if __name__ == "__main__":
    main()
