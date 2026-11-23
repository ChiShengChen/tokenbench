"""TokenBench's local baseline command-line interface."""

import asyncio
import sqlite3
import sys
from contextlib import closing
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path
from typing import Annotated

import typer

from tokenbench.config import ConfigurationError, load_suite
from tokenbench.experiment import load_experiment, run_experiment
from tokenbench.runner import run_baseline
from tokenbench.targets.python_callable import PythonCallableTarget

app = typer.Typer(
    help="Initialize and run local TokenBench baselines and policy experiments.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
)


@app.command()
def init(
    path: Annotated[Path, typer.Option(help="Directory for the bundled toy RAG suite.")],
) -> None:
    """Copy the offline suite, fixtures, and example policy experiment."""
    assets = files("tokenbench.demo")
    filenames = ("suite.yaml", "cases.jsonl", "prices.json", "experiment.yaml")
    try:
        path.mkdir(parents=True, exist_ok=True)
        if any((path / name).exists() or (path / name).is_symlink() for name in filenames):
            raise FileExistsError(f"Demo files already exist in {path}")
        for name in filenames:
            content = assets.joinpath(name).read_bytes()
            with (path / name).open("xb") as destination:
                destination.write(content)
    except OSError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Initialized toy RAG suite at {path / 'suite.yaml'}")


@app.command()
def doctor() -> None:
    """Check the Python and SQLite environment without network access."""
    try:
        with closing(sqlite3.connect(":memory:")) as connection:
            connection.execute("CREATE TABLE healthcheck (id INTEGER PRIMARY KEY)")
            connection.execute("INSERT INTO healthcheck VALUES (1)")
    except sqlite3.Error as exc:
        typer.echo(f"Error: SQLite check failed ({exc})", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"TokenBench {version('tokenbench')}")
    typer.echo(f"Python {sys.version.split()[0]} and SQLite {sqlite3.sqlite_version} OK")


@app.command()
def baseline(
    suite: Annotated[Path, typer.Option(help="Path to a v1 YAML suite.")],
    out: Annotated[
        Path | None, typer.Option(help="New or empty directory for the ledger and reports.")
    ] = None,
    dry_run: Annotated[
        bool, typer.Option(help="Validate suite, fixtures, and callable without running trials.")
    ] = False,
) -> None:
    """Run an unmodified target and write a SQLite ledger and JSON/HTML summaries."""
    if out is None and not dry_run:
        raise typer.BadParameter("--out is required unless --dry-run is set")
    try:
        loaded = load_suite(suite)
        target = PythonCallableTarget.from_spec(loaded.spec.target)
        if dry_run:
            typer.echo(f"Valid suite {loaded.spec.suite_id!r}: {len(loaded.cases)} cases")
            return
        assert out is not None
        summary = asyncio.run(run_baseline(loaded, target, out))
    except (ConfigurationError, OSError, sqlite3.Error) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    totals = summary.totals
    typer.echo(
        f"{totals.successful_trials}/{totals.trials} trials passed ({totals.attempts} attempts)"
    )
    typer.echo(f"Results written to {out}")


@app.command()
def run(
    experiment: Annotated[Path, typer.Option(help="Path to a v1 experiment YAML file.")],
    out: Annotated[
        Path | None, typer.Option(help="Output directory. Defaults to runs/<experiment-id>.")
    ] = None,
    dry_run: Annotated[
        bool, typer.Option(help="Validate the suite, target, and policies without running trials.")
    ] = False,
) -> None:
    """Compare explicitly configured policies against an unmodified baseline."""
    try:
        loaded = load_experiment(experiment)
        target = PythonCallableTarget.from_spec(loaded.suite.spec.target)
        if dry_run:
            typer.echo(
                f"Valid experiment {loaded.spec.experiment_id!r}: {len(loaded.configurations)} candidates"
            )
            return
        output = out if out is not None else Path("runs") / loaded.spec.experiment_id
        summary = asyncio.run(run_experiment(loaded, target, output))
    except (ConfigurationError, OSError, sqlite3.Error) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    for candidate in summary.candidates:
        typer.echo(
            f"{candidate.name}: {candidate.totals.successful_trials}/{candidate.totals.trials} "
            f"trials passed, {len(candidate.regressed_trials)} regressions ({candidate.status})"
        )
    typer.echo(f"Comparison written to {output}")
