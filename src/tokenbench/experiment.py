"""Execute explicitly named configurations on the same suite and grade paired outcomes."""

from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from tokenbench.config import (
    ConfigurationError,
    LoadedSuite,
    describe_validation,
    load_suite,
    parse_yaml,
    read_file,
)
from tokenbench.policies.compiler import (
    CompiledPolicies,
    PolicyConfigurationError,
    compile_policies,
)
from tokenbench.policies.models import HookName, PolicySpec, PolicyStatus
from tokenbench.report import BaselineSummary, Totals, html_document, table, write_artifacts
from tokenbench.runner import create_output, run_baseline, run_suite
from tokenbench.targets.python_callable import PythonCallableTarget

Slug = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9-]{0,63}$")]


class CandidateSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Slug
    policies: tuple[PolicySpec, ...]


class ExperimentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Annotated[int, Field(strict=True, ge=1, le=1)]
    experiment_id: Slug
    suite: Path
    candidates: Annotated[tuple[CandidateSpec, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_names(self) -> Self:
        names = {candidate.name for candidate in self.candidates}
        if "baseline" in names or len(names) != len(self.candidates):
            raise ValueError("candidate names must be unique and cannot be 'baseline'")
        return self


@dataclass(frozen=True)
class LoadedExperiment:
    spec: ExperimentSpec
    suite: LoadedSuite
    configurations: dict[str, CompiledPolicies]
    fingerprint: str


def load_experiment(path: Path) -> LoadedExperiment:
    content = read_file(path)
    try:
        spec = ExperimentSpec.model_validate(parse_yaml(content, path))
    except ValidationError as exc:
        raise ConfigurationError(
            f"{path} has invalid experiment data: {describe_validation(exc)}"
        ) from exc
    suite = load_suite(path.parent / spec.suite)
    configurations = {}
    for candidate in spec.candidates:
        try:
            configurations[candidate.name] = compile_policies(
                candidate.policies, suite.spec.target.hooks
            )
        except PolicyConfigurationError as exc:
            raise ConfigurationError(f"Candidate {candidate.name}: {exc}") from exc
    return LoadedExperiment(spec, suite, configurations, sha256(content).hexdigest())


class TrialKey(BaseModel):
    task_id: str
    repetition: int


class PolicyCoverage(BaseModel):
    policy_id: str
    hook: HookName
    events: dict[PolicyStatus, int]


class CandidateResult(BaseModel):
    name: str
    status: Literal["complete", "unsupported", "failed"]
    totals: Totals
    coverage: list[PolicyCoverage]
    regressed_trials: list[TrialKey]
    improved_trials: list[TrialKey]
    total_token_delta: int | None
    total_cost_delta_usd: Decimal | None


class ExperimentSummary(BaseModel):
    schema_version: Literal[1] = 1
    experiment_id: str
    config_sha256: str
    baseline: Totals
    candidates: list[CandidateResult]


def compare_candidate(
    name: str, baseline: BaselineSummary, candidate: BaselineSummary
) -> CandidateResult:
    before = {
        (attempt.task_id, attempt.repetition): attempt.passed is True
        for attempt in baseline.attempts
    }
    after = {
        (attempt.task_id, attempt.repetition): attempt.passed is True
        for attempt in candidate.attempts
    }
    events = Counter(
        (event.policy_id, event.hook, event.status)
        for attempt in candidate.attempts
        for event in attempt.policy_events
    )
    coverage = [
        PolicyCoverage(
            policy_id=policy.id,
            hook=policy.hook,
            events={
                status: count
                for (policy_id, hook, status), count in events.items()
                if (policy_id, hook) == (policy.id, policy.hook)
            },
        )
        for policy in candidate.manifest.policies
    ]
    status: Literal["complete", "failed", "unsupported"] = "complete"
    if any(item.events.get("failed") for item in coverage):
        status = "failed"
    elif any(not item.events for item in coverage):
        status = "unsupported"
    token_a, token_b = baseline.totals.total_tokens, candidate.totals.total_tokens
    cost_a, cost_b = baseline.totals.total_cost_usd, candidate.totals.total_cost_usd
    return CandidateResult(
        name=name,
        status=status,
        totals=candidate.totals,
        coverage=coverage,
        regressed_trials=[
            TrialKey(task_id=task_id, repetition=rep)
            for (task_id, rep), passed in before.items()
            if passed and not after[task_id, rep]
        ],
        improved_trials=[
            TrialKey(task_id=task_id, repetition=rep)
            for (task_id, rep), passed in before.items()
            if not passed and after[task_id, rep]
        ],
        total_token_delta=token_b - token_a
        if status == "complete" and token_a is not None and token_b is not None
        else None,
        total_cost_delta_usd=cost_b - cost_a
        if status == "complete" and cost_a is not None and cost_b is not None
        else None,
    )


def comparison_html(summary: ExperimentSummary) -> str:
    rows: list[tuple[object, ...]] = [
        (
            "baseline",
            "complete",
            f"{summary.baseline.successful_trials}/{summary.baseline.trials}",
            "",
            "",
            "",
            "",
        )
    ]
    for candidate in summary.candidates:
        rows.append(
            (
                candidate.name,
                candidate.status,
                f"{candidate.totals.successful_trials}/{candidate.totals.trials}",
                len(candidate.regressed_trials),
                len(candidate.improved_trials),
                candidate.total_token_delta
                if candidate.total_token_delta is not None
                else "Unknown",
                candidate.total_cost_delta_usd
                if candidate.total_cost_delta_usd is not None
                else "Unknown",
            )
        )
    results = table(
        (
            "Configuration",
            "Status",
            "Successful trials",
            "Regressions",
            "Improvements",
            "Token change",
            "USD change",
        ),
        rows,
    )
    return html_document(f"{summary.experiment_id} comparison", results)


async def run_experiment(
    experiment: LoadedExperiment,
    target: PythonCallableTarget,
    output: Path,
) -> ExperimentSummary:
    create_output(output)
    baseline = await run_baseline(experiment.suite, target, output / "baseline")
    candidates = []
    for name, configuration in experiment.configurations.items():
        result = await run_suite(experiment.suite, target, output / name, configuration)
        candidates.append(compare_candidate(name, baseline, result))
    summary = ExperimentSummary(
        experiment_id=experiment.spec.experiment_id,
        config_sha256=experiment.fingerprint,
        baseline=baseline.totals,
        candidates=candidates,
    )
    write_artifacts(output, "comparison", summary, comparison_html(summary))
    return summary
