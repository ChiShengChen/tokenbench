"""Execute policy configurations on the same suite and grade paired outcomes."""

from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
from html import escape
from importlib.metadata import version
from importlib.util import find_spec
from itertools import combinations
from os.path import relpath
from pathlib import Path
from typing import Annotated, Literal, Self, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from tokenbench.budget import BudgetLimits, BudgetStopped, ExperimentBudget, StopReason
from tokenbench.config import (
    ConfigurationError,
    LoadedSuite,
    describe_validation,
    load_suite,
    parse_yaml,
    read_file,
)
from tokenbench.pareto import (
    CandidateStatus,
    ParetoPoint,
    ParetoReport,
    frontier,
    point,
    write_pareto,
)
from tokenbench.policies.compiler import (
    CompiledPolicies,
    PolicyConfigurationError,
    compile_policies,
)
from tokenbench.policies.models import HookName, PolicySpec, PolicyStatus
from tokenbench.quality import QualityCheck, QualitySettings, assess_quality
from tokenbench.replay import ReplayProvenance, content_fingerprint, policy_fingerprint
from tokenbench.report import (
    BaselineSummary,
    Totals,
    final_outcomes,
    html_document,
    table,
    write_artifacts,
)
from tokenbench.runner import create_output, run_baseline, run_suite
from tokenbench.search import OptunaPlanner, SearchSettings
from tokenbench.statistics import (
    BootstrapSettings,
    Interval,
    PairedIntervals,
    TaskDifference,
    bootstrap,
    paired_tasks,
)
from tokenbench.targets.python_callable import PythonCallableTarget

Slug = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9-]{0,63}$")]


class CandidateSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Slug
    policies: tuple[PolicySpec, ...]


class ScreeningFactor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Slug
    policy: PolicySpec


class ExperimentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Annotated[int, Field(strict=True, ge=1, le=1)]
    experiment_id: Slug
    suite: Path
    candidates: Annotated[tuple[CandidateSpec, ...], Field(min_length=1)] | None = None
    screening: Annotated[tuple[ScreeningFactor, ...], Field(min_length=2)] | None = None
    search: SearchSettings | None = None
    budget: BudgetLimits | None = None
    quality: QualitySettings | None = None
    bootstrap: BootstrapSettings = Field(default_factory=BootstrapSettings)
    replay: ReplayProvenance | None = None

    @model_validator(mode="after")
    def validate_configurations(self) -> Self:
        if sum(mode is not None for mode in (self.candidates, self.screening, self.search)) != 1:
            raise ValueError("choose exactly one of candidates, screening, or search")
        if self.search is not None and (self.budget is None or self.budget.max_trials is None):
            raise ValueError("search requires budget.max_trials")
        if self.replay is not None and (
            self.quality is None or self.candidates is None or len(self.candidates) != 1
        ):
            raise ValueError("a replay requires quality settings and one fixed candidate")
        configurations = self.candidates or self.screening or ()
        names = {candidate.name for candidate in configurations}
        reserved = {"baseline", "validation"} if self.quality else {"baseline"}
        if reserved.intersection(names) or len(names) != len(configurations):
            raise ValueError(
                f"candidate names must be unique and cannot be {', '.join(sorted(reserved))}"
            )
        if self.screening and any(not factor.policy.enabled for factor in self.screening):
            raise ValueError("screening factors must use enabled policies")
        return self

    def replay_settings(self) -> dict[str, object]:
        return self.model_dump(mode="json", include={"budget", "quality", "bootstrap"})


@dataclass(frozen=True)
class ScreeningPair:
    name: str
    first: str
    second: str
    incompatibility: str | None = None


@dataclass(frozen=True)
class LoadedExperiment:
    spec: ExperimentSpec
    suite: LoadedSuite
    configurations: dict[str, CompiledPolicies]
    fingerprint: str
    suite_path: Path
    pairs: tuple[ScreeningPair, ...] = ()
    validation_suite: LoadedSuite | None = None


def load_experiment(path: Path) -> LoadedExperiment:
    content = read_file(path)
    try:
        spec = ExperimentSpec.model_validate(parse_yaml(content, path))
    except ValidationError as exc:
        raise ConfigurationError(
            f"{path} has invalid experiment data: {describe_validation(exc)}"
        ) from exc
    suite_path = path.parent / spec.suite
    suite = load_suite(
        suite_path, split="dev" if spec.screening or spec.search or spec.quality else None
    )
    if (
        spec.budget
        and spec.budget.max_usd is not None
        and not suite.spec.target.budgeted_model_calls
    ):
        raise ConfigurationError(
            "budget.max_usd requires target.budgeted_model_calls and per-call cost bounds"
        )
    validation_suite = None
    if spec.quality is not None:
        validation_suite = load_suite(suite_path, split="validation", replace_split=True)
        visible_ids = {task.task_id for task in (*suite.cases, *validation_suite.cases)}
        if unknown := sorted(set(spec.quality.critical_task_ids) - visible_ids):
            raise ConfigurationError(
                f"Critical task IDs must be on dev or validation: {', '.join(unknown)}"
            )
    if spec.search is not None:
        try:
            compile_policies(
                tuple(factor.choices[0] for factor in spec.search.factors), suite.spec.target.hooks
            )
        except PolicyConfigurationError as exc:
            raise ConfigurationError(f"Search space: {exc}") from exc
        if find_spec("optuna") is None:
            raise ConfigurationError(
                "Search requires the Optuna extra. Install it with uv sync --extra search"
            )
    factors = spec.screening or ()
    singles = spec.candidates or tuple(
        CandidateSpec(name=factor.name, policies=(factor.policy,)) for factor in factors
    )
    configurations = {}
    for candidate in singles:
        try:
            configurations[candidate.name] = compile_policies(
                candidate.policies, suite.spec.target.hooks
            )
        except PolicyConfigurationError as exc:
            raise ConfigurationError(f"Candidate {candidate.name}: {exc}") from exc
    pairs = []
    used_names = set(configurations)
    for first, second in combinations(factors, 2):
        name = f"{first.name}--{second.name}"
        if name in used_names:
            raise ConfigurationError(f"Screening name collision for generated pair {name!r}")
        used_names.add(name)
        try:
            configurations[name] = compile_policies(
                (first.policy, second.policy), suite.spec.target.hooks
            )
        except PolicyConfigurationError as exc:
            pairs.append(ScreeningPair(name, first.name, second.name, str(exc)))
        else:
            pairs.append(ScreeningPair(name, first.name, second.name))
    if spec.replay is not None:
        checked = spec.replay
        if checked.tokenbench_version != version("tokenbench"):
            raise ConfigurationError(f"Replay requires TokenBench {checked.tokenbench_version}")
        if checked.suite_fingerprints != suite.fingerprints:
            raise ConfigurationError("Replay suite, cases, or pricing fingerprint changed")
        if checked.policies_sha256 != policy_fingerprint(next(iter(configurations.values()))):
            raise ConfigurationError("Replay policy configuration or order changed")
        if checked.settings_sha256 != content_fingerprint(spec.replay_settings()):
            raise ConfigurationError("Replay budget, quality, or bootstrap settings changed")
    return LoadedExperiment(
        spec,
        suite,
        configurations,
        sha256(content).hexdigest(),
        suite_path.resolve(),
        tuple(pairs),
        validation_suite,
    )


def verify_replay_target(experiment: LoadedExperiment, target: PythonCallableTarget) -> None:
    replay = experiment.spec.replay
    if replay is not None and replay.target_module_sha256 is not None:
        if target.source_hash() != replay.target_module_sha256:
            raise ConfigurationError("Replay target module fingerprint changed")


class TrialKey(BaseModel):
    task_id: str
    repetition: int


class PolicyCoverage(BaseModel):
    policy_id: str
    hook: HookName
    events: dict[PolicyStatus, int]


class CandidateResult(BaseModel):
    name: str
    status: CandidateStatus
    totals: Totals
    coverage: list[PolicyCoverage]
    regressed_trials: list[TrialKey]
    improved_trials: list[TrialKey]
    total_token_delta: int | None
    total_cost_delta_usd: Decimal | None
    quality: QualityCheck | None = None
    search_choices: dict[str, int] | None = None
    intervals: PairedIntervals | None = None


class PairInteraction(BaseModel):
    name: str
    first: str
    second: str
    status: Literal["complete", "partial", "unsupported", "failed", "incompatible", "not_run"]
    reason: str | None = None
    successful_trial_interaction: int | None = None
    token_interaction: int | None = None
    cost_interaction_usd: Decimal | None = None


class ScreeningSummary(BaseModel):
    split: Literal["dev"] = "dev"
    singles: list[str]
    pairs: list[PairInteraction]


class BudgetSummary(BaseModel):
    limits: BudgetLimits
    search_trials_started: int
    validation_trials_started: int
    known_cost_usd: Decimal
    unpriced_calls: int
    committed_cost_upper_bound_usd: Decimal | None


class ValidationSummary(BaseModel):
    status: Literal["complete", "partial", "not_run"]
    reason: str | None = None
    selected: list[str]
    baseline: Totals | None = None
    candidates: list[CandidateResult] = Field(default_factory=list)
    confirmed: list[str] = Field(default_factory=list)
    replay_configs: dict[str, str] = Field(default_factory=dict)


class SearchSummary(BaseModel):
    sampler: Literal["optuna-nsgaii"] = "optuna-nsgaii"
    seed: int
    space_size: int


class ExperimentSummary(BaseModel):
    schema_version: Literal[1] = 1
    experiment_id: str
    config_sha256: str
    baseline: Totals
    baseline_stop_reason: StopReason | None = None
    candidates: list[CandidateResult]
    screening: ScreeningSummary | None = None
    status: Literal["complete", "partial"] = "complete"
    stop_reason: StopReason | None = None
    budget: BudgetSummary | None = None
    quality: QualitySettings | None = None
    validation: ValidationSummary | None = None
    search: SearchSummary | None = None
    bootstrap: BootstrapSettings = Field(default_factory=BootstrapSettings)


def compare_candidate(
    name: str,
    baseline: BaselineSummary,
    candidate: BaselineSummary,
    quality: QualitySettings | None = None,
) -> CandidateResult:
    before = final_outcomes(baseline.attempts, baseline.manifest.max_retries)
    after = final_outcomes(candidate.attempts, candidate.manifest.max_retries)
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
    status: CandidateStatus = "complete"
    if (
        baseline.stop_reason is not None
        or candidate.stop_reason is not None
        or before.keys() != after.keys()
    ):
        status = "partial"
    elif any(item.events.get("failed") for item in coverage):
        status = "failed"
    elif any(not item.events for item in coverage):
        status = "unsupported"
    token_a, token_b = baseline.totals.total_tokens, candidate.totals.total_tokens
    cost_a, cost_b = baseline.totals.total_cost_usd, candidate.totals.total_cost_usd
    regressions = [
        TrialKey(task_id=key[0], repetition=key[1])
        for key, passed in before.items()
        if passed and key in after and not after[key]
    ]
    return CandidateResult(
        name=name,
        status=status,
        totals=candidate.totals,
        coverage=coverage,
        regressed_trials=regressions,
        improved_trials=[
            TrialKey(task_id=task_id, repetition=rep)
            for (task_id, rep), passed in before.items()
            if not passed and after.get((task_id, rep), False)
        ],
        total_token_delta=token_b - token_a
        if status == "complete" and token_a is not None and token_b is not None
        else None,
        total_cost_delta_usd=cost_b - cost_a
        if status == "complete" and cost_a is not None and cost_b is not None
        else None,
        quality=assess_quality(
            quality,
            baseline.totals,
            candidate.totals,
            [trial.task_id for trial in regressions],
            status == "complete",
        )
        if quality is not None
        else None,
    )


Amount = TypeVar("Amount", int, Decimal)


def interaction(pair: Amount | None, first: Amount | None, second: Amount | None) -> Amount | None:
    if pair is None or first is None or second is None:
        return None
    return pair - first - second


def measure_pair(
    pair: ScreeningPair,
    baseline: Totals,
    results: dict[str, CandidateResult],
) -> PairInteraction:
    if pair.incompatibility is not None:
        return PairInteraction(
            name=pair.name,
            first=pair.first,
            second=pair.second,
            status="incompatible",
            reason=pair.incompatibility,
        )

    missing = [name for name in (pair.name, pair.first, pair.second) if name not in results]
    if missing:
        return PairInteraction(
            name=pair.name,
            first=pair.first,
            second=pair.second,
            status="not_run",
            reason=f"Missing observations: {', '.join(missing)}",
        )
    observed, first, second = results[pair.name], results[pair.first], results[pair.second]
    unavailable = [result for result in (observed, first, second) if result.status != "complete"]
    if unavailable:
        return PairInteraction(
            name=pair.name,
            first=pair.first,
            second=pair.second,
            status="partial"
            if any(result.status == "partial" for result in unavailable)
            else "failed"
            if any(result.status == "failed" for result in unavailable)
            else "unsupported",
            reason=", ".join(f"{result.name}: {result.status}" for result in unavailable),
        )

    return PairInteraction(
        name=pair.name,
        first=pair.first,
        second=pair.second,
        status="complete",
        successful_trial_interaction=(
            observed.totals.successful_trials
            + baseline.successful_trials
            - first.totals.successful_trials
            - second.totals.successful_trials
        ),
        token_interaction=interaction(
            observed.total_token_delta, first.total_token_delta, second.total_token_delta
        ),
        cost_interaction_usd=interaction(
            observed.total_cost_delta_usd, first.total_cost_delta_usd, second.total_cost_delta_usd
        ),
    )


def comparison_html(summary: ExperimentSummary) -> str:
    rows: list[tuple[object, ...]] = [
        (
            "baseline",
            "partial" if summary.baseline_stop_reason else "complete",
            f"{summary.baseline.successful_trials}/{summary.baseline.trials}",
            "",
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
                quality_label(candidate.quality),
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
            "Quality gate",
        ),
        rows,
    )
    status = f"{summary.status} ({summary.stop_reason})" if summary.stop_reason else summary.status
    results = (
        f'<p>Experiment status: {escape(status)}. <a href="pareto.html">Observed Pareto frontier</a></p>'
        + results
    )
    if summary.budget is not None:
        budget = summary.budget
        results += "<h2>Budget</h2>" + table(
            ("Measure", "Value"),
            [
                ("Search trials started", budget.search_trials_started),
                ("Validation trials started", budget.validation_trials_started),
                ("Known spend (USD)", budget.known_cost_usd),
                ("Unpriced calls", budget.unpriced_calls),
                (
                    "Committed upper bound (USD)",
                    budget.committed_cost_upper_bound_usd
                    if budget.committed_cost_upper_bound_usd is not None
                    else "Unset",
                ),
            ],
        )
    if summary.screening is not None:
        interactions = table(
            (
                "Pair",
                "Status",
                "Pass interaction",
                "Token interaction",
                "USD interaction",
                "Reason",
            ),
            [
                (
                    f"{pair.first} → {pair.second}",
                    pair.status,
                    pair.successful_trial_interaction
                    if pair.successful_trial_interaction is not None
                    else "Unknown",
                    pair.token_interaction if pair.token_interaction is not None else "Unknown",
                    pair.cost_interaction_usd
                    if pair.cost_interaction_usd is not None
                    else "Unknown",
                    pair.reason or "",
                )
                for pair in summary.screening.pairs
            ],
        )
        results += (
            "<h2>Pairwise interactions on dev</h2>"
            "<p>Interaction is the observed pair change minus the sum of the two individual changes. "
            "Positive token and USD values indicate less savings than the additive estimate.</p>"
            + interactions
        )
    regressions = [
        (candidate.name, trial.task_id, trial.repetition)
        for candidate in summary.candidates
        for trial in candidate.regressed_trials
    ]
    if regressions:
        results += "<h2>Regressed trials</h2>" + table(
            ("Configuration", "Task", "Repetition"), regressions
        )
    results += intervals_html(
        "Development" if summary.validation is not None else "Experiment",
        summary.candidates,
        summary.bootstrap,
    )
    if summary.validation is not None:
        validation = summary.validation
        results += (
            "<h2>Independent validation</h2>"
            + table(
                ("Measure", "Value"),
                [
                    ("Status", validation.status),
                    ("Stop reason", validation.reason or ""),
                    ("Selected on dev", ", ".join(validation.selected)),
                    ("Confirmed on validation", ", ".join(validation.confirmed)),
                ],
            )
            + table(
                ("Configuration", "Successful trials", "Quality gate", "Regressed tasks"),
                [
                    (
                        item.name,
                        f"{item.totals.successful_trials}/{item.totals.trials}",
                        quality_label(item.quality),
                        ", ".join(trial.task_id for trial in item.regressed_trials),
                    )
                    for item in validation.candidates
                ],
            )
        )
        results += intervals_html("Validation", validation.candidates, summary.bootstrap)
        if validation.replay_configs:
            links = "".join(
                f'<li><a href="{escape(path, quote=True)}">{escape(name)} replay YAML</a></li>'
                for name, path in validation.replay_configs.items()
            )
            results += "<h3>Confirmed replay configurations</h3><ul>" + links + "</ul>"
    return html_document(f"{summary.experiment_id} comparison", results)


def interval_cell(interval: Interval | None, scale: int = 1) -> str:
    if interval is None:
        return "Unavailable"
    estimate, lower, upper = (
        value * scale for value in (interval.estimate, interval.lower, interval.upper)
    )
    return f"{estimate:.6g} [{lower:.6g}, {upper:.6g}]"


def intervals_html(
    label: str, candidates: list[CandidateResult], settings: BootstrapSettings
) -> str:
    rows = [
        (
            candidate.name,
            result.status,
            result.task_pairs,
            result.trial_pairs,
            result.tasks_with_token_counts,
            result.tasks_with_costs,
            interval_cell(result.success_rate_delta, 100),
            interval_cell(result.tokens_per_trial_delta),
            interval_cell(result.cost_usd_per_trial_delta),
        )
        for candidate in candidates
        if (result := candidate.intervals) is not None
    ]
    if not rows:
        return ""
    return (
        f"<h3>{escape(label)} paired 95% confidence intervals</h3>"
        f"<p>Candidate minus baseline. {settings.resamples:,} percentile bootstrap resamples of whole tasks"
        f" with seed {settings.seed}. Small or uniform samples give limited information about other tasks.</p>"
        + table(
            (
                "Configuration",
                "Status",
                "Task pairs",
                "Trial pairs",
                "Token pairs",
                "USD pairs",
                "Success rate (percentage points)",
                "Tokens per trial",
                "USD per trial",
            ),
            rows,
        )
    )


def quality_label(check: QualityCheck | None) -> str:
    if check is None:
        return "—"
    return "Pass" if check.passed else ", ".join(check.reasons)


async def run_experiment(
    experiment: LoadedExperiment,
    target: PythonCallableTarget,
    output: Path,
) -> ExperimentSummary:
    verify_replay_target(experiment, target)
    create_output(output)
    spec = experiment.spec
    budget = ExperimentBudget(spec.budget) if spec.budget is not None else None
    baseline = await run_baseline(experiment.suite, target, output / "baseline", budget)
    phase = PhaseRunner(experiment.suite, target, output, baseline, budget, spec.quality)
    planner = (
        OptunaPlanner(spec.search, experiment.suite.spec.target.hooks)
        if spec.search is not None
        else None
    )
    evaluated, stopped = await evaluate_search(experiment, phase, planner)
    candidates = [item.result for item in evaluated]
    screening = None
    if spec.screening is not None:
        by_name = {candidate.name: candidate for candidate in candidates}
        screening = ScreeningSummary(
            singles=[factor.name for factor in spec.screening],
            pairs=[measure_pair(pair, baseline.totals, by_name) for pair in experiment.pairs],
        )
    validation = None
    validation_evaluations: tuple[Evaluation, ...] = ()
    phases = [
        frontier(
            experiment.suite.spec.split or "all",
            [phase.baseline_point(), *(item.metrics for item in evaluated)],
        )
    ]
    if spec.quality is not None and experiment.validation_suite is not None:
        outcome = await validate_finalists(
            experiment.validation_suite, phase, evaluated, spec.quality
        )
        validation = outcome.summary
        validation_evaluations = outcome.evaluations
        if outcome.baseline is not None:
            phases.append(
                frontier(
                    "validation",
                    [outcome.baseline, *(item.metrics for item in outcome.evaluations)],
                )
            )

    if budget is not None and budget.stop_reason is not None:
        stopped = budget.stop_reason
    elapsed_seconds = budget.elapsed_seconds if budget is not None else None
    for item in (*evaluated, *validation_evaluations):
        item.result.intervals = bootstrap(
            item.tasks, spec.bootstrap, complete=item.result.status == "complete"
        )
    if validation is not None and spec.replay is None:
        validation.replay_configs = write_confirmed_replays(
            experiment,
            baseline,
            validation,
            validation_evaluations,
            output,
        )
    summary = ExperimentSummary(
        experiment_id=spec.experiment_id,
        config_sha256=experiment.fingerprint,
        baseline=baseline.totals,
        baseline_stop_reason=baseline.stop_reason,
        candidates=candidates,
        screening=screening,
        status="partial" if stopped is not None else "complete",
        stop_reason=stopped,
        budget=BudgetSummary(
            limits=budget.limits,
            search_trials_started=len(candidates),
            validation_trials_started=len(validation.candidates) if validation is not None else 0,
            known_cost_usd=budget.known_cost,
            unpriced_calls=budget.unpriced_calls,
            committed_cost_upper_bound_usd=budget.committed_cost
            if budget.limits.max_usd is not None
            else None,
        )
        if budget is not None
        else None,
        quality=spec.quality,
        validation=validation,
        search=SearchSummary(seed=spec.search.seed, space_size=planner.space_size)
        if spec.search is not None and planner is not None
        else None,
        bootstrap=spec.bootstrap,
    )
    write_artifacts(output, "comparison", summary, comparison_html(summary))
    write_pareto(
        output,
        ParetoReport(
            experiment_id=spec.experiment_id,
            status=summary.status,
            phases=phases,
            elapsed_seconds=elapsed_seconds,
        ),
    )
    return summary


@dataclass(frozen=True)
class Evaluation:
    result: CandidateResult
    metrics: ParetoPoint
    policies: CompiledPolicies
    tasks: tuple[TaskDifference, ...]


@dataclass(frozen=True)
class ValidationRun:
    summary: ValidationSummary
    baseline: ParetoPoint | None = None
    evaluations: tuple[Evaluation, ...] = ()


@dataclass(frozen=True)
class PhaseRunner:
    suite: LoadedSuite
    target: PythonCallableTarget
    output: Path
    baseline: BaselineSummary
    budget: ExperimentBudget | None
    quality: QualitySettings | None

    def baseline_point(self) -> ParetoPoint:
        complete = self.baseline.stop_reason is None
        totals = self.baseline.totals
        gate = (
            assess_quality(self.quality, totals, totals, (), complete)
            if self.quality is not None
            else None
        )
        return point(
            "baseline",
            self.baseline,
            "complete" if complete else "partial",
            gate.passed if gate is not None else None,
        )

    async def evaluate(
        self,
        name: str,
        policies: CompiledPolicies,
        choices: dict[str, int] | None = None,
    ) -> Evaluation:
        observed = await run_suite(
            self.suite, self.target, self.output / name, policies, self.budget
        )
        result = compare_candidate(name, self.baseline, observed, self.quality)
        result.search_choices = choices
        passed = result.quality.passed if result.quality is not None else None
        return Evaluation(
            result,
            point(name, observed, result.status, passed),
            policies,
            paired_tasks(self.baseline, observed),
        )


def budget_cutoff(budget: ExperimentBudget | None) -> StopReason | None:
    if budget is not None:
        try:
            budget.remaining_seconds()
        except BudgetStopped as exc:
            return exc.reason
    return None


async def evaluate_search(
    experiment: LoadedExperiment,
    phase: PhaseRunner,
    planner: OptunaPlanner | None,
) -> tuple[list[Evaluation], StopReason | None]:
    measured: list[Evaluation] = []
    if phase.baseline.stop_reason is not None:
        return measured, phase.baseline.stop_reason
    configurations = iter(experiment.configurations.items())
    planned = planner.space_size if planner is not None else len(experiment.configurations)
    while len(measured) < planned:
        if reason := budget_cutoff(phase.budget):
            return measured, reason
        maximum = phase.budget.limits.max_trials if phase.budget is not None else None
        if maximum is not None and len(measured) >= maximum:
            return measured, "max_trials"
        if planner is None:
            name, policies = next(configurations)
            measured.append(await phase.evaluate(name, policies))
        else:
            proposal = planner.ask()
            observed = await phase.evaluate(proposal.name, proposal.policies, proposal.choices)
            measured.append(observed)
            planner.tell(proposal, observed.metrics.objectives())
        if phase.budget is not None and phase.budget.stop_reason is not None:
            return measured, phase.budget.stop_reason
    return measured, None


def finalist_order(evaluation: Evaluation) -> tuple[bool, Decimal, bool, int, int, str]:
    totals = evaluation.result.totals
    return (
        totals.total_cost_usd is None,
        totals.total_cost_usd or Decimal(0),
        totals.total_tokens is None,
        totals.total_tokens or 0,
        -totals.successful_trials,
        evaluation.result.name,
    )


async def validate_finalists(
    suite: LoadedSuite,
    development: PhaseRunner,
    evaluated: list[Evaluation],
    quality: QualitySettings,
) -> ValidationRun:
    qualified = [
        item for item in evaluated if item.result.quality is not None and item.result.quality.passed
    ]
    finalists = sorted(qualified, key=finalist_order)[: quality.validation_candidates]
    selected = [item.result.name for item in finalists]
    if not selected:
        return ValidationRun(
            ValidationSummary(
                status="not_run",
                reason="no_qualified_candidates",
                selected=selected,
            )
        )
    if reason := budget_cutoff(development.budget):
        return ValidationRun(ValidationSummary(status="not_run", reason=reason, selected=selected))
    output = development.output / "validation"
    baseline = await run_baseline(
        suite, development.target, output / "baseline", development.budget
    )
    phase = PhaseRunner(suite, development.target, output, baseline, development.budget, quality)
    measured = []
    stopped = baseline.stop_reason
    if stopped is None:
        for item in finalists:
            if stopped := budget_cutoff(development.budget):
                break
            measured.append(
                await phase.evaluate(item.result.name, item.policies, item.result.search_choices)
            )
            if development.budget is not None and development.budget.stop_reason is not None:
                stopped = development.budget.stop_reason
                break
    candidates = [item.result for item in measured]
    summary = ValidationSummary(
        status="partial" if stopped is not None else "complete",
        reason=stopped,
        selected=selected,
        baseline=baseline.totals,
        candidates=candidates,
        confirmed=[
            item.name for item in candidates if item.quality is not None and item.quality.passed
        ],
    )
    return ValidationRun(summary, phase.baseline_point(), tuple(measured))


def write_confirmed_replays(
    experiment: LoadedExperiment,
    baseline: BaselineSummary,
    validation: ValidationSummary,
    evaluated: tuple[Evaluation, ...],
    output: Path,
) -> dict[str, str]:
    paths = {}
    settings = experiment.spec.replay_settings()
    settings_hash = content_fingerprint(settings)
    for item in evaluated:
        name = item.result.name
        if name not in validation.confirmed:
            continue
        relative = Path("validation") / name / "replay.yaml"
        path = output / relative
        try:
            suite = relpath(experiment.suite_path, path.parent)
        except ValueError:
            suite = str(experiment.suite_path)
        provenance = ReplayProvenance(
            source_experiment=experiment.spec.experiment_id,
            source_config_sha256=experiment.fingerprint,
            source_candidate=name,
            tokenbench_version=baseline.manifest.tokenbench_version,
            suite_fingerprints=experiment.suite.fingerprints,
            target_module_sha256=baseline.manifest.fingerprints.get("target_module"),
            policies_sha256=policy_fingerprint(item.policies),
            settings_sha256=settings_hash,
        )
        document = {
            "schema_version": 1,
            "experiment_id": f"replay-{name[:40]}-{experiment.fingerprint[:12]}",
            "suite": suite,
            "candidates": [
                {
                    "name": name if len(name) <= 64 else "confirmed",
                    "policies": [
                        policy.model_dump(mode="json") for policy in item.policies.policies
                    ],
                }
            ],
            **settings,
            "replay": provenance.model_dump(mode="json"),
        }
        temporary = path.with_suffix(".yaml.tmp")
        temporary.write_text(
            yaml.safe_dump(document, sort_keys=False), encoding="utf-8", newline="\n"
        )
        temporary.replace(path)
        paths[name] = relative.as_posix()
    return paths
