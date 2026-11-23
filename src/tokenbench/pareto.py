"""Report the observed non-dominated frontier without combining objectives."""

from decimal import Decimal
from html import escape
from pathlib import Path
from typing import Literal, TypeAlias

from pydantic import BaseModel

from tokenbench.report import BaselineSummary, html_document, table, write_artifacts

CandidateStatus: TypeAlias = Literal["complete", "partial", "unsupported", "failed"]
Objectives: TypeAlias = tuple[float, int, Decimal, float]


class ParetoPoint(BaseModel):
    name: str
    status: CandidateStatus
    quality_passed: bool | None = None
    success_rate: float | None
    total_tokens: int | None
    total_cost_usd: Decimal | None
    p95_latency_ms: float | None

    def objectives(self) -> Objectives | None:
        if self.status != "complete" or self.quality_passed is False:
            return None
        success, tokens, cost, latency = (
            self.success_rate,
            self.total_tokens,
            self.total_cost_usd,
            self.p95_latency_ms,
        )
        if success is None or tokens is None or cost is None or latency is None:
            return None
        return -success, tokens, cost, latency


def point(
    name: str, summary: BaselineSummary, status: CandidateStatus, quality_passed: bool | None
) -> ParetoPoint:
    return ParetoPoint(
        name=name,
        status=status,
        quality_passed=quality_passed,
        success_rate=summary.totals.success_rate,
        total_tokens=summary.totals.total_tokens,
        total_cost_usd=summary.totals.total_cost_usd,
        p95_latency_ms=summary.p95_latency_ms,
    )


class ParetoPhase(BaseModel):
    name: str
    points: list[ParetoPoint]
    frontier: list[str]


def dominates(first: Objectives, second: Objectives) -> bool:
    return first != second and (
        first[0] <= second[0]
        and first[1] <= second[1]
        and first[2] <= second[2]
        and first[3] <= second[3]
    )


def frontier(name: str, points: list[ParetoPoint]) -> ParetoPhase:
    eligible = {item.name: values for item in points if (values := item.objectives()) is not None}
    names = [
        name
        for name, values in eligible.items()
        if not any(dominates(other, values) for other in eligible.values())
    ]
    return ParetoPhase(name=name, points=points, frontier=names)


class ParetoReport(BaseModel):
    experiment_id: str
    status: Literal["complete", "partial"]
    phases: list[ParetoPhase]
    elapsed_seconds: float | None = None


def write_pareto(output: Path, report: ParetoReport) -> None:
    sections = [
        f"<p>Experiment status: {report.status}. Frontiers use fully evaluated points with all four measurements.</p>"
    ]
    if report.elapsed_seconds is not None:
        sections.append(f"<p>Evaluation elapsed time: {report.elapsed_seconds:.3f} seconds.</p>")
    for phase in report.phases:
        rows = [
            (
                item.name,
                item.status,
                "Pass"
                if item.quality_passed is True
                else "Fail"
                if item.quality_passed is False
                else "—",
                f"{item.success_rate:.1%}" if item.success_rate is not None else "Unknown",
                item.total_tokens if item.total_tokens is not None else "Unknown",
                item.total_cost_usd if item.total_cost_usd is not None else "Unknown",
                f"{item.p95_latency_ms:.3f}" if item.p95_latency_ms is not None else "Unknown",
                "On frontier"
                if item.name in phase.frontier
                else "Dominated"
                if item.objectives() is not None
                else "Excluded",
            )
            for item in phase.points
        ]
        sections.append(
            f"<h2>{escape(phase.name)}</h2>"
            + table(
                (
                    "Configuration",
                    "Status",
                    "Quality gate",
                    "Success rate ↑",
                    "Tokens ↓",
                    "USD ↓",
                    "p95 ms ↓",
                    "Pareto",
                ),
                rows,
            )
        )
    write_artifacts(
        output,
        "pareto",
        report,
        html_document(f"{report.experiment_id} observed Pareto frontier", "\n".join(sections)),
    )
