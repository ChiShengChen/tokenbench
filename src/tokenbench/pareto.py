"""Report the observed non-dominated frontier without combining objectives."""

from collections import defaultdict
from collections.abc import Collection, Sequence
from decimal import Decimal
from html import escape
from pathlib import Path
from typing import Literal, TypeAlias
from urllib.parse import quote

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


_SCATTER_STYLES = """
.scatter-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 19rem), 1fr)); gap: 1rem; }
.scatter { margin: 0; border: 1px solid #a4acb3; border-radius: .5rem; padding: .75rem; }
.scatter figcaption { font-weight: 600; }
.plot { position: relative; height: 11rem; margin: 1rem 1.3rem 2rem 3rem;
        border-left: 1px solid #87919a; border-bottom: 1px solid #87919a; }
.plot-grid, .y-tick, .x-tick, .point-stack { position: absolute; }
.plot-grid { left: 0; width: 100%; border-top: 1px dotted #a4acb3; pointer-events: none; }
.y-tick { right: calc(100% + .5rem); transform: translateY(50%); font-size: .75rem; }
.x-tick { top: calc(100% + .3rem); transform: translateX(-50%); font-size: .75rem; white-space: nowrap; }
.point-stack { display: flex; gap: .23rem; transform: translate(-50%, 50%); }
.point { display: block; width: .85rem; height: .85rem; border-radius: 50%;
         background: Canvas; border: 2px solid #66737d; }
.point.frontier { background: #2169a1; border-color: #2169a1; }
.point.confirmed { box-shadow: 0 0 0 2px #22814d; }
.point:hover, .point:focus-visible { outline: 2px solid CanvasText; outline-offset: 3px; z-index: 1; }
"""


def scatter(
    metric: str,
    values: Sequence[tuple[ParetoPoint, int | float | Decimal | None]],
    phase: ParetoPhase,
    validation_names: Collection[str],
    confirmed_names: Collection[str],
    *,
    validation_phase: bool,
) -> str:
    coordinates: dict[tuple[Decimal, float], list[ParetoPoint]] = defaultdict(list)
    for item, value in values:
        if value is not None and item.success_rate is not None and item.status != "partial":
            coordinates[Decimal(str(value)), item.success_rate].append(item)
    title = f"Success rate vs. {metric}"
    if not coordinates:
        return f'<figure class="scatter"><figcaption>{escape(title)}</figcaption><p>No paired measurements.</p></figure>'

    minimum = min(value for value, _ in coordinates)
    maximum = max(value for value, _ in coordinates)
    ticks = (
        [(50, minimum)]
        if minimum == maximum
        else [(8, minimum), (50, (minimum + maximum) / 2), (92, maximum)]
    )
    axes = [
        f'<span class="plot-grid" style="bottom:{tick}%"></span>'
        f'<span class="y-tick" style="bottom:{tick}%">{tick}%</span>'
        for tick in (0, 25, 50, 75, 100)
    ]
    axes.extend(
        f'<span class="x-tick" style="left:{position}%">{value:.6g}</span>'
        for position, value in ticks
    )
    groups = []
    for (value, success), items in coordinates.items():
        x = Decimal(50) if minimum == maximum else 8 + 84 * (value - minimum) / (maximum - minimum)
        links = []
        for item in items:
            color = " frontier" if item.name in phase.frontier else ""
            verified = " confirmed" if item.name in confirmed_names else ""
            prefix = (
                "validation" if validation_phase or item.name in validation_names else "candidate"
            )
            href = f"comparison.html#{prefix}-{quote(item.name, safe='')}"
            label = f"{item.name}, {success:.1%} success, {metric} {value:.6g}"
            if verified:
                label += ", independently confirmed"
            attr = escape(label, quote=True)
            links.append(
                f'<a class="point{color}{verified}" href="{href}" title="{attr}" aria-label="{attr}"></a>'
            )
        groups.append(
            f'<div class="point-stack" style="left:{x:.4f}%;bottom:{100 * success:.4f}%">{"".join(links)}</div>'
        )

    return (
        f'<figure class="scatter"><figcaption>{escape(title)}</figcaption><div class="plot">'
        + "".join(axes + groups)
        + "</div></figure>"
    )


def write_pareto(
    output: Path,
    report: ParetoReport,
    *,
    validation_names: Collection[str] = (),
    confirmed_names: Collection[str] = (),
) -> None:
    sections = [
        f"<p>Experiment status: {report.status}. Frontiers use fully evaluated points with all four measurements.</p>"
    ]
    if report.elapsed_seconds is not None:
        sections.append(f"<p>Evaluation elapsed time: {report.elapsed_seconds:.3f} seconds.</p>")
    sections.append(
        "<p>Filled points are on the frontier. A green ring marks an independently confirmed candidate. "
        "Select a point for its paired outcomes and intervals. Points with equal coordinates are grouped.</p>"
    )
    for index, phase in enumerate(report.phases):
        plots = (
            scatter(
                "total tokens",
                [(item, item.total_tokens) for item in phase.points],
                phase,
                validation_names,
                confirmed_names,
                validation_phase=index > 0,
            ),
            scatter(
                "total USD",
                [(item, item.total_cost_usd) for item in phase.points],
                phase,
                validation_names,
                confirmed_names,
                validation_phase=index > 0,
            ),
            scatter(
                "p95 latency (ms)",
                [(item, item.p95_latency_ms) for item in phase.points],
                phase,
                validation_names,
                confirmed_names,
                validation_phase=index > 0,
            ),
        )
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
            + '<div class="scatter-grid">'
            + "".join(plots)
            + "</div>"
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
        html_document(
            f"{report.experiment_id} observed Pareto frontier",
            "\n".join(sections),
            styles=_SCATTER_STYLES,
        ),
    )
