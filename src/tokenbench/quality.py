"""Evaluate an objective quality gate on one paired split."""

from collections.abc import Sequence
from decimal import Decimal
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from tokenbench.policies.models import Identifier, PositiveInt
from tokenbench.report import Totals

Rate = Annotated[Decimal, Field(ge=0, le=1, allow_inf_nan=False)]
QualityReason = Literal[
    "incomplete_result", "min_success_rate", "max_absolute_drop", "critical_regression"
]


class QualitySettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    min_success_rate: Rate = Decimal(0)
    max_absolute_drop: Rate = Decimal(0)
    critical_task_ids: tuple[Identifier, ...] = ()
    validation_candidates: PositiveInt = 1


class QualityCheck(BaseModel):
    passed: bool
    success_rate_drop: Decimal | None = None
    critical_regressions: list[str] = Field(default_factory=list)
    reasons: list[QualityReason]


def assess_quality(
    settings: QualitySettings,
    baseline: Totals,
    candidate: Totals,
    regressed_task_ids: Sequence[str],
    complete: bool,
) -> QualityCheck:
    if not complete or not baseline.trials or baseline.trials != candidate.trials:
        return QualityCheck(passed=False, reasons=["incomplete_result"])

    trials = Decimal(candidate.trials)
    lost_passes = Decimal(baseline.successful_trials - candidate.successful_trials)
    critical = sorted(set(settings.critical_task_ids).intersection(regressed_task_ids))
    reasons: list[QualityReason] = []
    if candidate.successful_trials < settings.min_success_rate * trials:
        reasons.append("min_success_rate")
    if lost_passes > settings.max_absolute_drop * trials:
        reasons.append("max_absolute_drop")
    if critical:
        reasons.append("critical_regression")
    return QualityCheck(
        passed=not reasons,
        success_rate_drop=lost_passes / trials,
        critical_regressions=critical,
        reasons=reasons,
    )
