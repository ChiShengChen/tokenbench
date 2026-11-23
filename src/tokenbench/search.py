"""Optional Optuna planner for ordered combinations and parameter choices."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from itertools import product
from math import prod
from typing import TYPE_CHECKING, Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tokenbench.pareto import Objectives
from tokenbench.policies.compiler import CompiledPolicies, compile_policies
from tokenbench.policies.models import HookName, Identifier, PolicySpec

if TYPE_CHECKING:
    from optuna.study import Study
    from optuna.trial import Trial


class SearchFactor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: Identifier
    choices: Annotated[tuple[PolicySpec, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_choices(self) -> Self:
        if any(not choice.enabled for choice in self.choices):
            raise ValueError("search choices must be enabled policies; Optuna can omit each factor")
        if len({(choice.id, choice.hook) for choice in self.choices}) != 1:
            raise ValueError("all choices for a factor must use the same policy ID and hook")
        if len({choice.model_dump_json() for choice in self.choices}) != len(self.choices):
            raise ValueError("search choices must be unique")
        return self


class SearchSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: Annotated[int, Field(ge=0, le=2**32 - 1, strict=True)] = 42
    factors: Annotated[tuple[SearchFactor, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_factors(self) -> Self:
        if len({factor.name for factor in self.factors}) != len(self.factors):
            raise ValueError("search factor names must be unique")
        return self

    @property
    def space_size(self) -> int:
        return prod(len(factor.choices) + 1 for factor in self.factors) - 1


@dataclass(frozen=True)
class Proposal:
    trial: Trial
    name: str
    choices: dict[str, int]
    policies: CompiledPolicies


class OptunaPlanner:
    def __init__(self, settings: SearchSettings, hooks: tuple[HookName, ...]) -> None:
        optuna = import_module("optuna")
        self.study: Study = optuna.create_study(
            directions=["minimize"] * 4,
            sampler=optuna.samplers.NSGAIISampler(seed=settings.seed, population_size=4),
        )
        self._failure = optuna.trial.TrialState.FAIL
        self.factors = settings.factors
        self.hooks = hooks
        self._bounds = tuple(range(len(factor.choices) + 1) for factor in self.factors)
        self._seen = {tuple(0 for _ in self.factors)}
        self.space_size = settings.space_size

    def ask(self) -> Proposal:
        for _ in range(8):
            trial = self.study.ask()
            indices = tuple(
                trial.suggest_int(factor.name, 0, len(factor.choices)) for factor in self.factors
            )
            if indices not in self._seen:
                break
            self.study.tell(trial, state=self._failure)
        else:
            # A bounded fallback keeps progress near the end of a small discrete space.
            indices = next(value for value in product(*self._bounds) if value not in self._seen)
            self.study.enqueue_trial(
                {factor.name: index for factor, index in zip(self.factors, indices, strict=True)}
            )
            trial = self.study.ask()
            indices = tuple(
                trial.suggest_int(factor.name, 0, len(factor.choices)) for factor in self.factors
            )

        self._seen.add(indices)
        choices = {factor.name: index for factor, index in zip(self.factors, indices, strict=True)}
        specs = tuple(
            factor.choices[index - 1]
            for factor, index in zip(self.factors, indices, strict=True)
            if index
        )
        return Proposal(
            trial, f"trial-{len(self._seen) - 1}", choices, compile_policies(specs, self.hooks)
        )

    def tell(self, proposal: Proposal, objectives: Objectives | None) -> None:
        if objectives is None:
            self.study.tell(proposal.trial, state=self._failure)
        else:
            self.study.tell(proposal.trial, [float(value) for value in objectives])
