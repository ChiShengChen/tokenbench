"""Normalize reported usage and apply a frozen USD price snapshot."""

from decimal import Decimal
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

TokenCount = Annotated[int, Field(strict=True, ge=0, le=2**63 - 1)]
NonNegativeUSD = Annotated[Decimal, Field(ge=0, allow_inf_nan=False)]
UsageSource = Literal["provider", "estimated", "unknown"]
CostSource = Literal["provider", "price_snapshot", "unknown"]


class ProviderUsage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    input_tokens: TokenCount | None = None
    output_tokens: TokenCount | None = None
    cached_input_tokens: TokenCount | None = None
    reasoning_tokens: TokenCount | None = None
    cost_usd: NonNegativeUSD | None = None

    @model_validator(mode="after")
    def validate_subsets(self) -> Self:
        for detail, total in (
            (self.cached_input_tokens, self.input_tokens),
            (self.reasoning_tokens, self.output_tokens),
        ):
            if detail is not None and total is not None and detail > total:
                raise ValueError("token subset cannot exceed its reported total")
        return self


class ModelPrice(BaseModel):
    model_config = ConfigDict(extra="forbid", str_min_length=1)

    provider: str
    model: str
    input_usd_per_million: NonNegativeUSD
    output_usd_per_million: NonNegativeUSD
    cached_input_usd_per_million: NonNegativeUSD | None = None


class PriceSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", str_min_length=1)

    schema_version: Annotated[int, Field(strict=True, ge=1, le=1)]
    snapshot_id: str
    prices: list[ModelPrice]

    @model_validator(mode="after")
    def validate_prices(self) -> Self:
        if len({(price.provider, price.model) for price in self.prices}) != len(self.prices):
            raise ValueError("price snapshot has duplicate provider/model pairs")
        return self


class UsageRecord(ProviderUsage):
    usage_source: UsageSource
    cost_source: CostSource


def normalize_usage(
    reported: ProviderUsage | None,
    estimate: tuple[int, int] | None,
    price: ModelPrice | None,
) -> UsageRecord:
    usage_source: UsageSource = "unknown"
    cost_source: CostSource = "unknown"
    usage = ProviderUsage()
    if reported is not None:
        usage = reported
        if any(
            value is not None
            for value in (
                reported.input_tokens,
                reported.output_tokens,
                reported.cached_input_tokens,
                reported.reasoning_tokens,
            )
        ):
            usage_source = "provider"
    if usage_source == "unknown" and estimate is not None:
        usage = usage.model_copy(update={"input_tokens": estimate[0], "output_tokens": estimate[1]})
        usage_source = "estimated"

    cost = usage.cost_usd
    if cost is not None:
        cost_source = "provider"
    elif price is not None and usage.input_tokens is not None and usage.output_tokens is not None:
        cached = usage.cached_input_tokens or 0
        if not cached or price.cached_input_usd_per_million is not None:
            cost = (
                (usage.input_tokens - cached) * price.input_usd_per_million
                + cached * (price.cached_input_usd_per_million or Decimal(0))
                + usage.output_tokens * price.output_usd_per_million
            ) / 1_000_000
            cost_source = "price_snapshot"

    return UsageRecord(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cached_input_tokens=usage.cached_input_tokens,
        reasoning_tokens=usage.reasoning_tokens,
        cost_usd=cost,
        usage_source=usage_source,
        cost_source=cost_source,
    )
