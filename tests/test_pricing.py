from decimal import Decimal

import pytest
from pydantic import ValidationError

from tokenbench.pricing import ModelPrice, ProviderUsage, normalize_usage


@pytest.fixture
def rates() -> ModelPrice:
    return ModelPrice(
        provider="test",
        model="test-v1",
        input_usd_per_million=Decimal("1.50"),
        output_usd_per_million=Decimal("4.00"),
    )


def test_provider_totals_and_actual_cost_take_precedence(rates: ModelPrice) -> None:
    usage = normalize_usage(
        ProviderUsage(
            input_tokens=50,
            output_tokens=12,
            cached_input_tokens=30,
            reasoning_tokens=3,
            cost_usd=Decimal("0.005"),
        ),
        (100, 100),
        rates,
    )

    assert usage.usage_source == "provider"
    assert usage.cost_source == "provider"
    assert (usage.input_tokens, usage.output_tokens, usage.cost_usd) == (50, 12, Decimal("0.005"))


def test_estimate_uses_matching_snapshot(rates: ModelPrice) -> None:
    usage = normalize_usage(None, (40, 10), rates)

    assert usage.usage_source == "estimated"
    assert usage.cost_source == "price_snapshot"
    assert usage.cost_usd == Decimal("0.0001")
    assert usage.reasoning_tokens is None


def test_snapshot_uses_provider_token_counts_when_present(rates: ModelPrice) -> None:
    usage = normalize_usage(ProviderUsage(input_tokens=40, output_tokens=10), (100, 100), rates)

    assert usage.usage_source == "provider"
    assert usage.cost_source == "price_snapshot"
    assert usage.cost_usd == Decimal("0.0001")


def test_reported_cost_can_accompany_estimated_counts(rates: ModelPrice) -> None:
    usage = normalize_usage(ProviderUsage(cost_usd=Decimal("0")), (40, 10), rates)

    assert usage.usage_source == "estimated"
    assert usage.cost_source == "provider"
    assert usage.cost_usd == 0


def test_partial_provider_counts_stay_partial(rates: ModelPrice) -> None:
    usage = normalize_usage(ProviderUsage(input_tokens=0), (5, 8), rates)

    assert usage.usage_source == "provider"
    assert usage.input_tokens == 0
    assert usage.output_tokens is None
    assert usage.cost_source == "unknown"
    assert usage.cost_usd is None


def test_cached_counts_with_standard_rates_have_unknown_cost(rates: ModelPrice) -> None:
    usage = normalize_usage(
        ProviderUsage(input_tokens=40, output_tokens=10, cached_input_tokens=5),
        None,
        rates,
    )

    assert usage.input_tokens == 40
    assert usage.cached_input_tokens == 5
    assert usage.cost_source == "unknown"


def test_usage_rejects_invalid_subsets() -> None:
    with pytest.raises(ValidationError, match="token subset"):
        ProviderUsage(output_tokens=2, reasoning_tokens=3)

    with pytest.raises(ValidationError, match="token subset"):
        ProviderUsage(input_tokens=0, cached_input_tokens=1)


def test_usage_rejects_counts_outside_sqlite_integer_range() -> None:
    with pytest.raises(ValidationError, match="less than or equal"):
        ProviderUsage(input_tokens=2**63)
