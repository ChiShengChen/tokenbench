import json
import os
from contextlib import nullcontext
from importlib import import_module
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import httpx
import pytest
from pydantic import JsonValue
from typer.testing import CliRunner

from tokenbench.cli import app
from tokenbench.policies.deterministic import PolicyError, deduplicate, project
from tokenbench.policies.models import ContextHook, ContextSegment, Deduplicate, Projection
from tokenbench.trace import ModelCall, TraceSink

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def evidence_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(ROOT))
    return import_module("examples.hypogen.evidence")


@pytest.fixture
def evidence_row() -> dict[str, JsonValue]:
    return {
        "id": "qualification:1",
        "document_id": "qualification",
        "paragraph": "1",
        "quote": "完成認證",
        "source": "customer",
        "published_at": "2025-01-20T18:00:00+08:00",
        "obtained_at": "2025-01-21T09:00:00+08:00",
    }


def test_hypogen_dedup_preserves_distinct_citations(
    evidence_module: ModuleType, evidence_row: dict[str, JsonValue]
) -> None:
    distinct = {**evidence_row, "id": "qualification:2", "paragraph": "2"}
    payload = {"evidence": [evidence_row, evidence_row.copy(), distinct]}

    def apply(_hook: ContextHook, segments: list[ContextSegment]) -> tuple[ContextSegment, ...]:
        return deduplicate(tuple(segments), Deduplicate(id="filter.deduplicate"))

    assert evidence_module.optimize_evidence(payload, apply) == {
        "evidence": [evidence_row, distinct]
    }
    assert len(payload["evidence"]) == 3


def test_hypogen_projection_preserves_dates_citations_and_financial_units(
    evidence_module: ModuleType, evidence_row: dict[str, JsonValue]
) -> None:
    calculations = [{"amount": "0.048", "currency": "TWD", "unit": "currency_per_lot"}]
    payload = {"report": {"evidence": [evidence_row]}, "disclosed_calculations": calculations}
    projected = {
        key: value for key, value in evidence_row.items() if key not in {"document_id", "paragraph"}
    }
    policy = Projection.model_validate(
        {
            "id": "filter.schema_projection",
            "parameters": {"allow_keys": tuple(projected)},
        }
    )

    def apply(_hook: ContextHook, segments: list[ContextSegment]) -> tuple[ContextSegment, ...]:
        return tuple(project(segment, policy) for segment in segments)

    assert evidence_module.optimize_evidence(payload, apply) == {
        "report": {"evidence": [projected]},
        "disclosed_calculations": calculations,
    }
    for field in ("id", "quote", "source", "published_at", "obtained_at"):
        altered = {**projected, field: "unexpected"}
        with pytest.raises(PolicyError, match="hypogen_protected_evidence_changed"):
            evidence_module.verify_payload(payload, {**payload, "report": {"evidence": [altered]}})
    with pytest.raises(PolicyError, match="hypogen_protected_payload_changed"):
        evidence_module.verify_payload(
            payload,
            {
                **payload,
                "disclosed_calculations": [{**calculations[0], "unit": "currency"}],
            },
        )
    ambiguous = {
        **evidence_row,
        "id": "customer:qualification:1",
        "document_id": "customer:qualification",
    }
    with pytest.raises(PolicyError, match="hypogen_protected_evidence_changed"):
        evidence_module.optimize_evidence({"evidence": [ambiguous]}, apply)


def test_hypogen_cutoff_checks_publication_and_obtainment(
    evidence_module: ModuleType, evidence_row: dict[str, JsonValue]
) -> None:
    report = {"as_of": evidence_row["obtained_at"], "evidence": [evidence_row]}
    evidence_module.verify_cutoff(report)
    for field in ("published_at", "obtained_at"):
        future = {**evidence_row, field: "2025-01-21T10:00:00+08:00"}
        with pytest.raises(PolicyError, match="hypogen_evidence_after_cutoff"):
            evidence_module.verify_cutoff({**report, "evidence": [future]})


def test_hypogen_http_meter_records_serialized_input_and_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(ROOT))
    target = import_module("examples.hypogen.hypogen_target")
    trace, call = Mock(spec=TraceSink), Mock(spec=ModelCall)
    trace.model_call.return_value = nullcontext(call)

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.content == '{"query":"測試"}'.encode()
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok 完成"}}]})

    with httpx.Client(transport=httpx.MockTransport(respond), trust_env=False) as client:
        response = target.MeteredHTTP(client, trace).post(
            f"{target.MOCK_BASE}/chat/completions", json={"query": "測試"}
        )
    assert response.status_code == 200
    trace.model_call.assert_called_once_with(max_cost_usd=0)
    call.estimate_usage.assert_called_once_with(10, 3)
    call.report_usage.assert_not_called()


def test_real_hypogen_app_mock_comparison(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkout = os.environ.get("TOKENBENCH_HYPOGEN_ROOT")
    if not checkout:
        pytest.skip("Set TOKENBENCH_HYPOGEN_ROOT to run the Hypogen checkout integration")
    monkeypatch.syspath_prepend(checkout)
    monkeypatch.syspath_prepend(str(ROOT))
    output = tmp_path / "hypogen"
    result = CliRunner().invoke(
        app,
        [
            "run",
            "--experiment",
            str(ROOT / "examples/hypogen/experiment.yaml"),
            "--out",
            str(output),
        ],
    )
    assert result.exit_code == 0, (result.output, result.exception)

    comparison = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    baseline = comparison["baseline"]
    deduplication, projection = comparison["candidates"]
    for totals in (baseline, deduplication["totals"], projection["totals"]):
        assert totals["successful_trials"] == totals["trials"] == 6
        assert totals["model_calls"] == 8
        assert totals["usage_sources"] == {"estimated": 8}
        assert totals["total_cost_usd"] == "0"
        assert totals["known_output_tokens"] == baseline["known_output_tokens"]
    assert deduplication["total_token_delta"] == 0
    assert projection["totals"]["known_input_tokens"] < baseline["known_input_tokens"]
    assert projection["regressed_trials"] == []
    assert (output / "pareto.html").is_file()
