"""Run Hypogen's real live-adapter request path against an in-process HTTP mock."""

import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

import httpx
from pydantic import JsonValue

from tokenbench.tasks import canonical_json
from tokenbench.trace import TraceSink

from .evidence import evidence_owner, evidence_rows, optimize_evidence, verify_cutoff

MODEL = "hypogen-metered-mock"
MOCK_BASE = "http://127.0.0.1/v1"
TOKEN_UNITS = re.compile(r"[A-Za-z0-9]+|[^\s]")


class MeteredHTTP:
    """Replace the Hypogen adapter's HTTPX binding for a single generation."""

    def __init__(self, client: httpx.Client, trace: TraceSink):
        self.client = client
        self.trace = trace

    def __getattr__(self, name: str) -> Any:
        return getattr(httpx, name)

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        with self.trace.model_call(max_cost_usd=0) as call:
            response = self.client.post(url, **kwargs)
            response.raise_for_status()
            completion = response.json()["choices"][0]["message"]["content"]
            call.estimate_usage(
                len(TOKEN_UNITS.findall(response.request.content.decode("utf-8"))),
                len(TOKEN_UNITS.findall(completion)),
            )
        return response


def mock_engine(trace: TraceSink) -> Any:
    from research_app import adapters

    fixture = adapters.LinkedFixture()

    class MeteredLive(adapters.RankedLive):
        def __init__(self) -> None:
            self.key = "local-fixture"
            self.identity = {
                "mode": "local_fixture",
                "prompt_version": self.prompt_version,
                "model": MODEL,
                "base_url": MOCK_BASE,
            }

        def _generate(self, schema: Any, instruction: str, data: dict[str, JsonValue]) -> Any:
            projected = optimize_evidence(data, trace.apply_context)

            def respond(request: httpx.Request) -> httpx.Response:
                if request.method != "POST" or str(request.url) != f"{MOCK_BASE}/chat/completions":
                    raise ValueError("Unexpected Hypogen model endpoint")
                body = json.loads(request.content)
                received = json.loads(body["messages"][1]["content"])
                if body["model"] != MODEL or canonical_json(received) != canonical_json(projected):
                    raise ValueError("Hypogen model request differs from the checked payload")
                verify_cutoff(evidence_owner(received))

                if schema is self.hypothesis_schema:
                    result = fixture.hypotheses(received).model_dump(mode="json")
                elif schema is self.answer_schema:
                    question = received["question"]
                    answer = fixture.answer(received["report"], question)
                    result = {
                        **answer.model_dump(mode="json"),
                        "coverage": [
                            {
                                "requirement": question,
                                "status": "answered" if answer.evidence_ids else "unavailable",
                            }
                        ],
                        "calculation_ids": [],
                        "business_linkages": [],
                        "business_linkage_gap": None,
                    }
                else:
                    raise ValueError("Unsupported Hypogen mock response schema")
                completion = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
                return httpx.Response(200, json={"choices": [{"message": {"content": completion}}]})

            with httpx.Client(transport=httpx.MockTransport(respond), trust_env=False) as client:
                with patch.object(adapters, "httpx", MeteredHTTP(client, trace)):
                    return super()._generate(schema, instruction, projected)

    return MeteredLive()


def verify_financial_inputs(report: dict[str, JsonValue], expected: dict[str, JsonValue]) -> None:
    inputs = report["model_inputs"]
    if not isinstance(inputs, dict) or any(
        inputs[field] != expected[field] for field in ("currency", "unit", "evidence_id")
    ):
        raise ValueError("Hypogen financial units or source citation changed")
    if inputs["evidence_id"] not in {row["id"] for row in evidence_rows(report)}:
        raise ValueError("Hypogen financial source citation is not visible")


def run(task_input: dict[str, JsonValue], trace: TraceSink) -> JsonValue:
    import research_app
    from research_app.schema import Event, Question
    from research_app.store import Store

    operation = task_input["operation"]
    app_files = Path(research_app.__file__).resolve().parent
    fixture = app_files / (
        "examples/asml-q3.json" if operation == "historical-cutoff" else "fixtures/shutdown.json"
    )
    event_data = json.loads(fixture.read_text(encoding="utf-8"))
    event_data["search_mode"] = "off"
    if "as_of" in task_input:
        event_data["as_of"] = task_input["as_of"]
    if operation == "uncited-supply":
        event_data["documents"] = [doc for doc in event_data["documents"] if doc["id"] != "supply"]
    event = Event.model_validate(event_data)
    engine = mock_engine(trace)

    with TemporaryDirectory(prefix="tokenbench-hypogen-case-") as scratch:
        store = Store(Path(scratch) / "research.sqlite3")
        report_id = store.create(event, engine)["report_id"]
        report = store.revision(report_id, 1)
        verify_cutoff(report)
        if event.financial_inputs:
            verify_financial_inputs(report, event.financial_inputs.model_dump(mode="json"))

        if operation == "original":
            return {
                key: report["model_outputs"][key] for key in ("incremental_eps", "break_even_lots")
            }
        if operation == "recompute":
            question = Question(
                parent_revision=1, question="300 批還受益嗎？", overrides={"lots": 300}
            )
            answer = store.question(report_id, question, engine)
            child = store.revision(report_id, answer["child_revision"])
            verify_cutoff(child)
            verify_financial_inputs(child, report["model_inputs"])
            return {
                "eps": child["model_outputs"]["incremental_eps"],
                "delta": answer["incremental_eps_delta"],
                "parent_preserved": store.revision(report_id, 1) == report,
            }
        if operation == "qualification":
            question = Question(
                parent_revision=1, question="客戶已認證乙嗎？", evidence_search="off"
            )
            return store.question(report_id, question, engine)["evidence_ids"]
        if operation == "uncited-supply":
            return report["supply_edges"][0]["status"]
        if operation == "historical-cutoff":
            return {
                "future_q4_visible": any(
                    row["id"].startswith("asml-q4") for row in report["evidence"]
                )
            }
        raise ValueError("Unknown Hypogen fixture operation")
