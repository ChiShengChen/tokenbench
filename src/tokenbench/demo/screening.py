"""Frozen tool-response workload for the single-policy and pairwise demo."""

import json
from typing import cast

from pydantic import JsonValue

from tokenbench.policies.models import ContextSegment
from tokenbench.trace import TraceSink

_READINGS = {
    "north": ("18.4", "C", 2),
    "south": ("67.1", "F", 2),
    "east": ("24.8", "C", 2),
    "west": ("71.6", "F", 2),
    "central": ("21.5", "C", 2),
    "hill": ("14.7", "C", 1),
    "harbor": ("59.3", "F", 3),
    "river": ("17.2", "C", 2),
    "ridge": ("12.9", "C", 4),
    "lake": ("63.8", "F", 1),
    "valley": ("26.3", "C", 3),
    "park": ("68.5", "F", 4),
}
_BANNER = "<nav>Sensor catalogue menu and operator help</nav>"


def run(task_input: dict[str, JsonValue], trace: TraceSink) -> JsonValue:
    sensor, field = task_input.get("sensor"), task_input.get("field")
    if not isinstance(sensor, str) or not isinstance(field, str):
        raise ValueError("sensor and field must be strings")
    value, unit, banner_count = _READINGS[sensor]
    record: dict[str, JsonValue] = {
        "value": value,
        "unit": unit,
        "debug": "sample calibration notes kept by the local instrument collector",
    }
    results = trace.apply_context(
        "tool_response",
        (
            ContextSegment(
                segment_id="primary-reading",
                kind="tool_result",
                content=record,
                source_ids=("primary",),
                mutable=True,
            ),
            ContextSegment(
                segment_id="backup-reading",
                kind="tool_result",
                content=record,
                source_ids=("backup",),
                mutable=True,
            ),
            *(
                ContextSegment(
                    segment_id=f"banner-{index}",
                    kind="tool_result",
                    content=_BANNER,
                    mutable=True,
                )
                for index in range(banner_count)
            ),
        ),
    )
    prompt = json.dumps(
        {"field": field, "results": [result.content for result in results]}, sort_keys=True
    )
    with trace.model_call(max_cost_usd=0) as call:
        answer = fixture_completion(prompt)
        call.estimate_usage(len(prompt.split()), len(json.dumps(answer).split()))
    return answer


def fixture_completion(prompt: str) -> JsonValue:
    payload = cast("dict[str, JsonValue]", json.loads(prompt))
    field, results = payload["field"], payload["results"]
    assert isinstance(field, str) and isinstance(results, list)
    records = [item for item in results if isinstance(item, dict)]
    if not records:
        return None
    if field == "verified_unit":
        units = [record.get("unit") for record in records]
        return units[0] if len(units) >= 2 and all(unit == units[0] for unit in units) else None
    return records[0].get(field)
