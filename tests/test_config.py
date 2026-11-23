import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from tokenbench.config import ConfigurationError, SuiteSpec, load_cases
from tokenbench.tasks import ExactGrader


def test_exact_grader_preserves_json_structure_and_types() -> None:
    grader = ExactGrader(kind="exact", expected={"b": True, "a": [1, "2"]})

    assert grader.passes({"a": [1, "2"], "b": True})
    assert not grader.passes({"a": ["2", 1], "b": True})
    assert not grader.passes({"a": [1, "2"], "b": 1})
    assert not grader.passes({"a": [1.0, "2"], "b": True})


def test_cases_validate_duplicate_ids_and_bad_json() -> None:
    data = json.dumps(
        {
            "task_id": "example",
            "input": {},
            "task_type": "test",
            "grader_spec": {"kind": "exact", "expected": None},
        }
    ).encode()
    path = Path("cases.jsonl")

    assert load_cases(data, path)[0].grader_spec.passes(None)
    with pytest.raises(ConfigurationError, match="line 2 has a duplicate task_id"):
        load_cases(data + b"\n" + data, path)
    with pytest.raises(ConfigurationError, match="line 2 has invalid task data"):
        load_cases(data + b"\n{", path)


@pytest.mark.parametrize("repetitions", [0, True, "2"])
def test_repetitions_require_positive_integer(repetitions: object) -> None:
    with pytest.raises(ValidationError, match="repetitions"):
        SuiteSpec.model_validate(
            {
                "schema_version": 1,
                "suite_id": "local",
                "target": {
                    "adapter": "python",
                    "entrypoint": "agent:run",
                    "provider": "test",
                    "model": "test",
                },
                "cases": "cases.jsonl",
                "repetitions": repetitions,
            }
        )
