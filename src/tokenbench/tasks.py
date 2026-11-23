"""Task records and objective grading for the baseline runner."""

import json
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, JsonValue

Split: TypeAlias = Literal["dev", "validation", "test"]


class ExactGrader(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["exact"]
    expected: JsonValue

    def passes(self, output: JsonValue) -> bool:
        return canonical_json(output) == canonical_json(self.expected)


class TaskCase(BaseModel):
    model_config = ConfigDict(extra="forbid", str_min_length=1)

    schema_version: Annotated[int, Field(strict=True, ge=1, le=1)] = 1
    task_id: str
    input: dict[str, JsonValue]
    task_type: str
    grader_spec: ExactGrader
    split: Split = "dev"


def canonical_json(value: JsonValue) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False)
