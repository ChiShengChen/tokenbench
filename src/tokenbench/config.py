"""Load versioned suite, JSONL task, and price snapshot files."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Annotated, Literal, Self, TypeAlias

import yaml
from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from tokenbench.policies.models import HookName, PositiveInt
from tokenbench.pricing import NonNegativeUSD, PriceSnapshot
from tokenbench.tasks import Split, TaskCase, canonical_json


class ConfigurationError(ValueError):
    """An invalid suite or target configuration."""


class BaseTargetSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", str_min_length=1)

    provider: str
    model: str
    hooks: tuple[HookName, ...] = ()


class PythonTargetSpec(BaseTargetSpec):
    adapter: Literal["python"]
    entrypoint: Annotated[str, Field(pattern=r"^[\w.]+:[A-Za-z_]\w*$")]
    budgeted_model_calls: Annotated[bool, Field(strict=True)] = False


class OpenAICompatibleSpec(BaseTargetSpec):
    adapter: Literal["openai_compatible"]
    base_url: AnyHttpUrl
    hooks: tuple[()] = ()
    api_key_env: Annotated[str, Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")] | None = None
    timeout_seconds: Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)] = 30.0
    max_completion_tokens: PositiveInt | None = None
    max_cost_usd: NonNegativeUSD | None = None

    @field_validator("base_url")
    @classmethod
    def validate_url(cls, value: AnyHttpUrl) -> AnyHttpUrl:
        if any(
            part is not None
            for part in (value.username, value.password, value.query, value.fragment)
        ):
            raise ValueError("base_url cannot include credentials, a query string, or a fragment")
        return value

    @property
    def budgeted_model_calls(self) -> bool:
        return self.max_cost_usd is not None


TargetSpec: TypeAlias = Annotated[
    PythonTargetSpec | OpenAICompatibleSpec, Field(discriminator="adapter")
]
ExecutionMode: TypeAlias = Literal["python_callable", "openai_compatible", "local_fixture"]


class SuiteSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", str_min_length=1)

    schema_version: Annotated[int, Field(strict=True, ge=1, le=1)]
    suite_id: str
    execution: ExecutionMode = "python_callable"
    target: TargetSpec
    cases: Path
    split: Split | None = None
    repetitions: Annotated[int, Field(strict=True, ge=1)] = 1
    max_retries: Annotated[int, Field(strict=True, ge=0)] = 0
    price_snapshot: Path | None = None

    @model_validator(mode="after")
    def validate_execution(self) -> Self:
        default: ExecutionMode = (
            "python_callable" if self.target.adapter == "python" else "openai_compatible"
        )
        if "execution" not in self.model_fields_set:
            self.execution = default
        elif self.execution not in (default, "local_fixture"):
            raise ValueError("execution must match the adapter or be local_fixture")
        return self


@dataclass(frozen=True)
class LoadedSuite:
    spec: SuiteSpec
    cases: list[TaskCase]
    prices: PriceSnapshot | None
    fingerprints: dict[str, str]


def describe_validation(error: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(map(str, detail['loc'])) or 'document'} ({detail['msg']})"
        for detail in error.errors(include_input=False, include_context=False, include_url=False)
    )


def read_file(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ConfigurationError(f"Cannot read {path} ({exc.strerror})") from exc


def load_cases(data: bytes, path: Path) -> list[TaskCase]:
    try:
        lines = data.decode("utf-8").split("\n")
    except UnicodeDecodeError as exc:
        raise ConfigurationError(f"{path} must be UTF-8 JSONL") from exc

    cases: list[TaskCase] = []
    seen: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        location = f"{path}, line {line_number}"
        try:
            task = TaskCase.model_validate_json(line)
            canonical_json(task.input)
            canonical_json(task.grader_spec.expected)
        except ValidationError as exc:
            raise ConfigurationError(
                f"{location} has invalid task data: {describe_validation(exc)}"
            ) from exc
        except ValueError as exc:
            raise ConfigurationError(f"{location} has non-finite JSON numbers") from exc
        if task.task_id in seen:
            raise ConfigurationError(f"{location} has a duplicate task_id")
        seen.add(task.task_id)
        cases.append(task)
    return cases


def parse_yaml(data: bytes, path: Path) -> object:
    try:
        document: object = yaml.safe_load(data)
        return document
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"{path} is invalid YAML") from exc


def load_suite(
    path: Path, *, split: Split | None = None, replace_split: bool = False
) -> LoadedSuite:
    config_data = read_file(path)
    try:
        spec = SuiteSpec.model_validate(parse_yaml(config_data, path))
    except ValidationError as exc:
        raise ConfigurationError(
            f"{path} has invalid suite data: {describe_validation(exc)}"
        ) from exc
    if split is not None:
        if not replace_split and spec.split is not None and spec.split != split:
            raise ConfigurationError(
                f"{path} selects {spec.split!r} but screening requires {split!r}"
            )
        spec = spec.model_copy(update={"split": split})

    case_path = path.parent / spec.cases
    case_data = read_file(case_path)
    cases = load_cases(case_data, case_path)
    if spec.split is not None:
        cases = [case for case in cases if case.split == spec.split]
    if not cases:
        raise ConfigurationError(f"{case_path} has no cases for the selected suite or split")

    fingerprints = {
        "suite": sha256(config_data).hexdigest(),
        "cases": sha256(case_data).hexdigest(),
    }
    prices = None
    if spec.price_snapshot is not None:
        price_path = path.parent / spec.price_snapshot
        price_data = read_file(price_path)
        try:
            prices = PriceSnapshot.model_validate_json(price_data)
        except ValidationError as exc:
            raise ConfigurationError(
                f"{price_path} has invalid prices: {describe_validation(exc)}"
            ) from exc
        fingerprints["price_snapshot"] = sha256(price_data).hexdigest()

    return LoadedSuite(spec, cases, prices, fingerprints)
