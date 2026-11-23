from typing import Annotated, Literal, Self, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

ContextHook: TypeAlias = Literal["retrieval", "tool_response", "history"]
HookName: TypeAlias = Literal["retrieval", "tool_response", "history", "step"]
SegmentKind: TypeAlias = Literal[
    "instruction",
    "query",
    "history",
    "retrieved_doc",
    "tool_result",
    "agent_message",
    "code",
    "schema",
    "command",
    "credential",
    "citation",
    "other",
]
Identifier = Annotated[str, Field(min_length=1)]
NonnegativeInt = Annotated[int, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]
StrictBool = Annotated[bool, Field(strict=True)]


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ContextSegment(FrozenModel):
    """A target-provided block. Only explicitly mutable data can be transformed."""

    segment_id: Identifier
    kind: SegmentKind
    content: JsonValue
    role: Literal["system", "developer", "user", "assistant", "tool", "data"] = "data"
    source_ids: tuple[Identifier, ...] = ()
    mutable: StrictBool = False
    protected: StrictBool = False
    exact: StrictBool = False
    required_keys: tuple[Identifier, ...] = ()
    turn_id: Identifier | None = None
    format: Literal["default", "log"] = "default"

    @model_validator(mode="after")
    def validate_required_keys(self) -> Self:
        if self.required_keys and (
            not isinstance(self.content, dict) or not set(self.required_keys) <= self.content.keys()
        ):
            raise ValueError("required_keys must be present in the content object")
        return self

    @property
    def editable(self) -> bool:
        return (
            self.mutable
            and not self.protected
            and not self.exact
            and self.role not in {"system", "developer"}
            and self.kind
            not in {"instruction", "query", "code", "schema", "command", "credential", "citation"}
        )

    @property
    def droppable(self) -> bool:
        return self.editable and not self.required_keys and self.format != "log"


class PolicySettings(FrozenModel):
    enabled: StrictBool = True


class ProjectionParameters(FrozenModel):
    allow_keys: tuple[Identifier, ...]


class Projection(PolicySettings):
    id: Literal["filter.schema_projection"]
    hook: Literal["tool_response"] = "tool_response"
    parameters: ProjectionParameters


class BoilerplateParameters(FrozenModel):
    strip_ansi: StrictBool = True
    drop_lines: tuple[Identifier, ...] = ()


class Boilerplate(PolicySettings):
    id: Literal["filter.boilerplate"]
    hook: Literal["tool_response", "retrieval"] = "tool_response"
    parameters: BoilerplateParameters = BoilerplateParameters()


class DeduplicateParameters(FrozenModel):
    granularity: Literal["block"] = "block"


class Deduplicate(PolicySettings):
    id: Literal["filter.deduplicate"]
    hook: Literal["tool_response", "history"] = "tool_response"
    parameters: DeduplicateParameters = DeduplicateParameters()


class LogTailParameters(FrozenModel):
    head: NonnegativeInt = 2
    tail: NonnegativeInt = 10


class LogTail(PolicySettings):
    id: Literal["filter.log_tail"]
    hook: Literal["tool_response"] = "tool_response"
    parameters: LogTailParameters = LogTailParameters()


class TopKParameters(FrozenModel):
    k: PositiveInt


class TopK(PolicySettings):
    id: Literal["rag.topk"]
    hook: Literal["retrieval"] = "retrieval"
    parameters: TopKParameters


class HistoryParameters(FrozenModel):
    max_turns: NonnegativeInt


class HistoryWindow(PolicySettings):
    id: Literal["history.window"]
    hook: Literal["history"] = "history"
    parameters: HistoryParameters


class StepParameters(FrozenModel):
    max_steps: PositiveInt


class StepBudget(PolicySettings):
    id: Literal["agent.step_budget"]
    hook: Literal["step"] = "step"
    parameters: StepParameters


PolicySpec: TypeAlias = Annotated[
    Projection | Boilerplate | Deduplicate | LogTail | TopK | HistoryWindow | StepBudget,
    Field(discriminator="id"),
]
PolicyStatus: TypeAlias = Literal["applied", "unchanged", "failed", "blocked", "allowed"]


class PolicyAudit(FrozenModel):
    policy_id: str
    policy_version: Literal[1] = 1
    hook: HookName
    status: PolicyStatus
    before_tokens: int | None = None
    after_tokens: int | None = None
    token_source: Literal["estimated", "not_applicable"]
    segments_modified: tuple[str, ...] = ()
    preserved_references: tuple[str, ...] = ()
    removed_references: tuple[str, ...] = ()
    reason: str | None = None
    step: int | None = None
    compressor_api_tokens: Literal[0] = 0
    compressor_cost_usd: Literal["0"] = "0"
