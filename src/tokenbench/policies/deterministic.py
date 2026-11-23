import re
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from tokenbench.policies.models import (
    Boilerplate,
    ContextSegment,
    Deduplicate,
    HistoryWindow,
    Identifier,
    LogTail,
    NonnegativeInt,
    PolicySpec,
    Projection,
    TopK,
)
from tokenbench.tasks import canonical_json

Context = tuple[ContextSegment, ...]

_ANSI_STYLE = re.compile(r"\x1b\[[0-9;:]*m")
_LOG_DIAGNOSTIC = re.compile(r"\b(error|exception|fatal|panic|failed|failure)\b", re.IGNORECASE)
_STACK_START = re.compile(r"\b(?:traceback|stack trace|panic|[\w.]*exception)\b", re.IGNORECASE)
_REQUIRED_FIELDS = frozenset(
    {
        "id",
        "source_id",
        "source_ids",
        "url",
        "citation",
        "citations",
        "error",
        "errors",
        "error_type",
        "exit_code",
        "artifact_ref",
        "command",
        "arguments",
        "credentials",
        "instructions",
        "constraints",
        "tool_schema",
        "schema",
    }
)


class PolicyError(ValueError):
    """A policy could not preserve its declared input contract."""


class LogPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    lines: list[str]
    diagnostic_lines: list[NonnegativeInt]
    artifact_ref: Identifier
    exit_code: Annotated[int, Field(strict=True)]
    omitted_lines: NonnegativeInt = 0


def contains_required_fields(value: JsonValue) -> bool:
    if isinstance(value, dict):
        return bool(_REQUIRED_FIELDS.intersection(value)) or any(
            contains_required_fields(child) for child in value.values()
        )
    if isinstance(value, list):
        return any(contains_required_fields(child) for child in value)
    return False


def project(segment: ContextSegment, policy: Projection) -> ContextSegment:
    content = segment.content
    if not isinstance(content, dict) or segment.format == "log":
        return segment
    keep = _REQUIRED_FIELDS.union(policy.parameters.allow_keys, segment.required_keys)
    projected = {
        key: value
        for key, value in content.items()
        if key in keep or contains_required_fields(value)
    }
    return segment.model_copy(update={"content": projected})


def remove_boilerplate(segment: ContextSegment, policy: Boilerplate) -> ContextSegment:
    if not isinstance(segment.content, str) or segment.format == "log":
        return segment
    content = segment.content
    if policy.parameters.strip_ansi:
        content = _ANSI_STYLE.sub("", content)
    if policy.parameters.drop_lines:
        lines = set(policy.parameters.drop_lines)
        content = "".join(
            line for line in content.splitlines(keepends=True) if line.rstrip("\r\n") not in lines
        )
    return segment.model_copy(update={"content": content})


def tail_log(segment: ContextSegment, policy: LogTail) -> ContextSegment:
    if segment.format != "log":
        return segment
    if not isinstance(segment.content, dict):
        raise PolicyError("invalid_log_payload")
    try:
        log = LogPayload.model_validate(segment.content)
    except ValidationError as exc:
        raise PolicyError("invalid_log_payload") from exc
    size = len(log.lines)
    diagnostics = set(log.diagnostic_lines)
    if any(index >= size for index in diagnostics):
        raise PolicyError("log_diagnostic_index_out_of_range")
    for index, line in enumerate(log.lines):
        if _STACK_START.search(line):
            # Keep the entire remaining stack or chained diagnostics when its end is ambiguous.
            diagnostics.update(range(index, size))
            break
        if _LOG_DIAGNOSTIC.search(line):
            diagnostics.add(index)
    if log.exit_code != 0 and not diagnostics:
        raise PolicyError("log_failure_missing_diagnostics")
    head, tail = policy.parameters.head, policy.parameters.tail
    kept = sorted(diagnostics.union(range(min(head, size)), range(max(0, size - tail), size)))
    if len(kept) == size:
        return segment
    updated = dict(segment.content)
    updated.update(
        {
            "lines": [log.lines[index] for index in kept],
            "diagnostic_lines": [
                offset for offset, index in enumerate(kept) if index in diagnostics
            ],
            "omitted_lines": log.omitted_lines + size - len(kept),
        }
    )
    return segment.model_copy(update={"content": updated})


def deduplicate(context: Context, policy: Deduplicate) -> Context:
    kind = "history" if policy.hook == "history" else "tool_result"
    positions: dict[tuple[str, str | None, str], int] = {}
    kept: list[ContextSegment] = []
    for segment in context:
        if (
            not segment.droppable
            or segment.kind != kind
            or (kind == "history" and segment.turn_id is None)
        ):
            kept.append(segment)
            continue
        key = (segment.role, segment.turn_id, canonical_json(segment.content))
        if key not in positions:
            positions[key] = len(kept)
            kept.append(segment)
            continue
        index = positions[key]
        first = kept[index]
        references = tuple(dict.fromkeys(first.source_ids + segment.source_ids))
        kept[index] = first.model_copy(update={"source_ids": references})
    return tuple(kept)


def ranked_prefix(context: Context, policy: TopK) -> Context:
    ranked = [segment.segment_id for segment in context if segment.kind == "retrieved_doc"]
    first_k = set(ranked[: policy.parameters.k])
    return tuple(
        segment
        for segment in context
        if segment.kind != "retrieved_doc" or not segment.droppable or segment.segment_id in first_k
    )


def history_window(context: Context, policy: HistoryWindow) -> Context:
    turns: dict[str, bool] = {}
    previous = None
    for segment in context:
        if segment.kind != "history":
            continue
        turn = segment.turn_id
        if turn is None:
            if segment.droppable:
                raise PolicyError("history_turn_id_required")
            continue
        if turn in turns and turn != previous:
            raise PolicyError("history_turns_must_be_contiguous")
        turns[turn] = turns.get(turn, False) or not segment.droppable
        previous = turn

    maximum = policy.parameters.max_turns
    keep = set(list(turns)[-maximum:]) if maximum else set()
    keep.update(turn for turn, pinned in turns.items() if pinned)
    return tuple(
        segment
        for segment in context
        if segment.kind != "history" or not segment.droppable or segment.turn_id in keep
    )


def transform(context: Context, policy: PolicySpec) -> Context:
    match policy:
        case Deduplicate():
            return deduplicate(context, policy)
        case TopK():
            return ranked_prefix(context, policy)
        case HistoryWindow():
            return history_window(context, policy)
        case Projection():
            return tuple(
                project(segment, policy)
                if segment.editable and segment.kind == "tool_result"
                else segment
                for segment in context
            )
        case Boilerplate():
            kind = "retrieved_doc" if policy.hook == "retrieval" else "tool_result"
            return tuple(
                remove_boilerplate(segment, policy)
                if segment.editable and segment.kind == kind
                else segment
                for segment in context
            )
        case LogTail():
            return tuple(
                tail_log(segment, policy)
                if segment.editable and segment.kind == "tool_result"
                else segment
                for segment in context
            )
        case _:
            raise PolicyError("policy_requires_step_hook")
