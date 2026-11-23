from collections.abc import Sequence

import pytest
from pydantic import TypeAdapter, ValidationError

from tokenbench.policies.compiler import PolicyConfigurationError, compile_policies
from tokenbench.policies.deterministic import Context, PolicyError
from tokenbench.policies.models import ContextSegment, HookName, PolicyAudit, PolicySpec
from tokenbench.policies.runtime import PolicySession, StepBudgetExceeded
from tokenbench.tasks import ExactGrader

policy_adapter = TypeAdapter[PolicySpec](PolicySpec)


def policy(policy_id: str, **parameters: object) -> PolicySpec:
    return policy_adapter.validate_python({"id": policy_id, "parameters": parameters})


def session(
    *policies: PolicySpec,
    hooks: Sequence[HookName] = ("tool_response", "retrieval", "history", "step"),
) -> tuple[PolicySession, list[PolicyAudit]]:
    events: list[PolicyAudit] = []
    return PolicySession(
        compile_policies(policies, hooks), lambda event, _: events.append(event)
    ), events


def test_projection_retains_provenance_and_explicit_fields() -> None:
    source = ContextSegment(
        segment_id="result",
        kind="tool_result",
        mutable=True,
        required_keys=("limit",),
        content={
            "answer": 27,
            "noise": "one two three four",
            "limit": 50,
            "nested": {"source_id": "evidence"},
        },
        source_ids=("evidence",),
    )
    transform, events = session(policy("filter.schema_projection", allow_keys=["answer"]))
    (updated,) = transform.apply_context("tool_response", (source,))
    assert updated.content == {"answer": 27, "limit": 50, "nested": {"source_id": "evidence"}}
    assert events[0].segments_modified == ("result",)
    assert events[0].preserved_references == ("evidence",)
    assert events[0].before_tokens is not None and events[0].after_tokens is not None
    assert events[0].before_tokens > events[0].after_tokens
    assert events[0].compressor_api_tokens == 0

    aggressive, _ = session(policy("filter.schema_projection", allow_keys=[]))
    (result,) = aggressive.apply_context("tool_response", (source,))
    grader = ExactGrader(kind="exact", expected=27)
    assert isinstance(source.content, dict) and grader.passes(source.content.get("answer"))
    assert isinstance(result.content, dict) and not grader.passes(result.content.get("answer"))


def test_boilerplate_removes_allowlisted_lines_and_sgr_styles() -> None:
    raw = "<nav>Home</nav>\r\n\x1b[1;32manswer 27\x1b[0m\r\nA <nav>Home</nav> quote"
    content = ContextSegment(segment_id="page", kind="tool_result", mutable=True, content=raw)
    code = ContextSegment(segment_id="example", kind="code", mutable=True, content=raw)
    safe, _ = session(policy("filter.boilerplate", drop_lines=["<nav>Home</nav>"]))
    updated, untouched = safe.apply_context("tool_response", (content, code))
    assert updated.content == "answer 27\r\nA <nav>Home</nav> quote"
    assert untouched == code

    aggressive, _ = session(policy("filter.boilerplate", drop_lines=["answer 27"]))
    (removed,) = aggressive.apply_context("tool_response", (content,))
    grader = ExactGrader(kind="exact", expected=True)
    assert isinstance(updated.content, str) and grader.passes("answer 27" in updated.content)
    assert isinstance(removed.content, str) and not grader.passes("answer 27" in removed.content)


def test_deduplication_retains_references_but_changes_occurrence_count() -> None:
    first = ContextSegment(
        segment_id="a",
        kind="tool_result",
        mutable=True,
        content={"b": 2, "a": 1},
        source_ids=("a",),
    )
    duplicate = ContextSegment(
        segment_id="b",
        kind="tool_result",
        mutable=True,
        content={"a": 1, "b": 2},
        source_ids=("b",),
    )
    dedup, audit = session(policy("filter.deduplicate"))
    unique = dedup.apply_context("tool_response", (first, duplicate))
    assert len(unique) == 1
    assert unique[0].segment_id == "a"
    assert unique[0].source_ids == ("a", "b")
    assert audit[0].segments_modified == ("a", "b")
    assert audit[0].preserved_references == ("a", "b")
    assert not audit[0].removed_references

    grader = ExactGrader(kind="exact", expected=2)
    assert grader.passes(len((first, duplicate))) and not grader.passes(len(unique))
    pinned = duplicate.model_copy(update={"exact": True})
    assert dedup.apply_context("tool_response", (first, pinned)) == (first, pinned)


def test_log_tail_preserves_entire_trace_and_original_artifact() -> None:
    source = ContextSegment(
        segment_id="log",
        kind="tool_result",
        mutable=True,
        format="log",
        content={
            "lines": [
                "start",
                "progress percent 10",
                "answer 27",
                "worker timed out",
                "keepalive",
                "Traceback (most recent call last):",
                "  full frame",
                "RuntimeError: broken",
                "end",
            ],
            "diagnostic_lines": [3],
            "artifact_ref": "log:original",
            "exit_code": 9,
        },
    )
    trim, events = session(policy("filter.log_tail", head=1, tail=1))
    (result,) = trim.apply_context("tool_response", (source,))
    assert result.content == {
        "lines": [
            "start",
            "worker timed out",
            "Traceback (most recent call last):",
            "  full frame",
            "RuntimeError: broken",
            "end",
        ],
        "diagnostic_lines": [1, 2, 3, 4, 5],
        "artifact_ref": "log:original",
        "exit_code": 9,
        "omitted_lines": 3,
    }
    assert events[0].preserved_references == ("log:original",)
    grader = ExactGrader(kind="exact", expected=True)
    assert isinstance(source.content, dict) and isinstance(source.content["lines"], list)
    assert isinstance(result.content, dict) and isinstance(result.content["lines"], list)
    assert grader.passes("answer 27" in source.content["lines"])
    assert not grader.passes("answer 27" in result.content["lines"])


def test_failed_log_with_unclassified_cause_fails_closed() -> None:
    source = ContextSegment(
        segment_id="log",
        kind="tool_result",
        mutable=True,
        format="log",
        content={
            "lines": ["operation ran"],
            "diagnostic_lines": [],
            "artifact_ref": "log:original",
            "exit_code": 2,
        },
    )
    trim, events = session(policy("filter.log_tail", head=0, tail=0))
    with pytest.raises(PolicyError, match="log_failure_missing_diagnostics"):
        trim.apply_context("tool_response", (source,))
    assert events[0].status == "failed"
    assert events[0].after_tokens == events[0].before_tokens


def test_topk_retains_rank_order_and_pinned_evidence() -> None:
    docs = (
        ContextSegment(
            segment_id="noise", kind="retrieved_doc", content="ranked first", mutable=True
        ),
        ContextSegment(
            segment_id="answer",
            kind="retrieved_doc",
            content="answer 27",
            mutable=True,
            source_ids=("b",),
        ),
        ContextSegment(
            segment_id="pin",
            kind="retrieved_doc",
            content="existing citation",
            mutable=True,
            protected=True,
            source_ids=("c",),
        ),
        ContextSegment(segment_id="query", kind="query", content="find 27", mutable=True),
    )
    topk, events = session(policy("rag.topk", k=1))
    result = topk.apply_context("retrieval", docs)
    assert result == (docs[0], docs[2], docs[3])
    assert events[0].preserved_references == ("c",)
    assert events[0].removed_references == ("b",)
    grader = ExactGrader(kind="exact", expected=True)
    assert grader.passes(docs[1] in docs) and not grader.passes(docs[1] in result)


def test_history_window_keeps_entire_pinned_turns_and_recent_turns() -> None:
    turns = (
        ContextSegment(
            segment_id="state",
            kind="history",
            turn_id="1",
            content="state",
            mutable=True,
            protected=True,
        ),
        ContextSegment(
            segment_id="reply", kind="history", turn_id="1", content="reply to state", mutable=True
        ),
        ContextSegment(
            segment_id="old", kind="history", turn_id="2", content="answer 27", mutable=True
        ),
        ContextSegment(
            segment_id="new", kind="history", turn_id="3", content="new query", mutable=True
        ),
    )
    window, events = session(policy("history.window", max_turns=1))
    result = window.apply_context("history", turns)
    assert result == (turns[0], turns[1], turns[3])
    assert events[0].segments_modified == ("old",)
    grader = ExactGrader(kind="exact", expected=True)
    assert grader.passes(any(seg.content == "answer 27" for seg in turns))
    assert not grader.passes(any(seg.content == "answer 27" for seg in result))


def test_step_budget_abstains_and_stays_exhausted() -> None:
    unlimited, _ = session()
    limited, events = session(policy("agent.step_budget", max_steps=1))

    def finish_two_steps(runtime: PolicySession) -> int | None:
        try:
            runtime.step()
            runtime.step()
        except StepBudgetExceeded:
            return None
        return 27

    grader = ExactGrader(kind="exact", expected=27)
    assert grader.passes(finish_two_steps(unlimited))
    assert not grader.passes(finish_two_steps(limited))
    with pytest.raises(StepBudgetExceeded):
        limited.check()
    assert [(event.status, event.step) for event in events] == [("allowed", 1), ("blocked", 2)]
    assert events[1].reason == "step_budget_exceeded"
    assert events[1].before_tokens is None


def test_compiler_rejects_missing_hooks_and_bad_order() -> None:
    with pytest.raises(PolicyConfigurationError, match="requires the retrieval hook"):
        compile_policies((policy("rag.topk", k=1),), ())
    dedup = policy("filter.deduplicate")
    project = policy("filter.schema_projection", allow_keys=["answer"])
    with pytest.raises(PolicyConfigurationError, match="out of order"):
        compile_policies((dedup, project), ("tool_response",))
    with pytest.raises(PolicyConfigurationError, match="Duplicate policy"):
        compile_policies((project, project), ("tool_response",))
    assert compile_policies((project, dedup), ("tool_response",)).policies == (project, dedup)
    with pytest.raises(ValidationError, match="valid integer"):
        policy("rag.topk", k=True)
    with pytest.raises(ValidationError, match="rag.bm25"):
        policy("rag.bm25", k=4)


def test_disable_all_policies_preserves_context_and_does_not_require_hooks() -> None:
    configured = (
        policy("rag.topk", k=1),
        policy("filter.schema_projection", allow_keys=[]),
        policy("filter.boilerplate"),
        policy("filter.deduplicate"),
        policy("filter.log_tail"),
        policy("history.window", max_turns=0),
        policy("agent.step_budget", max_steps=1),
    )
    disabled = tuple(spec.model_copy(update={"enabled": False}) for spec in configured)
    no_op, events = session(*disabled, hooks=())
    context = (ContextSegment(segment_id="x", kind="retrieved_doc", mutable=True, content="data"),)
    assert no_op.apply_context("retrieval", context) == context
    no_op.step()
    no_op.step()
    assert not events


def test_ordered_pipeline_audits_projection_before_deduplication() -> None:
    project = policy("filter.schema_projection", allow_keys=["answer"])
    deduplicate = policy("filter.deduplicate")
    rules, events = session(project, deduplicate)
    segments = tuple(
        ContextSegment(
            segment_id=identifier,
            kind="tool_result",
            source_ids=(identifier,),
            mutable=True,
            content={"answer": 27, "debug": f"generated noise for {identifier}"},
        )
        for identifier in ("a", "b")
    )
    (result,) = rules.apply_context("tool_response", segments)
    assert result.content == {"answer": 27}
    assert result.source_ids == ("a", "b")
    assert [event.policy_id for event in events] == [project.id, deduplicate.id]
    assert all(event.status == "applied" for event in events)
    assert events[0].after_tokens == events[1].before_tokens


def test_runtime_rejects_internal_removal_of_protected_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_transform(context: Context, spec: PolicySpec) -> Context:
        return ()

    monkeypatch.setattr("tokenbench.policies.runtime.transform", broken_transform)
    rules, events = session(policy("rag.topk", k=1))
    instruction = ContextSegment(
        segment_id="instruction", kind="instruction", mutable=True, content="secret value"
    )
    with pytest.raises(PolicyError, match="protected_segment_removed"):
        rules.apply_context("retrieval", (instruction,))
    assert events[0].status == "failed"
    assert "secret value" not in events[0].model_dump_json()
