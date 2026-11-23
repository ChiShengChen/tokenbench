"""Policy hook and preservation checks for Hypogen's model-bound evidence."""

from collections import Counter
from collections.abc import Callable, Sequence
from datetime import datetime

from pydantic import JsonValue

from tokenbench.policies.deterministic import PolicyError
from tokenbench.policies.models import ContextHook, ContextSegment
from tokenbench.tasks import canonical_json

Hook = Callable[[ContextHook, Sequence[ContextSegment]], tuple[ContextSegment, ...]]


def evidence_owner(payload: dict[str, JsonValue]) -> dict[str, JsonValue]:
    owner = payload.get("report", payload)
    if not isinstance(owner, dict):
        raise PolicyError("invalid_hypogen_report_context")
    return owner


def evidence_rows(owner: dict[str, JsonValue]) -> list[dict[str, JsonValue]]:
    evidence = owner.get("evidence")
    if not isinstance(evidence, list):
        raise PolicyError("missing_hypogen_evidence")
    rows = []
    for row in evidence:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise PolicyError("invalid_hypogen_citation")
        rows.append(row)
    return rows


def essential_fields(row: dict[str, JsonValue]) -> dict[str, JsonValue]:
    locator = str(row["id"]).split(":")
    if locator == [row.get("document_id"), row.get("paragraph")]:
        return {key: value for key, value in row.items() if key not in ("document_id", "paragraph")}
    return row


def verify_payload(
    original: dict[str, JsonValue],
    projected: dict[str, JsonValue],
) -> None:
    original_owner, projected_owner = evidence_owner(original), evidence_owner(projected)
    original_rows, projected_rows = evidence_rows(original_owner), evidence_rows(projected_owner)
    before = [canonical_json(essential_fields(row)) for row in original_rows]
    after = [canonical_json(essential_fields(row)) for row in projected_rows]
    if list(dict.fromkeys(before)) != list(dict.fromkeys(after)) or Counter(after) - Counter(
        before
    ):
        raise PolicyError("hypogen_protected_evidence_changed")

    def without_evidence(
        payload: dict[str, JsonValue], owner: dict[str, JsonValue]
    ) -> dict[str, JsonValue]:
        protected = {key: value for key, value in owner.items() if key != "evidence"}
        return {**payload, "report": protected} if "report" in payload else protected

    if canonical_json(without_evidence(original, original_owner)) != canonical_json(
        without_evidence(projected, projected_owner)
    ):
        raise PolicyError("hypogen_protected_payload_changed")


def optimize_evidence(payload: dict[str, JsonValue], hook: Hook) -> dict[str, JsonValue]:
    owner = evidence_owner(payload)
    rows = evidence_rows(owner)
    segments = hook(
        "tool_response",
        [
            ContextSegment(
                segment_id=f"evidence-{index}",
                kind="tool_result",
                content=row,
                source_ids=(str(row["id"]),),
                mutable=True,
            )
            for index, row in enumerate(rows)
        ],
    )
    evidence: list[JsonValue] = [segment.content for segment in segments]
    updated = {**owner, "evidence": evidence}
    result = {**payload, "report": updated} if "report" in payload else updated
    verify_payload(payload, result)
    return result


def verify_cutoff(report: dict[str, JsonValue]) -> None:
    cutoff = datetime.fromisoformat(str(report["as_of"]))
    for row in evidence_rows(report):
        published = datetime.fromisoformat(str(row["published_at"]))
        obtained = datetime.fromisoformat(str(row["obtained_at"]))
        if max(published, obtained) > cutoff:
            raise PolicyError("hypogen_evidence_after_cutoff")
