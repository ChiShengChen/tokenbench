"""Provenance and integrity checks for exported confirmed configurations."""

import json
from hashlib import sha256
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from tokenbench.policies.compiler import CompiledPolicies

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class ReplayProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_experiment: str
    source_config_sha256: Digest
    source_candidate: str
    tokenbench_version: str
    suite_fingerprints: dict[str, Digest]
    target_module_sha256: Digest | None
    policies_sha256: Digest
    settings_sha256: Digest


def content_fingerprint(payload: object) -> str:
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def policy_fingerprint(compiled: CompiledPolicies) -> str:
    return content_fingerprint([policy.model_dump(mode="json") for policy in compiled.policies])
