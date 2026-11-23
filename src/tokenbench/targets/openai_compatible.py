"""Run non-streaming Chat Completions against an OpenAI-compatible endpoint."""

import asyncio
import os
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Annotated, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from tokenbench.budget import ExperimentBudget
from tokenbench.config import ConfigurationError, OpenAICompatibleSpec
from tokenbench.pricing import NonNegativeUSD, ProviderUsage, TokenCount
from tokenbench.targets.base import PermanentTargetError
from tokenbench.trace import ModelCall, TraceSink


class RequestTimeout(RuntimeError):
    """A Chat Completions attempt exceeded its request deadline."""


class TransportFailure(RuntimeError):
    """A connection or transfer failed before completion."""


class RetryableHTTPError(RuntimeError):
    """The endpoint returned a transient HTTP status."""


class HTTPRequestRejected(PermanentTargetError):
    """The endpoint rejected the request without a retryable status."""


class InvalidCompletion(PermanentTargetError):
    """The response did not contain a usable Chat Completions result."""


class TextMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "developer", "user", "assistant"]
    content: str


class ChatInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messages: Annotated[list[TextMessage], Field(min_length=1)]


class CostFields(BaseModel):
    cost_usd: NonNegativeUSD | None = None
    cost: NonNegativeUSD | None = None

    @property
    def billed_usd(self) -> Decimal | None:
        return self.cost_usd if self.cost_usd is not None else self.cost


class PromptDetails(BaseModel):
    cached_tokens: TokenCount | None = None


class CompletionDetails(BaseModel):
    reasoning_tokens: TokenCount | None = None


class ChatUsage(CostFields):
    prompt_tokens: TokenCount | None = None
    completion_tokens: TokenCount | None = None
    prompt_tokens_details: PromptDetails | None = None
    completion_tokens_details: CompletionDetails | None = None

    def normalized(self) -> ProviderUsage:
        return ProviderUsage(
            input_tokens=self.prompt_tokens,
            output_tokens=self.completion_tokens,
            cached_input_tokens=self.prompt_tokens_details.cached_tokens
            if self.prompt_tokens_details is not None
            else None,
            reasoning_tokens=self.completion_tokens_details.reasoning_tokens
            if self.completion_tokens_details is not None
            else None,
            cost_usd=self.billed_usd,
        )


class ChatOutputMessage(BaseModel):
    content: str


class ChatChoice(BaseModel):
    message: ChatOutputMessage
    finish_reason: Literal["stop"]


class ChatOutput(BaseModel):
    choices: Annotated[list[ChatChoice], Field(min_length=1, max_length=1)]


def read_completion(response: httpx.Response, call: ModelCall) -> str:
    payload: object = None
    try:
        payload = response.json(parse_float=Decimal)
        if isinstance(payload, dict) and (usage := payload.get("usage")) is not None:
            # Retain a valid charge even when the remaining usage or completion is malformed.
            billed = CostFields.model_validate(usage).billed_usd
            if billed is not None:
                call.report_usage(ProviderUsage(cost_usd=billed))
            call.report_usage(ChatUsage.model_validate(usage).normalized())
    except ValueError:
        if response.is_success:
            raise InvalidCompletion from None

    if response.status_code in (408, 429) or response.status_code >= 500:
        raise RetryableHTTPError
    if not response.is_success:
        raise HTTPRequestRejected
    try:
        return ChatOutput.model_validate(payload).choices[0].message.content
    except ValidationError:
        raise InvalidCompletion from None


@dataclass(frozen=True)
class OpenAICompatibleTarget:
    spec: OpenAICompatibleSpec
    api_key: str | None = field(default=None, repr=False)

    @classmethod
    def from_spec(cls, spec: OpenAICompatibleSpec) -> "OpenAICompatibleTarget":
        key = None
        if spec.api_key_env is not None:
            key = os.environ.get(spec.api_key_env)
            if not key or any(not 33 <= ord(char) <= 126 for char in key):
                raise ConfigurationError(f"Set {spec.api_key_env} to a valid bearer token")
        return cls(spec, key)

    def source_hash(self) -> None:
        return None

    async def run(
        self,
        task_input: dict[str, JsonValue],
        trace: TraceSink,
        budget: ExperimentBudget | None = None,
    ) -> str:
        try:
            request = ChatInput.model_validate(task_input).model_dump(mode="json")
        except ValidationError:
            raise HTTPRequestRejected from None
        request.update(model=self.spec.model, stream=False, n=1)
        if self.spec.max_completion_tokens is not None:
            request["max_completion_tokens"] = self.spec.max_completion_tokens
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        url = f"{str(self.spec.base_url).rstrip('/')}/chat/completions"
        with trace.model_call(max_cost_usd=self.spec.max_cost_usd) as call:
            try:
                async with (
                    asyncio.timeout(self.spec.timeout_seconds),
                    httpx.AsyncClient(
                        timeout=self.spec.timeout_seconds,
                        trust_env=False,
                        follow_redirects=False,
                    ) as client,
                ):
                    response = await client.post(url, json=request, headers=headers)
            except (TimeoutError, httpx.TimeoutException):
                raise RequestTimeout from None
            except httpx.RequestError:
                raise TransportFailure from None
            return read_completion(response, call)
