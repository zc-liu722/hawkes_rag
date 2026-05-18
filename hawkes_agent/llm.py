from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

from hawkes_agent.config import ModelRoutingConfig


class ContradictionResult(BaseModel):
    contradicted: list[str] = Field(default_factory=list)

    @field_validator("contradicted")
    @classmethod
    def dedupe_ids(cls, value: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for raw in value:
            item = str(raw)
            if item in seen:
                continue
            seen.add(item)
            out.append(item)
        return out


@dataclass
class LLMUsage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add_response(self, response: Any) -> None:
        self.calls += 1
        usage = getattr(response, "usage", None)
        if usage is None and isinstance(response, dict):
            usage = response.get("usage")
        if not usage:
            return
        get = usage.get if isinstance(usage, dict) else lambda k, d=0: getattr(usage, k, d)
        self.prompt_tokens += int(get("prompt_tokens", 0) or 0)
        self.completion_tokens += int(get("completion_tokens", 0) or 0)
        self.total_tokens += int(get("total_tokens", 0) or 0)


@dataclass
class LiteLLMRouter:
    models: ModelRoutingConfig
    usage_by_site: dict[str, LLMUsage] = field(default_factory=dict)
    strict_schema_retries: int = 1

    def _complete(self, *, call_site: str, messages: list[dict[str, str]], **kwargs: Any) -> str:
        import litellm

        model = getattr(self.models, call_site)
        response = litellm.completion(model=model, messages=messages, **kwargs)
        self.usage_by_site.setdefault(call_site, LLMUsage()).add_response(response)
        choice = response["choices"][0] if isinstance(response, dict) else response.choices[0]
        msg = choice["message"] if isinstance(choice, dict) else choice.message
        return msg["content"] if isinstance(msg, dict) else msg.content

    def _complete_with_optional_schema(
        self,
        *,
        call_site: str,
        messages: list[dict[str, str]],
        response_format: dict[str, Any],
        **kwargs: Any,
    ) -> str:
        try:
            return self._complete(
                call_site=call_site,
                messages=messages,
                response_format=response_format,
                **kwargs,
            )
        except Exception as exc:
            if not _looks_like_unsupported_response_format(exc):
                raise
            return self._complete(call_site=call_site, messages=messages, **kwargs)

    def main_answer(self, system_prompt: str, user_prompt: str) -> str:
        return self._complete(
            call_site="main_llm",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

    def classify_contradictions(
        self,
        *,
        user_turn: str,
        candidates: list[dict[str, str]],
    ) -> ContradictionResult:
        payload = {"user_turn": user_turn, "candidates": candidates}
        candidate_ids = {str(candidate["id"]) for candidate in candidates}
        messages = [
            {
                "role": "system",
                "content": (
                    "Classify direct memory contradictions. Return only the ids "
                    "of candidates directly overridden or falsified by the user turn. "
                    "If none are directly contradicted, return an empty list."
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        last_error = ""
        for attempt in range(self.strict_schema_retries + 1):
            content = self._complete_with_optional_schema(
                call_site="contradiction_micro",
                messages=messages,
                response_format=_contradiction_response_format(candidate_ids),
                temperature=0,
            )
            try:
                return _validated_contradiction_result(content, candidate_ids)
            except ValidationError as exc:
                last_error = str(exc)
                if attempt >= self.strict_schema_retries:
                    return ContradictionResult()
                messages = [
                    *messages,
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": (
                            "Repair the previous response so it satisfies this exact JSON schema "
                            "and uses only ids from the candidate list. "
                            f"Validation error: {last_error}"
                        ),
                    },
                ]
        return ContradictionResult()


def _contradiction_response_format(candidate_ids: set[str] | None = None) -> dict[str, Any]:
    schema = ContradictionResult.model_json_schema()
    schema["additionalProperties"] = False
    schema["required"] = ["contradicted"]
    ids = sorted(candidate_ids or [])
    if ids:
        schema["properties"]["contradicted"]["items"]["enum"] = ids
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "contradiction_result",
            "schema": schema,
            "strict": True,
        },
    }


def _validated_contradiction_result(content: str, candidate_ids: set[str]) -> ContradictionResult:
    try:
        result = ContradictionResult.model_validate_json(content)
    except ValidationError:
        start = content.find("{")
        end = content.rfind("}")
        if start < 0 or end <= start:
            raise
        result = ContradictionResult.model_validate_json(content[start : end + 1])
    invalid = [item for item in result.contradicted if item not in candidate_ids]
    if invalid:
        raise ValidationError.from_exception_data(
            ContradictionResult.__name__,
            [
                {
                    "type": "value_error",
                    "loc": ("contradicted",),
                    "msg": "contradicted ids must come from candidates",
                    "input": result.contradicted,
                    "ctx": {"error": ValueError(f"unknown ids: {invalid}")},
                }
            ],
        )
    return result


def _looks_like_unsupported_response_format(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "response_format" in text
        and (
            "unsupported" in text
            or "unavailable" in text
            or "not support" in text
            or "unknown parameter" in text
            or "unexpected keyword" in text
            or "invalid request" in text
            or "invalid_request" in text
        )
    )
