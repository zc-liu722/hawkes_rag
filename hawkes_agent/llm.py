from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from hawkes_agent.config import ModelRoutingConfig


class ContradictionResult(BaseModel):
    contradicted: list[str] = Field(default_factory=list)


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

    def _complete(self, *, call_site: str, messages: list[dict[str, str]], **kwargs: Any) -> str:
        import litellm

        model = getattr(self.models, call_site)
        response = litellm.completion(model=model, messages=messages, **kwargs)
        self.usage_by_site.setdefault(call_site, LLMUsage()).add_response(response)
        choice = response["choices"][0] if isinstance(response, dict) else response.choices[0]
        msg = choice["message"] if isinstance(choice, dict) else choice.message
        return msg["content"] if isinstance(msg, dict) else msg.content

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
        content = self._complete(
            call_site="contradiction_micro",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return strict JSON only: {\"contradicted\": [ids]}. "
                        "An id is contradicted only if the user turn directly "
                        "overrides or falsifies that old memory."
                    ),
                },
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0,
        )
        try:
            return ContradictionResult.model_validate_json(content)
        except Exception:
            start = content.find("{")
            end = content.rfind("}")
            if start >= 0 and end > start:
                return ContradictionResult.model_validate_json(content[start : end + 1])
            return ContradictionResult()
