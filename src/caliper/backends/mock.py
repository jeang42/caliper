"""Deterministic in-process backend used by tests and ``--dry-run``.

It fabricates plausible timings and canned answers so the full pipeline —
runner, validators, storage, reports — can be exercised without touching the
GPU. Results produced with this backend are tagged ``engine="mock"`` so they
can never be mistaken for measurements.
"""

from __future__ import annotations

import hashlib
from typing import Any

from ..results import GenerationMetrics, GenerationResult, ModelRecord

DEFAULT_ANSWERS: dict[str, str] = {}


class MockBackend:
    name = "mock"

    def __init__(
        self,
        models: list[str] | None = None,
        answers: dict[str, str] | None = None,
        default_answer: str = "mock response",
        fail_models: set[str] | None = None,
        tool_calls: dict[str, list[dict[str, Any]]] | None = None,
    ) -> None:
        self._models = models or ["mock-small:1b", "mock-moe:8b-a1b"]
        self._answers = dict(answers or DEFAULT_ANSWERS)
        self._tool_calls = dict(tool_calls or {})
        self._default_answer = default_answer
        self._fail_models = fail_models or set()
        self._loaded: set[str] = set()
        self.calls: list[dict[str, Any]] = []

    def version(self) -> str | None:
        return "mock-0"

    def available(self) -> bool:
        return True

    def list_models(self) -> list[ModelRecord]:
        return [self.describe_model(m) for m in self._models]

    def describe_model(self, model: str) -> ModelRecord:
        moe = "-a" in model
        return ModelRecord(
            name=model,
            engine=self.name,
            digest=hashlib.sha256(model.encode()).hexdigest()[:12],
            size_bytes=4 * 1024**3,
            family="mock",
            architecture="mock",
            parameter_size="8B",
            quantization="Q4_K_M",
            context_length=32768,
            expert_count=64 if moe else None,
            expert_used_count=8 if moe else None,
            is_moe=moe,
            capabilities=["completion", "tools"],
            supports_tools=True,
            supports_thinking=False,
            supports_vision=False,
        )

    def loaded_models(self) -> list[dict[str, Any]]:
        return [{"name": m, "size": 4 * 1024**3, "size_vram": 4 * 1024**3} for m in self._loaded]

    def is_loaded(self, model: str) -> bool:
        return model in self._loaded

    def residency(self, model: str) -> dict[str, Any] | None:
        if model not in self._loaded:
            return None
        return {
            "size_bytes": 4 * 1024**3,
            "size_vram_bytes": 4 * 1024**3,
            "gpu_fraction": 1.0,
            "fully_on_gpu": True,
            "source": "mock",
        }

    def answer_for(self, prompt: str) -> str:
        for key, value in self._answers.items():
            if key in prompt:
                return value
        return self._default_answer

    def tool_calls_for(self, prompt: str) -> list[dict[str, Any]]:
        for key, value in self._tool_calls.items():
            if key in prompt:
                return [dict(call) for call in value]
        return []

    def generate(
        self,
        model: str,
        prompt: str,
        *,
        system: str | None = None,
        options: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        think: bool | None = None,
        keep_alive: str | None = None,
        timeout_s: float | None = None,
    ) -> GenerationResult:
        self.calls.append({"model": model, "prompt": prompt, "options": dict(options or {})})
        was_loaded = model in self._loaded
        self._loaded.add(model)
        if model in self._fail_models:
            return GenerationResult(
                model=model,
                engine=self.name,
                error="mock failure",
                was_loaded_before=was_loaded,
                options=dict(options or {}),
            )
        text = self.answer_for(prompt)
        calls = self.tool_calls_for(prompt)
        # Deterministic pseudo-timings derived from the prompt hash.
        jitter = int(hashlib.sha256(prompt.encode()).hexdigest()[:4], 16) / 0xFFFF
        out_tokens = max(len(text.split()) + 8 * len(calls), 1)
        prompt_tokens = max(len(prompt.split()), 1)
        eval_s = out_tokens / (40.0 + 10.0 * jitter)
        prompt_s = prompt_tokens / 900.0
        load_s = 0.0 if was_loaded else 2.0
        metrics = GenerationMetrics(
            ttft_s=load_s + prompt_s + 0.02,
            total_latency_s=load_s + prompt_s + eval_s,
            engine_total_duration_s=load_s + prompt_s + eval_s,
            engine_load_duration_s=load_s,
            engine_prompt_eval_duration_s=prompt_s,
            engine_eval_duration_s=eval_s,
            prompt_tokens=prompt_tokens,
            output_tokens=out_tokens,
            output_tokens_per_s=out_tokens / eval_s,
            prompt_tokens_per_s=prompt_tokens / prompt_s,
            ttft_source="mock",
        )
        return GenerationResult(
            model=model,
            engine=self.name,
            text=text,
            tool_calls=calls,
            metrics=metrics,
            options=dict(options or {}),
            was_loaded_before=was_loaded,
            raw_final={"done_reason": "stop", "mock": True},
        )

    def unload(self, model: str) -> bool:
        self._loaded.discard(model)
        return True
