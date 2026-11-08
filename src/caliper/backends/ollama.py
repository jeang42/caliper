"""Ollama backend adapter.

Measurement notes
-----------------
* Time to first token is measured from just before the request is sent to the
  arrival of the first stream chunk carrying non-empty generated content. For
  thinking models the first ``thinking`` delta counts as generated content,
  because that is when the model starts producing tokens; ``ttft_source``
  records which stream field supplied it.
* Ollama reports ``load_duration``, ``prompt_eval_duration``, ``eval_duration``
  and ``total_duration`` in nanoseconds in the final stream object. These are
  kept as engine-reported values, separate from client-observed latency.
* Token/s figures are derived from the engine's own token counts and durations,
  which is the fairest available accounting for Ollama.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from .. import http
from ..results import GenerationResult, ModelRecord

log = logging.getLogger(__name__)

NS = 1_000_000_000.0

_CAP_FIELDS = {
    "tools": "supports_tools",
    "thinking": "supports_thinking",
    "vision": "supports_vision",
    "embedding": "supports_embedding",
}


def _ns(value: Any) -> float | None:
    return value / NS if isinstance(value, (int, float)) else None


def _rate(tokens: int | None, seconds: float | None) -> float | None:
    if tokens and seconds and seconds > 0:
        return tokens / seconds
    return None


class OllamaBackend:
    """Adapter for a local Ollama server. Read-only apart from generation."""

    name = "ollama"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        request_timeout_s: float = 600.0,
        connect_timeout_s: float = 10.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.request_timeout_s = request_timeout_s
        self.connect_timeout_s = connect_timeout_s

    # -- introspection ----------------------------------------------------
    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def version(self) -> str | None:
        try:
            return http.get_json(self._url("/api/version"), timeout=self.connect_timeout_s).get(
                "version"
            )
        except http.HttpError as exc:
            log.warning("ollama version unavailable: %s", exc)
            return None

    def available(self) -> bool:
        return self.version() is not None

    def list_models(self) -> list[ModelRecord]:
        data = http.get_json(self._url("/api/tags"), timeout=self.connect_timeout_s)
        records = []
        for entry in data.get("models", []):
            details = entry.get("details") or {}
            records.append(
                ModelRecord(
                    name=entry.get("name", "?"),
                    engine=self.name,
                    digest=entry.get("digest"),
                    size_bytes=entry.get("size"),
                    family=details.get("family"),
                    parameter_size=details.get("parameter_size"),
                    quantization=details.get("quantization_level"),
                    modified_at=entry.get("modified_at"),
                )
            )
        return records

    def _listed_size(self, model: str) -> int | None:
        """On-disk size from /api/tags, which /api/show does not report."""
        try:
            data = http.get_json(self._url("/api/tags"), timeout=self.connect_timeout_s)
        except http.HttpError:
            return None
        for entry in data.get("models", []):
            if entry.get("name") == model or entry.get("model") == model:
                return entry.get("size")
        return None

    def describe_model(self, model: str) -> ModelRecord:
        """Enrich a tag with /api/show metadata. Missing fields stay None."""
        data = http.post_json(
            self._url("/api/show"), {"model": model}, timeout=self.connect_timeout_s * 3
        )
        details = data.get("details") or {}
        info = data.get("model_info") or {}
        arch = info.get("general.architecture") or details.get("family")

        def arch_key(suffix: str) -> Any:
            return info.get(f"{arch}.{suffix}") if arch else None

        caps = list(data.get("capabilities") or [])
        expert_count = arch_key("expert_count")
        expert_used = arch_key("expert_used_count")
        is_moe: bool | None = None
        if expert_count is not None:
            is_moe = int(expert_count) > 1
        elif info:
            # Metadata was readable and contained no expert fields.
            is_moe = False

        record = ModelRecord(
            name=model,
            engine=self.name,
            digest=info.get("general.digest") or data.get("digest"),
            size_bytes=data.get("size") or self._listed_size(model),
            family=details.get("family"),
            architecture=arch,
            parameter_size=details.get("parameter_size"),
            parameter_count=info.get("general.parameter_count"),
            quantization=details.get("quantization_level"),
            context_length=arch_key("context_length"),
            embedding_length=arch_key("embedding_length"),
            block_count=arch_key("block_count"),
            expert_count=expert_count,
            expert_used_count=expert_used,
            is_moe=is_moe,
            capabilities=caps,
            raw_model_info={k: v for k, v in info.items() if not isinstance(v, (list, dict))},
        )
        for cap, field_name in _CAP_FIELDS.items():
            setattr(record, field_name, cap in caps)
        if is_moe and expert_used is None:
            record.notes.append("MoE detected but expert_used_count missing from metadata")
        if is_moe:
            record.notes.append("active parameter count not reported by the engine; left unknown")
        return record

    def loaded_models(self) -> list[dict[str, Any]]:
        try:
            data = http.get_json(self._url("/api/ps"), timeout=self.connect_timeout_s)
        except http.HttpError as exc:
            log.warning("ollama /api/ps unavailable: %s", exc)
            return []
        return data.get("models", []) or []

    def is_loaded(self, model: str) -> bool:
        return any(m.get("name") == model or m.get("model") == model for m in self.loaded_models())

    def residency(self, model: str) -> dict[str, Any] | None:
        """GPU vs CPU residency for a loaded model, as reported by Ollama.

        ``size`` is the total resident size and ``size_vram`` the part on GPU,
        so ``size - size_vram`` is the CPU/RAM offloaded portion.
        """
        for m in self.loaded_models():
            if m.get("name") == model or m.get("model") == model:
                total = m.get("size")
                vram = m.get("size_vram")
                frac = None
                if isinstance(total, (int, float)) and isinstance(vram, (int, float)) and total:
                    frac = vram / total
                return {
                    "size_bytes": total,
                    "size_vram_bytes": vram,
                    "gpu_fraction": frac,
                    "fully_on_gpu": (frac is not None and frac >= 0.999),
                    "expires_at": m.get("expires_at"),
                    "source": "ollama /api/ps",
                }
        return None

    # -- generation -------------------------------------------------------
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
        """Stream one chat completion and collect timings.

        ``/api/chat`` is used for every request so that system prompts and
        tool definitions follow one code path.
        """
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": dict(options or {}),
        }
        if tools:
            payload["tools"] = tools
        if think is not None:
            payload["think"] = think
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive

        was_loaded = self.is_loaded(model)
        result = GenerationResult(
            model=model,
            engine=self.name,
            options=dict(options or {}),
            was_loaded_before=was_loaded,
        )
        content: list[str] = []
        thinking: list[str] = []
        ttft: float | None = None
        ttft_source = "stream_first_content_token"
        final: dict[str, Any] = {}

        start = time.perf_counter()
        result.metrics.request_sent_at = time.time()
        try:
            for chunk in http.post_ndjson_stream(
                self._url("/api/chat"),
                payload,
                timeout=timeout_s or self.request_timeout_s,
            ):
                if err := chunk.get("error"):
                    result.error = f"engine error: {err}"
                    break
                msg = chunk.get("message") or {}
                piece = msg.get("content") or ""
                think_piece = msg.get("thinking") or ""
                calls = msg.get("tool_calls") or []
                if ttft is None and (piece or think_piece or calls):
                    ttft = time.perf_counter() - start
                    ttft_source = (
                        "stream_first_content_token"
                        if piece
                        else "stream_first_thinking_token"
                        if think_piece
                        else "stream_first_tool_call"
                    )
                if piece:
                    content.append(piece)
                if think_piece:
                    thinking.append(think_piece)
                if calls:
                    result.tool_calls.extend(calls)
                if chunk.get("done"):
                    final = chunk
        except http.HttpError as exc:
            result.error = str(exc)
        except KeyboardInterrupt:
            result.error = "interrupted"
            raise
        finally:
            elapsed = time.perf_counter() - start
            result.text = "".join(content)
            result.thinking = "".join(thinking)
            result.raw_final = final
            m = result.metrics
            m.ttft_s = ttft
            m.ttft_source = ttft_source
            m.total_latency_s = elapsed
            m.engine_total_duration_s = _ns(final.get("total_duration"))
            m.engine_load_duration_s = _ns(final.get("load_duration"))
            m.engine_prompt_eval_duration_s = _ns(final.get("prompt_eval_duration"))
            m.engine_eval_duration_s = _ns(final.get("eval_duration"))
            m.prompt_tokens = final.get("prompt_eval_count")
            m.output_tokens = final.get("eval_count")
            m.output_tokens_per_s = _rate(m.output_tokens, m.engine_eval_duration_s)
            m.prompt_tokens_per_s = _rate(m.prompt_tokens, m.engine_prompt_eval_duration_s)
        return result

    def unload(self, model: str) -> bool:
        """Evict a model by requesting keep_alive=0.

        Callers must only use this for models they loaded themselves.
        """
        try:
            http.post_json(
                self._url("/api/chat"),
                {"model": model, "messages": [], "keep_alive": 0},
                timeout=self.connect_timeout_s * 3,
            )
            return True
        except http.HttpError as exc:
            log.warning("could not unload %s: %s", model, exc)
            return False
