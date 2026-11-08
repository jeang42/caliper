"""Result records shared by backends, the runner and storage."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class ModelRecord:
    """Verifiable metadata for one installed model tag.

    Fields that cannot be read from the engine are ``None`` and recorded as
    unknown; nothing here is inferred from a model's name except
    ``active_params_source``, which names where a value came from.
    """

    name: str
    engine: str
    digest: str | None = None
    size_bytes: int | None = None
    family: str | None = None
    architecture: str | None = None
    parameter_size: str | None = None
    parameter_count: int | None = None
    quantization: str | None = None
    context_length: int | None = None
    embedding_length: int | None = None
    block_count: int | None = None
    expert_count: int | None = None
    expert_used_count: int | None = None
    is_moe: bool | None = None
    active_parameter_count: int | None = None
    active_params_source: str | None = None
    capabilities: list[str] = field(default_factory=list)
    supports_tools: bool | None = None
    supports_thinking: bool | None = None
    supports_vision: bool | None = None
    supports_embedding: bool | None = None
    modified_at: str | None = None
    raw_model_info: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class GenerationMetrics:
    """Timings for a single generation request.

    ``*_measured`` values are observed by the client; ``engine_*`` values are
    reported by the engine. They are kept separate and never mixed.
    """

    # Client-observed
    ttft_s: float | None = None
    total_latency_s: float | None = None
    request_sent_at: float | None = None
    # Engine-reported (Ollama: nanoseconds -> seconds)
    engine_total_duration_s: float | None = None
    engine_load_duration_s: float | None = None
    engine_prompt_eval_duration_s: float | None = None
    engine_eval_duration_s: float | None = None
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    # Derived
    output_tokens_per_s: float | None = None
    prompt_tokens_per_s: float | None = None
    ttft_source: str = "stream_first_content_token"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class GenerationResult:
    """A single completed (or failed) generation."""

    model: str
    engine: str
    text: str = ""
    thinking: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    metrics: GenerationMetrics = field(default_factory=GenerationMetrics)
    options: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    raw_final: dict[str, Any] = field(default_factory=dict)
    was_loaded_before: bool | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ok"] = self.ok
        return d


@dataclass(slots=True)
class TaskScore:
    """Outcome of a validator for one task attempt."""

    score: float | None  # 0.0-1.0, or None when grading is manual
    max_score: float = 1.0
    passed: bool | None = None
    validator: str = "unknown"
    graded: str = "automatic"  # "automatic" | "manual_rubric"
    detail: str = ""
    sub_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
