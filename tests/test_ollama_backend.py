"""Unit tests for the Ollama adapter using recorded stream fixtures."""

from __future__ import annotations

import pytest

from caliper import http
from caliper.backends.ollama import OllamaBackend

SHOW_RESPONSE = {
    "details": {"family": "qwen3moe", "parameter_size": "30.5B", "quantization_level": "Q4_K_M"},
    "capabilities": ["completion", "tools"],
    "model_info": {
        "general.architecture": "qwen3moe",
        "general.parameter_count": 30532122624,
        "qwen3moe.context_length": 262144,
        "qwen3moe.embedding_length": 2048,
        "qwen3moe.block_count": 48,
        "qwen3moe.expert_count": 128,
        "qwen3moe.expert_used_count": 8,
    },
}

DENSE_SHOW_RESPONSE = {
    "details": {"family": "qwen3", "parameter_size": "32.8B", "quantization_level": "Q4_K_M"},
    "capabilities": ["completion", "tools", "thinking"],
    "model_info": {
        "general.architecture": "qwen3",
        "qwen3.context_length": 40960,
        "qwen3.block_count": 64,
    },
}

STREAM = [
    {"message": {"role": "assistant", "content": ""}, "done": False},
    {"message": {"role": "assistant", "content": "Hello"}, "done": False},
    {"message": {"role": "assistant", "content": " world"}, "done": False},
    {
        "message": {"role": "assistant", "content": ""},
        "done": True,
        "done_reason": "stop",
        "total_duration": 3_000_000_000,
        "load_duration": 1_000_000_000,
        "prompt_eval_count": 900,
        "prompt_eval_duration": 1_000_000_000,
        "eval_count": 120,
        "eval_duration": 1_000_000_000,
    },
]


@pytest.fixture()
def backend() -> OllamaBackend:
    return OllamaBackend(base_url="http://127.0.0.1:11434")


def test_describe_model_flags_moe_and_keeps_active_params_unknown(backend, monkeypatch):
    monkeypatch.setattr(http, "post_json", lambda *a, **k: SHOW_RESPONSE)
    record = backend.describe_model("qwen3-coder:30b")
    assert record.is_moe is True
    assert record.expert_count == 128
    assert record.expert_used_count == 8
    assert record.context_length == 262144
    assert record.parameter_count == 30532122624
    assert record.active_parameter_count is None
    assert record.supports_tools is True
    assert record.supports_vision is False
    assert any("active parameter count" in n for n in record.notes)


def test_describe_model_marks_dense_models_not_moe(backend, monkeypatch):
    monkeypatch.setattr(http, "post_json", lambda *a, **k: DENSE_SHOW_RESPONSE)
    record = backend.describe_model("qwen3:32b")
    assert record.is_moe is False
    assert record.expert_count is None
    assert record.supports_thinking is True


def test_generate_collects_engine_timings(backend, monkeypatch):
    monkeypatch.setattr(http, "get_json", lambda *a, **k: {"models": []})
    monkeypatch.setattr(http, "post_ndjson_stream", lambda *a, **k: iter(STREAM))
    result = backend.generate("m", "hi", options={"temperature": 0.0})
    assert result.ok
    assert result.text == "Hello world"
    m = result.metrics
    assert m.prompt_tokens == 900
    assert m.output_tokens == 120
    assert m.engine_load_duration_s == 1.0
    assert m.engine_prompt_eval_duration_s == 1.0
    assert m.engine_eval_duration_s == 1.0
    assert m.output_tokens_per_s == pytest.approx(120.0)
    assert m.prompt_tokens_per_s == pytest.approx(900.0)
    assert result.was_loaded_before is False


def test_ttft_ignores_empty_leading_chunks(backend, monkeypatch):
    monkeypatch.setattr(http, "get_json", lambda *a, **k: {"models": []})
    monkeypatch.setattr(http, "post_ndjson_stream", lambda *a, **k: iter(STREAM))
    result = backend.generate("m", "hi")
    assert result.metrics.ttft_s is not None
    assert result.metrics.ttft_source == "stream_first_content_token"
    # TTFT must be at or before the end of the request.
    assert result.metrics.ttft_s <= result.metrics.total_latency_s


def test_ttft_from_thinking_token_is_labelled(backend, monkeypatch):
    stream = [
        {"message": {"thinking": "let me see"}, "done": False},
        {"message": {"content": "42"}, "done": False},
        {"message": {"content": ""}, "done": True, "eval_count": 2, "eval_duration": 1_000_000_000},
    ]
    monkeypatch.setattr(http, "get_json", lambda *a, **k: {"models": []})
    monkeypatch.setattr(http, "post_ndjson_stream", lambda *a, **k: iter(stream))
    result = backend.generate("m", "hi", think=True)
    assert result.metrics.ttft_source == "stream_first_thinking_token"
    assert result.thinking == "let me see"
    assert result.text == "42"


def test_tool_calls_are_captured(backend, monkeypatch):
    stream = [
        {
            "message": {
                "tool_calls": [{"function": {"name": "f", "arguments": {"a": 1}}}],
            },
            "done": False,
        },
        {"message": {"content": ""}, "done": True},
    ]
    monkeypatch.setattr(http, "get_json", lambda *a, **k: {"models": []})
    monkeypatch.setattr(http, "post_ndjson_stream", lambda *a, **k: iter(stream))
    result = backend.generate("m", "hi", tools=[{"type": "function"}])
    assert result.tool_calls[0]["function"]["name"] == "f"
    assert result.metrics.ttft_source == "stream_first_tool_call"


def test_engine_error_in_stream_becomes_a_failed_result(backend, monkeypatch):
    monkeypatch.setattr(http, "get_json", lambda *a, **k: {"models": []})
    monkeypatch.setattr(
        http, "post_ndjson_stream", lambda *a, **k: iter([{"error": "model not found"}])
    )
    result = backend.generate("missing:1b", "hi")
    assert not result.ok
    assert "model not found" in result.error


def test_http_failure_becomes_a_failed_result(backend, monkeypatch):
    def boom(*a, **k):
        raise http.HttpError("could not reach host")

    monkeypatch.setattr(http, "get_json", lambda *a, **k: {"models": []})
    monkeypatch.setattr(http, "post_ndjson_stream", boom)
    result = backend.generate("m", "hi")
    assert not result.ok
    assert "could not reach host" in result.error


def test_residency_reports_gpu_split(backend, monkeypatch):
    monkeypatch.setattr(
        http,
        "get_json",
        lambda *a, **k: {
            "models": [{"name": "m", "size": 20_000_000_000, "size_vram": 10_000_000_000}]
        },
    )
    residency = backend.residency("m")
    assert residency["gpu_fraction"] == pytest.approx(0.5)
    assert residency["fully_on_gpu"] is False


def test_residency_none_when_not_loaded(backend, monkeypatch):
    monkeypatch.setattr(http, "get_json", lambda *a, **k: {"models": []})
    assert backend.residency("m") is None
