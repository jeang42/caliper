"""Inference engine adapters."""

from .base import Backend
from .mock import MockBackend
from .ollama import OllamaBackend

__all__ = ["Backend", "MockBackend", "OllamaBackend", "get_backend", "mock_backend_for_suite"]


def mock_backend_for_suite(suite, models: list[str] | None = None) -> MockBackend:
    """A mock backend wired to the canned fixtures for ``suite``'s tasks."""
    from ..evals.fixtures import mock_fixtures_for_suite

    answers, tool_calls = mock_fixtures_for_suite(suite)
    return MockBackend(models=models, answers=answers, tool_calls=tool_calls)


def get_backend(name: str, config) -> Backend:
    """Construct a backend by name from a :class:`caliper.config.Config`."""
    if name == "ollama":
        cfg = config.engine("ollama")
        return OllamaBackend(
            base_url=cfg["base_url"],
            request_timeout_s=cfg["request_timeout_s"],
            connect_timeout_s=cfg.get("connect_timeout_s", 10.0),
        )
    if name == "mock":
        return MockBackend()
    if name == "vllm":
        raise NotImplementedError(
            "the vLLM adapter is not implemented yet; see docs/ROADMAP.md (Phase 6)"
        )
    raise ValueError(f"unknown backend {name!r}")
