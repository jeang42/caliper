"""Backend interface.

A backend wraps one inference engine. The surface is intentionally small: the
runner only needs to enumerate models, describe one, generate (streaming), and
report what is currently resident. A vLLM adapter will implement the same
protocol; nothing beyond it is abstracted until a second implementation exists.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ..results import GenerationResult, ModelRecord


@runtime_checkable
class Backend(Protocol):
    name: str

    def version(self) -> str | None:
        """Engine version string, or None when unavailable."""

    def available(self) -> bool:
        """True when the engine responds to a cheap request."""

    def list_models(self) -> list[ModelRecord]:
        """Installed/served models with whatever metadata the engine exposes."""

    def describe_model(self, model: str) -> ModelRecord:
        """Full metadata for one model."""

    def loaded_models(self) -> list[dict[str, Any]]:
        """Models currently resident in memory, as reported by the engine."""

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
        """Run one generation, measuring time to first generated token."""

    def unload(self, model: str) -> bool:
        """Ask the engine to evict a model. Returns True when accepted."""
