"""Configuration loading.

Config is TOML. Defaults live in ``DEFAULTS``; a user file overrides keys
shallowly per-section. No environment-specific paths are baked in: the data
directory defaults to ``$CALIPER_HOME`` or ``./data`` relative to the cwd.
"""

from __future__ import annotations

import copy
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULTS: dict[str, Any] = {
    "paths": {
        # Resolved relative to cwd unless absolute. Overridable via CALIPER_HOME.
        "data_dir": "data",
        "db_name": "caliper.sqlite",
    },
    "engines": {
        "ollama": {
            "base_url": "http://127.0.0.1:11434",
            "request_timeout_s": 600.0,
            "connect_timeout_s": 10.0,
        },
        "vllm": {
            # Placeholder; the vLLM adapter is not implemented yet.
            "base_url": "http://127.0.0.1:8000",
            "request_timeout_s": 600.0,
        },
    },
    "safety": {
        # Refuse to start heavy work when the machine is already busy.
        "max_gpu_utilization_pct": 25.0,
        "min_free_vram_mib": 20000.0,
        "min_free_ram_mib": 16000.0,
        "min_free_disk_gib": 50.0,
        # Foreign GPU processes are allowed by default (they are often idle
        # services holding a small allocation), but their VRAM counts against
        # min_free_vram_mib and they are always reported.
        "allow_foreign_gpu_processes": True,
        "max_foreign_gpu_mib": 4096.0,
        "gpu_sample_count": 3,
        "gpu_sample_interval_s": 1.0,
    },
    "run": {
        "warmup_runs": 1,
        "repetitions": 3,
        "temperature": 0.0,
        "seed": 42,
        "max_output_tokens": 512,
        "context_length": 8192,
        "keep_alive": "5m",
        "unload_after_run": True,
        "request_timeout_s": 300.0,
        "telemetry_interval_s": 0.25,
        "suite": "v1",
    },
}


@dataclass(slots=True)
class Config:
    raw: dict[str, Any]
    source: Path | None = None
    data_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        home = os.environ.get("CALIPER_HOME")
        raw_dir = self.raw["paths"]["data_dir"]
        base = Path(home) if home else Path(raw_dir)
        self.data_dir = base.expanduser().resolve()

    @property
    def db_path(self) -> Path:
        return self.data_dir / self.raw["paths"]["db_name"]

    def section(self, name: str) -> dict[str, Any]:
        return self.raw[name]

    @property
    def safety(self) -> dict[str, Any]:
        return self.raw["safety"]

    @property
    def run(self) -> dict[str, Any]:
        return self.raw["run"]

    def engine(self, name: str) -> dict[str, Any]:
        try:
            return self.raw["engines"][name]
        except KeyError as exc:  # pragma: no cover - guarded by CLI choices
            raise KeyError(f"no configuration for engine {name!r}") from exc


def _merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None) -> Config:
    """Load configuration, merging ``path`` (if given) over the defaults."""
    if path is None:
        env_path = os.environ.get("CALIPER_CONFIG")
        path = Path(env_path) if env_path else None
    if path is None:
        return Config(raw=copy.deepcopy(DEFAULTS))
    p = Path(path).expanduser()
    with p.open("rb") as fh:
        overlay = tomllib.load(fh)
    return Config(raw=_merge(DEFAULTS, overlay), source=p)
