"""Environment and model inventory.

Produces a machine-readable snapshot (JSON, also stored in SQLite) plus a
human-readable Markdown report. Nothing here mutates the engine or the host.

Metadata policy: every field comes from the engine's own API or from /proc and
nvidia-smi. Values that cannot be read are recorded as ``null`` and listed as
unknown in the report. The only derived fields are ``is_moe`` (from the
presence of an ``expert_count`` key) and ``estimated_*`` fields, which are
always labelled as estimates.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

from . import __version__
from .backends.base import Backend
from .hardware import MachineInfo, collect_machine_info
from .results import ModelRecord

log = logging.getLogger(__name__)

# Rule-of-thumb overhead for KV cache and runtime buffers on top of weights.
# Labelled an estimate everywhere it is used; never stored as a measurement.
RUNTIME_OVERHEAD_FRACTION = 0.15
MIB = 1024**2
GIB = 1024**3


@dataclass(slots=True)
class EngineInfo:
    name: str
    available: bool
    version: str | None = None
    base_url: str | None = None
    loaded_models: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


@dataclass(slots=True)
class Inventory:
    caliper_version: str
    machine: MachineInfo
    engines: list[EngineInfo]
    models: list[ModelRecord]
    unknown_fields: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "caliper_version": self.caliper_version,
            "machine": self.machine.to_dict(),
            "engines": [asdict(e) for e in self.engines],
            "models": [m.to_dict() for m in self.models],
            "unknown_fields": self.unknown_fields,
        }


_TRACKED_UNKNOWNS = (
    "digest",
    "architecture",
    "context_length",
    "parameter_count",
    "quantization",
)


def estimate_runtime_vram_gib(
    size_bytes: int | None, context_length: int | None = None
) -> float | None:
    """Rough resident-size estimate in GiB. ESTIMATE, not a measurement."""
    if not size_bytes:
        return None
    return (size_bytes / GIB) * (1.0 + RUNTIME_OVERHEAD_FRACTION)


def gpu_total_gib(machine: MachineInfo, gpu_index: int = 0) -> float | None:
    gpu = next((g for g in machine.gpus if g.index == gpu_index), None)
    if gpu is None or gpu.memory_total_mib is None:
        return None
    return gpu.memory_total_mib / 1024.0


def collect_inventory(
    backends: list[Backend],
    detail: bool = True,
    disk_paths: list[str] | None = None,
) -> Inventory:
    """Collect machine info plus per-engine model metadata.

    ``detail=False`` skips the per-model ``/api/show`` calls, which is useful
    on hosts with many tags.
    """
    machine = collect_machine_info(disk_paths)
    engines: list[EngineInfo] = []
    models: list[ModelRecord] = []

    for backend in backends:
        info = EngineInfo(name=backend.name, available=False)
        info.base_url = getattr(backend, "base_url", None)
        try:
            info.version = backend.version()
            info.available = info.version is not None
            if info.available:
                info.loaded_models = backend.loaded_models()
        except Exception as exc:
            info.error = repr(exc)
        engines.append(info)
        if not info.available:
            continue
        try:
            listed = backend.list_models()
        except Exception as exc:
            info.error = f"list_models failed: {exc!r}"
            continue
        for record in listed:
            if detail:
                try:
                    detailed = backend.describe_model(record.name)
                    detailed.size_bytes = detailed.size_bytes or record.size_bytes
                    detailed.digest = detailed.digest or record.digest
                    detailed.modified_at = record.modified_at
                    record = detailed
                except Exception as exc:
                    record.notes.append(f"describe_model failed: {exc!r}")
            models.append(record)

    unknown: dict[str, list[str]] = {}
    for record in models:
        missing = [f for f in _TRACKED_UNKNOWNS if getattr(record, f, None) in (None, "")]
        # A dense model legitimately has no expert fields, so their absence is
        # only "unknown" when the MoE classification itself could not be made.
        if record.is_moe is None:
            missing.append("is_moe")
        elif record.is_moe and record.expert_used_count is None:
            missing.append("expert_used_count")
        if missing:
            unknown[record.name] = missing

    return Inventory(
        caliper_version=__version__,
        machine=machine,
        engines=engines,
        models=sorted(models, key=lambda m: m.name),
        unknown_fields=unknown,
    )


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def _fmt(value: Any, suffix: str = "", nd: int = 1) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, float):
        return f"{value:.{nd}f}{suffix}"
    return f"{value}{suffix}"


def classify_residency(
    model: ModelRecord, machine: MachineInfo, gpu_index: int = 0
) -> tuple[str, str]:
    """Predict GPU residency from size alone. Returns (label, basis).

    This is an ESTIMATE used for planning only. Actual residency is recorded
    per run from the engine's own report (Ollama ``/api/ps`` size_vram).
    """
    total = gpu_total_gib(machine, gpu_index)
    est = estimate_runtime_vram_gib(model.size_bytes)
    if total is None or est is None:
        return "unknown", "missing GPU total or model size"
    basis = f"estimated {est:.1f} GiB resident vs {total:.1f} GiB VRAM"
    if est <= total * 0.80:
        return "likely fully on GPU", basis
    if est <= total * 0.97:
        return "tight fit; context length will decide", basis
    return "likely needs CPU/RAM offload", basis


def render_report(inventory: Inventory, gpu_index: int = 0) -> str:
    m = inventory.machine
    lines: list[str] = []
    add = lines.append
    add("# Caliper environment inventory")
    add("")
    add(f"- Host: `{m.hostname}`")
    add(f"- Platform: {m.platform}")
    add(f"- Kernel: {m.kernel}")
    add(f"- Python (collector): {m.python_version}")
    add(f"- Caliper: {inventory.caliper_version}")
    add("")
    add("## CPU and memory")
    add("")
    add(
        f"- CPU: {_fmt(m.cpu.model)} ({_fmt(m.cpu.logical_cores)} logical / "
        f"{_fmt(m.cpu.physical_cores)} physical cores, max {_fmt(m.cpu.max_mhz, ' MHz', 0)})"
    )
    if m.memory.total_mib:
        add(
            f"- RAM: {m.memory.total_mib / 1024:.1f} GiB total, "
            f"{(m.memory.available_mib or 0) / 1024:.1f} GiB available"
        )
    if m.memory.swap_total_mib:
        add(
            f"- Swap: {m.memory.swap_total_mib / 1024:.1f} GiB total, "
            f"{(m.memory.swap_free_mib or 0) / 1024:.1f} GiB free"
        )
    for disk in m.disks:
        add(f"- Disk `{disk.path}`: {disk.free_gib:.1f} GiB free of {disk.total_gib:.1f} GiB")
    add("")
    add("## GPU")
    add("")
    if not m.gpus:
        add("No GPU reported (nvidia-smi unavailable or no device).")
    for gpu in m.gpus:
        add(f"- GPU {gpu.index}: {gpu.name}")
        add(
            f"  - Driver {_fmt(gpu.driver_version)}, compute capability "
            f"{_fmt(gpu.compute_capability)}"
        )
        add(
            f"  - VRAM: {_fmt(gpu.memory_total_mib, ' MiB', 0)} total, "
            f"{_fmt(gpu.memory_free_mib, ' MiB', 0)} free"
        )
        add(
            f"  - At snapshot: {_fmt(gpu.utilization_gpu_pct, '%', 0)} utilization, "
            f"{_fmt(gpu.temperature_c, ' C', 0)}, {_fmt(gpu.power_draw_w, ' W')}"
        )
    if m.gpu_processes:
        add("")
        add("Pre-existing GPU processes (never disturbed by Caliper):")
        for proc in m.gpu_processes:
            add(f"- pid {proc.pid} `{proc.process_name}` {_fmt(proc.used_memory_mib, ' MiB', 0)}")
    add("")
    add("## Engines")
    add("")
    for engine in inventory.engines:
        status = f"version {engine.version}" if engine.available else "unavailable"
        add(f"- `{engine.name}` ({engine.base_url}): {status}")
        if engine.error:
            add(f"  - error: {engine.error}")
        if engine.loaded_models:
            for loaded in engine.loaded_models:
                add(f"  - currently loaded: {loaded.get('name')}")
        elif engine.available:
            add("  - currently loaded: none")
    add("")
    add(f"## Models ({len(inventory.models)})")
    add("")
    add("| Model | Size | Quant | Arch | Params | Context | MoE | Experts | Capabilities |")
    add("|---|---|---|---|---|---|---|---|---|")
    for mod in inventory.models:
        size = f"{mod.size_bytes / GIB:.2f} GiB" if mod.size_bytes else "unknown"
        moe = "unknown" if mod.is_moe is None else ("yes" if mod.is_moe else "no")
        experts = (
            f"{mod.expert_used_count}/{mod.expert_count}" if mod.expert_count is not None else "-"
        )
        ctx = f"{mod.context_length:,}" if mod.context_length else "unknown"
        add(
            f"| `{mod.name}` | {size} | {_fmt(mod.quantization)} | {_fmt(mod.architecture)} | "
            f"{_fmt(mod.parameter_size)} | {ctx} | {moe} | {experts} | "
            f"{', '.join(mod.capabilities) or '-'} |"
        )
    add("")
    add("## Estimated GPU residency (planning only, not measured)")
    add("")
    add("| Model | Estimated resident | Prediction |")
    add("|---|---|---|")
    for mod in inventory.models:
        label, basis = classify_residency(mod, m, gpu_index)
        est = estimate_runtime_vram_gib(mod.size_bytes)
        add(f"| `{mod.name}` | {_fmt(est, ' GiB', 1)} | {label} ({basis}) |")
    add("")
    add(
        f"Estimates add {RUNTIME_OVERHEAD_FRACTION:.0%} to on-disk size for KV cache and "
        "runtime buffers. Real residency is recorded per run from the engine's own report."
    )
    add("")
    if inventory.unknown_fields:
        add("## Unknown metadata")
        add("")
        add("Fields the engine did not report. These are left unknown, never guessed.")
        add("")
        for name, fields in sorted(inventory.unknown_fields.items()):
            add(f"- `{name}`: {', '.join(fields)}")
        add("")
    moe_models = [mod for mod in inventory.models if mod.is_moe]
    add(f"## MoE models detected ({len(moe_models)})")
    add("")
    add(
        "Detected from an `expert_count` key in the engine's model metadata. "
        "Active parameter counts are not reported by the engine and are left unknown."
    )
    add("")
    for mod in moe_models:
        add(
            f"- `{mod.name}`: {mod.expert_used_count}/{mod.expert_count} experts per token, "
            f"architecture `{mod.architecture}`, {_fmt(mod.parameter_size)} total"
        )
    add("")
    return "\n".join(lines)
