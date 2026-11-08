"""Preflight resource checks.

The framework refuses to start sustained inference when the machine is already
busy. Checks are advisory data structures; the caller decides whether to abort,
and ``--force`` is the only way past a failed check.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

from . import hardware

log = logging.getLogger(__name__)


@dataclass(slots=True)
class Check:
    name: str
    passed: bool
    detail: str
    measured: float | None = None
    threshold: float | None = None
    severity: str = "block"  # "block" | "warn"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class PreflightResult:
    checks: list[Check] = field(default_factory=list)
    machine: hardware.MachineInfo | None = None

    @property
    def blocking_failures(self) -> list[Check]:
        return [c for c in self.checks if not c.passed and c.severity == "block"]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if not c.passed and c.severity == "warn"]

    @property
    def notices(self) -> list[Check]:
        """Non-blocking checks worth printing even when they passed.

        Pre-existing GPU processes are the main case: they are tolerated but
        must always be visible, because a run shares the device with them.
        """
        return [c for c in self.checks if c.severity == "warn"]

    @property
    def ok(self) -> bool:
        return not self.blocking_failures

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "checks": [c.to_dict() for c in self.checks],
            "machine": self.machine.to_dict() if self.machine else None,
        }


def _own_pids() -> set[int]:
    return {os.getpid()}


def wait_until_clear(
    safety: dict,
    timeout_s: float,
    gpu_index: int = 0,
    disk_paths: list[str] | None = None,
    poll_interval_s: float = 10.0,
    on_wait: Callable[[PreflightResult, float], None] | None = None,
) -> PreflightResult:
    """Poll preflight until it passes or ``timeout_s`` elapses.

    Queued runs need this: a benchmark that has just finished leaves the GPU
    briefly busy, so a preflight taken immediately afterwards can block the
    next stage even though the machine is about to be free. Waiting is the
    correct response to a busy machine - lowering the threshold is not.

    Returns the last result either way; the caller decides what to do with a
    still-failing one.
    """
    deadline = time.monotonic() + max(timeout_s, 0.0)
    while True:
        result = preflight(safety, gpu_index=gpu_index, disk_paths=disk_paths)
        if result.ok:
            return result
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return result
        if on_wait is not None:
            on_wait(result, remaining)
        time.sleep(min(poll_interval_s, remaining))


def preflight(
    safety: dict,
    gpu_index: int = 0,
    disk_paths: list[str] | None = None,
    required_disk_gib: float | None = None,
) -> PreflightResult:
    """Run resource checks against the configured thresholds."""
    machine = hardware.collect_machine_info(disk_paths)
    checks: list[Check] = []

    if not machine.nvidia_smi_available:
        checks.append(
            Check(
                "nvidia_smi",
                False,
                "nvidia-smi not found; GPU telemetry will be unavailable",
                severity="warn",
            )
        )

    # --- GPU utilization: average several samples so a transient spike from
    # another process does not block a run, and a busy GPU is not missed.
    n = int(safety.get("gpu_sample_count", 3))
    interval = float(safety.get("gpu_sample_interval_s", 1.0))
    utils: list[float] = []
    free_vram: list[float] = []
    for i in range(max(n, 1)):
        gpus = hardware.sample_gpus()
        gpu = next((g for g in gpus if g.index == gpu_index), None)
        if gpu is None:
            break
        if gpu.utilization_gpu_pct is not None:
            utils.append(gpu.utilization_gpu_pct)
        if gpu.memory_free_mib is not None:
            free_vram.append(gpu.memory_free_mib)
        if i < n - 1:
            time.sleep(interval)

    if utils:
        mean_util = sum(utils) / len(utils)
        limit = float(safety["max_gpu_utilization_pct"])
        checks.append(
            Check(
                "gpu_utilization",
                mean_util <= limit,
                f"mean GPU utilization {mean_util:.1f}% over {len(utils)} samples "
                f"(limit {limit:.1f}%)",
                measured=mean_util,
                threshold=limit,
            )
        )
    else:
        checks.append(
            Check("gpu_utilization", False, "could not read GPU utilization", severity="warn")
        )

    if free_vram:
        min_free = min(free_vram)
        limit = float(safety["min_free_vram_mib"])
        checks.append(
            Check(
                "free_vram",
                min_free >= limit,
                f"minimum free VRAM {min_free:.0f} MiB (need {limit:.0f} MiB)",
                measured=min_free,
                threshold=limit,
            )
        )
    else:
        checks.append(Check("free_vram", False, "could not read free VRAM", severity="warn"))

    # --- Foreign GPU processes: never disrupt them, only account for them.
    foreign = [p for p in machine.gpu_processes if p.pid not in _own_pids()]
    foreign_mib = sum(p.used_memory_mib or 0.0 for p in foreign)
    if foreign:
        names = ", ".join(
            f"{p.process_name}(pid {p.pid}, {p.used_memory_mib:.0f} MiB)" for p in foreign
        )
        allowed = bool(safety.get("allow_foreign_gpu_processes", True))
        limit = float(safety.get("max_foreign_gpu_mib", 4096.0))
        passed = allowed and foreign_mib <= limit
        checks.append(
            Check(
                "foreign_gpu_processes",
                passed,
                f"{len(foreign)} pre-existing GPU process(es) using {foreign_mib:.0f} MiB: {names}",
                measured=foreign_mib,
                threshold=limit,
                severity="block" if not allowed else ("block" if foreign_mib > limit else "warn"),
            )
        )

    # --- System RAM
    avail = machine.memory.available_mib
    if avail is not None:
        limit = float(safety["min_free_ram_mib"])
        checks.append(
            Check(
                "free_ram",
                avail >= limit,
                f"available RAM {avail:.0f} MiB (need {limit:.0f} MiB)",
                measured=avail,
                threshold=limit,
            )
        )

    # --- Disk
    limit = float(required_disk_gib or safety["min_free_disk_gib"])
    for disk in machine.disks:
        checks.append(
            Check(
                f"free_disk:{disk.path}",
                disk.free_gib >= limit,
                f"{disk.path} free {disk.free_gib:.1f} GiB (need {limit:.1f} GiB)",
                measured=disk.free_gib,
                threshold=limit,
            )
        )

    return PreflightResult(checks=checks, machine=machine)
