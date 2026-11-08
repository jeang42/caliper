"""Hardware inspection and telemetry sampling.

Everything here is read-only. GPU data comes from ``nvidia-smi`` (no NVML
binding dependency); CPU and memory come from ``/proc``. When a source is
unavailable the corresponding fields are ``None`` rather than estimated.
"""

from __future__ import annotations

import logging
import platform
import shutil
import socket
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

NVIDIA_SMI = "nvidia-smi"

_GPU_QUERY_FIELDS = (
    "index",
    "name",
    "driver_version",
    "memory.total",
    "memory.used",
    "memory.free",
    "utilization.gpu",
    "utilization.memory",
    "temperature.gpu",
    "power.draw",
    "compute_cap",
)


@dataclass(slots=True)
class GpuSample:
    index: int
    name: str
    memory_total_mib: float | None
    memory_used_mib: float | None
    memory_free_mib: float | None
    utilization_gpu_pct: float | None
    utilization_mem_pct: float | None
    temperature_c: float | None
    power_draw_w: float | None
    driver_version: str | None = None
    compute_capability: str | None = None


@dataclass(slots=True)
class GpuProcess:
    pid: int
    process_name: str
    used_memory_mib: float | None


@dataclass(slots=True)
class CpuInfo:
    model: str | None
    logical_cores: int | None
    physical_cores: int | None
    max_mhz: float | None


@dataclass(slots=True)
class MemoryInfo:
    total_mib: float | None
    available_mib: float | None
    swap_total_mib: float | None
    swap_free_mib: float | None


@dataclass(slots=True)
class DiskInfo:
    path: str
    total_gib: float
    free_gib: float


@dataclass(slots=True)
class MachineInfo:
    hostname: str
    platform: str
    kernel: str
    python_version: str
    cpu: CpuInfo
    memory: MemoryInfo
    disks: list[DiskInfo]
    gpus: list[GpuSample]
    gpu_processes: list[GpuProcess]
    nvidia_smi_available: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _run(cmd: list[str], timeout: float = 15.0) -> str | None:
    if shutil.which(cmd[0]) is None:
        return None
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.warning("command %s failed: %s", cmd[0], exc)
        return None
    if proc.returncode != 0:
        log.warning("command %s exited %d: %s", cmd[0], proc.returncode, proc.stderr.strip())
        return None
    return proc.stdout


def _to_float(value: str) -> float | None:
    value = value.strip()
    if not value or value.lower().startswith(("n/a", "[n/a", "[not supported")):
        return None
    try:
        return float(value.split()[0])
    except ValueError:
        return None


def parse_gpu_query(stdout: str) -> list[GpuSample]:
    """Parse ``nvidia-smi --query-gpu`` CSV output (no header, no units)."""
    samples: list[GpuSample] = []
    for line in stdout.strip().splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(_GPU_QUERY_FIELDS):
            log.warning("unexpected nvidia-smi row: %r", line)
            continue
        idx = _to_float(parts[0])
        samples.append(
            GpuSample(
                index=int(idx) if idx is not None else -1,
                name=parts[1],
                driver_version=parts[2] or None,
                memory_total_mib=_to_float(parts[3]),
                memory_used_mib=_to_float(parts[4]),
                memory_free_mib=_to_float(parts[5]),
                utilization_gpu_pct=_to_float(parts[6]),
                utilization_mem_pct=_to_float(parts[7]),
                temperature_c=_to_float(parts[8]),
                power_draw_w=_to_float(parts[9]),
                compute_capability=parts[10] or None,
            )
        )
    return samples


def parse_gpu_processes(stdout: str) -> list[GpuProcess]:
    """Parse ``nvidia-smi --query-compute-apps`` CSV output."""
    procs: list[GpuProcess] = []
    for line in stdout.strip().splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        pid = _to_float(parts[0])
        if pid is None:
            continue
        procs.append(
            GpuProcess(
                pid=int(pid),
                process_name=parts[1],
                used_memory_mib=_to_float(parts[2]),
            )
        )
    return procs


def sample_gpus() -> list[GpuSample]:
    out = _run(
        [
            NVIDIA_SMI,
            f"--query-gpu={','.join(_GPU_QUERY_FIELDS)}",
            "--format=csv,noheader,nounits",
        ]
    )
    return parse_gpu_query(out) if out else []


def gpu_processes() -> list[GpuProcess]:
    out = _run(
        [
            NVIDIA_SMI,
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    return parse_gpu_processes(out) if out else []


def parse_meminfo(text: str) -> MemoryInfo:
    values: dict[str, float] = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        kb = _to_float(rest)
        if kb is not None:
            values[key] = kb / 1024.0
    return MemoryInfo(
        total_mib=values.get("MemTotal"),
        available_mib=values.get("MemAvailable"),
        swap_total_mib=values.get("SwapTotal"),
        swap_free_mib=values.get("SwapFree"),
    )


def memory_info() -> MemoryInfo:
    try:
        return parse_meminfo(Path("/proc/meminfo").read_text())
    except OSError:
        return MemoryInfo(None, None, None, None)


def parse_cpuinfo(text: str, lscpu: str | None = None) -> CpuInfo:
    model: str | None = None
    logical = 0
    physical_ids: set[tuple[str, str]] = set()
    cur_phys = ""
    for line in text.splitlines():
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if key == "processor":
            logical += 1
        elif key == "model name" and model is None:
            model = value
        elif key == "physical id":
            cur_phys = value
        elif key == "core id":
            physical_ids.add((cur_phys, value))
    max_mhz: float | None = None
    if lscpu:
        for line in lscpu.splitlines():
            if "CPU max MHz" in line:
                max_mhz = _to_float(line.split(":", 1)[1])
    return CpuInfo(
        model=model,
        logical_cores=logical or None,
        physical_cores=len(physical_ids) or None,
        max_mhz=max_mhz,
    )


def cpu_info() -> CpuInfo:
    try:
        text = Path("/proc/cpuinfo").read_text()
    except OSError:
        return CpuInfo(None, None, None, None)
    return parse_cpuinfo(text, _run(["lscpu"]))


def disk_info(paths: list[str] | None = None) -> list[DiskInfo]:
    out: list[DiskInfo] = []
    for p in paths or [str(Path.cwd())]:
        try:
            usage = shutil.disk_usage(p)
        except OSError:
            continue
        out.append(
            DiskInfo(
                path=p,
                total_gib=usage.total / 1024**3,
                free_gib=usage.free / 1024**3,
            )
        )
    return out


def collect_machine_info(disk_paths: list[str] | None = None) -> MachineInfo:
    gpus = sample_gpus()
    return MachineInfo(
        hostname=socket.gethostname(),
        platform=platform.platform(),
        kernel=platform.release(),
        python_version=platform.python_version(),
        cpu=cpu_info(),
        memory=memory_info(),
        disks=disk_info(disk_paths),
        gpus=gpus,
        gpu_processes=gpu_processes(),
        nvidia_smi_available=shutil.which(NVIDIA_SMI) is not None,
    )


def read_cpu_jiffies() -> tuple[float, float] | None:
    """Return (busy, total) jiffies from /proc/stat, or None."""
    try:
        first = Path("/proc/stat").read_text().splitlines()[0]
    except (OSError, IndexError):
        return None
    fields = [float(x) for x in first.split()[1:]]
    if len(fields) < 5:
        return None
    idle = fields[3] + fields[4]
    total = sum(fields)
    return total - idle, total


# --------------------------------------------------------------------------
# Telemetry sampler
# --------------------------------------------------------------------------


@dataclass(slots=True)
class TelemetrySummary:
    """Aggregated telemetry over a measured window. All values are measured."""

    sample_count: int = 0
    duration_s: float = 0.0
    gpu_util_mean_pct: float | None = None
    gpu_util_peak_pct: float | None = None
    gpu_mem_used_peak_mib: float | None = None
    gpu_mem_used_mean_mib: float | None = None
    gpu_mem_total_mib: float | None = None
    gpu_power_mean_w: float | None = None
    gpu_temp_peak_c: float | None = None
    ram_used_peak_mib: float | None = None
    ram_available_min_mib: float | None = None
    cpu_util_mean_pct: float | None = None
    baseline_gpu_mem_used_mib: float | None = None
    gpu_mem_delta_peak_mib: float | None = None
    samples: list[dict] = field(default_factory=list)

    def to_dict(self, include_samples: bool = False) -> dict:
        d = asdict(self)
        if not include_samples:
            d.pop("samples", None)
        return d


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


class TelemetrySampler:
    """Background sampler for GPU/CPU/RAM metrics.

    Sampling runs in a daemon thread so an interrupted benchmark cannot leave
    the process hanging. ``nvidia-smi`` costs ~20-40 ms per call, so intervals
    below ~0.2 s are not useful.
    """

    def __init__(self, interval_s: float = 0.25, gpu_index: int = 0) -> None:
        self.interval_s = max(interval_s, 0.05)
        self.gpu_index = gpu_index
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._samples: list[dict] = []
        self._started_at: float | None = None
        self._stopped_at: float | None = None
        self._baseline_gpu_mem: float | None = None

    def __enter__(self) -> TelemetrySampler:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("sampler already started")
        gpus = sample_gpus()
        gpu = next((g for g in gpus if g.index == self.gpu_index), None)
        self._baseline_gpu_mem = gpu.memory_used_mib if gpu else None
        self._started_at = time.monotonic()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="caliper-telemetry", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        prev = read_cpu_jiffies()
        while not self._stop.is_set():
            ts = time.monotonic()
            gpus = sample_gpus()
            gpu = next((g for g in gpus if g.index == self.gpu_index), None)
            mem = memory_info()
            cur = read_cpu_jiffies()
            cpu_pct: float | None = None
            if prev and cur and cur[1] > prev[1]:
                cpu_pct = 100.0 * (cur[0] - prev[0]) / (cur[1] - prev[1])
            prev = cur or prev
            self._samples.append(
                {
                    "t": ts,
                    "gpu_util_pct": gpu.utilization_gpu_pct if gpu else None,
                    "gpu_mem_used_mib": gpu.memory_used_mib if gpu else None,
                    "gpu_mem_total_mib": gpu.memory_total_mib if gpu else None,
                    "gpu_power_w": gpu.power_draw_w if gpu else None,
                    "gpu_temp_c": gpu.temperature_c if gpu else None,
                    "ram_total_mib": mem.total_mib,
                    "ram_available_mib": mem.available_mib,
                    "cpu_util_pct": cpu_pct,
                }
            )
            self._stop.wait(self.interval_s)

    def stop(self) -> TelemetrySummary:
        if self._thread is None:
            return TelemetrySummary()
        self._stop.set()
        self._thread.join(timeout=5.0)
        self._thread = None
        self._stopped_at = time.monotonic()
        if not self._samples:
            # A request that finished faster than one sampling interval still
            # deserves one data point, taken synchronously here.
            self._sample_once()
        return self.summary()

    def _sample_once(self) -> None:
        gpus = sample_gpus()
        gpu = next((g for g in gpus if g.index == self.gpu_index), None)
        mem = memory_info()
        self._samples.append(
            {
                "t": time.monotonic(),
                "gpu_util_pct": gpu.utilization_gpu_pct if gpu else None,
                "gpu_mem_used_mib": gpu.memory_used_mib if gpu else None,
                "gpu_mem_total_mib": gpu.memory_total_mib if gpu else None,
                "gpu_power_w": gpu.power_draw_w if gpu else None,
                "gpu_temp_c": gpu.temperature_c if gpu else None,
                "ram_total_mib": mem.total_mib,
                "ram_available_mib": mem.available_mib,
                "cpu_util_pct": None,
            }
        )

    def summary(self) -> TelemetrySummary:
        samples = list(self._samples)
        if not samples:
            return TelemetrySummary(baseline_gpu_mem_used_mib=self._baseline_gpu_mem)

        def col(key: str) -> list[float]:
            return [s[key] for s in samples if s.get(key) is not None]

        gpu_mem = col("gpu_mem_used_mib")
        ram_total = col("ram_total_mib")
        ram_avail = col("ram_available_mib")
        ram_used = [t - a for t, a in zip(ram_total, ram_avail, strict=False)]
        peak_mem = max(gpu_mem) if gpu_mem else None
        delta = (
            peak_mem - self._baseline_gpu_mem
            if peak_mem is not None and self._baseline_gpu_mem is not None
            else None
        )
        start = self._started_at or samples[0]["t"]
        end = self._stopped_at or samples[-1]["t"]
        util = col("gpu_util_pct")
        return TelemetrySummary(
            sample_count=len(samples),
            duration_s=end - start,
            gpu_util_mean_pct=_mean(util),
            gpu_util_peak_pct=max(util) if util else None,
            gpu_mem_used_peak_mib=peak_mem,
            gpu_mem_used_mean_mib=_mean(gpu_mem),
            gpu_mem_total_mib=max(col("gpu_mem_total_mib")) if col("gpu_mem_total_mib") else None,
            gpu_power_mean_w=_mean(col("gpu_power_w")),
            gpu_temp_peak_c=max(col("gpu_temp_c")) if col("gpu_temp_c") else None,
            ram_used_peak_mib=max(ram_used) if ram_used else None,
            ram_available_min_mib=min(ram_avail) if ram_avail else None,
            cpu_util_mean_pct=_mean(col("cpu_util_pct")),
            baseline_gpu_mem_used_mib=self._baseline_gpu_mem,
            gpu_mem_delta_peak_mib=delta,
            samples=samples,
        )
