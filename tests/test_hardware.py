from __future__ import annotations

import time

from caliper import hardware

GPU_CSV = "0, NVIDIA RTX 5000 Ada Generation, 580.178.04, 32760, 453, 31776, 0, 0, 39, 4.90, 8.9\n"
GPU_CSV_NA = "0, Some GPU, 580.178.04, 32760, 453, 31776, [N/A], 0, 39, [Not Supported], 8.9\n"
PROC_CSV = "2138, python, 430\n13867, /usr/lib/xorg/Xorg, 4\n"
MEMINFO = """MemTotal:       96411272 kB
MemFree:         6700000 kB
MemAvailable:   71307940 kB
SwapTotal:       8388604 kB
SwapFree:         129660 kB
"""
CPUINFO = """processor\t: 0
model name\t: AMD Ryzen 9 9900X 12-Core Processor
physical id\t: 0
core id\t\t: 0

processor\t: 1
model name\t: AMD Ryzen 9 9900X 12-Core Processor
physical id\t: 0
core id\t\t: 0

processor\t: 2
model name\t: AMD Ryzen 9 9900X 12-Core Processor
physical id\t: 0
core id\t\t: 1
"""


def test_parse_gpu_query():
    (gpu,) = hardware.parse_gpu_query(GPU_CSV)
    assert gpu.index == 0
    assert gpu.memory_total_mib == 32760
    assert gpu.memory_free_mib == 31776
    assert gpu.utilization_gpu_pct == 0
    assert gpu.compute_capability == "8.9"
    assert gpu.power_draw_w == 4.90


def test_parse_gpu_query_handles_unsupported_fields():
    (gpu,) = hardware.parse_gpu_query(GPU_CSV_NA)
    assert gpu.utilization_gpu_pct is None
    assert gpu.power_draw_w is None
    assert gpu.memory_total_mib == 32760


def test_parse_gpu_query_skips_malformed_rows():
    assert hardware.parse_gpu_query("0, broken\n") == []


def test_parse_gpu_processes():
    procs = hardware.parse_gpu_processes(PROC_CSV)
    assert [p.pid for p in procs] == [2138, 13867]
    assert procs[0].used_memory_mib == 430


def test_parse_meminfo_converts_to_mib():
    mem = hardware.parse_meminfo(MEMINFO)
    assert round(mem.total_mib) == 94152
    assert round(mem.available_mib) == 69637
    assert round(mem.swap_free_mib) == 127


def test_parse_cpuinfo_counts_cores():
    cpu = hardware.parse_cpuinfo(CPUINFO, "CPU max MHz:   5662.0161\n")
    assert cpu.logical_cores == 3
    assert cpu.physical_cores == 2
    assert cpu.max_mhz == 5662.0161
    assert "Ryzen" in cpu.model


def test_telemetry_summary_from_samples(monkeypatch):
    seq = iter(
        [
            [hardware.GpuSample(0, "g", 32760, 1000, 31760, 10, 5, 40, 50, None, None)],
            [hardware.GpuSample(0, "g", 32760, 21000, 11760, 90, 50, 60, 200, None, None)],
        ]
    )
    monkeypatch.setattr(hardware, "sample_gpus", lambda: next(seq, []))
    sampler = hardware.TelemetrySampler(interval_s=0.05)
    sampler._baseline_gpu_mem = 1000.0
    sampler._started_at = 0.0
    # Inject samples directly so the aggregation is tested deterministically
    # rather than racing a background thread.
    sampler._samples = [
        {
            "t": 0.0,
            "gpu_util_pct": 10,
            "gpu_mem_used_mib": 1000,
            "gpu_mem_total_mib": 32760,
            "gpu_power_w": 50,
            "gpu_temp_c": 40,
            "ram_total_mib": 1000,
            "ram_available_mib": 800,
            "cpu_util_pct": 5,
        },
        {
            "t": 1.0,
            "gpu_util_pct": 90,
            "gpu_mem_used_mib": 21000,
            "gpu_mem_total_mib": 32760,
            "gpu_power_w": 200,
            "gpu_temp_c": 60,
            "ram_total_mib": 1000,
            "ram_available_mib": 400,
            "cpu_util_pct": 55,
        },
    ]
    summary = sampler.summary()
    assert summary.sample_count == 2
    assert summary.gpu_util_peak_pct == 90
    assert summary.gpu_mem_used_peak_mib == 21000
    assert summary.gpu_mem_delta_peak_mib == 20000
    assert summary.ram_used_peak_mib == 600
    assert summary.cpu_util_mean_pct == 30
    assert summary.gpu_util_mean_pct == 50


def test_telemetry_sampler_runs_and_stops(monkeypatch):
    monkeypatch.setattr(
        hardware,
        "sample_gpus",
        lambda: [hardware.GpuSample(0, "g", 32760, 500, 32260, 0, 0, 35, 5, None, None)],
    )
    sampler = hardware.TelemetrySampler(interval_s=0.01)
    sampler.start()
    deadline = time.monotonic() + 5.0
    while not sampler._samples and time.monotonic() < deadline:
        time.sleep(0.01)
    summary = sampler.stop()
    assert summary.sample_count >= 1
    assert summary.baseline_gpu_mem_used_mib == 500


def test_empty_summary_is_safe():
    summary = hardware.TelemetrySampler().summary()
    assert summary.sample_count == 0
    assert summary.gpu_util_mean_pct is None
