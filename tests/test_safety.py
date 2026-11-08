from __future__ import annotations

from caliper import hardware, safety


def _gpu(util: float, free: float) -> list[hardware.GpuSample]:
    return [hardware.GpuSample(0, "g", 32760, 32760 - free, free, util, 0, 40, 20, None, None)]


def _machine(free_ram: float = 60000.0, free_disk: float = 600.0, procs=None):
    return hardware.MachineInfo(
        hostname="host",
        platform="linux",
        kernel="7.0",
        python_version="3.13",
        cpu=hardware.CpuInfo("cpu", 24, 12, 5000.0),
        memory=hardware.MemoryInfo(94000.0, free_ram, 8192.0, 128.0),
        disks=[hardware.DiskInfo("/", 3700.0, free_disk)],
        gpus=_gpu(0, 31000),
        gpu_processes=procs or [],
        nvidia_smi_available=True,
    )


THRESHOLDS = {
    "max_gpu_utilization_pct": 25.0,
    "min_free_vram_mib": 20000.0,
    "min_free_ram_mib": 16000.0,
    "min_free_disk_gib": 50.0,
    "allow_foreign_gpu_processes": True,
    "max_foreign_gpu_mib": 4096.0,
    "gpu_sample_count": 2,
    "gpu_sample_interval_s": 0.0,
}


def test_idle_machine_passes(monkeypatch):
    monkeypatch.setattr(safety.hardware, "collect_machine_info", lambda paths=None: _machine())
    monkeypatch.setattr(safety.hardware, "sample_gpus", lambda: _gpu(2, 31000))
    result = safety.preflight(THRESHOLDS)
    assert result.ok
    assert all(c.passed for c in result.checks)


def test_busy_gpu_blocks(monkeypatch):
    monkeypatch.setattr(safety.hardware, "collect_machine_info", lambda paths=None: _machine())
    monkeypatch.setattr(safety.hardware, "sample_gpus", lambda: _gpu(95, 31000))
    result = safety.preflight(THRESHOLDS)
    assert not result.ok
    assert any(c.name == "gpu_utilization" for c in result.blocking_failures)


def test_insufficient_vram_blocks(monkeypatch):
    monkeypatch.setattr(safety.hardware, "collect_machine_info", lambda paths=None: _machine())
    monkeypatch.setattr(safety.hardware, "sample_gpus", lambda: _gpu(0, 8000))
    result = safety.preflight(THRESHOLDS)
    assert not result.ok
    assert any(c.name == "free_vram" for c in result.blocking_failures)


def test_low_ram_and_disk_block(monkeypatch):
    machine = _machine(free_ram=2000.0, free_disk=10.0)
    monkeypatch.setattr(safety.hardware, "collect_machine_info", lambda paths=None: machine)
    monkeypatch.setattr(safety.hardware, "sample_gpus", lambda: _gpu(0, 31000))
    result = safety.preflight(THRESHOLDS)
    names = {c.name for c in result.blocking_failures}
    assert "free_ram" in names
    assert any(n.startswith("free_disk") for n in names)


def test_small_foreign_process_is_a_warning_not_a_block(monkeypatch):
    procs = [hardware.GpuProcess(2138, "python", 430.0)]
    monkeypatch.setattr(
        safety.hardware, "collect_machine_info", lambda paths=None: _machine(procs=procs)
    )
    monkeypatch.setattr(safety.hardware, "sample_gpus", lambda: _gpu(0, 31000))
    result = safety.preflight(THRESHOLDS)
    assert result.ok
    check = next(c for c in result.notices if c.name == "foreign_gpu_processes")
    assert check.passed is True
    assert "430" in check.detail


def test_large_foreign_process_blocks(monkeypatch):
    procs = [hardware.GpuProcess(999, "trainer", 20000.0)]
    monkeypatch.setattr(
        safety.hardware, "collect_machine_info", lambda paths=None: _machine(procs=procs)
    )
    monkeypatch.setattr(safety.hardware, "sample_gpus", lambda: _gpu(0, 31000))
    result = safety.preflight(THRESHOLDS)
    assert not result.ok
    assert any(c.name == "foreign_gpu_processes" for c in result.blocking_failures)


def test_required_disk_overrides_threshold(monkeypatch):
    monkeypatch.setattr(
        safety.hardware, "collect_machine_info", lambda paths=None: _machine(free_disk=100.0)
    )
    monkeypatch.setattr(safety.hardware, "sample_gpus", lambda: _gpu(0, 31000))
    result = safety.preflight(THRESHOLDS, required_disk_gib=500.0)
    assert not result.ok


def test_wait_until_clear_returns_as_soon_as_it_passes(monkeypatch):
    """A queued stage often starts while the previous one's GPU activity is
    still settling, so waiting is the correct response to a busy machine."""
    monkeypatch.setattr(safety.hardware, "collect_machine_info", lambda paths=None: _machine())
    busy = iter([95.0, 95.0, 2.0])
    monkeypatch.setattr(safety.hardware, "sample_gpus", lambda: _gpu(next(busy, 2.0), 31000))
    slept: list[float] = []
    monkeypatch.setattr(safety.time, "sleep", slept.append)

    seen: list[float] = []
    result = safety.wait_until_clear(
        dict(THRESHOLDS, gpu_sample_count=1),
        timeout_s=60.0,
        poll_interval_s=1.0,
        on_wait=lambda res, left: seen.append(left),
    )
    assert result.ok
    assert len(seen) == 2  # two failing polls, then success
    assert slept == [1.0, 1.0]


def test_wait_until_clear_gives_up_at_the_deadline(monkeypatch):
    monkeypatch.setattr(safety.hardware, "collect_machine_info", lambda paths=None: _machine())
    monkeypatch.setattr(safety.hardware, "sample_gpus", lambda: _gpu(99.0, 31000))
    monkeypatch.setattr(safety.time, "sleep", lambda _s: None)
    result = safety.wait_until_clear(
        dict(THRESHOLDS, gpu_sample_count=1), timeout_s=0.0, poll_interval_s=0.01
    )
    assert not result.ok
    assert any(c.name == "gpu_utilization" for c in result.blocking_failures)


def test_wait_until_clear_does_not_wait_when_already_clear(monkeypatch):
    monkeypatch.setattr(safety.hardware, "collect_machine_info", lambda paths=None: _machine())
    monkeypatch.setattr(safety.hardware, "sample_gpus", lambda: _gpu(0.0, 31000))
    calls: list[float] = []
    monkeypatch.setattr(safety.time, "sleep", calls.append)
    assert safety.wait_until_clear(dict(THRESHOLDS, gpu_sample_count=1), timeout_s=60.0).ok
    assert calls == []
