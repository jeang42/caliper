"""Inventory assembly and reporting tests."""

from __future__ import annotations

from caliper import hardware
from caliper.backends import MockBackend
from caliper.inventory import (
    classify_residency,
    collect_inventory,
    estimate_runtime_vram_gib,
    render_report,
)
from caliper.results import ModelRecord

GIB = 1024**3


def machine(vram_mib: float = 32760.0) -> hardware.MachineInfo:
    return hardware.MachineInfo(
        hostname="ada",
        platform="linux",
        kernel="7.0",
        python_version="3.12",
        cpu=hardware.CpuInfo("Ryzen", 24, 12, 5600.0),
        memory=hardware.MemoryInfo(94000.0, 69000.0, 8192.0, 128.0),
        disks=[hardware.DiskInfo("/", 3665.0, 653.0)],
        gpus=[
            hardware.GpuSample(
                0, "RTX 5000 Ada", vram_mib, 453, vram_mib - 453, 0, 0, 39, 5, "580", "8.9"
            )
        ],
        gpu_processes=[hardware.GpuProcess(2138, "python", 430.0)],
        nvidia_smi_available=True,
    )


def test_collect_inventory_includes_engine_and_models(monkeypatch):
    monkeypatch.setattr("caliper.inventory.collect_machine_info", lambda paths=None: machine())
    inv = collect_inventory([MockBackend(models=["a:1b", "b:8b-a1b"])])
    assert inv.engines[0].available is True
    assert {m.name for m in inv.models} == {"a:1b", "b:8b-a1b"}
    assert inv.machine.hostname == "ada"


def test_unavailable_engine_is_recorded_not_raised(monkeypatch):
    monkeypatch.setattr("caliper.inventory.collect_machine_info", lambda paths=None: machine())

    class Down(MockBackend):
        def version(self):
            return None

    inv = collect_inventory([Down()])
    assert inv.engines[0].available is False
    assert inv.models == []


def test_dense_models_are_not_listed_as_missing_expert_data(monkeypatch):
    monkeypatch.setattr("caliper.inventory.collect_machine_info", lambda paths=None: machine())
    inv = collect_inventory([MockBackend(models=["dense:8b"])])
    missing = inv.unknown_fields.get("dense:8b", [])
    assert "expert_count" not in missing
    assert "is_moe" not in missing


def test_unreadable_moe_classification_is_flagged(monkeypatch):
    monkeypatch.setattr("caliper.inventory.collect_machine_info", lambda paths=None: machine())

    class NoInfo(MockBackend):
        def describe_model(self, model):
            return ModelRecord(name=model, engine="mock", size_bytes=1000, is_moe=None)

    inv = collect_inventory([NoInfo(models=["mystery:1b"])])
    assert "is_moe" in inv.unknown_fields["mystery:1b"]


def test_residency_estimate_is_labelled_and_sized():
    small = ModelRecord(name="s", engine="mock", size_bytes=int(12.85 * GIB))
    big = ModelRecord(name="b", engine="mock", size_bytes=int(29.3 * GIB))
    tight = ModelRecord(name="t", engine="mock", size_bytes=int(26.0 * GIB))
    assert classify_residency(small, machine())[0] == "likely fully on GPU"
    assert classify_residency(big, machine())[0] == "likely needs CPU/RAM offload"
    assert classify_residency(tight, machine())[0] == "tight fit; context length will decide"


def test_residency_unknown_without_size():
    unknown = ModelRecord(name="u", engine="mock", size_bytes=None)
    assert classify_residency(unknown, machine())[0] == "unknown"
    assert estimate_runtime_vram_gib(None) is None


def test_report_marks_estimates_and_protected_processes(monkeypatch):
    monkeypatch.setattr("caliper.inventory.collect_machine_info", lambda paths=None: machine())
    inv = collect_inventory([MockBackend(models=["a:1b", "b:8b-a1b"])])
    text = render_report(inv)
    assert "planning only, not measured" in text
    assert "never disturbed by Caliper" in text
    assert "pid 2138" in text
    assert "MoE models detected" in text


def test_zero_size_tag_is_not_treated_as_a_tiny_model():
    cloud = ModelRecord(name="remote:cloud", engine="mock", size_bytes=0)
    assert estimate_runtime_vram_gib(cloud.size_bytes) is None
    assert classify_residency(cloud, machine())[0] == "unknown"
