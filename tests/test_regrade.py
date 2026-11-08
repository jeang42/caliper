"""Re-grading stored responses without re-running inference."""

from __future__ import annotations

import pytest

from caliper.backends import MockBackend
from caliper.evals.spec import Suite, Task
from caliper.hardware import TelemetrySampler
from caliper.regrade import regrade_run
from caliper.runner import BenchmarkRunner, RunSettings

THRESHOLDS = {
    "max_gpu_utilization_pct": 100.0,
    "min_free_vram_mib": 0.0,
    "min_free_ram_mib": 0.0,
    "min_free_disk_gib": 0.0,
    "allow_foreign_gpu_processes": True,
    "max_foreign_gpu_mib": 1e9,
    "gpu_sample_count": 1,
    "gpu_sample_interval_s": 0.0,
}


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    monkeypatch.setattr("caliper.hardware.sample_gpus", lambda: [])
    monkeypatch.setattr(TelemetrySampler, "interval_s", 0.01, raising=False)


def suite_named(name: str, expected: str) -> Suite:
    return Suite(
        name=name,
        suite_version="1",
        tasks=[
            Task(
                id="t.exact",
                category="coding",
                title="t",
                prompt="say OK",
                validator={"kind": "exact", "expected": expected},
            )
        ],
    )


def _run(store, suite):
    backend = MockBackend(models=["m:1b"], answers={"say OK": "OK"})
    settings = RunSettings(models=["m:1b"], repetitions=2, warmup_runs=0)
    return BenchmarkRunner(backend, store, settings, THRESHOLDS).run(suite)


def test_regrade_is_a_no_op_when_nothing_changed(store, monkeypatch):
    suite = suite_named("rg", "OK")
    monkeypatch.setattr("caliper.regrade.load_suite", lambda name: suite)
    outcome = _run(store, suite)
    result = regrade_run(store, outcome.run_id)
    assert result.changed == 0
    assert result.unchanged == 2


def test_regrade_applies_a_corrected_validator(store, monkeypatch):
    """The point of storing raw responses: fix grading without re-running."""
    outcome = _run(store, suite_named("rg", "OK"))
    assert all(r["score"] == 1.0 for r in store.attempts(outcome.run_id))

    # The validator is corrected to expect something else.
    monkeypatch.setattr("caliper.regrade.load_suite", lambda name: suite_named("rg", "DIFFERENT"))
    result = regrade_run(store, outcome.run_id)
    assert result.changed == 2
    rows = store.attempts(outcome.run_id)
    assert all(r["score"] == 0.0 and r["passed"] == 0 for r in rows)
    assert all("DIFFERENT" in r["score_detail"] for r in rows)


def test_regrade_dry_run_writes_nothing(store, monkeypatch):
    outcome = _run(store, suite_named("rg", "OK"))
    monkeypatch.setattr("caliper.regrade.load_suite", lambda name: suite_named("rg", "DIFFERENT"))
    result = regrade_run(store, outcome.run_id, dry_run=True)
    assert result.changed == 2
    assert all(r["score"] == 1.0 for r in store.attempts(outcome.run_id))


def test_regrade_leaves_measurements_untouched(store, monkeypatch):
    outcome = _run(store, suite_named("rg", "OK"))
    before = [
        (r["ttft_s"], r["output_tokens"], r["total_latency_s"], r["telemetry_json"])
        for r in store.attempts(outcome.run_id)
    ]
    monkeypatch.setattr("caliper.regrade.load_suite", lambda name: suite_named("rg", "DIFFERENT"))
    regrade_run(store, outcome.run_id)
    after = [
        (r["ttft_s"], r["output_tokens"], r["total_latency_s"], r["telemetry_json"])
        for r in store.attempts(outcome.run_id)
    ]
    assert before == after


def test_regrade_reports_tasks_no_longer_in_the_suite(store, monkeypatch):
    outcome = _run(store, suite_named("rg", "OK"))
    empty = Suite(name="rg", suite_version="1", tasks=[])
    monkeypatch.setattr("caliper.regrade.load_suite", lambda name: empty)
    result = regrade_run(store, outcome.run_id)
    assert result.missing_tasks == ["t.exact"]
    assert result.changed == 0


def test_regrade_unknown_run_raises(store):
    with pytest.raises(KeyError):
        regrade_run(store, "nope")
