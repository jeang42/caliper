"""End-to-end runner tests against the mock backend (no GPU involved)."""

from __future__ import annotations

from itertools import pairwise

import pytest

from caliper import safety as safety_mod
from caliper.backends import MockBackend
from caliper.evals import load_suite
from caliper.evals.longctx import render as longctx_render
from caliper.evals.spec import Suite, Task
from caliper.hardware import TelemetrySampler
from caliper.runner import (
    BenchmarkRunner,
    RunAborted,
    RunSettings,
    build_options,
    effective_filler_tokens,
    estimate_run_cost,
    filler_fit_problems,
)

THRESHOLDS = {
    "max_gpu_utilization_pct": 25.0,
    "min_free_vram_mib": 1.0,
    "min_free_ram_mib": 1.0,
    "min_free_disk_gib": 0.1,
    "allow_foreign_gpu_processes": True,
    "max_foreign_gpu_mib": 100000.0,
    "gpu_sample_count": 1,
    "gpu_sample_interval_s": 0.0,
}


@pytest.fixture(autouse=True)
def no_telemetry_threads(monkeypatch):
    """Keep tests fast and free of nvidia-smi calls."""
    monkeypatch.setattr("caliper.hardware.sample_gpus", lambda: [])
    monkeypatch.setattr(TelemetrySampler, "interval_s", 0.01, raising=False)


def tiny_suite() -> Suite:
    return Suite(
        name="test",
        suite_version="9.9.9",
        tasks=[
            Task(
                id="t.exact",
                category="instruction_following",
                title="exact",
                prompt="say OK",
                validator={"kind": "exact", "expected": "OK"},
            ),
            Task(
                id="t.rubric",
                category="agent",
                title="rubric",
                prompt="plan something",
                validator={"kind": "rubric", "criteria": ["a"]},
            ),
        ],
    )


def test_run_records_every_attempt(store):
    backend = MockBackend(models=["mock-small:1b"], answers={"say OK": "OK"})
    settings = RunSettings(models=["mock-small:1b"], repetitions=2, warmup_runs=1)
    runner = BenchmarkRunner(backend, store, settings, THRESHOLDS)
    outcome = runner.run(tiny_suite())

    assert outcome.status == "completed"
    measured = store.attempts(outcome.run_id)
    all_attempts = store.attempts(outcome.run_id, include_warmup=True)
    assert len(measured) == 4  # 2 tasks x 2 reps
    assert len(all_attempts) == 5  # plus 1 warmup
    assert sum(a["is_warmup"] for a in all_attempts) == 1

    exact = [a for a in measured if a["task_id"] == "t.exact"]
    assert all(a["score"] == 1.0 and a["passed"] == 1 for a in exact)
    rubric = [a for a in measured if a["task_id"] == "t.rubric"]
    assert all(a["score"] is None and a["graded"] == "manual_rubric" for a in rubric)


def test_run_stores_config_machine_and_model_metadata(store):
    backend = MockBackend(models=["mock-moe:8b-a1b"])
    settings = RunSettings(models=["mock-moe:8b-a1b"], repetitions=1, warmup_runs=0, seed=7)
    outcome = BenchmarkRunner(backend, store, settings, THRESHOLDS).run(tiny_suite())
    run = store.get_run(outcome.run_id)
    assert run["suite_version"] == "9.9.9"
    assert run["engine"] == "mock"
    assert '"seed": 7' in run["config_json"]
    assert run["machine_json"]
    assert run["preflight_json"]
    (model_row,) = store.run_models(outcome.run_id)
    assert model_row["is_moe"] == 1
    assert model_row["residency_json"]


def test_options_are_passed_to_the_backend(store):
    backend = MockBackend(models=["m:1b"])
    settings = RunSettings(
        models=["m:1b"],
        repetitions=1,
        warmup_runs=0,
        temperature=0.3,
        seed=11,
        max_output_tokens=64,
        context_length=4096,
    )
    BenchmarkRunner(backend, store, settings, THRESHOLDS).run(tiny_suite())
    options = backend.calls[0]["options"]
    assert options == {"temperature": 0.3, "num_predict": 64, "num_ctx": 4096, "seed": 11}


def test_task_overrides_beat_run_settings():
    settings = RunSettings(
        models=["m"], temperature=0.0, max_output_tokens=100, context_length=2048
    )
    task = Task(
        id="x",
        category="coding",
        title="x",
        prompt="p",
        validator={"kind": "exact", "expected": "y"},
        max_output_tokens=777,
        context_length=16384,
        temperature=0.9,
    )
    options = build_options(settings, task)
    assert options["num_predict"] == 777
    assert options["num_ctx"] == 16384
    assert options["temperature"] == 0.9


def test_seed_can_be_disabled():
    settings = RunSettings(models=["m"], seed=None)
    task = Task(
        id="x",
        category="coding",
        title="x",
        prompt="p",
        validator={"kind": "exact", "expected": "y"},
    )
    assert "seed" not in build_options(settings, task)


def test_generation_failure_is_recorded_and_counted(store):
    backend = MockBackend(models=["bad:1b"], fail_models={"bad:1b"})
    settings = RunSettings(models=["bad:1b"], repetitions=1, warmup_runs=0)
    outcome = BenchmarkRunner(backend, store, settings, THRESHOLDS).run(tiny_suite())
    assert outcome.status == "completed_with_errors"
    assert outcome.failures == 2
    rows = store.attempts(outcome.run_id)
    assert all(row["ok"] == 0 for row in rows)
    assert all(row["score"] == 0.0 for row in rows)


def test_preflight_failure_aborts_without_force(store, monkeypatch):
    monkeypatch.setattr(
        BenchmarkRunner,
        "preflight",
        lambda self: safety_mod.PreflightResult(
            checks=[safety_mod.Check("gpu_utilization", False, "GPU busy")]
        ),
    )
    settings = RunSettings(models=["m:1b"], repetitions=1, warmup_runs=0)
    runner = BenchmarkRunner(MockBackend(models=["m:1b"]), store, settings, THRESHOLDS)
    with pytest.raises(RunAborted):
        runner.run(tiny_suite())
    assert store.list_runs() == []


def test_force_overrides_preflight_failure(store, monkeypatch):
    monkeypatch.setattr(
        BenchmarkRunner,
        "preflight",
        lambda self: safety_mod.PreflightResult(
            checks=[safety_mod.Check("gpu_utilization", False, "GPU busy")]
        ),
    )
    settings = RunSettings(models=["m:1b"], repetitions=1, warmup_runs=0, force=True)
    runner = BenchmarkRunner(MockBackend(models=["m:1b"]), store, settings, THRESHOLDS)
    outcome = runner.run(tiny_suite())
    assert outcome.attempts == 2


def test_preloaded_model_is_never_unloaded(store):
    backend = MockBackend(models=["resident:1b"])
    backend._loaded.add("resident:1b")
    settings = RunSettings(
        models=["resident:1b"], repetitions=1, warmup_runs=0, unload_after_run=True
    )
    BenchmarkRunner(backend, store, settings, THRESHOLDS).run(tiny_suite())
    assert backend.is_loaded("resident:1b")


def test_model_loaded_by_the_run_is_unloaded_afterwards(store):
    backend = MockBackend(models=["fresh:1b"])
    settings = RunSettings(models=["fresh:1b"], repetitions=1, warmup_runs=0, unload_after_run=True)
    BenchmarkRunner(backend, store, settings, THRESHOLDS).run(tiny_suite())
    assert not backend.is_loaded("fresh:1b")


def test_no_unload_keeps_the_model_resident(store):
    backend = MockBackend(models=["fresh:1b"])
    settings = RunSettings(
        models=["fresh:1b"], repetitions=1, warmup_runs=0, unload_after_run=False
    )
    BenchmarkRunner(backend, store, settings, THRESHOLDS).run(tiny_suite())
    assert backend.is_loaded("fresh:1b")


def test_code_execution_tasks_are_skipped_by_default(store):
    suite = Suite(
        name="t",
        suite_version="1",
        tasks=[
            Task(
                id="c.exec",
                category="coding",
                title="exec",
                prompt="write code",
                validator={
                    "kind": "python_cases",
                    "cases": [{"name": "a", "expression": "f()", "expected": 1}],
                },
                requires_code_execution=True,
            )
        ],
    )
    settings = RunSettings(models=["m:1b"], repetitions=1, warmup_runs=0)
    outcome = BenchmarkRunner(MockBackend(models=["m:1b"]), store, settings, THRESHOLDS).run(suite)
    (row,) = store.attempts(outcome.run_id)
    assert row["graded"] == "skipped"


def test_task_and_category_selection(store):
    settings = RunSettings(models=["m:1b"], categories=["agent"], repetitions=1, warmup_runs=0)
    runner = BenchmarkRunner(MockBackend(models=["m:1b"]), store, settings, THRESHOLDS)
    assert [t.id for t in runner.select_tasks(tiny_suite())] == ["t.rubric"]

    settings.categories = []
    settings.task_ids = ["t.exact"]
    assert [t.id for t in runner.select_tasks(tiny_suite())] == ["t.exact"]

    settings.task_ids = ["nope"]
    with pytest.raises(KeyError):
        runner.select_tasks(tiny_suite())


def test_long_context_filler_is_substituted_and_deterministic(store):
    suite = Suite(
        name="t",
        suite_version="1",
        tasks=[
            Task(
                id="lc.x",
                category="long_context",
                title="lc",
                prompt="BEGIN {{FILLER}} END",
                validator={"kind": "exact", "expected": "x"},
                filler_tokens=200,
            )
        ],
    )
    backend = MockBackend(models=["m:1b"])
    settings = RunSettings(models=["m:1b"], repetitions=2, warmup_runs=0)
    BenchmarkRunner(backend, store, settings, THRESHOLDS).run(suite)
    prompts = [c["prompt"] for c in backend.calls]
    assert "{{FILLER}}" not in prompts[0]
    assert len(prompts[0]) > 500
    assert prompts[0] == prompts[1]  # same seed -> identical context


def test_interrupt_stops_but_keeps_completed_attempts(store, monkeypatch):
    backend = MockBackend(models=["m:1b"])
    settings = RunSettings(models=["m:1b"], repetitions=5, warmup_runs=0)
    runner = BenchmarkRunner(backend, store, settings, THRESHOLDS)
    calls = {"n": 0}

    def stop_after_three() -> bool:
        calls["n"] += 1
        return calls["n"] > 3

    monkeypatch.setattr("caliper.runner._interrupt_guard", lambda: stop_after_three)
    outcome = runner.run(tiny_suite())
    assert outcome.status == "interrupted"
    assert 0 < len(store.attempts(outcome.run_id)) < 10
    assert store.get_run(outcome.run_id)["status"] == "interrupted"


def test_unknown_model_is_skipped_not_fatal(store):
    class Broken(MockBackend):
        def describe_model(self, model):
            if model == "ghost:1b":
                raise RuntimeError("no such model")
            return super().describe_model(model)

    backend = Broken(models=["m:1b"])
    settings = RunSettings(models=["ghost:1b", "m:1b"], repetitions=1, warmup_runs=0)
    outcome = BenchmarkRunner(backend, store, settings, THRESHOLDS).run(tiny_suite())
    assert outcome.skipped_models == ["ghost:1b"]
    assert outcome.attempts == 2


def test_estimate_run_cost_counts_warmups():
    settings = RunSettings(models=["a", "b"], repetitions=3, warmup_runs=1)
    tasks = tiny_suite().tasks
    cost = estimate_run_cost(settings, tasks, seconds_per_attempt=10.0)
    assert cost["attempts_per_model"] == 7
    assert cost["total_attempts"] == 14
    assert cost["estimated_minutes"] == pytest.approx(140 / 60)


def test_real_suite_loads_and_every_validator_kind_is_known():
    from caliper.evals.validators import available_validators

    suite = load_suite("v1")
    known = set(available_validators())
    assert suite.tasks
    for task in suite.tasks:
        assert task.validator["kind"] in known, task.id


def test_cli_token_override_beats_a_task_budget():
    """An explicitly requested budget must win over the suite's own default."""
    settings = RunSettings(models=["m"], max_output_tokens=512, max_output_tokens_override=4096)
    task = Task(
        id="x",
        category="coding",
        title="x",
        prompt="p",
        validator={"kind": "exact", "expected": "y"},
        max_output_tokens=1536,
        context_length=16384,
    )
    options = build_options(settings, task)
    assert options["num_predict"] == 4096
    assert options["num_ctx"] == 16384  # untouched without its own override


def test_task_budget_still_wins_without_an_override():
    settings = RunSettings(models=["m"], max_output_tokens=512)
    task = Task(
        id="x",
        category="coding",
        title="x",
        prompt="p",
        validator={"kind": "exact", "expected": "y"},
        max_output_tokens=1536,
    )
    assert build_options(settings, task)["num_predict"] == 1536


def test_context_override_applies_to_every_task():
    settings = RunSettings(models=["m"], context_length=8192, context_length_override=65536)
    task = Task(
        id="x",
        category="long_context",
        title="x",
        prompt="p",
        validator={"kind": "exact", "expected": "y"},
        context_length=16384,
    )
    assert build_options(settings, task)["num_ctx"] == 65536


def test_tasks_are_ordered_so_context_length_changes_once(store):
    """Changing num_ctx forces the engine to evict and reload the model, which
    cost 14-20 s per switch on a 22 GiB model, so tasks sharing a context
    length must run consecutively."""
    suite = Suite(
        name="ctx",
        suite_version="1",
        tasks=[
            Task(
                id=f"t.{i}",
                category="coding",
                title="t",
                prompt=f"p{i}",
                validator={"kind": "exact", "expected": "x"},
                context_length=ctx,
            )
            for i, ctx in enumerate([8192, 16384, 8192, 16384, 8192])
        ],
    )
    settings = RunSettings(models=["m:1b"], repetitions=1, warmup_runs=0)
    runner = BenchmarkRunner(MockBackend(models=["m:1b"]), store, settings, THRESHOLDS)
    ordered = [t.context_length for t in runner.select_tasks(suite)]
    switches = sum(1 for a, b in pairwise(ordered) if a != b)
    assert switches == 1, ordered
    assert ordered == [8192, 8192, 8192, 16384, 16384]


def test_task_ordering_is_stable_within_a_context_group(store):
    suite = Suite(
        name="ctx",
        suite_version="1",
        tasks=[
            Task(
                id=f"t.{i}",
                category="coding",
                title="t",
                prompt="p",
                validator={"kind": "exact", "expected": "x"},
                context_length=8192,
            )
            for i in range(4)
        ],
    )
    settings = RunSettings(models=["m:1b"], repetitions=1, warmup_runs=0)
    runner = BenchmarkRunner(MockBackend(models=["m:1b"]), store, settings, THRESHOLDS)
    assert [t.id for t in runner.select_tasks(suite)] == ["t.0", "t.1", "t.2", "t.3"]


def test_cost_estimate_uses_measured_history_when_available(store):
    """A single global guess spanned 9x the real range across models, so a
    model that has been benchmarked here must be costed from its own data."""
    backend = MockBackend(models=["seen:1b"])
    settings = RunSettings(models=["seen:1b"], repetitions=1, warmup_runs=0)
    BenchmarkRunner(backend, store, settings, THRESHOLDS).run(tiny_suite())

    history = store.measured_seconds_per_attempt("seen:1b")
    assert history is not None
    measured_spa, samples = history
    assert samples == 2

    cost = estimate_run_cost(settings, tiny_suite().tasks, store=store)
    assert cost["assumed_models"] == []
    assert cost["assumed_seconds_per_attempt"] is None
    assert cost["measured_seconds_per_attempt"]["seen:1b"] == pytest.approx(measured_spa, abs=0.01)
    assert cost["estimated_minutes"] == pytest.approx(2 * measured_spa / 60.0, rel=0.01)
    assert "measured history" in cost["basis"]


def test_cost_estimate_falls_back_and_says_so(store):
    settings = RunSettings(models=["never-run:70b"], repetitions=1, warmup_runs=0)
    cost = estimate_run_cost(settings, tiny_suite().tasks, store=store, seconds_per_attempt=30.0)
    assert cost["assumed_models"] == ["never-run:70b"]
    assert cost["measured_seconds_per_attempt"] == {}
    assert cost["estimated_minutes"] == pytest.approx(2 * 30.0 / 60.0)
    assert "placeholder" in cost["basis"]


def test_cost_estimate_marks_a_mixed_basis(store):
    backend = MockBackend(models=["seen:1b"])
    BenchmarkRunner(
        backend, store, RunSettings(models=["seen:1b"], repetitions=1, warmup_runs=0), THRESHOLDS
    ).run(tiny_suite())
    settings = RunSettings(models=["seen:1b", "unseen:8b"], repetitions=1, warmup_runs=0)
    cost = estimate_run_cost(settings, tiny_suite().tasks, store=store)
    assert cost["basis"].startswith("mixed:")
    assert cost["assumed_models"] == ["unseen:8b"]
    assert "seen:1b" in cost["measured_seconds_per_attempt"]


def test_measured_history_ignores_dry_runs(store):
    """Mock timings are fabricated, so they must never cost a real run."""
    backend = MockBackend(models=["mocked:1b"])
    settings = RunSettings(models=["mocked:1b"], repetitions=1, warmup_runs=0, dry_run=True)
    BenchmarkRunner(backend, store, settings, THRESHOLDS).run(tiny_suite())
    assert store.measured_seconds_per_attempt("mocked:1b") is None


def test_runner_waits_for_a_busy_machine_when_asked(store, monkeypatch):
    """--wait turns a transient block into a delay rather than an abort."""
    attempts = {"n": 0}

    def fake_wait(safety, timeout_s, gpu_index=0, on_wait=None, **kw):
        attempts["n"] += 1
        return safety_mod.PreflightResult(checks=[safety_mod.Check("gpu_utilization", True, "ok")])

    monkeypatch.setattr(safety_mod, "wait_until_clear", fake_wait)
    settings = RunSettings(models=["m:1b"], repetitions=1, warmup_runs=0, wait_for_clear_s=120.0)
    runner = BenchmarkRunner(MockBackend(models=["m:1b"]), store, settings, THRESHOLDS)
    outcome = runner.run(tiny_suite())
    assert attempts["n"] == 1
    assert outcome.attempts == 2


def test_runner_does_not_wait_by_default(store, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("should not wait without --wait")

    monkeypatch.setattr(safety_mod, "wait_until_clear", boom)
    settings = RunSettings(models=["m:1b"], repetitions=1, warmup_runs=0)
    runner = BenchmarkRunner(MockBackend(models=["m:1b"]), store, settings, THRESHOLDS)
    assert runner.run(tiny_suite()).attempts == 2


def filler_suite(filler_tokens: int = 2000, context_length: int = 8192) -> Suite:
    """A one-task long-context suite plus a task with no document at all."""
    return Suite(
        name="test-longctx",
        suite_version="9.9.9",
        tasks=[
            Task(
                id="t.longctx",
                category="long_context",
                title="document",
                prompt="Document:\n{{FILLER:1}}\nQuestion: say OK",
                validator={"kind": "contains_all", "needles": ["OK"]},
                filler_tokens=filler_tokens,
                context_length=context_length,
                max_output_tokens=256,
            ),
            Task(
                id="t.short",
                category="instruction_following",
                title="short",
                prompt="say OK",
                validator={"kind": "exact", "expected": "OK"},
            ),
        ],
    )


def test_filler_override_changes_the_document_the_model_is_sent():
    """--filler is the only knob that moves document size.

    ``--context`` moves ``num_ctx`` alone; the document comes from the task's
    ``filler_tokens``. Stage 8b was mis-framed as a length test for exactly
    this reason, so the override has a test proving it reaches the prompt.
    """
    suite = filler_suite(filler_tokens=1000, context_length=32768)
    task = suite.get("t.longctx")
    base = RunSettings(models=["m:1b"])
    assert effective_filler_tokens(base, task) == 1000

    bigger = RunSettings(models=["m:1b"], filler_tokens_override=4000)
    assert effective_filler_tokens(bigger, task) == 4000

    small, _ = longctx_render(task.prompt, effective_filler_tokens(base, task))
    large, _ = longctx_render(task.prompt, effective_filler_tokens(bigger, task))
    assert len(large) > 3 * len(small)


def test_filler_override_never_pads_a_task_that_has_no_document():
    """The override scales the long-context category, not every prompt."""
    short = filler_suite().get("t.short")
    settings = RunSettings(models=["m:1b"], filler_tokens_override=40000)
    assert effective_filler_tokens(settings, short) is None


def test_the_override_reaches_the_prompt_the_backend_receives(store):
    backend = MockBackend(models=["m:1b"], default_answer="OK")
    settings = RunSettings(
        models=["m:1b"],
        repetitions=1,
        warmup_runs=0,
        filler_tokens_override=4000,
        context_length_override=32768,
    )
    BenchmarkRunner(backend, store, settings, THRESHOLDS).run(filler_suite(filler_tokens=1000))
    sent = {c["prompt"] for c in backend.calls}
    document = max(sent, key=len)
    assert "{{FILLER" not in document
    expected, _ = longctx_render(
        "Document:\n{{FILLER:1}}\nQuestion: say OK", 4000, seed=settings.seed
    )
    assert document == expected


def test_run_refuses_a_document_that_cannot_fit_its_context_window(store):
    """Ollama discards prompt overflow silently.

    An oversized document would then be scored as a long-context answer about
    a document the model was never shown - a wrong number that looks right,
    which is the one outcome this project treats as worse than no feature.
    """
    backend = MockBackend(models=["m:1b"], default_answer="OK")
    settings = RunSettings(
        models=["m:1b"],
        repetitions=1,
        warmup_runs=0,
        filler_tokens_override=40000,
    )
    runner = BenchmarkRunner(backend, store, settings, THRESHOLDS)
    with pytest.raises(RunAborted, match="would not fit"):
        runner.run(filler_suite(filler_tokens=1000, context_length=8192))
    assert backend.calls == []


def test_force_overrides_the_fit_check_and_the_run_proceeds(store):
    backend = MockBackend(models=["m:1b"], default_answer="OK")
    settings = RunSettings(
        models=["m:1b"],
        repetitions=1,
        warmup_runs=0,
        filler_tokens_override=40000,
        force=True,
    )
    outcome = BenchmarkRunner(backend, store, settings, THRESHOLDS).run(
        filler_suite(filler_tokens=1000, context_length=8192)
    )
    assert outcome.status == "completed"


def test_the_fit_check_passes_when_context_is_raised_to_match():
    suite = filler_suite(filler_tokens=1000, context_length=8192)
    tasks = suite.tasks
    tight = RunSettings(models=["m:1b"], filler_tokens_override=16000)
    assert filler_fit_problems(tight, tasks)
    roomy = RunSettings(
        models=["m:1b"], filler_tokens_override=16000, context_length_override=32768
    )
    assert filler_fit_problems(roomy, tasks) == []


def test_the_fit_check_counts_the_output_budget_too():
    """num_ctx holds the prompt and the generation together, not the prompt alone."""
    suite = filler_suite(filler_tokens=1000, context_length=8192)
    task = suite.get("t.longctx")
    task.max_output_tokens = 256
    settings = RunSettings(models=["m:1b"], filler_tokens_override=6200)
    # 6200 * 1.3 = 8060, under num_ctx 8192 on its own; with 256 output it is not.
    assert filler_fit_problems(settings, [task])
