"""Tests for the viewer's JSON API.

Every handler is a plain function of ``(Store, params)``, so these run without
opening a socket. Socket-level routing is covered in ``test_server.py``.
"""

from __future__ import annotations

import pytest

from caliper import webapi
from caliper.backends import MockBackend, mock_backend_for_suite
from caliper.evals import load_suite
from caliper.evals.spec import Suite, Task
from caliper.hardware import TelemetrySampler
from caliper.runner import BenchmarkRunner, RunSettings
from caliper.webapi import ApiError

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
    """Keep every test off the real GPU.

    webapi imports sample_gpus directly, so patching caliper.hardware alone
    leaves webapi bound to the real function and the test shells out to
    nvidia-smi.
    """
    monkeypatch.setattr("caliper.hardware.sample_gpus", lambda: [])
    monkeypatch.setattr("caliper.webapi.sample_gpus", lambda: [])
    monkeypatch.setattr(TelemetrySampler, "interval_s", 0.01, raising=False)


def mini_suite() -> Suite:
    return Suite(
        name="v1",
        suite_version=load_suite("v1").suite_version,
        tasks=[
            load_suite("v1").get("instruction.exact_format"),
            load_suite("v1").get("coding.refactor_decision"),
        ],
    )


@pytest.fixture()
def run_id(store):
    """A real run through the mock backend, using real v1 task definitions."""
    suite = mini_suite()
    backend = mock_backend_for_suite(suite, models=["m:1b"])
    settings = RunSettings(models=["m:1b"], repetitions=2, warmup_runs=1, suite="v1")
    outcome = BenchmarkRunner(backend, store, settings, THRESHOLDS).run(suite)
    return outcome.run_id


# ---------------------------------------------------------------- runs list


def test_list_runs_includes_models_and_progress(store, run_id):
    payload = webapi.list_runs(store)
    assert payload["caliper_version"]
    (entry,) = payload["runs"]
    assert entry["id"] == run_id
    assert entry["models"] == ["m:1b"]
    assert entry["attempts"] == 5  # 1 warmup + 2 tasks x 2 reps
    assert entry["failures"] == 0
    assert entry["running"] is False


def test_list_runs_is_empty_not_an_error(store):
    assert webapi.list_runs(store)["runs"] == []


# ----------------------------------------------------------------- run view


def test_get_run_returns_summary_and_progress(store, run_id):
    payload = webapi.get_run(store, run_id)
    assert payload["run"]["id"] == run_id
    assert payload["models"][0]["model"] == "m:1b"
    assert payload["profiles"]
    assert payload["progress"]["attempts"] == 5
    assert "_summaries" not in payload


def test_get_run_unknown_is_404(store):
    with pytest.raises(ApiError) as exc:
        webapi.get_run(store, "nope")
    assert exc.value.status == 404


# ------------------------------------------------------------------ attempts


def test_list_attempts_excludes_warmups_by_default(store, run_id):
    payload = webapi.list_attempts(store, run_id)
    assert payload["count"] == 4
    assert all(a["is_warmup"] == 0 for a in payload["attempts"])
    assert webapi.list_attempts(store, run_id, include_warmup=True)["count"] == 5


def test_list_attempts_omits_bulky_fields(store, run_id):
    (attempt, *_) = webapi.list_attempts(store, run_id)["attempts"]
    assert "response_text" not in attempt
    assert "thinking_text" not in attempt
    assert attempt["has_response"] is True
    assert "score_detail" in attempt


def test_list_attempts_filters(store, run_id):
    by_task = webapi.list_attempts(store, run_id, task_id="instruction.exact_format")
    assert by_task["count"] == 2
    by_cat = webapi.list_attempts(store, run_id, category="coding")
    assert by_cat["count"] == 2
    by_graded = webapi.list_attempts(store, run_id, graded="manual_rubric")
    assert by_graded["count"] == 2
    assert webapi.list_attempts(store, run_id, model="absent:1b")["count"] == 0


def test_list_attempts_unknown_run_is_404(store):
    with pytest.raises(ApiError) as exc:
        webapi.list_attempts(store, "nope")
    assert exc.value.status == 404


def test_get_attempt_returns_response_and_task_definition(store, run_id):
    first = webapi.list_attempts(store, run_id)["attempts"][0]
    payload = webapi.get_attempt(store, first["id"])
    assert payload["attempt"]["response_text"]
    assert isinstance(payload["attempt"]["options"], dict)
    assert payload["attempt"]["telemetry"] is not None
    assert "options_json" not in payload["attempt"]
    assert payload["task"]["id"] == first["task_id"]
    assert payload["task"]["prompt"]
    assert payload["run"]["engine"] == "mock"


def test_get_attempt_marks_rubric_tasks_as_manual(store, run_id):
    rubric = webapi.list_attempts(store, run_id, graded="manual_rubric")["attempts"][0]
    payload = webapi.get_attempt(store, rubric["id"])
    assert payload["task"]["manual"] is True
    assert payload["task"]["criteria"]
    assert payload["manual_score"] is None


def test_get_attempt_unknown_is_404(store):
    with pytest.raises(ApiError) as exc:
        webapi.get_attempt(store, 99999)
    assert exc.value.status == 404


# ------------------------------------------------------------------- grading


def test_grading_queue_lists_rubric_attempts_with_criteria(store, run_id):
    payload = webapi.grading_queue(store, run_id)
    assert payload["total"] == 2
    assert payload["scored"] == 0
    assert payload["remaining"] == 2
    entry = payload["queue"][0]
    assert entry["task_id"] == "coding.refactor_decision"
    assert entry["criteria"]
    assert entry["scored"] is False


def test_set_grade_from_criteria_computes_the_fraction(store, run_id):
    attempt = webapi.grading_queue(store, run_id)["queue"][0]
    result = webapi.set_grade(
        store,
        {
            "attempt_id": attempt["attempt_id"],
            "criteria": [
                {"text": "a", "met": True},
                {"text": "b", "met": True},
                {"text": "c", "met": False},
                {"text": "d", "met": False},
            ],
            "grader": "a-human",
            "notes": "partially argued",
        },
    )
    assert result == {
        "attempt_id": attempt["attempt_id"],
        "score": 2.0,
        "max_score": 4.0,
        "fraction": 0.5,
        "stored": True,
    }
    queue = webapi.grading_queue(store, run_id)
    assert queue["scored"] == 1
    assert queue["remaining"] == 1


def test_set_grade_accepts_an_explicit_score(store, run_id):
    attempt = webapi.grading_queue(store, run_id)["queue"][0]
    result = webapi.set_grade(
        store, {"attempt_id": attempt["attempt_id"], "score": 3, "max_score": 5}
    )
    assert result["fraction"] == pytest.approx(0.6)


def test_grade_appears_in_the_run_summary_separately(store, run_id):
    """A manual score must not be merged into the automatic mean."""
    attempt = webapi.grading_queue(store, run_id)["queue"][0]
    webapi.set_grade(store, {"attempt_id": attempt["attempt_id"], "score": 1, "max_score": 1})
    summary = webapi.get_run(store, run_id)
    coding = summary["models"][0]["categories"]["coding"]
    assert coding["manual_graded"] == 1
    assert coding["manual_mean_score"] == 1.0
    # The automatic mean for a rubric-only category stays absent.
    assert coding["mean_score"] is None


def test_grade_is_rejected_for_automatically_graded_attempts(store, run_id):
    auto = webapi.list_attempts(store, run_id, graded="automatic")["attempts"][0]
    with pytest.raises(ApiError, match="graded automatically"):
        webapi.set_grade(store, {"attempt_id": auto["id"], "score": 1, "max_score": 1})


def test_grade_validation(store, run_id):
    attempt = webapi.grading_queue(store, run_id)["queue"][0]
    aid = attempt["attempt_id"]
    with pytest.raises(ApiError, match="attempt_id is required"):
        webapi.set_grade(store, {})
    with pytest.raises(ApiError, match="no such attempt"):
        webapi.set_grade(store, {"attempt_id": 98765, "score": 1})
    with pytest.raises(ApiError, match="criteria must be a list"):
        webapi.set_grade(store, {"attempt_id": aid, "criteria": "nope"})
    with pytest.raises(ApiError, match="criteria list is empty"):
        webapi.set_grade(store, {"attempt_id": aid, "criteria": []})
    with pytest.raises(ApiError, match="either criteria, or score"):
        webapi.set_grade(store, {"attempt_id": aid})
    with pytest.raises(ApiError, match="outside"):
        webapi.set_grade(store, {"attempt_id": aid, "score": 9, "max_score": 1})


def test_grade_can_be_replaced_and_deleted(store, run_id):
    aid = webapi.grading_queue(store, run_id)["queue"][0]["attempt_id"]
    webapi.set_grade(store, {"attempt_id": aid, "score": 1, "max_score": 2})
    webapi.set_grade(store, {"attempt_id": aid, "score": 2, "max_score": 2})
    assert webapi.get_attempt(store, aid)["manual_score"]["score"] == 2.0
    webapi.delete_grade(store, aid)
    assert webapi.get_attempt(store, aid)["manual_score"] is None
    assert webapi.grading_queue(store, run_id)["scored"] == 0


def test_regrade_does_not_destroy_manual_scores(store, run_id, monkeypatch):
    """Human work must survive an automatic re-grade."""
    from caliper.regrade import regrade_run

    aid = webapi.grading_queue(store, run_id)["queue"][0]["attempt_id"]
    webapi.set_grade(store, {"attempt_id": aid, "score": 1, "max_score": 1})
    monkeypatch.setattr("caliper.regrade.load_suite", lambda name: mini_suite())
    regrade_run(store, run_id)
    assert webapi.get_attempt(store, aid)["manual_score"]["score"] == 1.0


# -------------------------------------------------------------------- status


def test_live_status_without_a_gpu_is_not_an_error(store):
    payload = webapi.live_status(store)
    assert payload["gpu"] is None
    assert payload["active_run"] is None


def test_live_status_reports_an_executing_run(store):
    """Attempts commit as they complete, so progress is visible mid-run."""
    from caliper.storage import RunHeader

    header = RunHeader(
        id="live0001",
        engine="mock",
        engine_version="0",
        hostname="h",
        suite="v1",
        suite_version="1",
        dry_run=False,
        label="in flight",
    )
    store.create_run(
        header,
        config={
            "settings": {
                "models": ["a:1b"],
                "repetitions": 3,
                "warmup_runs": 1,
                "categories": ["coding"],
                "task_ids": [],
            }
        },
        machine=None,
        preflight=None,
    )
    payload = webapi.live_status(store)
    assert payload["active_run"]["run_id"] == "live0001"
    assert payload["active_run"]["label"] == "in flight"
    assert payload["active_run"]["attempts_expected"] > 0


def test_live_status_reads_gpu_when_available(store, monkeypatch):
    from caliper.hardware import GpuSample

    monkeypatch.setattr(
        "caliper.webapi.sample_gpus",
        lambda: [GpuSample(0, "RTX 5000 Ada", 32760, 14000, 18760, 85, 40, 61, 180, "580", "8.9")],
    )
    gpu = webapi.live_status(store)["gpu"]
    assert gpu["utilization_pct"] == 85
    assert gpu["memory_used_mib"] == 14000
    assert gpu["name"] == "RTX 5000 Ada"


# ------------------------------------------------------------------- compare


def test_compare_flags_mismatched_suite_versions(store):
    suite_a = Suite(
        name="v1",
        suite_version="1.0.0",
        tasks=[
            Task(
                id="t.a",
                category="coding",
                title="t",
                prompt="say OK",
                validator={"kind": "exact", "expected": "OK"},
            )
        ],
    )
    suite_b = Suite(name="v1", suite_version="2.0.0", tasks=suite_a.tasks)
    ids = []
    for suite in (suite_a, suite_b):
        backend = MockBackend(models=["m:1b"], answers={"say OK": "OK"})
        settings = RunSettings(models=["m:1b"], repetitions=1, warmup_runs=0)
        ids.append(BenchmarkRunner(backend, store, settings, THRESHOLDS).run(suite).run_id)

    payload = webapi.compare_runs(store, ids)
    assert payload["comparable"] is False
    assert any("suite versions" in w for w in payload["warnings"])


def test_compare_accepts_matching_runs(store, run_id):
    payload = webapi.compare_runs(store, [run_id])
    assert payload["comparable"] is True
    assert payload["warnings"] == []
    assert payload["runs"][0]["run"]["id"] == run_id


def test_compare_requires_a_run(store):
    with pytest.raises(ApiError, match="at least one run id"):
        webapi.compare_runs(store, [])


def test_grade_is_rejected_for_a_warmup_attempt(store, run_id):
    """Regression: a warmup carries graded=None, which slipped past a
    "not automatic" denylist and could be given a score."""
    warmup = next(
        a
        for a in webapi.list_attempts(store, run_id, include_warmup=True)["attempts"]
        if a["is_warmup"]
    )
    with pytest.raises(ApiError, match="warmup attempt"):
        webapi.set_grade(store, {"attempt_id": warmup["id"], "score": 1, "max_score": 1})


def test_grade_is_rejected_for_a_skipped_validator(store):
    """A task whose validator was disabled has no response worth grading."""
    from caliper.evals.spec import Suite as S
    from caliper.evals.spec import Task as T

    suite = S(
        name="v1",
        suite_version="1",
        tasks=[
            T(
                id="c.exec",
                category="coding",
                title="t",
                prompt="write code",
                validator={
                    "kind": "python_cases",
                    "cases": [{"name": "a", "expression": "f()", "expected": 1}],
                },
                requires_code_execution=True,
            )
        ],
    )
    backend = MockBackend(models=["m:1b"])
    settings = RunSettings(models=["m:1b"], repetitions=1, warmup_runs=0)
    outcome = BenchmarkRunner(backend, store, settings, THRESHOLDS).run(suite)
    (attempt,) = webapi.list_attempts(store, outcome.run_id)["attempts"]
    assert attempt["graded"] == "skipped"
    with pytest.raises(ApiError, match="grading state"):
        webapi.set_grade(store, {"attempt_id": attempt["id"], "score": 1, "max_score": 1})


def test_attempt_detail_rebuilds_the_long_context_document(store):
    """A reviewer must see the document the model saw, not the placeholder.

    Rendered prompts are not stored per attempt - at 8000 tokens each they
    would dominate the database - so the attempt detail rebuilds one from the
    run's seed. This asserts the reconstruction is identical to what the runner
    sent, which only holds while both go through ``longctx.render`` with the
    same seed.
    """
    from caliper.evals import longctx

    v2 = load_suite("v2")
    suite = Suite(
        name="v2",
        suite_version=v2.suite_version,
        tasks=[v2.get("longctx.count_markers")],
    )
    sent: list[str] = []
    backend = mock_backend_for_suite(suite, models=["m:1b"])
    inner = backend.generate

    def record(model, prompt, **kwargs):
        sent.append(prompt)
        return inner(model, prompt, **kwargs)

    backend.generate = record  # type: ignore[method-assign]
    settings = RunSettings(models=["m:1b"], repetitions=1, warmup_runs=0, suite="v2", seed=7)
    outcome = BenchmarkRunner(backend, store, settings, THRESHOLDS).run(suite)

    (attempt,) = webapi.list_attempts(store, outcome.run_id)["attempts"]
    task = webapi.get_attempt(store, attempt["id"])["task"]
    assert task["rendered_prompt"] == sent[0]
    assert "{{FILLER" not in task["rendered_prompt"]
    # The seed is what makes this reproducible: a different one is a different
    # document, so a mismatched seed would be a silently wrong review surface.
    other, _ = longctx.render(task["prompt"], task["filler_tokens"], seed=8)
    assert other != task["rendered_prompt"]


def test_attempt_detail_has_no_rendered_prompt_for_a_short_task(store, run_id):
    """Only filler tasks pay the rebuild; everything else stays untouched."""
    attempt = next(
        a
        for a in webapi.list_attempts(store, run_id)["attempts"]
        if a["category"] != "long_context"
    )
    assert webapi.get_attempt(store, attempt["id"])["task"]["rendered_prompt"] is None


def test_attempt_detail_rebuilds_an_overridden_document_at_the_size_actually_used(store):
    """A run with --filler must rebuild at the run's size, not the task's.

    Rebuilding from ``task.filler_tokens`` would show a reviewer a document
    the model never saw, which is the exact failure the reconstruction exists
    to prevent - and it would look plausible, because the prose is the same
    generator either way.
    """
    v2 = load_suite("v2")
    suite = Suite(
        name="v2", suite_version=v2.suite_version, tasks=[v2.get("longctx.count_markers")]
    )
    sent: list[str] = []
    backend = mock_backend_for_suite(suite, models=["m:1b"])
    inner = backend.generate

    def record(model, prompt, **kwargs):
        sent.append(prompt)
        return inner(model, prompt, **kwargs)

    backend.generate = record  # type: ignore[method-assign]
    settings = RunSettings(
        models=["m:1b"],
        repetitions=1,
        warmup_runs=0,
        suite="v2",
        seed=7,
        filler_tokens_override=2000,
        context_length_override=32768,
    )
    outcome = BenchmarkRunner(backend, store, settings, THRESHOLDS).run(suite)

    (attempt,) = webapi.list_attempts(store, outcome.run_id)["attempts"]
    task = webapi.get_attempt(store, attempt["id"])["task"]
    assert task["rendered_prompt"] == sent[0]
    assert task["filler_tokens_rendered"] == 2000
    assert task["filler_tokens"] == suite.get("longctx.count_markers").filler_tokens
    assert task["filler_tokens_rendered"] != task["filler_tokens"]


# ------------------------------------------------- grouped rubric grading


def test_grading_queue_groups_repetitions_of_one_model_and_task(store, run_id):
    """96 attempts were 22 decisions, and that is why they went ungraded.

    Repetitions of one prompt at temperature 0 are usually the same text.
    Judging each separately multiplies the work by the repetition count for no
    extra information, so the queue reports groups alongside attempts.
    """
    payload = webapi.grading_queue(store, run_id)
    assert payload["total"] == 2, "two rubric attempts: one task x two repetitions"
    assert payload["decisions"] == 1, "but only one judgement to make"
    (group,) = payload["groups"]
    assert group["task_id"] == "coding.refactor_decision"
    assert group["model"] == "m:1b"
    assert group["repetitions"] == 2
    assert group["identical"] is True, "the mock returns one canned answer"
    assert group["distinct_responses"] == 1
    assert sorted(group["attempt_ids"]) == sorted(a["attempt_id"] for a in payload["queue"])


def test_one_judgement_grades_every_attempt_in_the_group(store, run_id):
    """Grouping must not merge scores: each attempt keeps its own row."""
    group = webapi.grading_queue(store, run_id)["groups"][0]
    criteria = [{"text": c, "met": True} for c in group["criteria"]]
    result = webapi.set_grade(
        store, {"attempt_ids": group["attempt_ids"], "criteria": criteria, "grader": "a-human"}
    )
    assert result["graded"] == len(group["attempt_ids"])
    assert result["divergent"] is False

    after = webapi.grading_queue(store, run_id)
    assert after["scored"] == after["total"], "every attempt in the group is scored"
    assert after["decisions_remaining"] == 0
    stored = store.manual_scores(run_id)
    assert len(stored) == len(group["attempt_ids"])
    assert {s["score"] for s in stored.values()} == {float(len(criteria))}


def test_a_group_grade_records_that_it_was_a_group(store, run_id):
    """Provenance: a reader of the score must know it was not read one by one."""
    group = webapi.grading_queue(store, run_id)["groups"][0]
    webapi.set_grade(store, {"attempt_ids": group["attempt_ids"], "score": 1, "max_score": 1})
    notes = {s["notes"] for s in store.manual_scores(run_id).values()}
    assert any("graded as a group of 2 repetitions with identical responses" in n for n in notes)


def test_a_group_grade_is_refused_when_the_responses_differ(store, run_id):
    """One grader's reading must not be applied to text they did not read.

    Repeated attempts at temperature 0 were observed on this engine to produce
    different output lengths for the same prompt, so divergence is a real
    case, not a hypothetical.
    """
    group = webapi.grading_queue(store, run_id)["groups"][0]
    ids = group["attempt_ids"]
    with store.raw() as conn:
        conn.execute(
            "UPDATE attempts SET response_text = ? WHERE id = ?",
            ("a materially different answer", ids[1]),
        )
        conn.commit()

    refreshed = webapi.grading_queue(store, run_id)["groups"][0]
    assert refreshed["identical"] is False
    assert refreshed["distinct_responses"] == 2

    with pytest.raises(webapi.ApiError, match="different responses"):
        webapi.set_grade(store, {"attempt_ids": ids, "score": 1, "max_score": 1})
    assert store.manual_scores(run_id) == {}, "nothing stored on a refusal"

    result = webapi.set_grade(
        store, {"attempt_ids": ids, "score": 1, "max_score": 1, "allow_divergent": True}
    )
    assert result["divergent"] is True
    notes = {s["notes"] for s in store.manual_scores(run_id).values()}
    assert any("allow_divergent" in n for n in notes)


def test_a_group_grade_refuses_to_span_two_models_or_tasks(store, run_id):
    """A judgement covers one model's repetitions of one task, nothing wider."""
    queue = webapi.grading_queue(store, run_id)
    ids = queue["groups"][0]["attempt_ids"]
    other = next(
        a["id"]
        for a in store.attempts(run_id)
        if a["graded"] == "automatic" and a["task_id"] != "coding.refactor_decision"
    )
    with pytest.raises(webapi.ApiError, match="same task and model"):
        webapi.set_grade(store, {"attempt_ids": [ids[0], other], "score": 1, "max_score": 1})


def test_a_group_grade_rejects_an_empty_or_malformed_list(store, run_id):
    for bad in ([], "7", [None]):
        with pytest.raises(webapi.ApiError):
            webapi.set_grade(store, {"attempt_ids": bad, "score": 1, "max_score": 1})
