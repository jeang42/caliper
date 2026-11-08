from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from caliper.backends import MockBackend
from caliper.evals.spec import Suite, Task
from caliper.export import export_attempts_csv, export_run_json, export_summary_csv
from caliper.hardware import TelemetrySampler
from caliper.report import (
    PROFILES,
    ModelSummary,
    Stat,
    rank_for_profile,
    render_markdown,
    stat,
    summarize_run,
)
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
def quiet_telemetry(monkeypatch):
    monkeypatch.setattr("caliper.hardware.sample_gpus", lambda: [])
    monkeypatch.setattr(TelemetrySampler, "interval_s", 0.01, raising=False)


def demo_suite() -> Suite:
    return Suite(
        name="demo",
        suite_version="1.0.0",
        tasks=[
            Task(
                id="a.ok",
                category="coding",
                title="t",
                prompt="say OK",
                validator={"kind": "exact", "expected": "OK"},
            ),
            Task(
                id="a.bad",
                category="reasoning",
                title="t",
                prompt="say NO",
                validator={"kind": "exact", "expected": "YES"},
            ),
            Task(
                id="a.manual",
                category="agent",
                title="t",
                prompt="plan",
                validator={"kind": "rubric", "criteria": ["x"]},
            ),
        ],
    )


@pytest.fixture()
def finished_run(store):
    backend = MockBackend(
        models=["fast:1b", "slow:8b"],
        answers={"say OK": "OK", "say NO": "NO", "plan": "a plan"},
    )
    settings = RunSettings(models=["fast:1b", "slow:8b"], repetitions=2, warmup_runs=1)
    outcome = BenchmarkRunner(backend, store, settings, THRESHOLDS).run(demo_suite())
    return outcome.run_id


def test_stat_reports_spread_and_count():
    s = stat([1.0, 2.0, 3.0, None])
    assert s.n == 3 and s.mean == 2.0 and s.median == 2.0
    assert s.minimum == 1.0 and s.maximum == 3.0
    assert s.stdev == pytest.approx(1.0)
    assert "±" in s.fmt()


def test_stat_of_nothing_is_not_an_error():
    assert stat([None, None]).fmt() == "n/a"


def test_summary_separates_categories_and_manual_grading(store, finished_run):
    summary = summarize_run(store, finished_run)
    assert {m["model"] for m in summary["models"]} == {"fast:1b", "slow:8b"}
    model = next(m for m in summary["models"] if m["model"] == "fast:1b")
    assert model["categories"]["coding"]["mean_score"] == 1.0
    assert model["categories"]["coding"]["graded_attempts"] == 2
    assert model["categories"]["reasoning"]["mean_score"] == 0.0
    assert model["categories"]["agent"]["mean_score"] is None
    assert model["categories"]["agent"]["pending_manual"] == 2
    assert model["reliability"] == 1.0


def test_summary_excludes_warmup_attempts(store, finished_run):
    summary = summarize_run(store, finished_run)
    model = summary["models"][0]
    # 3 tasks x 2 repetitions, warmup excluded.
    assert model["attempts"] == 6


def test_summary_records_performance_statistics(store, finished_run):
    summary = summarize_run(store, finished_run)
    perf = summary["models"][0]["performance"]
    assert perf["output_tokens_per_s"]["mean"] > 0
    assert perf["ttft_s"]["n"] == 6
    assert perf["model_load_s"]["n"] >= 0


def test_report_markdown_mentions_limits_and_categories(store, finished_run):
    text = render_markdown(summarize_run(store, finished_run))
    assert "Quality by category" in text
    assert "Limitations" in text
    assert "pending manual" in text
    assert "DRY RUN" not in text  # dry_run flag was not set
    for profile in PROFILES.values():
        assert profile["label"] in text


def test_dry_run_report_is_labelled(store):
    backend = MockBackend(models=["m:1b"])
    settings = RunSettings(models=["m:1b"], repetitions=1, warmup_runs=0, dry_run=True)
    outcome = BenchmarkRunner(backend, store, settings, THRESHOLDS).run(demo_suite())
    text = render_markdown(summarize_run(store, outcome.run_id))
    assert "DRY RUN" in text


def test_profile_ranking_prefers_quality_for_coding():
    good = ModelSummary(model="good", engine="mock")
    good.categories = {"coding": type("C", (), {"mean_score": 1.0})()}
    bad = ModelSummary(model="bad", engine="mock")
    bad.categories = {"coding": type("C", (), {"mean_score": 0.0})()}
    good.output_tps = Stat(n=1, mean=10.0)
    bad.output_tps = Stat(n=1, mean=100.0)
    ranking = rank_for_profile([good, bad], "coding_assistant")
    assert ranking.ranking[0][0] == "good"
    assert "not comparable across different model sets" in ranking.note


def test_profile_ranking_prefers_speed_for_interactive():
    a = ModelSummary(model="a", engine="mock")
    b = ModelSummary(model="b", engine="mock")
    for m in (a, b):
        m.categories = {
            "instruction_following": type("C", (), {"mean_score": 0.8})(),
            "reasoning": type("C", (), {"mean_score": 0.8})(),
            "coding": type("C", (), {"mean_score": 0.8})(),
        }
    a.output_tps = Stat(n=1, mean=120.0)
    a.ttft_s = Stat(n=1, mean=0.2)
    b.output_tps = Stat(n=1, mean=20.0)
    b.ttft_s = Stat(n=1, mean=2.0)
    assert rank_for_profile([a, b], "fast_interactive").ranking[0][0] == "a"


def test_every_profile_ranks_without_data():
    empty = [ModelSummary(model="x", engine="mock")]
    for name in PROFILES:
        ranking = rank_for_profile(empty, name)
        assert ranking.ranking[0][0] == "x"


def test_exports_write_json_and_csv(store, finished_run, tmp_path: Path):
    json_path = export_run_json(store, finished_run, tmp_path / "run.json")
    attempts_path = export_attempts_csv(store, finished_run, tmp_path / "attempts.csv")
    summary_path = export_summary_csv(store, finished_run, tmp_path / "summary.csv")

    payload = json.loads(json_path.read_text())
    assert payload["run"]["id"] == finished_run
    assert "attempts" not in payload
    assert len(payload["models"]) == 2
    assert payload["profiles"]

    rows = list(csv.DictReader(attempts_path.open()))
    assert len(rows) == 14  # 2 models x (1 warmup + 3 tasks x 2 reps)
    assert {r["is_warmup"] for r in rows} == {"0", "1"}

    summary_rows = list(csv.DictReader(summary_path.open()))
    assert {r["model"] for r in summary_rows} == {"fast:1b", "slow:8b"}
    assert summary_rows[0]["score_coding"] not in (None, "")


def test_export_json_can_include_raw_responses(store, finished_run, tmp_path: Path):
    path = export_run_json(store, finished_run, tmp_path / "full.json", include_responses=True)
    payload = json.loads(path.read_text())
    assert payload["attempts"]
    assert any(a["response_text"] for a in payload["attempts"])


def test_summarize_unknown_run_raises(store):
    with pytest.raises(KeyError):
        summarize_run(store, "nope")


def test_truncation_is_counted_and_surfaced(store):
    """A response that stopped at the token limit must be visible in the report."""
    backend = MockBackend(models=["chatty:8b"], answers={"say OK": ""})

    original = backend.generate

    def truncating(*args, **kwargs):
        result = original(*args, **kwargs)
        result.raw_final = {"done_reason": "length"}
        return result

    backend.generate = truncating
    settings = RunSettings(models=["chatty:8b"], repetitions=1, warmup_runs=0)
    outcome = BenchmarkRunner(backend, store, settings, THRESHOLDS).run(demo_suite())
    summary = summarize_run(store, outcome.run_id)
    assert summary["models"][0]["truncated"] == 3
    text = render_markdown(summary)
    assert "truncated" in text
    assert "stopped at the token limit" in text


def test_cold_load_comes_from_the_warmup_attempt(store):
    """Cold-start cost exists only on the attempt that loaded the model."""
    backend = MockBackend(models=["fresh:8b"])
    settings = RunSettings(models=["fresh:8b"], repetitions=1, warmup_runs=1)
    outcome = BenchmarkRunner(backend, store, settings, THRESHOLDS).run(demo_suite())
    summary = summarize_run(store, outcome.run_id)
    perf = summary["models"][0]["performance"]
    # The mock reports a 2 s load on the first (warmup) call and none after.
    assert perf["cold_load_s"]["n"] == 1
    assert perf["cold_load_s"]["mean"] == pytest.approx(2.0)
    # Warm load is measured, not omitted: a resident model costs ~0 to "load".
    assert perf["model_load_s"]["n"] == 3
    assert perf["model_load_s"]["mean"] == pytest.approx(0.0)
    assert summary["models"][0]["reloads"] == 0
    assert "cold versus warm" in render_markdown(summary)


def test_profile_drops_categories_the_run_never_measured(store):
    """A profile must not build a composite out of untested categories."""
    backend = MockBackend(models=["m:1b"])
    settings = RunSettings(models=["m:1b"], categories=["coding"], repetitions=1, warmup_runs=0)
    outcome = BenchmarkRunner(backend, store, settings, THRESHOLDS).run(demo_suite())
    summary = summarize_run(store, outcome.run_id)
    reasoning = next(p for p in summary["profiles"] if p["profile"] == "reasoning_assistant")
    assert "reasoning" in reasoning["unmeasured_categories"]
    assert "did not measure" in reasoning["note"] or "none of the categories" in reasoning["note"]
    assert "**Partial:**" in render_markdown(summary)


def test_profile_keeps_the_categories_that_were_measured(store, finished_run):
    """Only the untested categories are dropped; the tested ones still count.

    The demo suite covers coding and reasoning automatically and agent only by
    rubric, so the coding profile should drop instruction_following and keep
    the other two weighted.
    """
    summary = summarize_run(store, finished_run)
    coding = next(p for p in summary["profiles"] if p["profile"] == "coding_assistant")
    assert coding["unmeasured_categories"] == ["instruction_following"]
    components = coding["ranking"][0]["components"]
    assert "quality:coding" in components
    assert "quality:reasoning" in components
    assert "quality:instruction_following" not in components


def test_rubric_only_category_counts_as_unmeasured(store, finished_run):
    """Agent tasks in the demo suite are rubric-graded, so they have no
    automatic score and must not be silently treated as zero."""
    summary = summarize_run(store, finished_run)
    agent_profile = next(p for p in summary["profiles"] if p["profile"] == "multi_agent_worker")
    assert "agent" in agent_profile["unmeasured_categories"]


def test_report_states_the_token_limits_actually_sent(store):
    """The run default is not what the engine received when tasks override it."""
    suite = Suite(
        name="t",
        suite_version="1",
        tasks=[
            Task(
                id="a.big",
                category="coding",
                title="t",
                prompt="say OK",
                validator={"kind": "exact", "expected": "OK"},
                max_output_tokens=1536,
                context_length=16384,
            )
        ],
    )
    backend = MockBackend(models=["m:1b"], answers={"say OK": "OK"})
    settings = RunSettings(
        models=["m:1b"], repetitions=1, warmup_runs=0, max_output_tokens=512, context_length=8192
    )
    outcome = BenchmarkRunner(backend, store, settings, THRESHOLDS).run(suite)
    summary = summarize_run(store, outcome.run_id)
    assert summary["observed_options"]["num_predict"] == [1536]
    assert summary["observed_options"]["num_ctx"] == [16384]
    text = render_markdown(summary)
    assert "num_predict 1536" in text
    assert "num_ctx 16384" in text
    assert "num_predict 512" not in text


def test_prompt_cache_hits_are_excluded_from_prompt_throughput(store):
    """An impossible prompt rate means the engine reused a cached prompt.

    Averaging those in produced a reported 61,560 tok/s on real hardware that
    sustains roughly 1,000-2,000.
    """
    from caliper.report import PROMPT_RATE_PLAUSIBILITY_CEILING

    backend = MockBackend(models=["m:1b"], answers={"say OK": "OK"})
    settings = RunSettings(models=["m:1b"], repetitions=2, warmup_runs=0)
    outcome = BenchmarkRunner(backend, store, settings, THRESHOLDS).run(demo_suite())

    # Half the attempts get a plausible prompt rate, half an impossible one.
    with store.raw() as conn:
        ids = [
            r["id"]
            for r in conn.execute(
                "SELECT id FROM attempts WHERE run_id = ? ORDER BY id", (outcome.run_id,)
            )
        ]
        assert len(ids) == 6
        for i, attempt_id in enumerate(ids):
            rate = 1500.0 if i % 2 == 0 else 750_000.0
            conn.execute(
                "UPDATE attempts SET prompt_tokens_per_s = ? WHERE id = ?", (rate, attempt_id)
            )

    summary = summarize_run(store, outcome.run_id)
    model = summary["models"][0]
    assert model["prompt_cache_hits"] == 3
    prompt_rate = model["performance"]["prompt_tokens_per_s"]
    assert prompt_rate["n"] == 3
    assert prompt_rate["max"] < PROMPT_RATE_PLAUSIBILITY_CEILING
    text = render_markdown(summary)
    assert "cached prompts" in text
    assert "physically impossible" in text


def test_only_the_first_repetition_of_a_task_counts_as_a_cold_prompt(store, finished_run):
    """Repeating an identical prompt hits the engine's prompt cache, whatever
    rate it reports, so only the first repetition of each task measures real
    prompt throughput. Observed undetected by a pure rate threshold: 9,433
    tokens in 0.108 s, which is 87,000 tok/s against real rates of 1,000-13,000."""
    summary = summarize_run(store, finished_run)
    model = summary["models"][0]
    # The demo suite has 3 tasks at 2 repetitions: 3 cold, 3 cached.
    assert model["attempts"] == 6
    assert model["prompt_cache_hits"] == 3
    assert model["performance"]["prompt_tokens_per_s"]["n"] == 3


def test_single_repetition_runs_have_no_cached_prompts(store):
    backend = MockBackend(
        models=["m:1b"], answers={"say OK": "OK", "say NO": "NO", "plan": "a plan"}
    )
    settings = RunSettings(models=["m:1b"], repetitions=1, warmup_runs=0)
    outcome = BenchmarkRunner(backend, store, settings, THRESHOLDS).run(demo_suite())
    model = summarize_run(store, outcome.run_id)["models"][0]
    assert model["prompt_cache_hits"] == 0
    assert model["performance"]["prompt_tokens_per_s"]["n"] == 3
