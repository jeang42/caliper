from __future__ import annotations

from caliper import SCHEMA_VERSION
from caliper.results import GenerationMetrics, GenerationResult, TaskScore
from caliper.storage import RunHeader, Store, new_id


def header(**kw) -> RunHeader:
    base = dict(
        id=new_id(),
        engine="mock",
        engine_version="0",
        hostname="host",
        suite="v1",
        suite_version="1.0.0",
        dry_run=False,
    )
    base.update(kw)
    return RunHeader(**base)


def test_run_round_trip(store: Store):
    h = header(label="demo")
    store.create_run(h, {"a": 1}, {"hostname": "host"}, {"ok": True}, notes="hello")
    row = store.get_run(h.id)
    assert row["label"] == "demo"
    assert row["schema_version"] == SCHEMA_VERSION
    assert row["status"] == "running"
    assert row["notes"] == "hello"
    store.finish_run(h.id, "completed")
    assert store.get_run(h.id)["status"] == "completed"
    assert store.get_run(h.id)["finished_at"]


def test_attempt_round_trip_preserves_raw_output(store: Store):
    h = header()
    store.create_run(h, {}, None, None)
    result = GenerationResult(
        model="m",
        engine="mock",
        text="the answer",
        thinking="hmm",
        tool_calls=[{"function": {"name": "f"}}],
        metrics=GenerationMetrics(ttft_s=0.4, output_tokens=12, output_tokens_per_s=30.0),
        options={"temperature": 0.0},
        raw_final={"done_reason": "stop"},
    )
    score = TaskScore(score=0.75, passed=False, validator="contains_all", detail="3/4")
    store.add_attempt(
        h.id, "m", "t1", "coding", 0, False, result, score, {"gpu_util_mean_pct": 42.0}
    )
    (row,) = store.attempts(h.id)
    assert row["response_text"] == "the answer"
    assert row["thinking_text"] == "hmm"
    assert "f" in row["tool_calls_json"]
    assert row["score"] == 0.75
    assert row["passed"] == 0
    assert row["ttft_s"] == 0.4
    assert '"temperature": 0.0' in row["options_json"]
    assert "42" in row["telemetry_json"]


def test_warmups_are_filtered_by_default(store: Store):
    h = header()
    store.create_run(h, {}, None, None)
    result = GenerationResult(model="m", engine="mock", text="x")
    store.add_attempt(h.id, "m", "t", "coding", 0, True, result, None, None)
    store.add_attempt(h.id, "m", "t", "coding", 0, False, result, None, None)
    assert len(store.attempts(h.id)) == 1
    assert len(store.attempts(h.id, include_warmup=True)) == 2


def test_model_metadata_and_residency(store: Store):
    h = header()
    store.create_run(h, {}, None, None)
    store.add_run_model(
        h.id,
        {
            "name": "m",
            "digest": "abc",
            "size_bytes": 100,
            "quantization": "Q4_K_M",
            "architecture": "qwen3moe",
            "is_moe": True,
            "context_length": 262144,
        },
    )
    store.set_residency(h.id, "m", {"gpu_fraction": 0.5})
    (row,) = store.run_models(h.id)
    assert row["is_moe"] == 1
    assert row["context_length"] == 262144
    assert "0.5" in row["residency_json"]


def test_latest_run_id_can_exclude_dry_runs(store: Store):
    dry = header(dry_run=True)
    real = header(dry_run=False)
    store.create_run(dry, {}, None, None)
    store.create_run(real, {}, None, None)
    assert store.latest_run_id(dry_run=False) == real.id
    assert store.latest_run_id(dry_run=True) == dry.id


def test_attempts_can_be_filtered_by_model(store: Store):
    h = header()
    store.create_run(h, {}, None, None)
    for name in ("a", "b"):
        store.add_attempt(
            h.id,
            name,
            "t",
            "coding",
            0,
            False,
            GenerationResult(model=name, engine="mock"),
            None,
            None,
        )
    assert len(store.attempts(h.id, model="a")) == 1


def test_inventory_round_trip(store: Store):
    inv_id = store.save_inventory({"models": []}, "host", "ollama")
    with store.raw() as conn:
        row = conn.execute("SELECT * FROM inventories WHERE id = ?", (inv_id,)).fetchone()
    assert row["hostname"] == "host"
    assert row["engine"] == "ollama"


def test_done_reason_and_truncation_are_stored(store: Store):
    h = header()
    store.create_run(h, {}, None, None)
    for reason in ("stop", "length"):
        store.add_attempt(
            h.id,
            "m",
            "t",
            "coding",
            0,
            False,
            GenerationResult(model="m", engine="mock", text="x", raw_final={"done_reason": reason}),
            None,
            None,
        )
    rows = store.attempts(h.id)
    assert [r["done_reason"] for r in rows] == ["stop", "length"]
    assert [r["truncated"] for r in rows] == [0, 1]


def test_migration_adds_columns_to_an_older_database(tmp_path):
    """A database created before a column existed must still open."""
    import sqlite3

    path = tmp_path / "old.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE attempts (id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, "
        "model TEXT, task_id TEXT, repetition INTEGER, is_warmup INTEGER, ok INTEGER);"
    )
    conn.commit()
    conn.close()

    with Store(path) as store, store.raw() as conn:
        columns = {r["name"] for r in conn.execute("PRAGMA table_info(attempts)")}
    assert {"done_reason", "truncated"} <= columns
