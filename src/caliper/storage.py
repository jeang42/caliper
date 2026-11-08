"""SQLite persistence for benchmark runs.

One file holds every run so results stay comparable over time. The schema is
deliberately wide and flat: raw model responses, full option sets, telemetry
summaries and scores are all stored, because re-grading old outputs with an
improved validator must not require re-running inference.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import SCHEMA_VERSION, __version__

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id                TEXT PRIMARY KEY,
    created_at        TEXT NOT NULL,
    finished_at       TEXT,
    status            TEXT NOT NULL,
    engine            TEXT NOT NULL,
    engine_version    TEXT,
    hostname          TEXT,
    suite             TEXT,
    suite_version     TEXT,
    caliper_version   TEXT NOT NULL,
    schema_version    INTEGER NOT NULL,
    dry_run           INTEGER NOT NULL DEFAULT 0,
    label             TEXT,
    config_json       TEXT,
    machine_json      TEXT,
    preflight_json    TEXT,
    notes             TEXT
);

CREATE TABLE IF NOT EXISTS run_models (
    run_id         TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    model          TEXT NOT NULL,
    digest         TEXT,
    size_bytes     INTEGER,
    quantization   TEXT,
    architecture   TEXT,
    is_moe         INTEGER,
    context_length INTEGER,
    model_json     TEXT,
    residency_json TEXT,
    PRIMARY KEY (run_id, model)
);

CREATE TABLE IF NOT EXISTS attempts (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id                      TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    model                       TEXT NOT NULL,
    task_id                     TEXT NOT NULL,
    category                    TEXT,
    repetition                  INTEGER NOT NULL,
    is_warmup                   INTEGER NOT NULL DEFAULT 0,
    started_at                  TEXT,
    ok                          INTEGER NOT NULL,
    error                       TEXT,
    prompt_tokens               INTEGER,
    output_tokens               INTEGER,
    ttft_s                      REAL,
    ttft_source                 TEXT,
    total_latency_s             REAL,
    engine_total_duration_s     REAL,
    engine_load_duration_s      REAL,
    engine_prompt_eval_duration_s REAL,
    engine_eval_duration_s      REAL,
    output_tokens_per_s         REAL,
    prompt_tokens_per_s         REAL,
    was_loaded_before           INTEGER,
    done_reason                 TEXT,
    truncated                   INTEGER,
    score                       REAL,
    max_score                   REAL,
    passed                      INTEGER,
    validator                   TEXT,
    graded                      TEXT,
    score_detail                TEXT,
    sub_scores_json             TEXT,
    options_json                TEXT,
    response_text               TEXT,
    thinking_text               TEXT,
    tool_calls_json             TEXT,
    telemetry_json              TEXT,
    raw_final_json              TEXT
);

CREATE INDEX IF NOT EXISTS idx_attempts_run ON attempts(run_id);
CREATE INDEX IF NOT EXISTS idx_attempts_model ON attempts(run_id, model);
CREATE INDEX IF NOT EXISTS idx_attempts_task ON attempts(run_id, task_id);

CREATE TABLE IF NOT EXISTS inventories (
    id          TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    hostname    TEXT,
    engine      TEXT,
    payload_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def new_id() -> str:
    return uuid.uuid4().hex[:16]


def _dumps(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, default=str, sort_keys=True)


@dataclass(slots=True)
class RunHeader:
    id: str
    engine: str
    engine_version: str | None
    hostname: str | None
    suite: str | None
    suite_version: str | None
    dry_run: bool
    label: str | None = None
    status: str = "running"


class Store:
    """Thin SQLite wrapper. Each public method commits, so an interrupted run
    keeps every attempt already written."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA)
        self._migrate()
        self._conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )

    def _migrate(self) -> None:
        """Add columns introduced after a database was first created.

        CREATE TABLE IF NOT EXISTS leaves an existing table alone, so new
        columns are added here. Each entry is append-only and nullable, which
        keeps old rows readable.
        """
        added: dict[str, list[tuple[str, str]]] = {
            "attempts": [("done_reason", "TEXT"), ("truncated", "INTEGER")],
        }
        for table, columns in added.items():
            existing = {
                row["name"] for row in self._conn.execute(f"PRAGMA table_info({table})").fetchall()
            }
            for name, decl in columns:
                if name not in existing:
                    log.info("migrating %s: adding column %s", table, name)
                    self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- writes -----------------------------------------------------------
    def create_run(
        self,
        header: RunHeader,
        config: dict[str, Any],
        machine: dict[str, Any] | None,
        preflight: dict[str, Any] | None,
        notes: str = "",
    ) -> str:
        self._conn.execute(
            """INSERT INTO runs (id, created_at, status, engine, engine_version, hostname,
                                 suite, suite_version, caliper_version, schema_version,
                                 dry_run, label, config_json, machine_json, preflight_json, notes)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                header.id,
                utcnow(),
                header.status,
                header.engine,
                header.engine_version,
                header.hostname,
                header.suite,
                header.suite_version,
                __version__,
                SCHEMA_VERSION,
                int(header.dry_run),
                header.label,
                _dumps(config),
                _dumps(machine),
                _dumps(preflight),
                notes,
            ),
        )
        return header.id

    def finish_run(self, run_id: str, status: str) -> None:
        self._conn.execute(
            "UPDATE runs SET status = ?, finished_at = ? WHERE id = ?",
            (status, utcnow(), run_id),
        )

    def add_run_model(
        self, run_id: str, model: dict[str, Any], residency: dict[str, Any] | None = None
    ) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO run_models
               (run_id, model, digest, size_bytes, quantization, architecture, is_moe,
                context_length, model_json, residency_json)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                run_id,
                model["name"],
                model.get("digest"),
                model.get("size_bytes"),
                model.get("quantization"),
                model.get("architecture"),
                None if model.get("is_moe") is None else int(model["is_moe"]),
                model.get("context_length"),
                _dumps(model),
                _dumps(residency),
            ),
        )

    def set_residency(self, run_id: str, model: str, residency: dict[str, Any] | None) -> None:
        self._conn.execute(
            "UPDATE run_models SET residency_json = ? WHERE run_id = ? AND model = ?",
            (_dumps(residency), run_id, model),
        )

    def add_attempt(
        self,
        run_id: str,
        model: str,
        task_id: str,
        category: str | None,
        repetition: int,
        is_warmup: bool,
        result: Any,
        score: Any | None,
        telemetry: dict[str, Any] | None,
    ) -> int:
        m = result.metrics
        cur = self._conn.execute(
            """INSERT INTO attempts
               (run_id, model, task_id, category, repetition, is_warmup, started_at, ok, error,
                prompt_tokens, output_tokens, ttft_s, ttft_source, total_latency_s,
                engine_total_duration_s, engine_load_duration_s, engine_prompt_eval_duration_s,
                engine_eval_duration_s, output_tokens_per_s, prompt_tokens_per_s,
                was_loaded_before, done_reason, truncated, score, max_score, passed, validator,
                graded, score_detail, sub_scores_json, options_json, response_text, thinking_text,
                tool_calls_json, telemetry_json, raw_final_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                run_id,
                model,
                task_id,
                category,
                repetition,
                int(is_warmup),
                datetime.fromtimestamp(m.request_sent_at, tz=UTC).isoformat(timespec="seconds")
                if m.request_sent_at
                else utcnow(),
                int(result.ok),
                result.error,
                m.prompt_tokens,
                m.output_tokens,
                m.ttft_s,
                m.ttft_source,
                m.total_latency_s,
                m.engine_total_duration_s,
                m.engine_load_duration_s,
                m.engine_prompt_eval_duration_s,
                m.engine_eval_duration_s,
                m.output_tokens_per_s,
                m.prompt_tokens_per_s,
                None if result.was_loaded_before is None else int(result.was_loaded_before),
                result.done_reason,
                int(result.truncated),
                getattr(score, "score", None),
                getattr(score, "max_score", None),
                None if getattr(score, "passed", None) is None else int(score.passed),
                getattr(score, "validator", None),
                getattr(score, "graded", None),
                getattr(score, "detail", None),
                _dumps(getattr(score, "sub_scores", None)),
                _dumps(result.options),
                result.text,
                result.thinking,
                _dumps(result.tool_calls),
                _dumps(telemetry),
                _dumps(result.raw_final),
            ),
        )
        return int(cur.lastrowid or 0)

    def save_inventory(self, payload: dict[str, Any], hostname: str, engine: str) -> str:
        inv_id = new_id()
        self._conn.execute(
            "INSERT INTO inventories (id, created_at, hostname, engine, payload_json) "
            "VALUES (?,?,?,?,?)",
            (inv_id, utcnow(), hostname, engine, _dumps(payload)),
        )
        return inv_id

    # -- reads ------------------------------------------------------------
    def list_runs(self, limit: int = 50) -> list[sqlite3.Row]:
        return list(
            self._conn.execute(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        )

    def get_run(self, run_id: str) -> sqlite3.Row | None:
        return self._conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()

    def latest_run_id(self, dry_run: bool | None = None) -> str | None:
        sql = "SELECT id FROM runs"
        params: tuple = ()
        if dry_run is not None:
            sql += " WHERE dry_run = ?"
            params = (int(dry_run),)
        sql += " ORDER BY created_at DESC LIMIT 1"
        row = self._conn.execute(sql, params).fetchone()
        return row["id"] if row else None

    def run_models(self, run_id: str) -> list[sqlite3.Row]:
        return list(
            self._conn.execute(
                "SELECT * FROM run_models WHERE run_id = ? ORDER BY model", (run_id,)
            ).fetchall()
        )

    def attempts(
        self, run_id: str, include_warmup: bool = False, model: str | None = None
    ) -> list[sqlite3.Row]:
        sql = "SELECT * FROM attempts WHERE run_id = ?"
        params: list[Any] = [run_id]
        if not include_warmup:
            sql += " AND is_warmup = 0"
        if model:
            sql += " AND model = ?"
            params.append(model)
        sql += " ORDER BY model, task_id, repetition"
        return list(self._conn.execute(sql, params).fetchall())

    @contextmanager
    def raw(self) -> Iterator[sqlite3.Connection]:
        yield self._conn
