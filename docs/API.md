# API reference

Two surfaces, both intended for other programs and agents to consume: the HTTP
API behind the viewer, and the Python functions the CLI itself uses. Both are
versioned by `caliper_version` and the database's `schema_version`.

Field names in payloads are stable. New fields may be added; existing ones are
not renamed or repurposed without a version bump.

## HTTP API

Served by `caliper serve` (default `http://127.0.0.1:8777`). All responses are
JSON. Errors return `{"error": "...", "status": <code>}` with that status.

There is **no authentication**, and no endpoint can start, stop or configure a
benchmark. `webapi.py` and `server.py` do not import the runner, and a test
asserts they never will.

### Read endpoints

| Method | Path | Returns |
|---|---|---|
| GET | `/api/version` | `{caliper_version}` |
| GET | `/api/runs?limit=N` | Run list with models, attempt counts, failures, `running` flag |
| GET | `/api/runs/<run_id>` | Full aggregation: run metadata, per-model category scores, performance, telemetry, per-profile rankings, live progress |
| GET | `/api/runs/<run_id>/attempts` | Attempt list (bulky text fields omitted) |
| GET | `/api/runs/<run_id>/grading` | Rubric attempts with their criteria and any human score |
| GET | `/api/attempts/<id>` | One attempt in full: response, thinking, tool calls, telemetry, options, plus the task definition |
| GET | `/api/status` | Current GPU telemetry and any run still executing |
| GET | `/api/compare?run=A&run=B` | Several runs side by side, with comparability warnings |

Filters for `/api/runs/<id>/attempts`: `model`, `category`, `task`, `graded`
(`automatic` \| `manual_rubric` \| `no_output` \| `skipped` \| `error`),
`warmup=1`, `ungraded=1`.

### Write endpoints

Recording a human rubric score is the only write.

```
POST /api/grades
{
  "attempt_id": 287,
  "criteria": [{"text": "...", "met": true}, {"text": "...", "met": false}],
  "grader": "optional name",
  "notes": "optional"
}
→ {"attempt_id": 287, "score": 1.0, "max_score": 2.0, "fraction": 0.5, "stored": true}
```

Pass `score` and `max_score` instead of `criteria` to set a score directly.

```
DELETE /api/grades/<attempt_id>   → {"attempt_id": ..., "deleted": true}
```

A grade is accepted only for an attempt whose grading state is
`manual_rubric` or `no_output`. Automatic scores, warmups and skipped
validators are refused — the allowlist is `webapi.GRADEABLE_STATES`.

### Live progress

Attempts are committed to SQLite as each one completes, so polling
`/api/status` or `/api/runs/<id>` shows an executing run's progress without
the runner publishing anything. The viewer polls `/api/status` every 3
seconds. `active_run.attempts_expected` is derived from the stored run
configuration and is an estimate when a run was filtered to a subset of tasks.

## Python API

For in-process consumers (Nexus, Trykode, an agent with repository access).
Import from these modules only; everything else is internal.

```python
from caliper.config import load_config
from caliper.storage import Store
from caliper.report import summarize_run, rank_for_profile, PROFILES

config = load_config()  # honours CALIPER_CONFIG / CALIPER_HOME
with Store(config.db_path) as store:
    run_id = store.latest_run_id(dry_run=False)
    summary = summarize_run(store, run_id)  # plain dicts, JSON-serialisable

    # Workload-weighted ranking, with the weights and components attached.
    ranking = rank_for_profile(summary["_summaries"], "coding_assistant")
    best_model, composite, components = ranking.ranking[0]
```

| Function | Purpose |
|---|---|
| `summarize_run(store, run_id)` | Everything about one run as nested dicts. `_summaries` holds the dataclasses; drop it before serialising. |
| `rank_for_profile(summaries, profile)` | A ranking plus its weights, per-component values, and which categories the run did not measure. |
| `PROFILES` | The workload profiles and their weights. |
| `regrade_run(store, run_id)` | Re-score stored responses with the current validators. No inference. |
| `collect_inventory(backends)` | Machine, engine and model snapshot. |
| `preflight(config.safety)` | Resource-safety checks. Call before any inference. |

`rank_for_profile` reports `unmeasured_categories`: categories the profile
weights that the run did not measure. They are dropped from the weighting
rather than scored zero, so **check this field before acting on a composite**.

### Deliberately absent

There is no automatic model-routing API. The ranking functions hand a consumer
the numbers and the weights behind them; choosing a model is a policy decision
that belongs to the consumer, and it should not start making itself before the
benchmark data is trusted.

## Stability notes

- `schema_version` (in `meta` and on every run) changes when the SQLite schema
  changes. Migrations are additive and nullable; old rows stay readable.
- Quality scores are comparable across runs only when `suite_version`, engine
  version and model digest match. `/api/compare` checks all three and returns
  `comparable: false` with reasons rather than lining up numbers that are not.
- Composites are normalised across the models within one comparison, so they
  are **not** comparable across different model sets.
- `dry_run: true` means the mock backend produced the responses. Never present
  those as measurements.
