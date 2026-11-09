# Architecture

## Components

| Component | Module | Responsibility |
|---|---|---|
| Model inventory | `caliper/inventory.py` | Machine + engine + model snapshot, JSON and Markdown |
| Hardware telemetry | `caliper/hardware.py` | `nvidia-smi` and `/proc` readers, background sampler |
| Resource safety | `caliper/safety.py` | Preflight checks against configurable thresholds |
| Backend adapter | `caliper/backends/ollama.py` | Ollama API, streaming generation, timing capture |
| Backend interface | `caliper/backends/base.py` | The `Backend` protocol a vLLM adapter will implement |
| Mock backend | `caliper/backends/mock.py` | Deterministic in-process engine for tests and dry runs |
| Evaluation dataset | `caliper/evals/suites/` | Versioned task definitions in TOML; a suite may extend another |
| Validators | `caliper/evals/validators.py` | Automated grading; manual rubric passthrough |
| Benchmark runner | `caliper/runner.py` | Serial execution, telemetry attribution, persistence |
| Storage | `caliper/storage.py` | SQLite schema and access |
| Reporting | `caliper/report.py` | Aggregation, workload profiles, Markdown |
| Export | `caliper/export.py` | JSON and CSV |
| Re-grading | `caliper/regrade.py` | Re-score stored responses, no inference |
| Web API | `caliper/webapi.py` | JSON handlers as pure functions of (Store, params) |
| Web server | `caliper/server.py` | Routing and serialisation on `http.server` |
| Viewer | `caliper/web/index.html` | Single-page viewer and grading queue |
| CLI | `caliper/cli.py` | Command surface |

Dependencies point one way: `cli` → `runner`/`report` → `backends`/`evals`/
`storage` → `hardware`/`http`/`results`. Nothing below the runner knows the
CLI exists.

## Why these boundaries

**One backend protocol, no plugin framework.** `backends/base.py` defines
seven methods. There is no registry, entry-point discovery or capability
negotiation, because there is exactly one real implementation so far. The
`get_backend()` factory is an `if` chain on purpose; it grows into something
else when, and only when, a second engine is working.

**The runner owns measurement, the backend owns the engine.** The backend
knows how to stream from Ollama and what its timing fields mean. The runner
knows about warmups, repetitions, telemetry windows and persistence. A vLLM
adapter therefore changes nothing in the runner.

**Validators are pure functions of (spec, result).** Grading never calls an
engine, so stored responses can be re-graded by an improved validator without
re-running inference. That is also why raw responses are stored in full.

**SQLite, one file, wide tables.** Runs are small and comparisons are
cross-run, so a single queryable file beats a directory of JSON. The schema is
flat and denormalized where that keeps reporting queries simple.

## Data model

```
runs           one row per benchmark run: identity, engine version, machine
               snapshot, full run configuration, preflight result, status
  run_models   one row per (run, model): digest, size, quantization,
               architecture, MoE flag, context capability, GPU/CPU residency
  manual_scores one row per hand-graded attempt: score, criteria, grader,
               notes, timestamp - separate from attempts so an automatic
               re-grade can never destroy human work
  attempts     one row per (run, model, task, repetition), warmups included
               and flagged: engine timings, client timings, token counts,
               telemetry summary, score, validator detail, raw response,
               thinking text, tool calls, options, errors
inventories    standalone environment snapshots
meta           schema version
```

Identity recorded with every run: run id, machine hostname and full hardware
snapshot, engine name and version, model digest, Caliper version, suite name
and `suite_version`, schema version, timestamps, and the complete settings
object. Two runs are comparable when engine version, model digest and
`suite_version` match; the report prints all three.

Every attempt is committed as it completes, which is what makes interruption
safe.

## Measurement attribution

For each attempt the runner starts a background telemetry sampler, issues one
streaming request, then stops the sampler. Peak GPU memory, mean GPU
utilization, peak RAM and mean CPU therefore belong to a single generation.
The sampler records a baseline GPU memory reading before the request so the
reported delta excludes memory held by other processes. If a request finishes
faster than one sampling interval, one sample is taken synchronously at stop
so a fast attempt is never left with no telemetry.

See [MEASUREMENT.md](MEASUREMENT.md) for what each field means.

## The viewer

`webapi.py` holds every handler as a plain function of `(Store, params)`, so
the API is tested without opening a socket; `server.py` only routes and
serialises. The viewer is read-only except for manual rubric grades, and
neither module imports the runner — a test asserts that, so the inability to
launch inference from a browser is structural rather than a matter of which
routes happen to exist.

Live progress needs no machinery: attempts are committed as they complete (for
interruption safety), so polling the database shows an executing run advancing.

The rubric queue groups attempts by (task, model) rather than listing them one
by one, because repetitions of one prompt at temperature 0 are usually the same
text. The grouping lives in `webapi.grading_queue`, not in the page, so the
rule that decides whether a group may be graded once — identical responses
after whitespace normalisation — is enforced on the write path as well as
displayed on the read path. A browser cannot grade a divergent group by
omitting a field.

## Internal API for other projects

Nexus, Trykode or anything else should consume Caliper through these stable
surfaces, not by importing internals:

1. **The CLI with machine-readable output.** `caliper report --json`,
   `caliper export`, `caliper inventory --json`. Stable field names, versioned
   by `schema_version` and `caliper_version` in the payload.
2. **`caliper.report.summarize_run(store, run_id)`** → a plain dict of run
   metadata, per-model category scores, performance statistics, telemetry and
   per-profile rankings.
3. **`caliper.report.rank_for_profile(summaries, profile)`** → an explicit
   ranking with its weights and components attached, for workload-specific
   model selection.
4. **The SQLite file itself**, which is a documented, queryable schema.

Automatic model routing is deliberately **not** implemented. The ranking
functions give a consumer what it needs to decide; the decision stays outside.

## What is intentionally absent

- No parallel or concurrent execution (it would destroy telemetry attribution
  on a single GPU, and concurrency belongs in the vLLM comparison instead).
- No LLM judge.
- No model downloading, pulling or pruning.
- No service management.
- No automatic routing.
- No plugin system.
