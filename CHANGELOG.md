# Changelog

## 0.1.2 — 2026-11-08

Stage 3 comparison run (3 models, 228 attempts, 22 min, zero request
failures). It exposed two defects that made the headline quality column
unfair, plus a measurement artefact worth more than the run itself.

- **Rubric tasks with no output no longer corrupt category means.** The
  empty-output guard was scoring them 0 as if automatic, so a model that
  failed to answer had those attempts inside the automatic mean while other
  models' attempts at the same task stayed outside it — category means covered
  a different task set per model. They now land in a `no_output` bucket that is
  counted and reported separately. This moved `qwen3.5:35b` coding from a
  misleading 0.35 to 0.44 over the same tasks as every other model.
- **Model reloads were mislabelled as warm load time and were inflating
  latency.** Changing `num_ctx` between tasks forces Ollama to evict and
  reload: 2.9 s for `gpt-oss:20b`, 16.0 s for `qwen3.5:35b`. That cost sat
  inside the affected attempt's latency, which is why `qwen3.5:35b` appeared
  to have a 1.20 s TTFT. Reloads are now counted and costed in their own
  columns and excluded from TTFT and latency, which are steady-state figures.
  Corrected TTFT: 0.09, 0.09, 0.11 s.
- **Tasks are now ordered so that all tasks sharing a context length run
  consecutively**, reducing reloads to at most one per context group.
- **New `caliper regrade`** re-scores a stored run with the current validators
  and no inference, which is what storing every raw response was for. Applied
  to stages 2 and 3: 6 attempts corrected, 219 reproduced identically, and no
  measurement altered.
- Warm load time is now reported as the measured ~0.002 s rather than "n/a".

## 0.1.1 — 2026-11-08

First real measurements taken (stage 1 smoke test, `gpt-oss:20b`). The run
exposed three defects, all fixed here, plus one genuine model finding.

- **Empty output no longer scores full marks.** Negative-constraint and
  lookahead patterns match the empty string, so a response that was never
  emitted used to pass. Any attempt with no visible output now scores zero,
  with the reason recorded. This changed the stage 1 instruction-following
  score from a false 0.88 to an honest 0.50.
- **Truncation is tracked.** `done_reason` and `truncated` are stored per
  attempt (schema version 2, with migration) and surfaced in the report.
- **Cold load is reported.** Model load time and the VRAM the weights occupy
  come from the warmup attempt, which is excluded from all other aggregates,
  so cold-start cost was previously invisible.
- **Profiles no longer invent composites.** A workload profile that weights
  categories a run did not measure drops them and says so, instead of scoring
  them zero.
- `--max-tokens` and `--context` are now true overrides that beat a task's own
  budget; previously they were silently ignored for every task in the suite.
- Fixed `describe_model` dropping a model's on-disk size.
- Suite bumped to 1.1.0: per-task output budgets raised so a thinking model's
  reasoning tokens do not consume the answer. Versioning policy documented.
- Added `scripts/benchmark-stage.sh` to run the approved stages end to end.

## 0.1.0 — 2026-11-07

Initial foundation. Nothing has been benchmarked against a live model yet; the
only run executed so far is a dry run against the mock backend.

- Environment and model inventory (`caliper inventory`, `caliper models`),
  including MoE detection from engine metadata and estimated GPU residency.
- Preflight resource safety checks with configurable thresholds
  (`caliper preflight`).
- Ollama backend adapter: streaming generation, time to first generated token,
  engine timing capture, GPU/CPU residency reporting.
- Backend protocol plus a deterministic mock backend.
- Hardware telemetry sampler (GPU, RAM, CPU) attributed per request.
- Serial benchmark runner with warmups, repetitions, interruption safety and
  per-attempt persistence.
- Evaluation suite `v1` (version 1.0.1): 25 original tasks across coding,
  reasoning, instruction following, tool use, long context and agent
  behaviour.
- Nine validators including a sandboxed-ish Python execution validator (opt-in)
  and a manual rubric passthrough.
- SQLite storage, Markdown reports with per-category scores and workload
  profiles, JSON and CSV export.
- Canned response fixtures so a dry run exercises every validator path.
