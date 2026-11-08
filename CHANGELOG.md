# Changelog

## 0.3.1 — 2026-11-08

Suite `v2` measured, in stages 8a and 8b. **The result is negative: `v2` is
saturated too.** Full figures in [docs/RESULTS.md](docs/RESULTS.md).

- Four of five models score 1.00 on all eight new long-context tasks, at
  `num_ctx` 16384 and again at 32768. Harder task structure was the wrong
  hypothesis: multi-hop chains in reverse order, whole-document aggregation,
  supersession, near-miss distractors, absent facts, order reconstruction and
  scattered arithmetic are all easy for these models at an 8,000-token
  document.
- The only model below 1.00, `glm-4.7-flash`, fails for a reason that is not
  about long context: it does not stop reasoning. On
  `longctx.embedded_instruction_deep` it produced 45,995 characters of
  thinking and no answer, three times out of three; a diagnostic run tripled
  the budget to 12,288 tokens and it produced exactly 45,995 characters again
  and still no answer. A genuine failure to terminate, not a budget artifact.
- **Validator fix, suite version 2.0.1.** `qwen3-coder:30b` answered
  `RESULT: 13 incidents.` and scored 0.50, because the trailing full stop fell
  outside the pattern's `incidents\s*$` anchor — a correct answer marked a
  long-context failure over punctuation, which is exactly what the task's own
  validator note said this category must not do. Optional trailing punctuation
  is now allowed, both runs were re-graded from stored responses (5 attempts
  changed, 150 reproduced identically, no measurement altered), and
  `qwen3-coder:30b`'s category mean moved 0.94 to 1.00.
- **A correction to an earlier claim in this project.** Stage 8b was described
  as testing document length. It does not: `--context` raises `num_ctx`, which
  is the window, while the document comes from each task's `filler_tokens` and
  stayed at roughly 9,200 prompt tokens. Document length has never been varied
  in any run — v1, v2 or stage 5 — and is now the top evaluation priority, with
  a `--filler` override the cheapest way to get at it.
- 245 tests.

## 0.3.0 — 2026-11-08

Suite `v2`: a harder long-context category. **Nothing has been measured
against it yet** — this release is task authoring and the machinery to carry
it, not results.

- **Suite `v2`**, 30 tasks: `v1`'s three long-context tasks replaced by eight
  that require multi-hop chains stated in reverse order, aggregation over the
  whole document, resolving two supersessions, rejecting three near-identical
  distractors, reporting an absent fact instead of inventing one,
  reconstructing an order from scattered dates, arithmetic over far-apart
  figures, and obeying a formatting rule stated once before the document and
  never recalled. Each has a specific wrong answer a partial read produces,
  and a test asserts the validator rejects it.
- **Why only this category.** With five models measured, five of six
  categories discriminate with spreads of 0.33-0.56. `long_context` was the
  only saturated one: 1.00 for every model at every context length in stage 5,
  including the 29%-offloaded case. So this is a one-category replacement, not
  a suite rewrite.
- **Suites can extend other suites.** A `suite.toml` manifest names the base,
  the base version it was composed against, and the categories it replaces;
  everything else is inherited rather than copied, so `v2` runs report all six
  categories and the five it did not touch stay directly comparable to the
  stored `v1` runs. The version pin makes the one real hazard loud: edit `v1`
  and loading `v2` fails, instead of silently changing what `v2` measures
  while `v2`'s own `suite_version` stays at 2.0.0. The loader also refuses to
  replace a category without providing it, and refuses to add tasks to a
  category that was not replaced — both would put a category mean over
  different task sets in different runs.
- **Indexed filler placeholders** `{{FILLER:1}}`, `{{FILLER:2}}`, ... give
  each site distinct prose with entry numbering continuing across segments,
  and `filler_tokens` becomes the approximate total rather than the per-site
  size. The unindexed `{{FILLER}}` substitutes one identical block everywhere;
  it is unchanged and verified to render v1's prompts byte-for-byte, because
  v1's results were measured with it.
- **A correction this exposed.** v1's long-context documents are about twice
  the size its `filler_tokens` settings suggest, because two of its tasks use
  `{{FILLER}}` twice and got the same block both times: its stored runs
  average 7064-8280 prompt tokens, not 4000-5000. v2's documents are held at
  that same size deliberately, so a v1-to-v2 score change is attributable to
  task difficulty rather than to length.
- **`prompt_keys` no longer returns a key containing a placeholder.** It falls
  back to the whole prompt when head and tail are not unique, and for a
  long-context task that prompt still holds `{{FILLER}}` — a key that can
  never match a rendered prompt, so the fixture would be skipped and the task
  graded on an empty response, which is indistinguishable from a model that
  answered nothing.
- **Stages `8a` and `8b`** added to `scripts/benchmark-stage.sh` for the v2
  long-context runs. Neither has been run.
- 243 tests.

## 0.2.1 — 2026-11-08

Benchmark stages 4 and 5 executed. Five models and a full context-scaling
curve are now measured; see [docs/RESULTS.md](docs/RESULTS.md).

- **Prompt-cache detection is now structural.** The rate ceiling missed real
  cache hits (9,433 tokens in 0.108 s = 87,000 tok/s passed under a 100,000
  ceiling). Only the first repetition of a task has a cold prompt cache, so
  later ones are cache hits by construction and no threshold is needed.
  Corrected prompt throughput for the five models: 3,549 / 3,196 / 1,937 /
  2,222 / 800 tok/s, replacing figures inflated two- to ninefold. The metric
  now shows the dense model paying 4x per prompt token, which the inflated
  numbers hid.
- **`--wait` on `preflight` and `run`**, so a queued stage waits out a busy
  GPU instead of aborting. Stage 5b was blocked by the tail of stage 5a's own
  activity; waiting is the right response to a busy machine, lowering the
  threshold is not. The stage script now waits `CALIPER_WAIT` seconds
  (default 180).
- **Cost estimates come from measured history** per model instead of one
  global 25 s/attempt guess. Measured values span 1.3 to 11.8 s/attempt, a 9x
  range. The output names which models were measured and which assumed.
- The context-grouped task ordering added in 0.1.2 is confirmed working:
  reloads fell to the single unavoidable context switch.
- Correction to an earlier claim: the suite does **not** have a general
  quality ceiling. With five models measured, five of six categories
  discriminate with spreads of 0.33-0.56. Only `long_context` is saturated.

## 0.2.0 — 2026-11-08

Web viewer, manual rubric grading, and re-grading without inference.

- **`caliper serve`** starts a single-page viewer over the same SQLite
  database: runs, category scores, performance, memory and loading behaviour,
  an attempt browser showing every raw response, thinking text and tool call,
  and cross-run comparison that states whether runs are actually comparable
  rather than lining up numbers that are not. Live GPU telemetry and
  in-flight run progress come free, because attempts are committed as they
  complete.
- **Rubric grading queue.** Each response beside its criteria, a checkbox per
  criterion. Scores live in a `manual_scores` table, keyed by attempt and
  separate from automatic scores, so `caliper regrade` cannot destroy them;
  reports show them as their own table and never merge them into automatic
  means.
- The viewer cannot start, stop or configure a benchmark. `webapi.py` and
  `server.py` do not import the runner and a test asserts they never will.
- `serve` binds loopback and refuses any other address without
  `--allow-remote`, because it has no authentication and serves telemetry,
  inventory and raw model output.
- Fixed a hole in the grading guard: it denied `graded == "automatic"`, but a
  warmup carries `graded = None` and slipped through, so a warmup could be
  given a score. Replaced with an allowlist (`GRADEABLE_STATES`).
- Zero new runtime dependencies: `http.server` and one hand-written page.
- New docs: `docs/API.md` (HTTP and Python APIs for other programs and
  agents) and `CLAUDE.md` (working agreement for coding agents — invariants,
  metric definitions, and the mistakes already made once).
- Tests: 203 passing, including socket-level server tests, path-traversal
  refusal, and the grading write path.

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
