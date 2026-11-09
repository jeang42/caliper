# Caliper — working agreement for coding agents

Read this before changing anything. It encodes decisions that were paid for
with real measurements; several were learned by shipping a bug and catching it
in the data.

## What this project is

A local LLM benchmarking and evaluation framework. It measures performance and
grades models on a versioned task suite, stores everything in SQLite, and
reports. Ollama is the only implemented backend; vLLM is planned.

The product is **trustworthy numbers**. A fast feature that makes a number
subtly wrong is worse than no feature.

## Non-negotiable invariants

Break these and the output stops being evidence.

1. **Never mix measured, engine-reported and estimated values.** They are
   separate fields. Anything estimated says "estimate" in its name, docstring
   and rendered output.
2. **Unknown stays unknown.** If the engine does not report something, store
   `None`. Do not infer it from a model's name, size or family. The one
   allowed derivation is `is_moe` from the presence of an `expert_count` key.
3. **Never collapse quality into one unexplained score.** Category scores are
   always shown with their sample count. A composite may only appear next to
   the weights that produced it.
4. **Category means must cover the same task set for every model compared.**
   If a validator's behaviour depends on what the model produced, it can
   silently change the denominator. This happened: an empty-output guard
   scored rubric tasks 0 as if automatic, so one model had those attempts
   inside the mean while others had them outside. See `MANUAL_KINDS` and the
   `no_output` grading state.
5. **Automatic and manual scores never merge.** Manual scores live in their
   own table and their own report table.
6. **No LLM judge.** An LLM judge is not ground truth. Rubric tasks are graded
   by a human.
7. **The host comes first.** Preflight before inference; never unload a model
   another process loaded; never stop, start or reconfigure a service; no
   sudo; no model downloads.
8. **Serial execution only.** Concurrency destroys telemetry attribution on a
   single GPU. The only place concurrency is allowed is a future vLLM batching
   comparison that is explicitly about batching.
9. **The viewer cannot start inference.** `webapi.py` and `server.py` must not
   import the runner; a test asserts this. Launching stays in the CLI where
   preflight and serialisation live.
10. **Grading is a pure function of (validator spec, stored response).** This
    is what makes `caliper regrade` possible. Never make a validator depend on
    live engine state.

## Metric definitions you must not quietly change

- **TTFT** is client-observed, from just before the request to the first
  stream chunk carrying generated content. Empty leading chunks do not count.
  `ttft_source` records which field supplied it.
- **Throughput** is derived from the engine's own token counts and durations.
- **Output tokens include a thinking model's reasoning tokens**, because the
  engine counts them as generated. They share the `num_predict` budget with
  the answer, which is why per-task budgets are large.
- **Prompt throughput excludes prompt-cache hits.** A repeated identical
  prompt makes Ollama report a physically impossible rate (observed: 8,168
  tokens in 0.0108 s). See `PROMPT_RATE_PLAUSIBILITY_CEILING`.
- **TTFT and latency exclude attempts that reloaded the model.** Changing
  `num_ctx` between tasks forces Ollama to evict and reload (2.9 s for a
  13 GiB model, 16 s for a 22 GiB one). That is loading cost, not inference
  cost, and it is reported separately as `reloads` / `reload_s`.
- **GPU memory delta** is peak minus a baseline taken just before the request,
  so it excludes other processes' memory.

## Conventions

- Python ≥ 3.11, type hints everywhere, `from __future__ import annotations`.
- **Zero runtime dependencies.** Standard library only. `pytest` and `ruff`
  are dev-only. Do not add a dependency; if you think you need one, say why in
  the PR instead of adding it.
- `ruff check .` and `ruff format .` must both pass. Line length 100.
- Docstrings explain *why*, not *what*. A comment that restates the code is
  noise; a comment recording why a threshold exists is the point.
- No host-specific paths in code. Data location comes from config or
  `CALIPER_HOME`.
- Tests never touch the GPU, the network or a real engine. The Ollama adapter
  is tested against recorded stream fixtures, the runner against
  `MockBackend`.

## Testing expectations

- Every bug fix gets a regression test whose docstring states the symptom that
  was observed. Look at existing ones for the pattern — they read like
  incident notes, deliberately.
- New aggregation logic gets a test that proves the *wrong* answer is no
  longer produced, not just that the function runs.
- When you patch a module that imports a symbol directly
  (`from .hardware import sample_gpus`), patch it in the importing module too.
  Patching only the source module leaves the import bound to the real
  function, and a test will shell out to `nvidia-smi` without telling you.
- Run the whole suite, not just the file you touched: `.venv/bin/python -m pytest`.

## Gotchas that have bitten

- **`ruff format` reflows string literals, so `old_string` edits against
  wrapped text fail silently.** This happened twice and shipped a doc claim
  that was false. Assert the anchor exists before replacing, and verify
  against rendered output.
- **Python's `re` has no `\z`.** Use `\Z`. A suite regex used `\z` and two
  tasks failed to grade for a whole run.
- **Validators that can match the empty string award free marks.** Negative
  constraints and lookaheads do. `grade()` guards empty output centrally — do
  not bypass it.
- **JSON/TOML cannot express tuples**, so the code-execution harness compares
  tuples and lists as equal. A correct `rle_encode` scored 0.2 before that.
- **`sqlite3` connections are not shareable across threads.** The server keeps
  one per thread.
- **Denylists on grading state are wrong.** A warmup carries `graded=None` and
  slipped past a "not automatic" check. Use `GRADEABLE_STATES`.
- **`{{FILLER}}` is substituted identically at every site**, so v1's
  long-context documents repeat themselves and their entry numbers. v1's
  stored prompts are 7000-8300 tokens, not the 4000-5000 its `filler_tokens`
  suggests. Do not read `filler_tokens` as the document size for v1 tasks.
- **`--context` does not change the document.** It sets `num_ctx`, the
  window. Document size lives in the task's `filler_tokens` and moves only
  with `--filler`. A whole stage was described as a length test before this
  was noticed; length was in fact never varied until 0.4.0.
- **An oversized prompt is truncated silently.** Ollama's `num_ctx` covers the
  prompt and the generation together and drops the overflow without a word, so
  a document too large for its window yields a long-context score for a
  document the model never saw. `filler_fit_problems` refuses the run;
  `--force` is the only way past, and the estimate is deliberately generous
  (1.3 prompt tokens per filler token) so it fails safe.
- **A long-context answer can be a property of the generated document.** The
  counting task's expected value is the number of escalated entries in the
  rendered prompt. Change the filler size or the generator and that number can
  move, scoring every model zero for a suite bug; tests recompute it at every
  sweep size.
- **A fixture key must survive filler substitution.** `prompt_keys` falls back
  to the whole prompt, which for a long-context task still contains the
  placeholder; such a key can never match, and the task would be graded on an
  empty response - which looks exactly like a model that answered nothing.
  Placeholder-bearing candidates are now rejected.

## Suite versioning

`suite_version` changes when anything affecting grading changes: a prompt, a
validator, a token budget, a context setting. Comments and documentation notes
do **not** bump it, so purely editorial changes keep results comparable.

Two runs are comparable when engine version, model digest and `suite_version`
all match. `webapi.compare_runs` asserts this rather than assuming it.

Adding tasks: prefer a new suite directory over mutating an existing one. A new
suite does not copy the tasks it keeps. It declares a `suite.toml` manifest
with `extends`, `extends_version` and `replaces`, inherits every category it
does not name, and provides its own files for the ones it does. `v2` is `v1`
with `long_context` replaced.

`extends_version` pins the base suite's version, and loading fails if the base
has moved. That is deliberate: without the pin, editing `v1` would change what
`v2` measures while `v2`'s own `suite_version` stayed put, and stored `v2` runs
would claim comparability they no longer had. When you do change the base,
update the pin and the extending suite's `suite_version` together.

Two further rules the loader enforces, both about keeping a category mean over
one task set: a replaced category must actually be provided, and a suite may
not add tasks to a category it has not replaced.

## How to extend

**A new task**: add it to the right `src/caliper/evals/suites/<suite>/*.toml`,
add a fixture in `evals/fixtures.py` (a test asserts every task has one), bump
`suite_version` in every file of that suite, and give it an output budget that
leaves a thinking model room to reach its answer.

`FIXTURES` is keyed by task id across *all* suites, so a task id reused in two
suites silently shares one canned response. Give a replacement task a new id.

**A long-context task**: use the indexed placeholders `{{FILLER:1}}`,
`{{FILLER:2}}`, ... rather than `{{FILLER}}`. The unindexed form substitutes
the same block at every site, which repeats the prose and the entry numbers;
it is kept only because v1 was measured with it. `filler_tokens` is the
approximate total across all segments. Where the answer is a property of the
generated document - a count, for instance - add a test that recomputes it
from the rendered prompt, so the filler generator cannot drift away from the
expected answer and score every model zero for a suite bug.

**A new validator**: register it in `validators.py`, make it return partial
credit where meaningful, and ask yourself whether it can match empty output.
Add it to the README list.

**A new backend**: implement the seven methods in `backends/base.py`. Nothing
in the runner, storage or reporting should need to change — if it does, that
is a signal the protocol is wrong, not the runner. Document any sampling
parameter whose meaning differs from Ollama's rather than assuming parity.

## Commands

```sh
.venv/bin/python -m pytest          # whole suite, no GPU needed
.venv/bin/ruff check . && .venv/bin/ruff format .
.venv/bin/caliper plan --dry-run --model mock-small:1b   # costs nothing
.venv/bin/caliper run --dry-run --allow-code-execution   # full pipeline, no GPU
./scripts/benchmark-stage.sh list   # the approved real-benchmark stages
```

`caliper run` without `--dry-run` is the only command that performs inference.
Do not run it on your own initiative — it consumes a shared GPU. Ask first.
