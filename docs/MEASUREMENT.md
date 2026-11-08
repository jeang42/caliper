# What Caliper measures

Metrics fall into three kinds, which are never mixed:

- **Engine-reported** — numbers the inference engine itself returns.
- **Client-observed** — numbers Caliper times from the calling side.
- **Estimated** — planning figures, always labelled as estimates.

## Performance metrics

| Metric | Kind | Source and meaning |
|---|---|---|
| `ttft_s` | client-observed | Time from just before the request is sent to the first stream chunk carrying generated content. Empty leading chunks are ignored. |
| `ttft_source` | — | Which stream field supplied TTFT: `stream_first_content_token`, `stream_first_thinking_token`, or `stream_first_tool_call`. |
| `total_latency_s` | client-observed | Wall clock for the whole request, including model load when a load occurs. |
| `engine_total_duration_s` | engine-reported | Ollama `total_duration`, ns → s. |
| `engine_load_duration_s` | engine-reported | Ollama `load_duration`. Near zero on a warm model. |
| `engine_prompt_eval_duration_s` | engine-reported | Prompt processing time only. |
| `engine_eval_duration_s` | engine-reported | Generation time only. |
| `prompt_tokens` | engine-reported | Ollama `prompt_eval_count`. |
| `output_tokens` | engine-reported | Ollama `eval_count`. |
| `output_tokens_per_s` | derived | `output_tokens / engine_eval_duration_s`. |
| `prompt_tokens_per_s` | derived | `prompt_tokens / engine_prompt_eval_duration_s`. |
| `was_loaded_before` | engine-reported | Whether the model was already resident, from `/api/ps`. Distinguishes warm from cold. |

Throughput is computed from the engine's own token counts and its own
durations, which is the fairest accounting available for Ollama. TTFT is
client-observed because it is a property of the request as experienced by a
caller. The two are not interchangeable and are reported separately.

Why TTFT uses the first *generated* token and not the first HTTP chunk: Ollama
sends stream objects with empty content before generation begins, so timing
the first chunk would under-report TTFT. For a thinking model the first
`thinking` delta counts, because that is when the model starts producing
tokens — `ttft_source` records which field it came from, so thinking and
non-thinking models are never silently compared on different definitions.

## Telemetry

Sampled by a background thread for the duration of each request via
`nvidia-smi` and `/proc`. The interval is configurable
(`run.telemetry_interval_s`, default 0.25 s) and recorded with the run.

| Metric | Meaning |
|---|---|
| `gpu_mem_used_peak_mib` | Highest GPU memory in use during the request, device-wide. |
| `baseline_gpu_mem_used_mib` | GPU memory in use just before the request. |
| `gpu_mem_delta_peak_mib` | Peak minus baseline — the part attributable to this request rather than to other processes. |
| `gpu_util_mean_pct`, `gpu_util_peak_pct` | GPU utilization over the window. |
| `gpu_power_mean_w`, `gpu_temp_peak_c` | Power and thermals. |
| `ram_used_peak_mib` | `MemTotal - MemAvailable`, peak over the window, host-wide. |
| `ram_available_min_mib` | Lowest available RAM over the window. |
| `cpu_util_mean_pct` | From `/proc/stat` deltas, host-wide. |

Sampling means a memory spike shorter than the interval can be missed. GPU and
RAM figures are device- and host-wide, not per-process; the delta column is
the attributable figure.

## GPU versus CPU residency

From Ollama `/api/ps`, which reports a loaded model's total size and the part
resident in VRAM:

- `gpu_fraction` = `size_vram / size`
- `fully_on_gpu` = `gpu_fraction >= 0.999`
- The remainder is CPU/RAM offload.

This is the engine's own report, captured after a model's attempts complete.
The residency *predictions* in `caliper inventory` are a different thing: they
are estimates from on-disk size plus a 15% overhead allowance, labelled as
such, and only used for planning which models to try.

## Quality scores

Each attempt gets a `score` in 0.0–1.0 from one validator, plus `graded`:

- `automatic` — a deterministic validator produced the score.
- `manual_rubric` — a rubric task: no score, output stored for hand grading.
- `skipped` — a validator that was disabled, for example `python_cases`
  without `--allow-code-execution`.
- `error` — the validator itself raised. Excluded from means and counted
  separately in the report, so a broken validator cannot quietly inflate or
  deflate a category.

Category scores are the mean of automatic scores in that category, with the
sample count printed. Several validators give partial credit (`contains_all`,
`regex_all`, `python_cases`, `tool_call`, `json_schema`), so a score between 0
and 1 on a single attempt is meaningful, not a rounding artefact.

## Uncertainty

Aggregates report `n`, mean, median, sample standard deviation, min and max.
The report prints `mean ±stdev` and the sample count everywhere. With a suite
of this size, category differences smaller than roughly one task's worth of
score should be treated as noise.

## Thinking models and the output budget

A thinking model's reasoning tokens are counted by the engine as generated
output and draw on the same `num_predict` budget as the answer. Three
consequences, all observed on `gpt-oss:20b` on 2026-11-08:

1. **`output_tokens` and `output_tokens_per_s` include reasoning tokens.**
   That is the correct accounting — they really were generated — but it means
   tokens/s is not a measure of answer production rate for a thinking model.
2. **A budget sized for a non-thinking model truncates the answer away.** The
   suite's per-task budgets were raised in suite version 1.1.0 for this reason.
3. **Raising the budget does not always help.** On two tight negative-constraint
   tasks, `gpt-oss:20b` produced 18,300 and 12,817 characters of reasoning at a
   6,144-token budget and still emitted no answer. That is a failure to
   terminate, not a measurement artefact, and it scores zero.

An attempt that stops at the limit is recorded with `done_reason = "length"`
and `truncated = 1`, and the report prints a truncation count per model. An
attempt with no visible output is scored zero regardless of validator, because
several validators — negative constraints and lookahead patterns in particular
— match the empty string and would otherwise award full marks for an answer
that was never given.

## Suite versioning policy

`suite_version` changes when anything that affects grading changes: a prompt, a
validator, a token budget or a context setting. Comments and documentation
notes do not bump it, so results stay comparable across purely editorial
changes. Two runs are comparable when engine version, model digest and
`suite_version` all match; the report prints all three.

## Known limitations

- Engine-reported durations come from Ollama's own instrumentation and are
  taken at face value.
- Tool-use tasks are simulated: tools are declared to the model but never
  executed, so only selection and argument correctness are measured.
- Rubric tasks carry no automatic score, so categories containing them are
  partially measured until hand grading is done.
- `temperature = 0` and a fixed `seed` reduce but do not guarantee
  determinism.
- Sampling parameters are not guaranteed to mean the same thing across
  engines. The vLLM adapter must document its mapping rather than assume
  parity.
