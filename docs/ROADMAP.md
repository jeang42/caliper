# Roadmap

What exists today is the Ollama path, end to end. This is what is not built,
roughly in the order it becomes worth building.

## Near term

**Write manual rubric scores back into the database.** Five tasks per run are
hand-graded and the scores currently live outside Caliper (see
[GRADING.md](GRADING.md)). A `caliper grade` command that reads a CSV keyed by
`(run_id, model, task_id, repetition)` and fills in the attempt rows would let
rubric categories appear in reports alongside the automatic ones.

**Cross-run comparison.** `caliper report` covers one run. A `caliper compare
--run A --run B` that refuses to compare mismatched `suite_version`, engine
version or model digest, and states what differs when it does proceed.

**Grow the suite.** Suite `v1` is 25 tasks, which is enough to separate models
coarsely and not enough to resolve small differences. Additions should land as
`v2` rather than mutating `v1`, so existing results stay comparable. Priorities:
more partial-credit coding tasks, multi-turn instruction following, and
long-context tasks at several lengths of the same question.

**Real tool execution.** Tool-use tasks are simulated: tools are declared but
never called. A small in-process tool loop with deterministic fake tools would
turn `tool.multistep_plan` from a written plan into an executed workflow.

## Phase 6: the vLLM comparison

vLLM 0.11.0 is installed on this host in a separate conda environment
(Python 3.11.13, torch 2.8.0+cu128) and is
not running. Nothing about it has been started or reconfigured.

### What has to be true first

1. The Ollama path has produced stable, repeated results.
2. vLLM is started deliberately, by the user, with a known configuration.
3. The weights question below is settled.

### The weights problem

This is the central obstacle, and it needs to be stated plainly rather than
assumed away. Ollama stores GGUF (and MXFP4 for `gpt-oss`). vLLM does not
consume Ollama's model store, and its strongest paths are unquantized or
AWQ/GPTQ/FP8 safetensors. So:

- **Identical weights and quantization are usually not available on both
  engines.** Where they are not, the comparison compares two *deployments*, not
  two engines, and the report must say which differences are in play:
  quantization, KV cache dtype, attention implementation and batching policy.
- Where a model does exist in a form both engines can load at the same
  quantization, that model should carry the headline comparison.
- A comparison across different quantizations is still useful — it is the
  realistic choice a user faces — but it must never be presented as an
  engine-versus-engine result.

### Implementation shape

`caliper/backends/vllm.py` implements the same seven-method `Backend` protocol.
No change to the runner, storage or reporting is expected; if one turns out to
be needed, that is the signal the protocol was wrong, not the runner.

Specific work:

- Map run settings to vLLM's OpenAI-compatible API (`temperature`, `max_tokens`,
  `seed`), and **document the mapping**, because `num_ctx` has no direct
  equivalent — context is a server-launch property in vLLM, not a per-request
  one. That asymmetry has to be recorded with the run, not papered over.
- Derive TTFT the same way: first stream chunk with non-empty content.
- Report the engine's own metrics where they exist, and leave fields `null`
  where they do not, rather than substituting client timings into
  engine-reported columns.
- Residency: vLLM preallocates a VRAM fraction at startup, so
  `size_vram / size` has no meaning. Report what vLLM reports and mark the
  rest unknown.

### What the comparison should cover

Single-request latency, streaming latency, throughput under concurrency, GPU
memory utilization, context scaling, model and quantization compatibility,
operational complexity, CPU offloading behaviour, and scheduling/batching
differences.

Concurrency is the one dimension where the serial-only rule is relaxed, and
only inside a vLLM-versus-vLLM or deployment-versus-deployment measurement
that is explicitly about batching. Caliper should not assume vLLM wins: the
likely outcome is that vLLM leads on concurrent throughput while Ollama leads
on operational simplicity and offloading flexibility, and single-request
latency goes either way depending on configuration.

## Later, if ever

**A second machine.** The Windows RTX 4090 workstation. The storage schema
already records a full machine snapshot per run and the hostname, so results
from several machines can share one database. A remote worker needs only a way
to reach a remote engine URL and a run label identifying the host — no
architectural change. Distributed GPU inference is explicitly out of scope.

**A web dashboard.** Only after the CLI reports stop changing shape.

**Nexurite integration.** Through the stable surfaces in
[ARCHITECTURE.md](ARCHITECTURE.md) — the JSON exports, `summarize_run`,
`rank_for_profile` and the SQLite file — not by importing internals.

**Automatic model routing.** Deliberately not built. The ranking functions give
a consumer the numbers and the weights behind them; choosing a model is a
policy decision that belongs to the consumer, and it should not start making
itself before the benchmark data is trusted.
