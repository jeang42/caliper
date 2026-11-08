# Roadmap

What exists today is the Ollama path, end to end. This is what is not built,
roughly in the order it becomes worth building.

## Near term

**A CLI cross-run comparison.** `/api/compare` and the viewer do this already;
`caliper compare --run A --run B` would make it available without a browser.

**Harder long-context tasks — tried, and the hypothesis was wrong.** Suite
`v2` replaced v1's three saturated long-context tasks with eight that need
multi-hop chains in reverse order, whole-document aggregation, supersession,
near-miss distractors, an absent fact, an order to reconstruct and arithmetic
over scattered figures. Measured in stage 8a: four of five models score 1.00 on
all eight, and the fifth fails only by never terminating. See
[docs/RESULTS.md](RESULTS.md). Task *structure* is not what these models find
hard at an 8,000-token document.

**The one axis still untested: document length.** `filler_tokens` is a
per-task TOML field with no command-line override, so every long-context
measurement to date — v1, v2, and all of stage 5 — used documents of roughly
8,000-10,000 tokens. Stage 5 and stage 8b varied `num_ctx`, which is the
window, not the document. A `--filler` override, or a `v2` variant sized at
30,000 and 60,000 tokens, would answer whether these models degrade with length
and would cost about the same as stage 8a. This is now the highest-value
evaluation work outstanding, ahead of writing more tasks of the same kind.

**Camouflaged needles, as a possible `v3`.** `v2`'s needles use entry numbers
in the 9000s, well above the filler's own numbering, so they are findable by
their shape. That makes `v2` a test of retrieval and reasoning over a long
document, not of search under camouflage. Blending needles into the filler's
numbering would add that axis, and would need the filler generator to place
them rather than the task author. Worth doing after the length axis, since
length is the cheaper experiment and the more likely explanation.

**Reasoning-budget exhaustion deserves its own measurement.** Two separate
models have now been observed spending an entire output budget on reasoning
and emitting no answer: `gpt-oss:20b` on v1's negative-constraint instruction
tasks, and `glm-4.7-flash` on `longctx.embedded_instruction_deep`, which still
produced exactly 45,995 characters of thinking and an empty answer when its
budget was tripled to 12,288 tokens. This is a real operational property — the
request costs full price and returns nothing — and it is currently visible only
as a zero in a category mean plus a truncation count. It belongs in the report
as a named failure mode.

**Grow the other categories.** The remaining priorities are more
partial-credit coding tasks, multi-turn instruction following, and the same
long-context question asked at several document lengths. The suite-extension
mechanism `v2` introduced makes each of these a new suite that replaces one
category and inherits the rest, so existing results stay comparable.

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

**Nexurite integration.** Through the stable surfaces in
[ARCHITECTURE.md](ARCHITECTURE.md) — the JSON exports, `summarize_run`,
`rank_for_profile` and the SQLite file — not by importing internals.

**Automatic model routing.** Deliberately not built. The ranking functions give
a consumer the numbers and the weights behind them; choosing a model is a
policy decision that belongs to the consumer, and it should not start making
itself before the benchmark data is trusted.
