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

**Document length was the missing axis, and it is now measured.** Stages 9a
and 9b took the documents from ~9,000 to ~17,000 and ~40,000 tokens with
`--filler`. Both models tested at ~40,000 fell to 0.88, and every loss came
from `longctx.event_ordering` — correct retrieval followed by a wrong sort.
See [RESULTS.md](RESULTS.md). The category discriminates now, but through one
task out of eight, which is a thin instrument.

**What that suggests next, in order.**

1. **More whole-document aggregation tasks.** The one task that fails at
   length is the one that must combine four facts scattered through the
   document and order them. The seven that hold are single-needle retrieval,
   supersession, distractor rejection, absence, counting and arithmetic. A
   `v3` weighted toward cross-document aggregation — ordering, grouping,
   reconciling two lists, finding the one inconsistency — would likely
   discriminate at ~17,000 tokens rather than needing 40,000, which matters
   because only two models here fit a 64k window.
2. **A length curve, not two points.** 9a and 9b are two sizes. The shape
   between them is unknown: the drop could be a cliff somewhere near 24,000
   tokens or a slope. `--filler 24000 --context 49152` on `gpt-oss:20b` and
   `qwen3-coder:30b` would cost about ten minutes and place the knee.
3. **A per-model prompt-token ratio.** The fit guard assumes 1.3 prompt tokens
   per filler token for every model. Measured ratios are 1.013 for
   `gpt-oss:20b`, 1.051 for `glm-4.7-flash` and 1.14 for the qwen3 tokenizers,
   so the guard refused a legitimate glm diagnostic that `--force` then ran
   safely. Storing a measured ratio per model and using it when one exists
   would remove the false refusal without weakening the check.
4. **Reasoning budget as a declared model property.** `glm-4.7-flash` needs
   3x the output budget other models finish inside, and recovers 4 of 5 failed
   tasks when given it. That is a measurable per-model number, not a suite
   constant, and it belongs in the model record rather than being rediscovered
   by every run that scores glm zero.

The ceiling is the hardware, not the suite. Measured at `num_ctx` 65536,
`gpt-oss:20b` peaks at 19,447 MiB and stays fully resident, while
`qwen3-coder:30b` peaks at 31,195 MiB of 32,760 and the engine offloads 29% of
it — output throughput falls to a quarter and prompt processing to an eighth,
with a 24-second TTFT. The three 20+ GiB models cannot reach a 64k window at
all without offload, so anything past ~40,000-token documents on this host is
a `gpt-oss:20b`-only experiment. Beyond roughly 260,000 filler tokens the
filler's own entry numbering would reach the 9000s and collide with the
needles; a test asserts the margin, but a future host with more VRAM would
need the needles renumbered.

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

vLLM 0.11.0 is installed on the benchmark host in a separate conda
environment (Python 3.11.13, torch 2.8.0+cu128) and is not running. Nothing about it has been started or reconfigured.

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
