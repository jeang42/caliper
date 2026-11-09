# Measured results

All figures here were measured on Ada (`ub2404srv`, RTX 5000 Ada, 32760 MiB
VRAM, Ollama 0.33.2) between 2026-11-07 and 2026-11-08, suite `v1` version
`1.2.0`, temperature 0, seed 42, 3 repetitions unless stated. Nothing in this
file is an estimate.

Regenerate any of it from the stored runs:

```sh
caliper runs
caliper report --run <id>
caliper serve          # same data in the browser
```

## Models benchmarked

| Model | Size | Arch | MoE | Context capability |
|---|---|---|---|---|
| `gpt-oss:20b` | 12.85 GiB | gptoss | 4/32 experts | 131072 |
| `qwen3-coder:30b` | 17.28 GiB | qwen3moe | 8/128 experts | 262144 |
| `qwen3.5:35b` | 22.23 GiB | qwen35moe | 8/256 experts | 262144 |
| `glm-4.7-flash:latest` | 17.71 GiB | glm4moelite | 4/64 experts | 202752 |
| `qwen3.8:27b` | 16.52 GiB | qwen35 | dense | 262144 |

## Quality by category

Mean automatic score, 0.0–1.0, at `num_ctx` 8192. Rubric tasks are excluded —
45 of them are still awaiting manual grading, so these cover the automatic
tasks only.

| Model | agent | coding | instruction | long ctx | reasoning | tool use |
|---|---|---|---|---|---|---|
| `qwen3.8:27b` | **1.00** | **1.00** | **0.94** | 1.00 | **1.00** | **1.00** |
| `gpt-oss:20b` | **1.00** | **1.00** | 0.50 | 1.00 | **1.00** | **1.00** |
| `qwen3-coder:30b` | 0.50 | **1.00** | 0.69 | 1.00 | 0.92 | 0.89 |
| `glm-4.7-flash` | 0.50 | 0.96 | 0.92 | 1.00 | 0.50 | 0.89 |
| `qwen3.5:35b` | 0.50 | 0.44 | 0.50 | 1.00 | 0.75 | 0.67 |

`long_context` is saturated: every model scores 1.00, so that category
currently carries no information. The other five discriminate, with spreads of
0.33 to 0.56.

## Suite v2: the harder long-context tasks did not help

Measured 2026-11-08. Stage 8a, run `810b3fa6c8924e70`, 16.1 minutes, 125
attempts, 0 failures, `num_ctx` 16384, 3 repetitions. Stage 8b, run
`7f3d373f024a4fcf`, 1.5 minutes, 34 attempts, `num_ctx` 32768, 2 repetitions.
Both re-graded from their stored responses under suite version 2.0.1; see the
correction below.

Suite `v2` replaces v1's three long-context tasks with eight that require
multi-hop chains stated in reverse order, aggregation over the whole document,
resolving two supersessions, rejecting near-identical distractors, reporting an
absent fact, reconstructing an order, arithmetic over scattered figures, and a
formatting rule stated once before the document. Per-task mean score:

| Task | gpt-oss:20b | qwen3-coder:30b | qwen3.5:35b | glm-4.7-flash | qwen3.8:27b |
|---|---|---|---|---|---|
| `multi_hop_chain` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `resolve_supersession` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `count_markers` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `distractor_precision` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `event_ordering` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `sum_scattered` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `absent_fact` | 1.00 | 1.00 | 1.00 | **0.67** | 1.00 |
| `embedded_instruction_deep` | 1.00 | 1.00 | 1.00 | **0.00** | 1.00 |
| **category mean** | 1.00 | 1.00 | 1.00 | **0.83** | 1.00 |

**The conclusion is negative: v2 is saturated too.** Four of the five models
score 1.00 on all eight tasks. Harder task *structure* was the wrong
hypothesis; these models solve multi-hop retrieval, whole-document
aggregation, supersession and near-miss discrimination at this document size
without difficulty.

The single model below 1.00 fails for a reason that is not about long context.
`glm-4.7-flash` does not stop reasoning: on `embedded_instruction_deep` it
produced 45,995 characters of thinking and **no answer at all**, three times
out of three, identically. A diagnostic run (`9967b2fe5d5d4689`) raised the
budget from 4,096 to 12,288 tokens and it again produced exactly 45,995
characters of thinking and an empty answer, stopping at the limit. So this is a
genuine failure to terminate, not a budget artifact — the same behaviour
`gpt-oss:20b` shows on v1's negative-constraint instruction tasks. Its 0.67 on
`absent_fact` is one further truncation of the same kind.

### Context length is not the explanation; document length is

Stage 8b re-ran the same eight tasks at `num_ctx` 32768. Every score stayed
1.00 for both models. This is consistent with stage 5: a larger window costs
nothing in quality.

It is **not** a test of longer documents, and an earlier note in this project
said it was. `--context` raises `num_ctx` only; the documents are generated
from each task's `filler_tokens` and stayed at roughly 9,200 prompt tokens in
both runs, so nothing in stages 5 or 8 speaks to document length. Stages 9a
and 9b, below, do.

### Document length is what breaks them

Measured 2026-11-09. Stage 9a, run `a81725976273479e`, 17.3 minutes, 85
attempts, 0 failures, `--filler 16000 --context 32768`, 2 repetitions, five
models. Stage 9b, run `65a2911f4f2f4b6f`, 34 attempts, 0 failures,
`--filler 40000 --context 65536`, 2 repetitions, the two models whose weights
leave room for a 64k window. Stage 9c, run `bb9d5dba62e54ad6`, 34 attempts, 0
failures, `--filler 24000 --context 40960`, 2 repetitions, the same two models,
run to place the knee between 9a and 9b. All under suite version 2.0.1.

Stages 9a and 9b varied the document itself with `--filler`. This is the one
axis that produced a failure in a healthy model.

| Model | ~9k doc (8a) | ~17-19k doc (9a) | ~24-27k doc (9c) | ~40-45k doc (9b) |
|---|---|---|---|---|
| `gpt-oss:20b` | 1.00 | 1.00 | 1.00 | **0.88** |
| `qwen3-coder:30b` | 1.00 | **0.94** | **0.88** | **0.88** |
| `qwen3.5:35b` | 1.00 | 1.00 | not run | not run (no 64k window) |
| `qwen3.8:27b` | 1.00 | 1.00 | not run | not run (no 64k window) |
| `glm-4.7-flash` | 0.83 | 0.56 | not run | not run (no 64k window) |

**The two models degrade at different lengths, and that is the useful part.**
`qwen3-coder:30b` slides: perfect at ~9,500 tokens, then one of two attempts
at ~18,000, then both attempts failing from ~27,000 onward. `gpt-oss:20b`
holds 1.00 all the way to ~24,200 and breaks somewhere between there and
~40,000. So the category now separates two models that were indistinguishable
at every length measured before stage 9 — which is what a benchmark is for.
It is a slope for one model and a later cliff for the other, not a single
threshold.

Every loss in that table comes from **one task**, `longctx.event_ordering`,
which asks for four scattered events in chronological order. At ~40,000 tokens
both models fail it on every attempt (4 of 4), and the failure is not
retrieval. `qwen3-coder:30b` quotes all four entries correctly and then sorts
them wrong, identically on both repetitions:

```
- KAPPA: day 59
- THETA: day 212
- OMEGA: day 134
- SIGMA: day 341
Order: KAPPA, THETA, OMEGA, SIGMA
```

The correct order is KAPPA, OMEGA, THETA, SIGMA. The model contradicts its own
list two lines above it. `gpt-oss:20b` fails the same task two ways: one
attempt mis-sorts (`KAPPA, SIGMA, OMEGA, THETA`), the other spends its whole
budget on reasoning and emits nothing. Retrieval over 40,000 tokens succeeds;
ordering four numbers it has already written down fails.

The seven other v2 tasks stay at 1.00 for both models at every size measured.
So the degradation is specific, not general: aggregation across the whole
document is what length attacks first, while single-needle retrieval,
supersession, distractor rejection, absence and arithmetic hold.

Performance degrades at the same time, and for `qwen3-coder:30b` it falls off
a cliff because a 64k window no longer fits:

| Model | stage | prompt tok | out tok/s | prompt tok/s | TTFT s | residency |
|---|---|---|---|---|---|---|
| `gpt-oss:20b` | 8a | 8,498 | 116.6 | 9,109 | 0.38 | 100% GPU |
| `gpt-oss:20b` | 9a | 16,202 | 111.4 | 8,882 | 0.96 | 100% GPU |
| `gpt-oss:20b` | 9c | 24,200 | 107.0 | 8,404 | 1.48 | 100% GPU |
| `gpt-oss:20b` | 9b | 40,170 | 99.0 | 6,935 | 2.88 | 100% GPU |
| `qwen3-coder:30b` | 8a | 9,506 | 137.5 | 7,215 | 0.48 | 100% GPU |
| `qwen3-coder:30b` | 9a | 18,190 | 111.9 | 5,089 | 1.84 | 100% GPU |
| `qwen3-coder:30b` | 9c | 27,206 | 73.3 | 2,427 | 5.33 | 91% GPU |
| `qwen3-coder:30b` | 9b | 45,200 | 35.8 | 879 | 23.99 | **71% GPU** |

`qwen3-coder:30b` runs out of VRAM before it runs out of accuracy. It peaks at
31,257 MiB of 32,760 at a 40,960 window (9% offloaded, 73.3 tok/s) and 31,195
MiB at 65,536 (29% offloaded, 35.8 tok/s, 24-second TTFT). Stage 5 measured
quality unchanged under 29% offload, so the scores above are comparable; the
throughput is not. On this host, documents past ~24,000 tokens are a
`gpt-oss:20b` job on speed grounds alone, and past ~40,000 on quality grounds
too.

### glm-4.7-flash needs three times the output budget, not a better memory

glm's 0.56 at ~17,000 tokens looked like a long-context collapse. It is mostly
a budget artifact. Every one of its seven failures was `done_reason=length`
with an empty answer and 13,900-22,000 characters of thinking. Re-running the
five failing tasks with `--max-tokens 12288` (3x the task budgets, same
documents, same `num_ctx`) recovered four of them:

| Task | 9a (4096-5120 budget) | diagnostic (12288) |
|---|---|---|
| `resolve_supersession` | 1.00 | 1.00 |
| `sum_scattered` | 1.00 | 1.00 |
| `absent_fact` | 0.50 | **1.00** (3,799 output tokens) |
| `embedded_instruction_deep` | 0.00 | **1.00** (7,234 output tokens) |
| `multi_hop_chain` | 0.00 | 0.00 (12,288 exhausted, 49,184 chars of thinking) |

So glm is a model whose reasoning scales with document length faster than its
answer arrives, and one task where it genuinely never terminates. Judged at a
budget it can use, its long-context retrieval is sound. Judged at the budget
every other model finishes comfortably inside, it scores 0.56. Both statements
are true and the second is the one that matters operationally: it costs full
price and returns nothing.

This diagnostic was run with `--force`, because the fit guard's generic
estimate of 1.3 prompt tokens per filler token is too pessimistic for glm's
tokenizer (1.051 as measured in stage 9a; 1.057 is the maximum observed
across every run since), and `num_ctx` was held at 9a's 32768 so the two runs
stay comparable.

### A validator correction

`qwen3-coder:30b` answered `RESULT: 13 incidents.` three times out of three and
was first scored 0.50, because the trailing full stop fell outside the
pattern's `incidents\s*$` anchor. That is a correct answer marked a
long-context failure over punctuation — the precise failure mode the task's own
validator note said this category must avoid. Suite version 2.0.1 allows
optional trailing punctuation, and both runs were re-graded from their stored
responses: 5 attempts changed, 150 reproduced identically, no measurement
altered. `qwen3-coder:30b`'s category mean went 0.94 to 1.00, which removed the
last apparent discrimination other than glm's truncations.

Both runs record `suite_version` 2.0.0, the version that graded them first,
while their scores now come from 2.0.1. `/api/compare` will therefore call them
not comparable with any future 2.0.1 run. That is the conservative direction,
but it is a provenance wrinkle to know about.

## Performance

| Model | output tok/s | prompt tok/s | TTFT s | latency s | truncated |
|---|---|---|---|---|---|
| `qwen3-coder:30b` | **164.8 ±10.6** | 3196 ±2014 | 0.09 ±0.23 | **1.0 ±1.0** | **0/75** |
| `qwen3.5:35b` | 137.8 ±3.6 | 1937 ±1419 | 0.11 ±0.27 | 10.3 ±4.8 | 30/75 |
| `glm-4.7-flash` | 125.5 ±5.0 | 2222 ±1300 | 0.09 ±0.34 | 7.7 ±4.3 | 7/75 |
| `gpt-oss:20b` | 122.3 ±2.4 | **3549 ±2161** | 0.09 ±0.13 | 4.2 ±4.2 | 6/75 |
| `qwen3.8:27b` | 76.7 ±8.2 | 800 ±417 | 0.28 ±0.79 | 8.1 ±6.5 | 3/75 |

Every model completed every request: 375 of 375 attempts across all runs
returned without error.

**MoE versus dense, measured.** The dense `qwen3.8:27b` processes prompts about
four times slower than the MoE models (800 vs 1937–3549 tok/s) and generates
about half as fast, while scoring highest on quality. That is the trade-off
stated as measurement rather than assumption.

## Loading and residency at `num_ctx` 8192

| Model | cold load s | weights to GPU | warm load s | reloads | residency |
|---|---|---|---|---|---|
| `gpt-oss:20b` | 2.32 | 13496 MiB | 0.002 | 2/75 | 100% GPU |
| `qwen3-coder:30b` | 11.30 | 21162 MiB | 0.002 | 2/75 | 100% GPU |
| `qwen3.5:35b` | 11.62 | 23184 MiB | 0.002 | 5/75 | 100% GPU |
| `glm-4.7-flash` | 16.55 | 20130 MiB | 0.002 | 3/75 | 100% GPU |
| `qwen3.8:27b` | 12.89 | 18690 MiB | 0.002 | 1/75 | 100% GPU |

A *reload* is a measured attempt that had to bring the model back in. Changing
`num_ctx` between tasks forces one; `qwen3.5:35b`'s extra reloads are memory
pressure at 23 GiB of weights.

## Context scaling

Long-context category only, 2 repetitions. This is the operationally important
table.

| `num_ctx` | Model | output tok/s | TTFT s | peak GPU | on GPU |
|---|---|---|---|---|---|
| 16384 | `gpt-oss:20b` | 116.9 | 0.34 | 15065 MiB | 100% |
| 32768 | `gpt-oss:20b` | 116.9 | 0.34 | 16665 MiB | 100% |
| 65536 | `gpt-oss:20b` | 117.3 | 0.33 | 19447 MiB | 100% |
| 16384 | `qwen3-coder:30b` | 143.9 | 0.45 | 24717 MiB | 100% |
| 32768 | `qwen3-coder:30b` | 145.3 | 0.53 | 30767 MiB | 100% |
| 65536 | `qwen3-coder:30b` | **48.7** | **2.99** | 31195 MiB | **71%** |

- **`gpt-oss:20b` scales to 64k for free.** Throughput and TTFT are flat from
  16k to 64k and it never leaves the GPU, growing only from 15.1 to 19.4 GiB.
- **`qwen3-coder:30b` falls off a cliff between 32k and 64k.** At 32k it is
  already using 30767 MiB of 32760, roughly 2 GB of headroom. At 64k, 29% of
  the model is offloaded to system RAM, throughput drops 66% and TTFT rises
  5.6×.

So on this GPU: `qwen3-coder:30b` is safe up to 32k and should not be run at
64k; `gpt-oss:20b` is the long-context choice.

Quality stayed 1.00 at every context length for both models, including the
offloaded case. Note what that does and does not say: a larger *window* costs
nothing in quality, and stages 9a and 9b then showed that a larger *document*
does. Window size and document size are different variables and only the
second one degraded these models.

## Recommendations

Derived from the tables above. Composites and their weights are in each run's
report under *Recommendations by workload*.

| Workload | Model | Why |
|---|---|---|
| Coding assistant | `qwen3-coder:30b` | coding 1.00 at the highest throughput (164.8 tok/s) and lowest latency (1.0 s), zero truncations |
| Fast interactive | `qwen3-coder:30b` | best speed and latency with instruction-following 0.69 |
| Highest quality | `qwen3.8:27b` | top or joint-top in all six categories, including instruction 0.94; costs half the speed |
| Reasoning | `gpt-oss:20b` or `qwen3.8:27b` | both 1.00; `gpt-oss` is 1.6× faster, `qwen3.8` follows instructions far better |
| Long-context analysis | `gpt-oss:20b` | the only model to hold full GPU residency at 64k; at ~40,000-token documents `qwen3-coder:30b` offloads 29% and drops to 35.8 tok/s. Both score 0.88 there, failing the same ordering task |
| Lightweight / multi-agent worker | `gpt-oss:20b` | tool use and agent both 1.00 at the smallest footprint (12.85 GiB, 13.5 GiB resident) |
| Vision | not yet benchmarked | `qwen3.5:35b` and `qwen3.8:27b` report vision capability; the suite has no vision tasks |

Avoid `qwen3.5:35b` for these workloads: it was last or joint-last on every
quality axis and truncated 30 of 75 attempts.

## Caveats

- 45 rubric attempts are ungraded, so the agent, coding, reasoning and tool-use
  categories are currently measured by their automatic tasks only.
- `long_context` discriminates nothing at present, in `v1` or in `v2`. Both
  are saturated at roughly 8,000-10,000-token documents; see the v2 section
  above.
- The v1 long-context documents are larger than the suite's `filler_tokens`
  settings suggest. Two of the three tasks use the `{{FILLER}}` placeholder
  twice and receive the same block both times, so the prompts above averaged
  7064-8280 tokens rather than the 4000-5000 implied. The figures in this file
  are the engine's own token counts and are unaffected; only a reading of the
  task definitions would mislead.
- Tool-use tasks are simulated: tools are declared but never executed.
- `gpt-oss:20b` fails to terminate on two tight negative-constraint
  instruction tasks, producing 18,300 and 12,817 characters of reasoning at a
  6,144-token budget without emitting an answer. That is most of its 0.50
  instruction-following score.
- Repeated attempts at temperature 0 with a fixed seed produced different
  output lengths, so sampling controls reduce run-to-run variation without
  removing it.
- Quality scores are comparable only within suite version `1.2.0`.
