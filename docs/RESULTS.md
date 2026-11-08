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
offloaded case — more evidence that the long-context tasks are too easy rather
than that offloading is harmless to quality.

## Recommendations

Derived from the tables above. Composites and their weights are in each run's
report under *Recommendations by workload*.

| Workload | Model | Why |
|---|---|---|
| Coding assistant | `qwen3-coder:30b` | coding 1.00 at the highest throughput (164.8 tok/s) and lowest latency (1.0 s), zero truncations |
| Fast interactive | `qwen3-coder:30b` | best speed and latency with instruction-following 0.69 |
| Highest quality | `qwen3.8:27b` | top or joint-top in all six categories, including instruction 0.94; costs half the speed |
| Reasoning | `gpt-oss:20b` or `qwen3.8:27b` | both 1.00; `gpt-oss` is 1.6× faster, `qwen3.8` follows instructions far better |
| Long-context analysis | `gpt-oss:20b` | the only model measured to hold full GPU residency and flat throughput at 64k |
| Lightweight / multi-agent worker | `gpt-oss:20b` | tool use and agent both 1.00 at the smallest footprint (12.85 GiB, 13.5 GiB resident) |
| Vision | not yet benchmarked | `qwen3.5:35b` and `qwen3.8:27b` report vision capability; the suite has no vision tasks |

Avoid `qwen3.5:35b` for these workloads: it was last or joint-last on every
quality axis and truncated 30 of 75 attempts.

## Caveats

- 45 rubric attempts are ungraded, so the agent, coding, reasoning and tool-use
  categories are currently measured by their automatic tasks only.
- `long_context` discriminates nothing at present.
- Tool-use tasks are simulated: tools are declared but never executed.
- `gpt-oss:20b` fails to terminate on two tight negative-constraint
  instruction tasks, producing 18,300 and 12,817 characters of reasoning at a
  6,144-token budget without emitting an answer. That is most of its 0.50
  instruction-following score.
- Repeated attempts at temperature 0 with a fixed seed produced different
  output lengths, so sampling controls reduce run-to-run variation without
  removing it.
- Quality scores are comparable only within suite version `1.2.0`.
