# Caliper

A local LLM benchmarking and evaluation framework. CLI-first, Ollama today, a
vLLM adapter planned. Built to answer a practical question: *which locally
installed model should I use for which job on this hardware?*

Caliper measures throughput, latency, memory and GPU residency, and grades
models on a small versioned suite of coding, reasoning, instruction-following,
tool-use, long-context and agent tasks. Results go into SQLite and come back
out as Markdown reports, JSON and CSV.

## Design principles

- **Measurements are never mixed with estimates.** Engine-reported timings,
  client-observed latency and planning estimates are separate fields, and
  anything estimated says so.
- **Unknown stays unknown.** Metadata the engine does not report is recorded
  as `null`, never inferred.
- **No single quality score.** Category scores are always shown. A composite
  appears only under a named workload profile, printed with its weights.
- **The host comes first.** A preflight check refuses to start work on a busy
  GPU, pre-existing workloads are never disturbed, and a model that was
  already resident is never evicted.
- **Minimal dependencies.** The runtime needs only the Python standard
  library. `pytest` and `ruff` are dev-only.

## Install

```sh
cd /path/to/llm-benchmark
uv venv --python 3.12 .venv            # or: python3 -m venv .venv
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/caliper --version
```

## Quick start

Everything below is read-only or mock-backed; nothing touches the GPU.

```sh
# What is on this machine, and what models does Ollama have?
caliper inventory --disk / --json inventory/env.json --markdown inventory/env.md
caliper models --detail
caliper models --moe-only

# Is it safe to run right now?
caliper preflight

# What would a run do, and what would it cost?
caliper plan --model gpt-oss:20b --model qwen3-coder:30b

# Exercise the whole pipeline with canned responses - no inference at all
caliper run --dry-run --allow-code-execution --label "dry run"
caliper report
caliper export --out-dir reports
```

A real run only happens when you ask for one:

```sh
caliper run --engine ollama --model gpt-oss:20b --repetitions 3 --warmup 1
```

`caliper run` is the only command that performs inference. Downloads, runs and
reports are deliberately separate operations; Caliper never downloads a model.

## Commands

| Command | What it does | Touches the GPU? |
|---|---|---|
| `inventory` | Machine + engine + model snapshot (JSON and Markdown) | no |
| `models` | List models, optionally with full metadata or MoE only | no |
| `preflight` | Resource-safety check against configured thresholds | no |
| `tasks` | List the evaluation tasks in a suite | no |
| `profiles` | Show the workload weighting profiles | no |
| `plan` | Task selection, controls, cost estimate, preflight | no |
| `run` | Execute a benchmark run | **yes** (unless `--dry-run`) |
| `report` | Markdown or JSON report for a stored run | no |
| `runs` | List stored runs | no |
| `export` | JSON + CSV export of a run | no |

## Configuration

Defaults live in `src/caliper/config.py`. Override them with a TOML file:

```sh
caliper --config configs/caliper.example.toml preflight
# or
export CALIPER_CONFIG=/path/to/caliper.toml
```

See [`configs/caliper.example.toml`](configs/caliper.example.toml) for every
setting with comments. Results are stored in `./data/caliper.sqlite` by
default; set `CALIPER_HOME` to put them elsewhere. No host-specific paths are
baked into the code.

## Safety model

`caliper run` runs `preflight` first and refuses to start when any blocking
check fails. Thresholds are configurable:

| Check | Default | Severity |
|---|---|---|
| Mean GPU utilization over several samples | ≤ 25% | block |
| Free VRAM | ≥ 20000 MiB | block |
| Available system RAM | ≥ 16000 MiB | block |
| Free disk | ≥ 50 GiB | block |
| Pre-existing GPU processes | reported; block above 4096 MiB | note / block |

Other guarantees:

- Execution is serial. There is no parallel mode.
- A model that was resident before the run is never unloaded. A model Caliper
  loaded is unloaded afterwards unless `--no-unload` is passed.
- `--force` is the only way past a blocking check, and the override is
  recorded in the run's stored preflight result.
- Ctrl-C finishes the attempt in flight, marks the run `interrupted` and keeps
  every completed attempt.
- Caliper never starts, stops or reconfigures an engine or any other service.

## Evaluation suite

Tasks live in `src/caliper/evals/suites/<version>/*.toml` and are original
content. Every suite declares a `suite_version` that is stored with each run,
so results are only compared when they were graded by the same definitions.

Validators: `exact`, `contains_all`, `regex_all`, `choice`, `numeric`,
`json_schema`, `tool_call`, `python_cases`, `rubric`.

An attempt that produced no visible output scores zero whatever the validator
says, because negative-constraint patterns match the empty string. Thinking
models make this a live concern: their reasoning tokens consume the same
`num_predict` budget as the answer, so a tight budget can truncate the answer
away entirely. Per-task budgets are sized for this; `--max-tokens` overrides
them when you need more.

`python_cases` executes model-written Python and is **off by default**; pass
`--allow-code-execution` to enable it. It runs in a separate isolated
interpreter with a memory cap and timeout — a guard against runaway test code,
not a security sandbox. Only run suites you wrote.

`rubric` tasks produce no automatic score. The output is stored and graded by
hand against the criteria in the task definition; see
[docs/GRADING.md](docs/GRADING.md). No LLM judge is used, because an LLM judge
is not ground truth.

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — components, data model, the
  internal API other projects can consume
- [docs/MEASUREMENT.md](docs/MEASUREMENT.md) — exactly what each metric means
  and how it is obtained
- [docs/GRADING.md](docs/GRADING.md) — manual rubric grading procedure
- [docs/CANDIDATES.md](docs/CANDIDATES.md) — model candidate analysis and the
  proposed experiment order for this host
- [docs/ROADMAP.md](docs/ROADMAP.md) — what is not built yet, including the
  vLLM comparison plan

## Development

```sh
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
.venv/bin/ruff format .
```

Tests never touch the GPU or the network: the Ollama adapter is tested against
recorded stream fixtures and the runner against the mock backend.
