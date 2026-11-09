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

Python 3.11 or newer; no runtime dependencies. An NVIDIA GPU is optional —
without `nvidia-smi` the GPU checks report *unknown* and the evaluation side
works unchanged.

```sh
git clone <this repository>
cd caliper
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
| `regrade` | Re-score a stored run with current validators | no |
| `serve` | Local web viewer and grading queue | no |

## Web viewer

```sh
caliper serve                       # http://127.0.0.1:8777
caliper serve --host 0.0.0.0 --allow-remote   # reachable from the network
```

A single-page viewer over the same SQLite database: runs, per-category scores,
performance, memory and loading behaviour, an attempt browser showing every
raw response, thinking text and tool call, and a cross-run comparison that
states whether the runs are actually comparable. Live GPU telemetry and the
progress of any executing run appear in the header, because attempts are
committed as they complete.

It also carries the **rubric grading queue**. Rubric tasks have no automatic
score by design; the queue shows each response beside its criteria with a
checkbox per criterion, and the score is stored in its own table where an
automatic re-grade cannot touch it.

The viewer cannot start, stop or configure a benchmark. Runs are launched from
the CLI, where preflight and serial execution are enforced; a browser button
would invite two people starting runs on one GPU. Recording a manual grade is
the only write it performs.

**Exposure.** `caliper serve` binds loopback and refuses any other address
unless `--allow-remote` is passed, because the viewer has no authentication and
serves machine telemetry, model inventory and raw model output. Over loopback,
reach it from elsewhere with `ssh -L 8777:127.0.0.1:8777 <host>`. If you do
bind outward, note that `0.0.0.0` covers every interface — including a VPN
such as Tailscale, not just the local subnet — and put a reverse proxy with
authentication in front of it if the network is not trusted.

## Re-grading without re-running

Grading is a pure function of the validator and the stored response, so a
corrected validator does not require new inference:

```sh
caliper regrade --run <id> --dry-run    # report what would change
caliper regrade --run <id>              # apply
```

Measurements are never altered. Manual rubric scores are in a separate table
and survive a re-grade.

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

Two suites are installed:

| Suite | Tasks | What it is |
|---|---|---|
| `v1` | 25 | The measured baseline. Every stored result to date uses it. |
| `v2` | 30 | `v1` with its three long-context tasks replaced by eight harder ones. |

`v1` remains the default, because `v2` has not been measured yet. Select a
suite with `--suite v2` on `run`, `plan` and `tasks`.

A suite may extend another through a `suite.toml` manifest, naming the
categories it replaces and inheriting the rest instead of copying them. The
manifest pins the base suite's version: edit `v1` and loading `v2` fails and
says so, rather than quietly changing what `v2` measures while `v2`'s own
`suite_version` stays put.

Long-context tasks substitute deterministic generated filler for a
placeholder. `{{FILLER}}` (v1) puts the *same* block at every site;
`{{FILLER:1}}`, `{{FILLER:2}}`, ... (v2) put distinct prose at each, with entry
numbering continuing across them, and `filler_tokens` then means the
approximate total across all segments rather than the size of each.

`--filler N` on `run` and `plan` overrides each task's `filler_tokens`, which
is the only way to vary document size: `--context` sets `num_ctx`, the window,
not the document. Raise both together — `num_ctx` holds the prompt and the
generation, and Ollama discards the overflow silently, so a run whose document
cannot fit is refused rather than scored against a document the model never
saw. `caliper plan` reports the same thing before any GPU time is spent.

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
- [docs/RESULTS.md](docs/RESULTS.md) — measured results for the models
  benchmarked so far, and which model to use for which workload
- [docs/API.md](docs/API.md) — the HTTP API and the stable Python API for
  other programs and agents
- [docs/ROADMAP.md](docs/ROADMAP.md) — what is not built yet, including the
  vLLM comparison plan
- [CLAUDE.md](CLAUDE.md) — working agreement for coding agents: invariants,
  metric definitions, and the mistakes already made once

## License

MIT. See [LICENSE](LICENSE).

## Development

```sh
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
.venv/bin/ruff format .
```

Tests never touch the GPU or the network: the Ollama adapter is tested against
recorded stream fixtures and the runner against the mock backend. CI runs the
same three commands plus a full dry run on Python 3.11, 3.12 and 3.13.

The measured results in [docs/RESULTS.md](docs/RESULTS.md) come from one
host — an RTX 5000 Ada with 32,760 MiB of VRAM — and are provenance for those
numbers, not a claim about your machine. Re-run the suite on your own hardware
before trusting any of the rankings there.
