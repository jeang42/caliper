# Changelog

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
