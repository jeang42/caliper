"""JSON and CSV exports of stored runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .report import summarize_run
from .storage import Store

ATTEMPT_COLUMNS = (
    "run_id",
    "model",
    "task_id",
    "category",
    "repetition",
    "is_warmup",
    "ok",
    "error",
    "prompt_tokens",
    "output_tokens",
    "ttft_s",
    "ttft_source",
    "total_latency_s",
    "engine_total_duration_s",
    "engine_load_duration_s",
    "engine_prompt_eval_duration_s",
    "engine_eval_duration_s",
    "output_tokens_per_s",
    "prompt_tokens_per_s",
    "was_loaded_before",
    "score",
    "max_score",
    "passed",
    "validator",
    "graded",
    "score_detail",
)


def export_run_json(store: Store, run_id: str, path: Path, include_responses: bool = False) -> Path:
    """Write the aggregated summary, optionally with every raw response."""
    summary = summarize_run(store, run_id)
    summary.pop("_summaries", None)
    if include_responses:
        summary["attempts"] = [dict(row) for row in store.attempts(run_id, include_warmup=True)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, default=str))
    return path


def export_attempts_csv(store: Store, run_id: str, path: Path) -> Path:
    """One row per attempt, warmups included and flagged."""
    rows = store.attempts(run_id, include_warmup=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(ATTEMPT_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            data = dict(row)
            writer.writerow({k: data[k] for k in ATTEMPT_COLUMNS if k in data})
    return path


def export_summary_csv(store: Store, run_id: str, path: Path) -> Path:
    """One row per model with headline quality and performance figures."""
    summary = summarize_run(store, run_id)
    models: list[dict[str, Any]] = summary["models"]
    categories = sorted({c for m in models for c in m["categories"]})
    fields = [
        "run_id",
        "model",
        "engine",
        "digest",
        "quantization",
        "architecture",
        "is_moe",
        "size_gib",
        "attempts",
        "errors",
        "reliability",
        "output_tokens_per_s_mean",
        "output_tokens_per_s_stdev",
        "prompt_tokens_per_s_mean",
        "ttft_s_mean",
        "total_latency_s_mean",
        "model_load_s_mean",
        "gpu_mem_peak_mib_mean",
        "gpu_mem_delta_mib_mean",
        "ram_used_peak_mib_mean",
        "gpu_fraction",
        *[f"score_{c}" for c in categories],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for m in models:
            perf = m["performance"]
            tel = m["telemetry"]
            row = {
                "run_id": summary["run"]["id"],
                "model": m["model"],
                "engine": m["engine"],
                "digest": m["digest"],
                "quantization": m["quantization"],
                "architecture": m["architecture"],
                "is_moe": m["is_moe"],
                "size_gib": m["size_gib"],
                "attempts": m["attempts"],
                "errors": m["errors"],
                "reliability": m["reliability"],
                "output_tokens_per_s_mean": perf["output_tokens_per_s"]["mean"],
                "output_tokens_per_s_stdev": perf["output_tokens_per_s"]["stdev"],
                "prompt_tokens_per_s_mean": perf["prompt_tokens_per_s"]["mean"],
                "ttft_s_mean": perf["ttft_s"]["mean"],
                "total_latency_s_mean": perf["total_latency_s"]["mean"],
                "model_load_s_mean": perf["model_load_s"]["mean"],
                "gpu_mem_peak_mib_mean": tel["gpu_mem_peak_mib"]["mean"],
                "gpu_mem_delta_mib_mean": tel["gpu_mem_delta_mib"]["mean"],
                "ram_used_peak_mib_mean": tel["ram_used_peak_mib"]["mean"],
                "gpu_fraction": (m["residency"] or {}).get("gpu_fraction"),
            }
            for cat in categories:
                row[f"score_{cat}"] = (m["categories"].get(cat) or {}).get("mean_score")
            writer.writerow(row)
    return path
