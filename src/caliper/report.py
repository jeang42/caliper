"""Aggregation and report generation.

Design rules
------------
* Quality is never collapsed into one unexplained number. Category scores are
  always shown, and a composite only appears under a named workload profile
  whose weights are printed alongside it.
* Automatically graded and manually graded tasks are counted separately.
  Rubric tasks contribute no automatic score and are reported as pending.
* Every aggregate carries its sample count, and spread is reported as the
  sample standard deviation so uncertainty is visible.
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .evals.spec import CATEGORIES
from .storage import Store

GIB = 1024**3

#: Backstop for detecting a prompt-cache hit by implausible rate alone.
#:
#: Prompt throughput is only real when the engine actually evaluated the
#: prompt. Repeating an identical prompt - which every benchmark repetition
#: does - makes Ollama report the full token count against a near-zero
#: duration. Observed on this host: 8,168 tokens in 0.0108 s (754,000 tok/s)
#: and 9,433 tokens in 0.108 s (87,000 tok/s), against measured cold rates of
#: 1,000-13,000 tok/s.
#:
#: The primary rule is structural rather than numeric: for a given model and
#: task, only the first measured repetition has a cold prompt cache, so later
#: repetitions are cache hits by construction and no threshold is needed. This
#: ceiling then catches the remaining case - a first repetition whose prompt
#: was already warmed by the run's warmup attempt. It is set well below the
#: lowest observed cache-hit rate and far above the highest real one.
PROMPT_RATE_PLAUSIBILITY_CEILING = 50_000.0


@dataclass(slots=True)
class Stat:
    n: int = 0
    mean: float | None = None
    median: float | None = None
    stdev: float | None = None
    minimum: float | None = None
    maximum: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "mean": self.mean,
            "median": self.median,
            "stdev": self.stdev,
            "min": self.minimum,
            "max": self.maximum,
        }

    def fmt(self, nd: int = 1, unit: str = "") -> str:
        if self.mean is None:
            return "n/a"
        out = f"{self.mean:.{nd}f}{unit}"
        if self.stdev is not None and self.n > 1:
            out += f" ±{self.stdev:.{nd}f}"
        return out


def stat(values: Iterable[float | None]) -> Stat:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return Stat()
    return Stat(
        n=len(vals),
        mean=statistics.fmean(vals),
        median=statistics.median(vals),
        stdev=statistics.stdev(vals) if len(vals) > 1 else None,
        minimum=min(vals),
        maximum=max(vals),
    )


@dataclass(slots=True)
class CategoryScore:
    category: str
    mean_score: float | None
    graded_attempts: int
    pending_manual: int
    skipped: int
    errors: int
    grading_errors: int
    no_output: int
    tasks: int
    manual_mean_score: float | None = None
    manual_graded: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "mean_score": self.mean_score,
            "graded_attempts": self.graded_attempts,
            "pending_manual": self.pending_manual,
            "skipped": self.skipped,
            "errors": self.errors,
            "grading_errors": self.grading_errors,
            "no_output": self.no_output,
            "tasks": self.tasks,
            "manual_mean_score": self.manual_mean_score,
            "manual_graded": self.manual_graded,
        }


@dataclass(slots=True)
class ModelSummary:
    model: str
    engine: str
    digest: str | None = None
    quantization: str | None = None
    architecture: str | None = None
    is_moe: bool | None = None
    size_gib: float | None = None
    context_length: int | None = None
    residency: dict[str, Any] | None = None
    categories: dict[str, CategoryScore] = field(default_factory=dict)
    output_tps: Stat = field(default_factory=Stat)
    prompt_tps: Stat = field(default_factory=Stat)
    ttft_s: Stat = field(default_factory=Stat)
    latency_s: Stat = field(default_factory=Stat)
    load_s: Stat = field(default_factory=Stat)
    gpu_mem_peak_mib: Stat = field(default_factory=Stat)
    gpu_mem_delta_mib: Stat = field(default_factory=Stat)
    gpu_util_mean_pct: Stat = field(default_factory=Stat)
    ram_used_peak_mib: Stat = field(default_factory=Stat)
    cpu_util_mean_pct: Stat = field(default_factory=Stat)
    prompt_tokens: Stat = field(default_factory=Stat)
    cold_load_s: Stat = field(default_factory=Stat)
    cold_load_gpu_mem_mib: Stat = field(default_factory=Stat)
    reload_s: Stat = field(default_factory=Stat)
    reloads: int = 0
    attempts: int = 0
    errors: int = 0
    truncated: int = 0
    no_output: int = 0
    prompt_cache_hits: int = 0

    @property
    def reliability(self) -> float | None:
        if not self.attempts:
            return None
        return 1.0 - self.errors / self.attempts

    def quality_mean(self) -> float | None:
        scores = [c.mean_score for c in self.categories.values() if c.mean_score is not None]
        return statistics.fmean(scores) if scores else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "engine": self.engine,
            "digest": self.digest,
            "quantization": self.quantization,
            "architecture": self.architecture,
            "is_moe": self.is_moe,
            "size_gib": self.size_gib,
            "context_length": self.context_length,
            "residency": self.residency,
            "attempts": self.attempts,
            "errors": self.errors,
            "truncated": self.truncated,
            "no_output": self.no_output,
            "reloads": self.reloads,
            "prompt_cache_hits": self.prompt_cache_hits,
            "reliability": self.reliability,
            "categories": {k: v.to_dict() for k, v in self.categories.items()},
            "performance": {
                "output_tokens_per_s": self.output_tps.to_dict(),
                "prompt_tokens_per_s": self.prompt_tps.to_dict(),
                "ttft_s": self.ttft_s.to_dict(),
                "total_latency_s": self.latency_s.to_dict(),
                "model_load_s": self.load_s.to_dict(),
                "cold_load_s": self.cold_load_s.to_dict(),
                "cold_load_gpu_mem_mib": self.cold_load_gpu_mem_mib.to_dict(),
                "reload_s": self.reload_s.to_dict(),
                "prompt_tokens": self.prompt_tokens.to_dict(),
            },
            "telemetry": {
                "gpu_mem_peak_mib": self.gpu_mem_peak_mib.to_dict(),
                "gpu_mem_delta_mib": self.gpu_mem_delta_mib.to_dict(),
                "gpu_util_mean_pct": self.gpu_util_mean_pct.to_dict(),
                "ram_used_peak_mib": self.ram_used_peak_mib.to_dict(),
                "cpu_util_mean_pct": self.cpu_util_mean_pct.to_dict(),
            },
        }


# ---------------------------------------------------------------------------
# Workload profiles
# ---------------------------------------------------------------------------

#: Named workloads. ``quality`` weights are per category; ``speed`` and
#: ``memory`` weight normalized throughput and memory frugality. Weights are
#: printed with every composite so a number is never unexplained.
PROFILES: dict[str, dict[str, Any]] = {
    "fast_interactive": {
        "label": "Fast interactive assistant",
        "quality": {"instruction_following": 1.0, "reasoning": 0.5, "coding": 0.5},
        "speed": 2.0,
        "latency": 1.5,
        "memory": 0.5,
    },
    "coding_assistant": {
        "label": "Coding assistant",
        "quality": {"coding": 3.0, "instruction_following": 1.0, "reasoning": 1.0},
        "speed": 1.0,
        "latency": 0.5,
        "memory": 0.5,
    },
    "reasoning_assistant": {
        "label": "Reasoning assistant",
        "quality": {"reasoning": 3.0, "coding": 0.5, "long_context": 1.0},
        "speed": 0.5,
        "latency": 0.25,
        "memory": 0.25,
    },
    "lightweight_agent": {
        "label": "Lightweight agent",
        "quality": {"tool_use": 2.0, "instruction_following": 1.5, "agent": 1.5},
        "speed": 1.5,
        "latency": 1.5,
        "memory": 1.5,
    },
    "multi_agent_worker": {
        "label": "Multi-agent worker",
        "quality": {"tool_use": 2.0, "agent": 2.0, "instruction_following": 1.0},
        "speed": 1.5,
        "latency": 1.0,
        "memory": 2.0,
    },
    "long_context_analysis": {
        "label": "Long-context analysis",
        "quality": {"long_context": 3.0, "reasoning": 1.0, "instruction_following": 1.0},
        "speed": 0.5,
        "latency": 0.25,
        "memory": 1.0,
    },
    "high_quality_slow": {
        "label": "High quality, slower acceptable",
        "quality": {c: 1.0 for c in CATEGORIES},
        "speed": 0.1,
        "latency": 0.1,
        "memory": 0.1,
    },
}


def _normalize(values: dict[str, float | None], higher_is_better: bool = True) -> dict[str, float]:
    """Scale values into 0..1 across models. Missing values score 0."""
    present = {k: v for k, v in values.items() if v is not None and math.isfinite(v)}
    if not present:
        return {k: 0.0 for k in values}
    lo, hi = min(present.values()), max(present.values())
    out: dict[str, float] = {}
    for key in values:
        val = present.get(key)
        if val is None:
            out[key] = 0.0
        elif hi == lo:
            out[key] = 1.0
        else:
            frac = (val - lo) / (hi - lo)
            out[key] = frac if higher_is_better else 1.0 - frac
    return out


@dataclass(slots=True)
class ProfileRanking:
    profile: str
    label: str
    weights: dict[str, Any]
    ranking: list[tuple[str, float, dict[str, float]]]
    note: str = ""
    unmeasured_categories: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "label": self.label,
            "weights": self.weights,
            "note": self.note,
            "unmeasured_categories": self.unmeasured_categories,
            "ranking": [{"model": m, "composite": s, "components": c} for m, s, c in self.ranking],
        }


def rank_for_profile(
    summaries: Sequence[ModelSummary], profile: str, profiles: dict[str, Any] | None = None
) -> ProfileRanking:
    spec = (profiles or PROFILES)[profile]
    speed_norm = _normalize({s.model: s.output_tps.mean for s in summaries}, True)
    latency_norm = _normalize({s.model: s.ttft_s.mean for s in summaries}, False)
    mem_norm = _normalize({s.model: s.gpu_mem_peak_mib.mean for s in summaries}, False)

    # A category with no automatic score for any model in the run was not
    # measured. Scoring it as zero would silently punish every model and make
    # the composite meaningless, so such categories are dropped from the
    # weighting and named in the note instead.
    measured: dict[str, dict[str, float]] = {}
    unmeasured: list[str] = []
    for cat in spec["quality"]:
        values = {
            s.model: (s.categories[cat].mean_score if cat in s.categories else None)
            for s in summaries
        }
        if all(v is None for v in values.values()):
            unmeasured.append(cat)
        else:
            measured[cat] = {m: (v or 0.0) for m, v in values.items()}
    cat_norms = measured

    ranking: list[tuple[str, float, dict[str, float]]] = []
    for s in summaries:
        components: dict[str, float] = {}
        total_weight = 0.0
        acc = 0.0
        for cat, weight in spec["quality"].items():
            if cat not in cat_norms:
                continue
            value = cat_norms[cat].get(s.model, 0.0)
            components[f"quality:{cat}"] = value
            acc += weight * value
            total_weight += weight
        for name, norm in (
            ("speed", speed_norm),
            ("latency", latency_norm),
            ("memory", mem_norm),
        ):
            weight = float(spec.get(name, 0.0))
            if weight:
                value = norm.get(s.model, 0.0)
                components[name] = value
                acc += weight * value
                total_weight += weight
        ranking.append((s.model, acc / total_weight if total_weight else 0.0, components))
    ranking.sort(key=lambda row: row[1], reverse=True)
    note = (
        "Composite is a weighted mean of category quality scores (absolute, 0-1) and "
        "normalized speed/latency/memory (relative to the models in this comparison), "
        "so composites are not comparable across different model sets."
    )
    if unmeasured:
        note += (
            " This profile weights " + ", ".join(sorted(unmeasured)) + ", which this run "
            "did not measure; those categories were dropped from the weighting, so the "
            "composite covers only part of the profile."
        )
    if not measured:
        note = (
            "This run measured none of the categories this profile weights, so the "
            "composite reflects speed, latency and memory only."
        )
    return ProfileRanking(
        profile, spec["label"], spec, ranking, note, unmeasured_categories=sorted(unmeasured)
    )


# ---------------------------------------------------------------------------
# Building summaries from storage
# ---------------------------------------------------------------------------


def _loads(value: Any) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None


def summarize_run(store: Store, run_id: str) -> dict[str, Any]:
    """Aggregate one run into per-model summaries plus run metadata."""
    run = store.get_run(run_id)
    if run is None:
        raise KeyError(f"no such run: {run_id}")
    rows = store.attempts(run_id)
    model_meta = {r["model"]: r for r in store.run_models(run_id)}
    manual = store.manual_scores(run_id)

    by_model: dict[str, list[Any]] = {}
    for row in rows:
        by_model.setdefault(row["model"], []).append(row)

    # Cold-start behaviour only shows up on the attempt that actually loaded
    # the model, which is normally a warmup. Warmups stay out of every other
    # aggregate, but they are the only place the cold figures exist.
    warm_rows = store.attempts(run_id, include_warmup=True)
    cold_by_model: dict[str, list[Any]] = {}
    for row in warm_rows:
        if row["engine_load_duration_s"] and row["engine_load_duration_s"] > 0.5:
            cold_by_model.setdefault(row["model"], []).append(row)

    summaries: list[ModelSummary] = []
    for model, attempts in by_model.items():
        meta = model_meta.get(model)
        summary = ModelSummary(
            model=model,
            engine=run["engine"],
            digest=meta["digest"] if meta else None,
            quantization=meta["quantization"] if meta else None,
            architecture=meta["architecture"] if meta else None,
            is_moe=None if not meta or meta["is_moe"] is None else bool(meta["is_moe"]),
            size_gib=(meta["size_bytes"] / GIB) if meta and meta["size_bytes"] else None,
            context_length=meta["context_length"] if meta else None,
            residency=_loads(meta["residency_json"]) if meta else None,
        )
        summary.attempts = len(attempts)
        summary.errors = sum(1 for a in attempts if not a["ok"])
        ok_attempts = [a for a in attempts if a["ok"]]

        # Cold load is the warmup attempt that first brought the model in.
        cold = [a for a in cold_by_model.get(model, []) if a["is_warmup"]]
        summary.cold_load_s = stat(a["engine_load_duration_s"] for a in cold)
        summary.cold_load_gpu_mem_mib = stat(
            (_loads(a["telemetry_json"]) or {}).get("gpu_mem_delta_peak_mib") for a in cold
        )
        # A measured attempt that also paid a load is a reload: the engine
        # evicted and re-admitted the model mid-run. Changing num_ctx between
        # tasks forces this, and so does memory pressure.
        reloaded = [a for a in cold_by_model.get(model, []) if not a["is_warmup"]]
        summary.reloads = len(reloaded)
        summary.reload_s = stat(a["engine_load_duration_s"] for a in reloaded)

        summary.output_tps = stat(a["output_tokens_per_s"] for a in ok_attempts)
        # Prompt throughput is only meaningful when the prompt was actually
        # evaluated. Only the first measured repetition of a given task sees a
        # cold prompt cache; see PROMPT_RATE_PLAUSIBILITY_CEILING.
        seen_tasks: set[str] = set()
        cold_prompt = []
        for a in sorted(ok_attempts, key=lambda r: (r["task_id"], r["repetition"])):
            first_time = a["task_id"] not in seen_tasks
            seen_tasks.add(a["task_id"])
            rate = a["prompt_tokens_per_s"]
            implausible = rate is not None and rate >= PROMPT_RATE_PLAUSIBILITY_CEILING
            if first_time and not implausible:
                cold_prompt.append(a)
        summary.prompt_cache_hits = len(ok_attempts) - len(cold_prompt)
        summary.prompt_tps = stat(a["prompt_tokens_per_s"] for a in cold_prompt)
        # A request that had to load the model is not a steady-state request:
        # its latency is dominated by loading, which is reported on its own.
        reload_ids = {a["id"] for a in reloaded}
        steady = [a for a in ok_attempts if a["id"] not in reload_ids]
        summary.ttft_s = stat(a["ttft_s"] for a in steady)
        summary.latency_s = stat(a["total_latency_s"] for a in steady)
        summary.prompt_tokens = stat(a["prompt_tokens"] for a in ok_attempts)
        # Model load time only means something when a load actually happened.
        summary.load_s = stat(
            a["engine_load_duration_s"]
            for a in ok_attempts
            if a["engine_load_duration_s"] is not None and a["id"] not in reload_ids
        )

        summary.truncated = sum(1 for a in attempts if a["truncated"])
        summary.no_output = sum(1 for a in attempts if a["graded"] == "no_output")

        telemetry = [_loads(a["telemetry_json"]) or {} for a in attempts]
        summary.gpu_mem_peak_mib = stat(t.get("gpu_mem_used_peak_mib") for t in telemetry)
        summary.gpu_mem_delta_mib = stat(t.get("gpu_mem_delta_peak_mib") for t in telemetry)
        summary.gpu_util_mean_pct = stat(t.get("gpu_util_mean_pct") for t in telemetry)
        summary.ram_used_peak_mib = stat(t.get("ram_used_peak_mib") for t in telemetry)
        summary.cpu_util_mean_pct = stat(t.get("cpu_util_mean_pct") for t in telemetry)

        by_cat: dict[str, list[Any]] = {}
        for a in attempts:
            by_cat.setdefault(a["category"] or "unknown", []).append(a)
        for cat, cat_attempts in by_cat.items():
            graded = [
                a["score"]
                for a in cat_attempts
                if a["graded"] == "automatic" and a["score"] is not None
            ]
            manual_values = [
                manual[a["id"]]["score"] / manual[a["id"]]["max_score"]
                for a in cat_attempts
                if a["id"] in manual and manual[a["id"]]["max_score"]
            ]
            summary.categories[cat] = CategoryScore(
                category=cat,
                mean_score=statistics.fmean(graded) if graded else None,
                graded_attempts=len(graded),
                pending_manual=sum(1 for a in cat_attempts if a["graded"] == "manual_rubric"),
                skipped=sum(1 for a in cat_attempts if a["graded"] == "skipped"),
                errors=sum(1 for a in cat_attempts if not a["ok"]),
                grading_errors=sum(1 for a in cat_attempts if a["graded"] == "error"),
                no_output=sum(1 for a in cat_attempts if a["graded"] == "no_output"),
                tasks=len({a["task_id"] for a in cat_attempts}),
                manual_mean_score=statistics.fmean(manual_values) if manual_values else None,
                manual_graded=len(manual_values),
            )
        summaries.append(summary)

    summaries.sort(key=lambda s: s.model)
    observed: dict[str, list[Any]] = {}
    for key in ("num_ctx", "num_predict", "temperature", "seed"):
        values = sorted(
            {
                (_loads(a["options_json"]) or {}).get(key)
                for a in rows
                if (_loads(a["options_json"]) or {}).get(key) is not None
            }
        )
        if values:
            observed[key] = values

    return {
        "run": {
            "id": run["id"],
            "created_at": run["created_at"],
            "finished_at": run["finished_at"],
            "status": run["status"],
            "engine": run["engine"],
            "engine_version": run["engine_version"],
            "hostname": run["hostname"],
            "suite": run["suite"],
            "suite_version": run["suite_version"],
            "caliper_version": run["caliper_version"],
            "dry_run": bool(run["dry_run"]),
            "label": run["label"],
            "config": _loads(run["config_json"]),
            "preflight": _loads(run["preflight_json"]),
        },
        "observed_options": observed,
        "models": [s.to_dict() for s in summaries],
        "profiles": [rank_for_profile(summaries, p).to_dict() for p in PROFILES],
        "_summaries": summaries,
    }


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def render_markdown(summary: dict[str, Any]) -> str:
    run = summary["run"]
    summaries: list[ModelSummary] = summary["_summaries"]
    lines: list[str] = []
    add = lines.append

    add(f"# Benchmark report: run `{run['id']}`")
    add("")
    if run["dry_run"]:
        add(
            "> **DRY RUN** - responses came from the mock backend. These are not "
            "measurements of any real model."
        )
        add("")
    add(f"- Engine: `{run['engine']}` {run['engine_version'] or ''}")
    add(f"- Host: `{run['hostname']}`")
    add(f"- Suite: `{run['suite']}` version `{run['suite_version']}`")
    add(f"- Status: {run['status']}")
    add(f"- Started: {run['created_at']} | finished: {run['finished_at'] or 'n/a'}")
    add(f"- Caliper: {run['caliper_version']}")
    settings = (run.get("config") or {}).get("settings") or {}
    if settings:
        add(
            f"- Controls: temperature={settings.get('temperature')}, "
            f"seed={settings.get('seed')}, "
            f"repetitions={settings.get('repetitions')}, "
            f"warmup={settings.get('warmup_runs')}, "
            f"keep_alive={settings.get('keep_alive')!r}"
        )
        observed = summary.get("observed_options") or {}
        ctx = observed.get("num_ctx") or [settings.get("context_length")]
        predict = observed.get("num_predict") or [settings.get("max_output_tokens")]
        add(
            "- Token limits actually sent: num_ctx "
            + ", ".join(str(v) for v in ctx)
            + "; num_predict "
            + ", ".join(str(v) for v in predict)
            + " (tasks carry their own budgets unless overridden on the command line)"
        )
    add("")

    add("## Quality by category")
    add("")
    add(
        "Mean automatic score per category, 0.0-1.0. `n` is the number of graded "
        "attempts. Rubric tasks are excluded and counted under *pending manual*."
    )
    add("")
    cats = sorted({c for s in summaries for c in s.categories})
    add(
        "| Model | "
        + " | ".join(cats)
        + " | pending manual | no output | skipped | grading errors |"
    )
    add("|---" * (len(cats) + 5) + "|")
    for s in summaries:
        cells = []
        for cat in cats:
            cs = s.categories.get(cat)
            if cs is None or cs.mean_score is None:
                cells.append("n/a")
            else:
                cells.append(f"{cs.mean_score:.2f} (n={cs.graded_attempts})")
        pending = sum(c.pending_manual for c in s.categories.values())
        skipped = sum(c.skipped for c in s.categories.values())
        grading_errors = sum(c.grading_errors for c in s.categories.values())
        no_output = sum(c.no_output for c in s.categories.values())
        add(
            f"| `{s.model}` | "
            + " | ".join(cells)
            + f" | {pending} | {no_output} | {skipped} | {grading_errors} |"
        )
    if any(c.manual_graded for s in summaries for c in s.categories.values()):
        add("")
        add("### Manually graded tasks")
        add("")
        add(
            "Rubric scores entered by a human, as a fraction of the criteria met. "
            "These are kept in a separate table rather than merged into the "
            "automatic means, because they come from a different process and "
            "should not be mistaken for a validator's output."
        )
        add("")
        add("| Model | " + " | ".join(cats) + " |")
        add("|---" * (len(cats) + 1) + "|")
        for s in summaries:
            cells = []
            for cat in cats:
                cs = s.categories.get(cat)
                if cs is None or cs.manual_mean_score is None:
                    cells.append("-")
                else:
                    cells.append(f"{cs.manual_mean_score:.2f} (n={cs.manual_graded})")
            add(f"| `{s.model}` | " + " | ".join(cells) + " |")
    if any(s.no_output for s in summaries):
        add("")
        add(
            "> *No output* counts attempts at a manually graded task where the "
            "model emitted nothing to grade. They are kept out of the automatic "
            "means so that every model's category score covers the same set of "
            "tasks, and they are a failure in their own right: see the truncation "
            "and reload columns below."
        )
    if any(c.grading_errors for s in summaries for c in s.categories.values()):
        add("")
        add(
            "> Some attempts could not be graded (a validator raised). Those "
            "attempts are excluded from the means above; see the `score_detail` "
            "column in the attempts export for the cause."
        )
    add("")

    add("## Performance")
    add("")
    add(
        "Throughput and prompt-processing rates are the engine's own token "
        "accounting. TTFT and total latency are client-observed. Spread is ±1 "
        "sample standard deviation."
    )
    add("")
    add(
        "| Model | output tok/s | prompt tok/s | TTFT s | total latency s "
        "| truncated | cached prompts | reliability |"
    )
    add("|---|---|---|---|---|---|---|---|")
    for s in summaries:
        rel = (
            "n/a"
            if s.reliability is None
            else f"{s.reliability:.0%} ({s.attempts - s.errors}/{s.attempts})"
        )
        trunc = f"{s.truncated}/{s.attempts}" if s.attempts else "n/a"
        cached = f"{s.prompt_cache_hits}/{s.attempts}" if s.attempts else "n/a"
        add(
            f"| `{s.model}` | {s.output_tps.fmt(1)} | {s.prompt_tps.fmt(0)} | "
            f"{s.ttft_s.fmt(2)} | {s.latency_s.fmt(1)} | {trunc} | {cached} | {rel} |"
        )
    add("")
    add(
        "Output tokens include a thinking model's reasoning tokens, because the "
        "engine counts them as generated. They draw on the same `num_predict` "
        "budget as the answer."
    )
    add("")
    if any(s.prompt_cache_hits for s in summaries):
        add(
            "*Cached prompts* counts attempts whose prompt the engine reused from "
            "its cache instead of evaluating. Only the first measured repetition of "
            "each task sees a cold cache, so the rest are cache hits by construction "
            "and are excluded from the prompt tok/s column, as is any first "
            "repetition whose reported rate is physically impossible. Output "
            "throughput is unaffected, though a cached prompt does lower TTFT."
        )
        add("")
    if any(s.truncated for s in summaries):
        add(
            "> Some attempts stopped at the token limit rather than finishing. A "
            "truncated response cannot be judged on output format, and a thinking "
            "model can spend its whole budget before emitting any answer, which "
            "scores zero. Raise `--max-tokens` and re-run those tasks."
        )
        add("")

    add("## Model loading: cold versus warm")
    add("")
    add(
        "Cold figures come from the attempt that actually loaded the model, which "
        "is normally a warmup and is excluded from every other aggregate. Warm "
        "load time is what a request pays once the model is already resident."
    )
    add("")
    add("| Model | cold load s | weights loaded to GPU | warm load s | reloads | reload cost s |")
    add("|---|---|---|---|---|---|")
    for s in summaries:
        reloads = f"{s.reloads}/{s.attempts}" if s.attempts else "n/a"
        add(
            f"| `{s.model}` | {s.cold_load_s.fmt(2)} | "
            f"{s.cold_load_gpu_mem_mib.fmt(0, ' MiB')} | {s.load_s.fmt(3)} | "
            f"{reloads} | {s.reload_s.fmt(1)} |"
        )
    add("")
    add(
        "A *reload* is a measured attempt that had to bring the model back in "
        "because the engine had evicted it. Changing `num_ctx` between tasks "
        "forces one, and so does memory pressure on a model that only just "
        "fits. Reload attempts are excluded from the TTFT and latency figures "
        "above, because their cost is loading rather than inference; the "
        "reload cost is reported here instead."
    )
    add("")

    add("## Memory and residency")
    add("")
    add(
        "| Model | size | peak GPU mem | GPU mem delta | mean GPU util "
        "| peak RAM used | residency |"
    )
    add("|---|---|---|---|---|---|---|")
    for s in summaries:
        res = "unknown"
        if s.residency and s.residency.get("gpu_fraction") is not None:
            frac = s.residency["gpu_fraction"]
            res = f"{frac:.0%} on GPU" + (
                " (full)" if s.residency.get("fully_on_gpu") else " (offloaded)"
            )
        size = f"{s.size_gib:.2f} GiB" if s.size_gib else "unknown"
        add(
            f"| `{s.model}` | {size} | {s.gpu_mem_peak_mib.fmt(0, ' MiB')} | "
            f"{s.gpu_mem_delta_mib.fmt(0, ' MiB')} | {s.gpu_util_mean_pct.fmt(0, '%')} | "
            f"{s.ram_used_peak_mib.fmt(0, ' MiB')} | {res} |"
        )
    add("")
    add(
        "GPU memory delta is peak usage minus the baseline sampled just before each "
        "request, so it excludes memory held by other processes. Residency is the "
        "engine's own GPU/CPU split report."
    )
    add("")

    add("## Context length observed")
    add("")
    add("| Model | configured num_ctx | model max context | mean prompt tokens |")
    add("|---|---|---|---|")
    sent_ctx = ", ".join(
        str(v) for v in (summary.get("observed_options") or {}).get("num_ctx", [])
    ) or str(settings.get("context_length", "n/a"))
    for s in summaries:
        ctx = f"{s.context_length:,}" if s.context_length else "unknown"
        add(f"| `{s.model}` | {sent_ctx} | {ctx} | {s.prompt_tokens.fmt(0)} |")
    add("")

    add("## Recommendations by workload")
    add("")
    for prof in summary["profiles"]:
        add(f"### {prof['label']}")
        add("")
        weights = prof["weights"]
        qual = ", ".join(f"{k}x{v}" for k, v in weights["quality"].items())
        add(
            f"Weights - quality: {qual}; speedx{weights.get('speed', 0)}, "
            f"latencyx{weights.get('latency', 0)}, memoryx{weights.get('memory', 0)}."
        )
        if prof.get("unmeasured_categories"):
            add("")
            add(f"**Partial:** {prof['note']}")
        add("")
        for i, row in enumerate(prof["ranking"][:3], start=1):
            add(f"{i}. `{row['model']}` - composite {row['composite']:.3f}")
        add("")
    add("")

    add("## Limitations")
    add("")
    add(
        "- Composite scores are relative to the models in this run; speed, latency "
        "and memory are min-max normalized within the comparison set."
    )
    add(
        "- Rubric tasks carry no automatic score. Until they are graded by hand, "
        "the categories containing them are measured only by their automatic tasks."
    )
    add(
        "- The evaluation set is small, so category scores have wide uncertainty. "
        "Treat differences under roughly one task's worth of score as noise."
    )
    add("- Tool-use tasks are simulated: tools are declared but never executed.")
    add(
        "- A truncated attempt with no visible output scores zero. For a thinking "
        "model that is a failure to terminate reasoning, which is a different "
        "fault from answering incorrectly; the `truncated` column distinguishes "
        "them and raising the budget does not always fix it."
    )
    add(
        "- Telemetry is sampled, so a brief memory peak between samples can be "
        "missed. The sampling interval is recorded in the run config."
    )
    add(
        "- Engine-reported durations and client-observed latency are measured "
        "differently and are not interchangeable."
    )
    add(
        "- Prompt throughput is measured only on the first repetition of each "
        "task, which is the only one whose prompt the engine had to evaluate. "
        "Later repetitions reuse the cached prompt and are excluded, so this "
        "column has a smaller sample than the others."
    )
    add(
        "- Repeated attempts at temperature 0 with a fixed seed were observed on "
        "this engine to produce different output lengths for the same prompt, so "
        "the sampling controls reduce run-to-run variation without removing it."
    )
    if run["dry_run"]:
        add("- This was a dry run against the mock backend: no model was executed.")
    add("")
    return "\n".join(lines)
