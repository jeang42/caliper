"""Read-mostly JSON API behind the local web viewer.

Every handler is a plain function of ``(Store, params) -> dict``, so the API is
tested without opening a socket. The HTTP layer in :mod:`caliper.server` only
routes and serialises.

Scope: the viewer observes runs and records human rubric grades. It cannot
start, stop or configure a benchmark. Launching inference stays in the CLI,
where preflight and serial execution are enforced — a browser button would
invite exactly the concurrent-run failure that the safety model exists to
prevent.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from . import __version__
from .evals import load_suite, longctx
from .evals.validators import MANUAL_KINDS
from .hardware import sample_gpus
from .report import PROFILES, summarize_run
from .storage import Store

log = logging.getLogger(__name__)

#: Columns small enough to send for every attempt in a list view.
#: Grading states that accept a human score: a rubric attempt awaiting one, or
#: a rubric attempt where the model emitted nothing (which a human grades as a
#: failure). Everything else - automatic scores, warmups, skipped validators -
#: is refused.
GRADEABLE_STATES = frozenset({"manual_rubric", "no_output"})

ATTEMPT_LIST_COLUMNS = (
    "id",
    "model",
    "task_id",
    "category",
    "repetition",
    "is_warmup",
    "ok",
    "error",
    "done_reason",
    "truncated",
    "prompt_tokens",
    "output_tokens",
    "ttft_s",
    "total_latency_s",
    "output_tokens_per_s",
    "engine_load_duration_s",
    "score",
    "max_score",
    "passed",
    "validator",
    "graded",
    "score_detail",
)


class ApiError(Exception):
    """Raised for a request the client got wrong."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def _loads(value: Any, default: Any = None) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def _row(row: Any, columns: tuple[str, ...] | None = None) -> dict[str, Any]:
    data = dict(row)
    if columns is None:
        return data
    return {k: data.get(k) for k in columns}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


def list_runs(store: Store, limit: int = 50) -> dict[str, Any]:
    runs = []
    for row in store.list_runs(limit=limit):
        progress = store.run_progress(row["id"])
        models = [m["model"] for m in store.run_models(row["id"])]
        runs.append(
            {
                "id": row["id"],
                "created_at": row["created_at"],
                "finished_at": row["finished_at"],
                "status": row["status"],
                "engine": row["engine"],
                "engine_version": row["engine_version"],
                "hostname": row["hostname"],
                "suite": row["suite"],
                "suite_version": row["suite_version"],
                "dry_run": bool(row["dry_run"]),
                "label": row["label"],
                "models": models,
                "attempts": progress["attempts"],
                "failures": progress["failures"],
                "running": row["finished_at"] is None and row["status"] == "running",
            }
        )
    return {"runs": runs, "caliper_version": __version__}


def get_run(store: Store, run_id: str) -> dict[str, Any]:
    try:
        summary = summarize_run(store, run_id)
    except KeyError as exc:
        raise ApiError(f"no such run: {run_id}", status=404) from exc
    summary.pop("_summaries", None)
    summary["progress"] = store.run_progress(run_id)
    return summary


def list_attempts(
    store: Store,
    run_id: str,
    model: str | None = None,
    category: str | None = None,
    task_id: str | None = None,
    graded: str | None = None,
    include_warmup: bool = False,
    only_ungraded: bool = False,
) -> dict[str, Any]:
    if store.get_run(run_id) is None:
        raise ApiError(f"no such run: {run_id}", status=404)
    rows = store.attempts(run_id, include_warmup=include_warmup, model=model)
    manual = store.manual_scores(run_id)
    out = []
    for row in rows:
        if category and row["category"] != category:
            continue
        if task_id and row["task_id"] != task_id:
            continue
        if graded and row["graded"] != graded:
            continue
        entry = _row(row, ATTEMPT_LIST_COLUMNS)
        entry["has_response"] = bool(row["response_text"])
        entry["has_thinking"] = bool(row["thinking_text"])
        entry["tool_call_count"] = len(_loads(row["tool_calls_json"], []) or [])
        score = manual.get(row["id"])
        entry["manual_score"] = (
            {
                "score": score["score"],
                "max_score": score["max_score"],
                "grader": score["grader"],
                "notes": score["notes"],
                "graded_at": score["graded_at"],
            }
            if score
            else None
        )
        if only_ungraded and (entry["manual_score"] or row["graded"] == "automatic"):
            continue
        out.append(entry)
    return {"run_id": run_id, "attempts": out, "count": len(out)}


def get_attempt(store: Store, attempt_id: int) -> dict[str, Any]:
    """One attempt in full, including the raw response and the task definition."""
    row = store.attempt(attempt_id)
    if row is None:
        raise ApiError(f"no such attempt: {attempt_id}", status=404)
    run = store.get_run(row["run_id"])
    data = _row(row)
    data["options"] = _loads(row["options_json"], {})
    data["tool_calls"] = _loads(row["tool_calls_json"], [])
    data["telemetry"] = _loads(row["telemetry_json"], {})
    data["raw_final"] = _loads(row["raw_final_json"], {})
    data["sub_scores"] = _loads(row["sub_scores_json"], {})
    for key in (
        "options_json",
        "tool_calls_json",
        "telemetry_json",
        "raw_final_json",
        "sub_scores_json",
    ):
        data.pop(key, None)

    settings = (_loads(run["config_json"], {}) or {}).get("settings", {}) if run else {}
    task = _task_definition(
        run["suite"] if run else "v1",
        row["task_id"],
        seed=settings.get("seed"),
        render=True,
        filler_tokens=settings.get("filler_tokens_override"),
    )
    manual = store.manual_scores(row["run_id"]).get(attempt_id)
    return {
        "attempt": data,
        "task": task,
        "manual_score": dict(manual) if manual else None,
        "run": {
            "id": run["id"],
            "engine": run["engine"],
            "engine_version": run["engine_version"],
            "suite": run["suite"],
            "suite_version": run["suite_version"],
            "dry_run": bool(run["dry_run"]),
        }
        if run
        else None,
    }


def _task_definition(
    suite_name: str,
    task_id: str,
    seed: int | None = None,
    render: bool = False,
    filler_tokens: int | None = None,
) -> dict[str, Any] | None:
    """The task as the installed suite defines it, for grading context.

    With ``render``, a long-context task also carries the document the model
    actually saw, rebuilt from the run's seed. Rendered prompts are not stored
    per attempt - at 8000 tokens each that would dominate the database - but
    they are deterministic, so they can be reconstructed exactly. Without this
    a reviewer looking at a wrong long-context answer sees only the
    placeholder, which is the one thing the model did not see.
    """
    try:
        suite = load_suite(suite_name or "v1")
        task = suite.get(task_id)
    except (KeyError, FileNotFoundError, ValueError):
        return None
    rendered: str | None = None
    # The run's --filler override, not the task's own value: rebuilding with
    # the task default would show a reviewer a document the model never saw,
    # which is the exact failure this reconstruction exists to prevent.
    size = (filler_tokens or task.filler_tokens) if task.filler_tokens else None
    if render and size:
        rendered, substituted = longctx.render(task.prompt, size, seed=seed or 1234)
        if not substituted:
            rendered = None
    return {
        "rendered_prompt": rendered,
        "id": task.id,
        "category": task.category,
        "title": task.title,
        "prompt": task.prompt,
        "system": task.system,
        "tools": task.tools,
        "validator_kind": task.validator.get("kind"),
        "criteria": task.validator.get("criteria", []),
        "signal_phrases": task.validator.get("signal_phrases", []),
        "manual": task.validator.get("kind") in MANUAL_KINDS,
        "max_output_tokens": task.max_output_tokens,
        "context_length": task.context_length,
        "filler_tokens": task.filler_tokens,
        "filler_tokens_rendered": size,
        "suite_version": suite.suite_version,
    }


def _response_digest(text: str | None) -> str:
    """Stable fingerprint of a response, for deciding whether two match.

    Whitespace is normalised because trailing newlines differ between
    otherwise identical generations and are not something a grader would
    judge differently.
    """
    return hashlib.sha256(" ".join((text or "").split()).encode()).hexdigest()[:16]


def grading_queue(store: Store, run_id: str) -> dict[str, Any]:
    """Attempts at manually graded tasks, with their criteria and any score.

    This is the view that makes the long tail of evaluation sustainable: every
    rubric attempt in one place, with the criteria beside the response.

    Attempts are also grouped by (task, model), because repetitions of one
    prompt at temperature 0 are usually the same text and judging each one
    separately is what makes a queue of 96 attempts go ungraded. A group whose
    responses are byte-identical after whitespace normalisation can be graded
    once; a group whose responses differ says so and must be graded per
    attempt. Grouping never merges scores - every attempt still gets its own
    row in ``manual_scores``.
    """
    run = store.get_run(run_id)
    if run is None:
        raise ApiError(f"no such run: {run_id}", status=404)
    manual = store.manual_scores(run_id)
    queue = []
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for row in store.attempts(run_id):
        if row["graded"] not in GRADEABLE_STATES:
            continue
        task = _task_definition(run["suite"], row["task_id"])
        score = manual.get(row["id"])
        digest = _response_digest(row["response_text"])
        entry = {
            "attempt_id": row["id"],
            "model": row["model"],
            "task_id": row["task_id"],
            "category": row["category"],
            "repetition": row["repetition"],
            "graded": row["graded"],
            "truncated": bool(row["truncated"]),
            "criteria": (task or {}).get("criteria", []),
            "title": (task or {}).get("title"),
            "scored": score is not None,
            "score": score["score"] if score else None,
            "max_score": score["max_score"] if score else None,
            "response_digest": digest,
            "response_chars": len(row["response_text"] or ""),
        }
        queue.append(entry)
        key = (row["task_id"], row["model"])
        group = groups.setdefault(
            key,
            {
                "task_id": row["task_id"],
                "model": row["model"],
                "category": row["category"],
                "title": entry["title"],
                "criteria": entry["criteria"],
                "attempts": [],
                "digests": [],
            },
        )
        group["attempts"].append(entry)
        group["digests"].append(digest)

    grouped = []
    for group in groups.values():
        digests = group.pop("digests")
        attempts = group["attempts"]
        group["identical"] = len(set(digests)) == 1
        group["distinct_responses"] = len(set(digests))
        group["repetitions"] = len(attempts)
        group["scored"] = sum(1 for a in attempts if a["scored"])
        group["attempt_ids"] = [a["attempt_id"] for a in attempts]
        group["truncated"] = any(a["truncated"] for a in attempts)
        grouped.append(group)
    grouped.sort(key=lambda g: (g["category"], g["task_id"], g["model"]))

    done = sum(1 for q in queue if q["scored"])
    return {
        "run_id": run_id,
        "queue": queue,
        "groups": grouped,
        "total": len(queue),
        "scored": done,
        "remaining": len(queue) - done,
        "decisions": len(grouped),
        "decisions_remaining": sum(1 for g in grouped if g["scored"] < g["repetitions"]),
    }


def set_grade(store: Store, payload: dict[str, Any]) -> dict[str, Any]:
    """Record a human score. The only write this API performs.

    ``attempt_ids`` grades several attempts with one judgement, which is how a
    group of repetitions is graded once. The responses must be identical after
    whitespace normalisation, or the call is refused: applying one grader's
    reading to text they did not read is exactly the shortcut this queue exists
    to avoid. ``allow_divergent`` overrides that for a grader who has read all
    of them and judges them equivalent, and the reason is recorded in the
    stored note.
    """
    ids = payload.get("attempt_ids")
    if ids is not None:
        return _set_grades(store, payload, ids)
    try:
        attempt_id = int(payload["attempt_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ApiError("attempt_id is required and must be an integer") from exc

    row = store.attempt(attempt_id)
    if row is None:
        raise ApiError(f"no such attempt: {attempt_id}", status=404)
    # Allowlist, not denylist: a warmup attempt carries graded=None and would
    # pass a "not automatic" check, but a warmup is not part of the evaluation
    # and must never receive a score.
    if row["graded"] not in GRADEABLE_STATES:
        reason = (
            f"it was graded automatically by the {row['validator']!r} validator"
            if row["graded"] == "automatic"
            else "it is a warmup attempt"
            if row["is_warmup"]
            else f"its grading state is {row['graded']!r}"
        )
        raise ApiError(
            f"attempt {attempt_id} cannot be graded by hand: {reason}. "
            "Manual scores apply only to attempts at rubric tasks."
        )

    criteria = payload.get("criteria")
    if criteria is not None:
        if not isinstance(criteria, list):
            raise ApiError("criteria must be a list of {text, met} objects")
        met = sum(1 for c in criteria if c.get("met"))
        score, max_score = float(met), float(len(criteria))
        if max_score == 0:
            raise ApiError("criteria list is empty")
    else:
        try:
            score = float(payload["score"])
            max_score = float(payload.get("max_score", 1.0))
        except (KeyError, TypeError, ValueError) as exc:
            raise ApiError("provide either criteria, or score and max_score") from exc

    try:
        store.set_manual_score(
            attempt_id=attempt_id,
            run_id=row["run_id"],
            score=score,
            max_score=max_score,
            criteria=criteria,
            grader=payload.get("grader") or None,
            notes=payload.get("notes") or "",
        )
    except ValueError as exc:
        raise ApiError(str(exc)) from exc
    return {
        "attempt_id": attempt_id,
        "score": score,
        "max_score": max_score,
        "fraction": score / max_score,
        "stored": True,
    }


def _set_grades(store: Store, payload: dict[str, Any], ids: Any) -> dict[str, Any]:
    """Apply one judgement to a group of attempts, all or nothing."""
    if not isinstance(ids, list) or not ids:
        raise ApiError("attempt_ids must be a non-empty list of attempt ids")
    try:
        attempt_ids = [int(i) for i in ids]
    except (TypeError, ValueError) as exc:
        raise ApiError("attempt_ids must all be integers") from exc

    rows = []
    for attempt_id in attempt_ids:
        row = store.attempt(attempt_id)
        if row is None:
            raise ApiError(f"no such attempt: {attempt_id}", status=404)
        rows.append(row)

    tasks = {r["task_id"] for r in rows}
    models = {r["model"] for r in rows}
    if len(tasks) > 1 or len(models) > 1:
        raise ApiError(
            "attempt_ids must all be the same task and model: got "
            f"{sorted(tasks)} and {sorted(models)}. One judgement covers one "
            "model's repetitions of one task, nothing wider."
        )

    digests = {_response_digest(r["response_text"]) for r in rows}
    divergent = len(digests) > 1
    if divergent and not payload.get("allow_divergent"):
        raise ApiError(
            f"these {len(rows)} attempts produced {len(digests)} different responses, so one "
            "judgement cannot cover them. Grade them individually, or pass allow_divergent "
            "to record that you read all of them and judge them equivalent."
        )

    note = payload.get("notes") or ""
    if len(rows) > 1:
        how = (
            "with differing responses (allow_divergent)"
            if divergent
            else "with identical responses"
        )
        provenance = f"graded as a group of {len(rows)} repetitions {how}"
        note = f"{note}\n{provenance}".strip() if note else provenance

    results = []
    for attempt_id in attempt_ids:
        single = {k: v for k, v in payload.items() if k not in {"attempt_ids", "notes"}}
        single["attempt_id"] = attempt_id
        single["notes"] = note
        results.append(set_grade(store, single))
    return {
        "attempt_ids": attempt_ids,
        "graded": len(results),
        "divergent": divergent,
        "distinct_responses": len(digests),
        "score": results[0]["score"],
        "max_score": results[0]["max_score"],
        "results": results,
        "stored": True,
    }


def delete_grade(store: Store, attempt_id: int) -> dict[str, Any]:
    if store.attempt(attempt_id) is None:
        raise ApiError(f"no such attempt: {attempt_id}", status=404)
    store.delete_manual_score(attempt_id)
    return {"attempt_id": attempt_id, "deleted": True}


def live_status(store: Store, gpu_index: int = 0) -> dict[str, Any]:
    """Current GPU telemetry plus any run still in flight.

    Attempts are committed as they complete, so an executing run's progress is
    visible here without the runner publishing anything.
    """
    gpus = sample_gpus()
    gpu = next((g for g in gpus if g.index == gpu_index), None)
    active = None
    for row in store.list_runs(limit=10):
        if row["status"] == "running" and row["finished_at"] is None:
            progress = store.run_progress(row["id"])
            config = _loads(row["config_json"], {}) or {}
            settings = config.get("settings") or {}
            models = settings.get("models") or []
            expected = None
            if models and settings.get("repetitions"):
                expected = len(models) * (
                    settings.get("warmup_runs", 0)
                    + settings["repetitions"] * _suite_task_count(row["suite"], settings)
                )
            active = {
                "run_id": row["id"],
                "label": row["label"],
                "models": models,
                "attempts_done": progress["attempts"],
                "attempts_expected": expected,
                "failures": progress["failures"],
                "last_attempt": progress["last_attempt"],
                "started_at": row["created_at"],
            }
            break
    return {
        "gpu": {
            "name": gpu.name if gpu else None,
            "utilization_pct": gpu.utilization_gpu_pct if gpu else None,
            "memory_used_mib": gpu.memory_used_mib if gpu else None,
            "memory_total_mib": gpu.memory_total_mib if gpu else None,
            "temperature_c": gpu.temperature_c if gpu else None,
            "power_draw_w": gpu.power_draw_w if gpu else None,
        }
        if gpu
        else None,
        "active_run": active,
    }


def _suite_task_count(suite_name: str | None, settings: dict[str, Any]) -> int:
    try:
        suite = load_suite(suite_name or "v1")
    except (FileNotFoundError, ValueError):
        return 0
    tasks = suite.by_category(settings.get("categories") or [])
    if settings.get("task_ids"):
        wanted = set(settings["task_ids"])
        tasks = [t for t in tasks if t.id in wanted]
    return len(tasks)


def compare_runs(store: Store, run_ids: list[str]) -> dict[str, Any]:
    """Category scores and headline performance across several runs.

    Comparability is asserted, not assumed: the response states whether engine
    version, suite version and model digests match, and the viewer shows the
    warning rather than quietly putting incomparable numbers side by side.
    """
    if not run_ids:
        raise ApiError("at least one run id is required")
    entries: list[dict[str, Any]] = []
    suite_versions: set[str] = set()
    engine_versions: set[str] = set()
    digests: dict[str, set[str]] = {}
    for run_id in run_ids:
        summary = get_run(store, run_id)
        run = summary["run"]
        suite_versions.add(f"{run['suite']}@{run['suite_version']}")
        engine_versions.add(f"{run['engine']}@{run['engine_version']}")
        for model in summary["models"]:
            digests.setdefault(model["model"], set()).add(model["digest"] or "unknown")
        entries.append({"run": run, "models": summary["models"]})

    mismatched = sorted(name for name, seen in digests.items() if len(seen) > 1)
    warnings = []
    if len(suite_versions) > 1:
        warnings.append(
            "different suite versions: "
            + ", ".join(sorted(suite_versions))
            + " - task definitions differ, so quality scores are not comparable"
        )
    if len(engine_versions) > 1:
        warnings.append(
            "different engine versions: "
            + ", ".join(sorted(engine_versions))
            + " - performance figures are not comparable"
        )
    if mismatched:
        warnings.append(
            "same model name, different digest: "
            + ", ".join(mismatched)
            + " - the weights differ between these runs"
        )
    return {
        "runs": entries,
        "comparable": not warnings,
        "warnings": warnings,
        "profiles": list(PROFILES),
    }
