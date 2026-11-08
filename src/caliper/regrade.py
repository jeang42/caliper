"""Re-score stored responses with the current validators.

Grading is a pure function of (validator spec, stored response), so improving
a validator or correcting a grading rule does not require re-running
inference. This is the reason raw responses, thinking text, tool calls and the
engine's final payload are all stored in full.

Re-grading rewrites the score columns of an existing run in place and leaves
every measurement untouched.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from .evals import load_suite
from .evals.validators import grade
from .results import GenerationResult
from .storage import Store

log = logging.getLogger(__name__)


@dataclass(slots=True)
class RegradeOutcome:
    run_id: str
    attempts: int
    changed: int
    unchanged: int
    missing_tasks: list[str]

    def summary_line(self) -> str:
        return (
            f"re-graded {self.attempts} attempt(s) of run {self.run_id}: "
            f"{self.changed} changed, {self.unchanged} unchanged"
            + (
                f", {len(self.missing_tasks)} task(s) no longer in the suite"
                if self.missing_tasks
                else ""
            )
        )


def _result_from_row(row) -> GenerationResult:
    """Rebuild the object a validator sees from a stored attempt."""

    def loads(value, default):
        if not value:
            return default
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return default

    return GenerationResult(
        model=row["model"],
        engine="stored",
        text=row["response_text"] or "",
        thinking=row["thinking_text"] or "",
        tool_calls=loads(row["tool_calls_json"], []),
        error=row["error"],
        raw_final=loads(row["raw_final_json"], {}),
    )


def regrade_run(
    store: Store,
    run_id: str,
    allow_code_execution: bool = False,
    dry_run: bool = False,
) -> RegradeOutcome:
    """Re-apply the suite's validators to every stored attempt of a run.

    The suite is loaded by the name and version the run recorded, so a run is
    always re-graded by the definitions it was measured against.
    """
    run = store.get_run(run_id)
    if run is None:
        raise KeyError(f"no such run: {run_id}")
    suite = load_suite(run["suite"] or "v1")
    if run["suite_version"] and suite.suite_version != run["suite_version"]:
        log.warning(
            "run %s was graded by suite version %s but %s is installed; "
            "re-grading uses the installed definitions",
            run_id,
            run["suite_version"],
            suite.suite_version,
        )

    rows = store.attempts(run_id, include_warmup=False)
    changed = unchanged = 0
    missing: list[str] = []
    updates: list[tuple] = []

    for row in rows:
        try:
            task = suite.get(row["task_id"])
        except KeyError:
            missing.append(row["task_id"])
            continue
        spec = dict(task.validator)
        if spec.get("kind") == "python_cases":
            spec["allow_code_execution"] = allow_code_execution
        score = grade(spec, _result_from_row(row))
        same = (
            row["score"] == score.score
            and row["graded"] == score.graded
            and (row["passed"] is None) == (score.passed is None)
        )
        if same:
            unchanged += 1
            continue
        changed += 1
        updates.append(
            (
                score.score,
                None if score.passed is None else int(score.passed),
                score.validator,
                score.graded,
                score.detail,
                json.dumps(score.sub_scores, sort_keys=True),
                row["id"],
            )
        )

    if updates and not dry_run:
        store.update_scores(updates)
    return RegradeOutcome(
        run_id=run_id,
        attempts=len(rows),
        changed=changed,
        unchanged=unchanged,
        missing_tasks=sorted(set(missing)),
    )
