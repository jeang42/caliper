"""Benchmark runner.

Execution model
---------------
* Serial by default: one model, one task, one repetition at a time. Concurrency
  would make the telemetry attribution meaningless on a single GPU.
* Per model: optional warmup run(s) (recorded, flagged ``is_warmup``), then N
  measured repetitions of every selected task.
* Telemetry is sampled in the background for the duration of each request, so
  peak GPU memory is attributed to a single generation.
* Every attempt is written to SQLite as soon as it completes, so Ctrl-C never
  loses finished work.
* Models that were already resident before the run are never unloaded.
"""

from __future__ import annotations

import logging
import signal
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from . import safety as safety_mod
from .backends.base import Backend
from .evals import longctx, validators
from .evals.spec import Suite, Task
from .hardware import TelemetrySampler
from .results import GenerationResult, TaskScore
from .storage import RunHeader, Store, new_id

log = logging.getLogger(__name__)

ProgressFn = Callable[[str], None]


class RunAborted(RuntimeError):
    """Raised when preflight blocks a run."""


@dataclass(slots=True)
class RunSettings:
    """Experimental controls for one benchmark run. Recorded verbatim."""

    models: list[str]
    suite: str = "v1"
    categories: list[str] = field(default_factory=list)
    task_ids: list[str] = field(default_factory=list)
    repetitions: int = 3
    warmup_runs: int = 1
    temperature: float = 0.0
    seed: int | None = 42
    max_output_tokens: int = 512
    context_length: int = 8192
    keep_alive: str = "5m"
    unload_after_run: bool = True
    request_timeout_s: float = 300.0
    telemetry_interval_s: float = 0.25
    allow_code_execution: bool = False
    gpu_index: int = 0
    dry_run: bool = False
    label: str | None = None
    force: bool = False

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


@dataclass(slots=True)
class RunOutcome:
    run_id: str
    status: str
    attempts: int
    failures: int
    skipped_models: list[str] = field(default_factory=list)


def build_options(settings: RunSettings, task: Task) -> dict[str, Any]:
    """Assemble engine sampling options for one task.

    Per-task overrides win over run-level settings. Ollama option names are
    used directly; a future vLLM adapter must translate them and record that
    the mapping is not guaranteed identical.
    """
    options: dict[str, Any] = {
        "temperature": task.temperature if task.temperature is not None else settings.temperature,
        "num_predict": task.max_output_tokens or settings.max_output_tokens,
        "num_ctx": task.context_length or settings.context_length,
    }
    if settings.seed is not None:
        options["seed"] = settings.seed
    return options


def _interrupt_guard() -> Callable[[], bool]:
    """Install a SIGINT handler that requests a graceful stop.

    Returns a predicate that reports whether a stop was requested. The handler
    is only installed when running on the main thread.
    """
    state = {"stop": False}

    def handler(signum: int, frame: object) -> None:
        state["stop"] = True
        log.warning("interrupt received; finishing current attempt then stopping")

    try:
        signal.signal(signal.SIGINT, handler)
    except ValueError:  # not the main thread
        log.debug("SIGINT handler not installed (not main thread)")
    return lambda: state["stop"]


class BenchmarkRunner:
    def __init__(
        self,
        backend: Backend,
        store: Store,
        settings: RunSettings,
        safety_thresholds: dict[str, Any],
        progress: ProgressFn | None = None,
    ) -> None:
        self.backend = backend
        self.store = store
        self.settings = settings
        self.safety = safety_thresholds
        self.progress = progress or (lambda msg: None)

    # -- preparation ------------------------------------------------------
    def preflight(self) -> safety_mod.PreflightResult:
        return safety_mod.preflight(self.safety, gpu_index=self.settings.gpu_index)

    def select_tasks(self, suite: Suite) -> list[Task]:
        tasks = suite.by_category(self.settings.categories)
        if self.settings.task_ids:
            wanted = set(self.settings.task_ids)
            tasks = [t for t in tasks if t.id in wanted]
            missing = wanted - {t.id for t in tasks}
            if missing:
                raise KeyError(f"unknown task ids: {sorted(missing)}")
        if not self.settings.allow_code_execution:
            skipped = [t.id for t in tasks if t.requires_code_execution]
            if skipped:
                log.info(
                    "%d task(s) need code execution and will be recorded as skipped: %s",
                    len(skipped),
                    ", ".join(skipped),
                )
        return tasks

    # -- execution --------------------------------------------------------
    def run(self, suite: Suite) -> RunOutcome:
        settings = self.settings
        tasks = self.select_tasks(suite)
        if not tasks:
            raise ValueError("no tasks selected")

        pre = self.preflight()
        if not pre.ok and not settings.dry_run:
            detail = "; ".join(c.detail for c in pre.blocking_failures)
            if not settings.force:
                raise RunAborted(f"preflight blocked the run: {detail}")
            log.warning("preflight failures overridden by --force: %s", detail)
        for warn in pre.notices:
            log.warning("preflight note: %s", warn.detail)

        header = RunHeader(
            id=new_id(),
            engine=self.backend.name,
            engine_version=self.backend.version(),
            hostname=pre.machine.hostname if pre.machine else None,
            suite=suite.name,
            suite_version=suite.suite_version,
            dry_run=settings.dry_run,
            label=settings.label,
        )
        self.store.create_run(
            header,
            config={"settings": settings.to_dict(), "safety": self.safety},
            machine=pre.machine.to_dict() if pre.machine else None,
            preflight=pre.to_dict(),
        )
        self.progress(
            f"run {header.id} started: {len(settings.models)} model(s) x {len(tasks)} task(s) "
            f"x {settings.repetitions} rep(s)" + (" [DRY RUN]" if settings.dry_run else "")
        )

        should_stop = _interrupt_guard()
        attempts = 0
        failures = 0
        skipped: list[str] = []
        status = "completed"

        for model in settings.models:
            if should_stop():
                status = "interrupted"
                break
            try:
                record = self.backend.describe_model(model)
            except Exception as exc:
                log.error("skipping %s: metadata unavailable (%r)", model, exc)
                skipped.append(model)
                continue
            self.store.add_run_model(header.id, record.to_dict())
            preloaded = self._is_loaded(model)
            if preloaded:
                self.progress(f"{model}: already resident before this run; it will not be unloaded")

            model_attempts, model_failures, stopped = self._run_model(
                header.id, model, tasks, should_stop
            )
            attempts += model_attempts
            failures += model_failures
            self.store.set_residency(header.id, model, self._residency(model))

            evict = settings.unload_after_run and not preloaded and not settings.dry_run
            if evict and self.backend.unload(model):
                self.progress(f"{model}: unloaded (loaded by this run)")
            if stopped:
                status = "interrupted"
                break

        if failures and status == "completed":
            status = "completed_with_errors"
        self.store.finish_run(header.id, status)
        self.progress(f"run {header.id} {status}: {attempts} attempt(s), {failures} failure(s)")
        return RunOutcome(header.id, status, attempts, failures, skipped)

    def _is_loaded(self, model: str) -> bool:
        checker = getattr(self.backend, "is_loaded", None)
        return bool(checker(model)) if checker else False

    def _residency(self, model: str) -> dict[str, Any] | None:
        fn = getattr(self.backend, "residency", None)
        return fn(model) if fn else None

    def _run_model(
        self,
        run_id: str,
        model: str,
        tasks: list[Task],
        should_stop: Callable[[], bool],
    ) -> tuple[int, int, bool]:
        settings = self.settings
        attempts = 0
        failures = 0

        warmup_task = tasks[0]
        for i in range(settings.warmup_runs):
            if should_stop():
                return attempts, failures, True
            self.progress(f"{model}: warmup {i + 1}/{settings.warmup_runs}")
            self._attempt(run_id, model, warmup_task, repetition=i, is_warmup=True)
            attempts += 1

        for task in tasks:
            for rep in range(settings.repetitions):
                if should_stop():
                    return attempts, failures, True
                self.progress(f"{model}: {task.id} rep {rep + 1}/{settings.repetitions}")
                result, _ = self._attempt(run_id, model, task, repetition=rep, is_warmup=False)
                attempts += 1
                if not result.ok:
                    failures += 1
                    log.warning("%s / %s rep %d failed: %s", model, task.id, rep, result.error)
        return attempts, failures, False

    def _attempt(
        self, run_id: str, model: str, task: Task, repetition: int, is_warmup: bool
    ) -> tuple[GenerationResult, TaskScore | None]:
        settings = self.settings
        options = build_options(settings, task)
        prompt, _ = longctx.render(task.prompt, task.filler_tokens, seed=settings.seed or 1234)

        sampler = TelemetrySampler(
            interval_s=settings.telemetry_interval_s, gpu_index=settings.gpu_index
        )
        sampler.start()
        try:
            result = self.backend.generate(
                model,
                prompt,
                system=task.system,
                options=options,
                tools=task.tools or None,
                think=task.think,
                keep_alive=settings.keep_alive,
                timeout_s=settings.request_timeout_s,
            )
        finally:
            telemetry = sampler.stop()

        score: TaskScore | None = None
        if not is_warmup:
            spec = dict(task.validator)
            if spec.get("kind") == "python_cases":
                spec["allow_code_execution"] = settings.allow_code_execution
            score = validators.grade(spec, result)

        self.store.add_attempt(
            run_id=run_id,
            model=model,
            task_id=task.id,
            category=task.category,
            repetition=repetition,
            is_warmup=is_warmup,
            result=result,
            score=score,
            telemetry=telemetry.to_dict(include_samples=False),
        )
        return result, score


def estimate_run_cost(
    settings: RunSettings, tasks: list[Task], seconds_per_attempt: float = 25.0
) -> dict[str, Any]:
    """Rough wall-clock estimate for a planned run. ESTIMATE only."""
    per_model = settings.warmup_runs + len(tasks) * settings.repetitions
    total = per_model * len(settings.models)
    return {
        "models": len(settings.models),
        "tasks": len(tasks),
        "repetitions": settings.repetitions,
        "attempts_per_model": per_model,
        "total_attempts": total,
        "assumed_seconds_per_attempt": seconds_per_attempt,
        "estimated_minutes": total * seconds_per_attempt / 60.0,
        "basis": "estimate; actual time depends on model speed and output length",
    }
