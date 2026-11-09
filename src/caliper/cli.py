"""Caliper command-line interface.

Operations are deliberately separate: ``inventory`` and ``preflight`` are
read-only, ``plan`` costs nothing, ``run`` is the only command that performs
inference, and ``report``/``export`` only read stored results.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import __version__
from . import safety as safety_mod
from .backends import get_backend, mock_backend_for_suite
from .config import Config, load_config
from .evals import load_suite
from .evals.spec import CATEGORIES
from .export import export_attempts_csv, export_run_json, export_summary_csv
from .inventory import collect_inventory, render_report
from .logging_setup import setup_logging
from .regrade import regrade_run
from .report import PROFILES, render_markdown, summarize_run
from .runner import (
    BenchmarkRunner,
    RunAborted,
    RunSettings,
    estimate_run_cost,
    filler_fit_problems,
    prompt_token_ratios,
)
from .server import serve
from .storage import Store

log = logging.getLogger("caliper")


def _store(config: Config) -> Store:
    return Store(config.db_path)


def _write(path: Path | None, text: str) -> None:
    if path is None:
        sys.stdout.write(text if text.endswith("\n") else text + "\n")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_inventory(args: argparse.Namespace, config: Config) -> int:
    backends = [get_backend(name, config) for name in args.engines]
    inv = collect_inventory(backends, detail=not args.fast, disk_paths=args.disk or ["/"])
    payload = inv.to_dict()

    if args.json:
        _write(Path(args.json), json.dumps(payload, indent=2, default=str))
    if args.markdown:
        _write(Path(args.markdown), render_report(inv, gpu_index=args.gpu_index))
    if not args.json and not args.markdown:
        sys.stdout.write(render_report(inv, gpu_index=args.gpu_index) + "\n")
    if not args.no_store:
        with _store(config) as store:
            inv_id = store.save_inventory(payload, inv.machine.hostname, ",".join(args.engines))
            print(f"stored inventory {inv_id} in {config.db_path}", file=sys.stderr)
    return 0


def cmd_models(args: argparse.Namespace, config: Config) -> int:
    backend = get_backend(args.engine, config)
    if not backend.available():
        print(f"engine {args.engine} is not reachable", file=sys.stderr)
        return 2
    records = backend.list_models()
    if args.detail or args.moe_only:
        records = [backend.describe_model(r.name) for r in records]
    if args.moe_only:
        records = [r for r in records if r.is_moe]
    records.sort(key=lambda r: r.name)
    if args.json:
        print(json.dumps([r.to_dict() for r in records], indent=2, default=str))
        return 0
    for r in records:
        size = f"{r.size_bytes / 1024**3:6.2f} GiB" if r.size_bytes else "   unknown"
        extra = ""
        if r.expert_count is not None:
            extra = f"  MoE {r.expert_used_count}/{r.expert_count}"
        caps = f"  [{', '.join(r.capabilities)}]" if r.capabilities else ""
        print(f"{r.name:60s} {size}  {r.quantization or '?':10s}{extra}{caps}")
    print(f"\n{len(records)} model(s)")
    return 0


def cmd_preflight(args: argparse.Namespace, config: Config) -> int:
    if args.wait:
        result = safety_mod.wait_until_clear(
            config.safety,
            timeout_s=args.wait,
            gpu_index=args.gpu_index,
            disk_paths=args.disk or ["/"],
            on_wait=lambda res, left: print(
                f"waiting for the machine to free up ({left:.0f}s left): "
                + "; ".join(c.detail for c in res.blocking_failures),
                file=sys.stderr,
            ),
        )
    else:
        result = safety_mod.preflight(
            config.safety, gpu_index=args.gpu_index, disk_paths=args.disk or ["/"]
        )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
    else:
        for check in result.checks:
            if check.passed:
                mark = "NOTE" if check.severity == "warn" else "PASS"
            else:
                mark = "WARN" if check.severity == "warn" else "FAIL"
            print(f"[{mark}] {check.name}: {check.detail}")
        print()
        print("preflight: OK to run" if result.ok else "preflight: BLOCKED")
    return 0 if result.ok else 1


def _run_settings(args: argparse.Namespace, config: Config) -> RunSettings:
    run_cfg = config.run
    return RunSettings(
        models=args.model,
        suite=args.suite or run_cfg["suite"],
        categories=args.category or [],
        task_ids=args.task or [],
        repetitions=args.repetitions if args.repetitions is not None else run_cfg["repetitions"],
        warmup_runs=args.warmup if args.warmup is not None else run_cfg["warmup_runs"],
        temperature=args.temperature if args.temperature is not None else run_cfg["temperature"],
        seed=run_cfg["seed"] if args.seed is None else (None if args.seed < 0 else args.seed),
        max_output_tokens=run_cfg["max_output_tokens"],
        max_output_tokens_override=args.max_tokens,
        context_length=run_cfg["context_length"],
        context_length_override=args.context,
        filler_tokens_override=args.filler,
        keep_alive=args.keep_alive or run_cfg["keep_alive"],
        unload_after_run=not args.no_unload and run_cfg["unload_after_run"],
        request_timeout_s=args.timeout
        if args.timeout is not None
        else run_cfg["request_timeout_s"],
        telemetry_interval_s=run_cfg["telemetry_interval_s"],
        allow_code_execution=args.allow_code_execution,
        gpu_index=args.gpu_index,
        dry_run=args.dry_run,
        label=args.label,
        force=getattr(args, "force", False),
        wait_for_clear_s=getattr(args, "wait", None) or 0.0,
    )


def cmd_plan(args: argparse.Namespace, config: Config) -> int:
    settings = _run_settings(args, config)
    suite = load_suite(settings.suite)
    backend = mock_backend_for_suite(suite) if args.dry_run else get_backend(args.engine, config)
    store = Store(config.db_path)
    runner = BenchmarkRunner(backend, store, settings, config.safety)
    tasks = runner.select_tasks(suite)
    cost = estimate_run_cost(settings, tasks, store=store)
    print(f"Suite `{suite.name}` version {suite.suite_version}")
    print(f"Engine: {backend.name} (version {backend.version()})")
    print(f"Models: {', '.join(settings.models)}")
    print(f"Tasks selected: {len(tasks)}")
    by_cat: dict[str, int] = {}
    for t in tasks:
        by_cat[t.category] = by_cat.get(t.category, 0) + 1
    for cat, count in sorted(by_cat.items()):
        print(f"  {cat}: {count}")
    needs_exec = [t.id for t in tasks if t.requires_code_execution]
    if needs_exec and not settings.allow_code_execution:
        print(f"\n{len(needs_exec)} task(s) will be SKIPPED without --allow-code-execution:")
        for t in needs_exec:
            print(f"  {t}")
    fit = filler_fit_problems(settings, tasks, prompt_token_ratios(settings, store))
    if fit:
        print()
        print("Long-context documents that will NOT fit their context window:")
        for problem in fit:
            print(f"  {problem}")
        print("  `caliper run` refuses this without --force.")
    print()
    print("Estimated cost (ESTIMATE):")
    for key, value in cost.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.1f}")
        elif value in (None, {}, []):
            continue
        else:
            print(f"  {key}: {value}")
    print()
    print("Controls:")
    for key, value in settings.to_dict().items():
        print(f"  {key}: {value}")
    print()
    pre = runner.preflight()
    print("Preflight:", "OK" if pre.ok else "BLOCKED")
    for check in pre.checks:
        if not check.passed:
            print(f"  [{check.severity.upper()}] {check.name}: {check.detail}")
    return 0


def cmd_run(args: argparse.Namespace, config: Config) -> int:
    settings = _run_settings(args, config)
    suite = load_suite(settings.suite)
    engine = "mock" if settings.dry_run else args.engine
    backend = mock_backend_for_suite(suite) if settings.dry_run else get_backend(engine, config)
    if settings.dry_run and not settings.models:
        settings.models = [r.name for r in backend.list_models()]
    if not backend.available():
        print(f"engine {engine} is not reachable", file=sys.stderr)
        return 2

    with _store(config) as store:
        runner = BenchmarkRunner(
            backend, store, settings, config.safety, progress=lambda m: print(m, flush=True)
        )
        try:
            outcome = runner.run(suite)
        except RunAborted as exc:
            print(f"aborted: {exc}", file=sys.stderr)
            print(
                "Re-check with `caliper preflight`, or pass --force to override.", file=sys.stderr
            )
            return 3
        print()
        print(f"run id: {outcome.run_id}")
        if outcome.skipped_models:
            print(f"skipped models: {', '.join(outcome.skipped_models)}")
        print(f"next: caliper report --run {outcome.run_id}")
    return 0 if outcome.status in {"completed", "completed_with_errors"} else 1


def cmd_report(args: argparse.Namespace, config: Config) -> int:
    with _store(config) as store:
        run_id = args.run or store.latest_run_id()
        if run_id is None:
            print("no runs stored yet", file=sys.stderr)
            return 2
        summary = summarize_run(store, run_id)
        if args.json:
            summary.pop("_summaries", None)
            _write(
                Path(args.output) if args.output else None,
                json.dumps(summary, indent=2, default=str),
            )
        else:
            _write(Path(args.output) if args.output else None, render_markdown(summary))
    return 0


def cmd_regrade(args: argparse.Namespace, config: Config) -> int:
    with _store(config) as store:
        run_id = args.run or store.latest_run_id()
        if run_id is None:
            print("no runs stored yet", file=sys.stderr)
            return 2
        outcome = regrade_run(
            store,
            run_id,
            allow_code_execution=args.allow_code_execution,
            dry_run=args.dry_run,
        )
        print(outcome.summary_line())
        if args.dry_run:
            print("--dry-run: nothing was written")
        for task_id in outcome.missing_tasks:
            print(f"  skipped {task_id}: not in the installed suite", file=sys.stderr)
    return 0


def cmd_runs(args: argparse.Namespace, config: Config) -> int:
    with _store(config) as store:
        rows = store.list_runs(limit=args.limit)
        if not rows:
            print("no runs stored yet")
            return 0
        print(f"{'run id':18s} {'created':22s} {'engine':8s} {'status':24s} {'dry':4s} label")
        for row in rows:
            print(
                f"{row['id']:18s} {row['created_at']:22s} {row['engine']:8s} "
                f"{row['status']:24s} {'yes' if row['dry_run'] else 'no':4s} {row['label'] or ''}"
            )
    return 0


def cmd_export(args: argparse.Namespace, config: Config) -> int:
    with _store(config) as store:
        run_id = args.run or store.latest_run_id()
        if run_id is None:
            print("no runs stored yet", file=sys.stderr)
            return 2
        out_dir = Path(args.out_dir)
        written = [
            export_run_json(
                store, run_id, out_dir / f"run-{run_id}.json", include_responses=args.responses
            ),
            export_attempts_csv(store, run_id, out_dir / f"run-{run_id}-attempts.csv"),
            export_summary_csv(store, run_id, out_dir / f"run-{run_id}-summary.csv"),
        ]
    for path in written:
        print(f"wrote {path}")
    return 0


def cmd_tasks(args: argparse.Namespace, config: Config) -> int:
    suite = load_suite(args.suite)
    print(f"Suite `{suite.name}` version {suite.suite_version}: {len(suite.tasks)} tasks")
    for task in suite.by_category(args.category or []):
        flags = []
        if task.requires_code_execution:
            flags.append("code-exec")
        if task.tools:
            flags.append(f"{len(task.tools)} tool(s)")
        if task.filler_tokens:
            flags.append(f"filler~{task.filler_tokens}tok")
        suffix = f"  ({', '.join(flags)})" if flags else ""
        print(f"  [{task.category:20s}] {task.id:36s} {task.validator['kind']:14s}{suffix}")
        if args.verbose_tasks:
            print(f"      {task.title}")
    return 0


def cmd_serve(args: argparse.Namespace, config: Config) -> int:
    """Start the local viewer.

    Binding anywhere but loopback needs --allow-remote, because the viewer
    exposes machine telemetry, model inventory and raw model output and has no
    authentication. Use an SSH tunnel to reach it from another machine.
    """
    host = args.host
    if host not in {"127.0.0.1", "localhost", "::1"} and not args.allow_remote:
        print(
            f"refusing to bind {host}: the viewer has no authentication and exposes "
            "telemetry, inventory and raw model output.\n"
            "Use an SSH tunnel (ssh -L 8777:127.0.0.1:8777 <host>), or pass "
            "--allow-remote if you really mean it.",
            file=sys.stderr,
        )
        return 2
    if not config.db_path.exists():
        print(
            f"no database at {config.db_path}; run a benchmark first (try: caliper run --dry-run)",
            file=sys.stderr,
        )
        return 2
    serve(config, host=host, port=args.port)
    return 0


def cmd_profiles(args: argparse.Namespace, config: Config) -> int:
    for key, spec in PROFILES.items():
        print(f"{key}: {spec['label']}")
        quality = ", ".join(f"{k}x{v}" for k, v in spec["quality"].items())
        print(f"  quality: {quality}")
        print(
            f"  speedx{spec.get('speed', 0)}  latencyx{spec.get('latency', 0)}  "
            f"memoryx{spec.get('memory', 0)}"
        )
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="caliper",
        description="Local LLM benchmarking and evaluation framework.",
    )
    parser.add_argument("--version", action="version", version=f"caliper {__version__}")
    parser.add_argument("--config", help="path to a TOML config file")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument("--gpu-index", type=int, default=0)
    sub = parser.add_subparsers(dest="command", required=True)

    p_inv = sub.add_parser("inventory", help="collect a read-only environment inventory")
    p_inv.add_argument("--engines", nargs="+", default=["ollama"])
    p_inv.add_argument("--json", help="write the machine-readable inventory here")
    p_inv.add_argument("--markdown", help="write the human-readable report here")
    p_inv.add_argument("--fast", action="store_true", help="skip per-model metadata calls")
    p_inv.add_argument("--disk", nargs="+", help="disk paths to report on")
    p_inv.add_argument("--no-store", action="store_true", help="do not save to SQLite")
    p_inv.set_defaults(func=cmd_inventory)

    p_models = sub.add_parser("models", help="list models known to an engine")
    p_models.add_argument("--engine", default="ollama")
    p_models.add_argument("--detail", action="store_true", help="fetch full metadata per model")
    p_models.add_argument(
        "--moe-only", action="store_true", help="only MoE models (implies --detail)"
    )
    p_models.add_argument("--json", action="store_true")
    p_models.set_defaults(func=cmd_models)

    p_pre = sub.add_parser("preflight", help="check whether it is safe to run now")
    p_pre.add_argument("--json", action="store_true")
    p_pre.add_argument("--disk", nargs="+")
    p_pre.add_argument(
        "--wait",
        type=float,
        metavar="SECONDS",
        help="poll until the machine is free, up to this long, instead of failing at once",
    )
    p_pre.set_defaults(func=cmd_preflight)

    def add_run_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--engine", default="ollama", choices=["ollama", "mock"])
        p.add_argument("--model", action="append", default=[], help="repeatable")
        p.add_argument("--suite", help="suite name (default from config)")
        p.add_argument("--category", action="append", choices=list(CATEGORIES), default=[])
        p.add_argument("--task", action="append", default=[], help="task id, repeatable")
        p.add_argument("--repetitions", type=int)
        p.add_argument("--warmup", type=int)
        p.add_argument("--temperature", type=float)
        p.add_argument("--seed", type=int, help="negative value disables seeding")
        p.add_argument(
            "--max-tokens",
            type=int,
            help="num_predict; overrides each task's own budget when given",
        )
        p.add_argument(
            "--context", type=int, help="num_ctx; overrides each task's own value when given"
        )
        p.add_argument(
            "--filler",
            type=int,
            help=(
                "long-context document size in filler tokens; overrides each task's own "
                "filler_tokens. Raise --context to match, or the run is refused"
            ),
        )
        p.add_argument("--keep-alive")
        p.add_argument(
            "--no-unload", action="store_true", help="leave models resident after the run"
        )
        p.add_argument("--timeout", type=float, help="per-request timeout in seconds")
        p.add_argument(
            "--allow-code-execution",
            action="store_true",
            help="run model-written Python in a subprocess for code tasks",
        )
        p.add_argument("--label")
        p.add_argument(
            "--dry-run", action="store_true", help="use the mock backend; no GPU inference"
        )

    p_plan = sub.add_parser("plan", help="show what a run would do, without running it")
    add_run_args(p_plan)
    p_plan.set_defaults(func=cmd_plan)

    p_run = sub.add_parser("run", help="execute a benchmark run")
    add_run_args(p_run)
    p_run.add_argument("--force", action="store_true", help="override preflight failures")
    p_run.add_argument(
        "--wait",
        type=float,
        metavar="SECONDS",
        help="wait up to this long for a busy machine instead of aborting",
    )
    p_run.set_defaults(func=cmd_run)

    p_report = sub.add_parser("report", help="render a report for a stored run")
    p_report.add_argument("--run", help="run id (default: most recent)")
    p_report.add_argument("--json", action="store_true")
    p_report.add_argument("-o", "--output", help="write to this path instead of stdout")
    p_report.set_defaults(func=cmd_report)

    p_regrade = sub.add_parser(
        "regrade",
        help="re-score a stored run with the current validators (no inference)",
    )
    p_regrade.add_argument("--run", help="run id (default: most recent)")
    p_regrade.add_argument("--allow-code-execution", action="store_true")
    p_regrade.add_argument(
        "--dry-run", action="store_true", help="report what would change without writing"
    )
    p_regrade.set_defaults(func=cmd_regrade)

    p_runs = sub.add_parser("runs", help="list stored runs")
    p_runs.add_argument("--limit", type=int, default=25)
    p_runs.set_defaults(func=cmd_runs)

    p_export = sub.add_parser("export", help="export a run to JSON and CSV")
    p_export.add_argument("--run")
    p_export.add_argument("--out-dir", default="reports")
    p_export.add_argument(
        "--responses", action="store_true", help="include every raw model response in the JSON"
    )
    p_export.set_defaults(func=cmd_export)

    p_tasks = sub.add_parser("tasks", help="list evaluation tasks")
    p_tasks.add_argument("--suite", default="v1")
    p_tasks.add_argument("--category", action="append", choices=list(CATEGORIES), default=[])
    p_tasks.add_argument("--verbose-tasks", action="store_true")
    p_tasks.set_defaults(func=cmd_tasks)

    p_serve = sub.add_parser("serve", help="start the local read-only web viewer")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8777)
    p_serve.add_argument(
        "--allow-remote",
        action="store_true",
        help="permit binding a non-loopback address (no authentication is provided)",
    )
    p_serve.set_defaults(func=cmd_serve)

    p_prof = sub.add_parser("profiles", help="show workload weighting profiles")
    p_prof.set_defaults(func=cmd_profiles)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(verbose=args.verbose, quiet=args.quiet)
    config = load_config(args.config)
    try:
        return int(args.func(args, config))
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except (FileNotFoundError, KeyError, ValueError, NotImplementedError) as exc:
        log.error("%s", exc)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
