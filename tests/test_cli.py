"""CLI tests. Every command exercised here is read-only or mock-backed."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from caliper.cli import main
from caliper.hardware import TelemetrySampler


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CALIPER_HOME", str(tmp_path / "data"))
    monkeypatch.delenv("CALIPER_CONFIG", raising=False)
    monkeypatch.setattr("caliper.hardware.sample_gpus", lambda: [])
    monkeypatch.setattr("caliper.hardware.gpu_processes", lambda: [])
    monkeypatch.setattr(TelemetrySampler, "interval_s", 0.01, raising=False)


def test_tasks_command_lists_the_suite(capsys):
    assert main(["tasks"]) == 0
    out = capsys.readouterr().out
    assert "coding.rle_encode" in out
    assert "version 1.2.0" in out


def test_tasks_command_filters_by_category(capsys):
    assert main(["tasks", "--category", "tool_use"]) == 0
    out = capsys.readouterr().out
    assert "tool.select_single" in out
    assert "coding.rle_encode" not in out


def test_profiles_command(capsys):
    assert main(["profiles"]) == 0
    assert "Coding assistant" in capsys.readouterr().out


def test_plan_is_free_and_shows_estimates(capsys):
    code = main(["plan", "--dry-run", "--model", "mock-small:1b", "--repetitions", "2"])
    assert code == 0
    out = capsys.readouterr().out
    assert "Estimated cost (ESTIMATE)" in out
    assert "total_attempts" in out
    assert "SKIPPED without --allow-code-execution" in out


def test_dry_run_then_report_and_export(capsys, tmp_path: Path):
    assert (
        main(
            [
                "run",
                "--dry-run",
                "--model",
                "mock-small:1b",
                "--category",
                "instruction_following",
                "--repetitions",
                "1",
                "--warmup",
                "0",
                "--label",
                "ci-smoke",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "run id:" in out

    assert main(["runs"]) == 0
    runs_out = capsys.readouterr().out
    assert "ci-smoke" in runs_out
    assert "yes" in runs_out  # dry_run column

    assert main(["report"]) == 0
    report = capsys.readouterr().out
    assert "DRY RUN" in report
    assert "Quality by category" in report

    assert main(["export", "--out-dir", str(tmp_path / "out")]) == 0
    export_out = capsys.readouterr().out
    assert "wrote" in export_out
    files = list((tmp_path / "out").iterdir())
    assert len(files) == 3
    payload = json.loads(next(f for f in files if f.suffix == ".json").read_text())
    assert payload["run"]["dry_run"] is True


def test_report_json_output(capsys, tmp_path: Path):
    main(
        [
            "run",
            "--dry-run",
            "--model",
            "mock-small:1b",
            "--task",
            "instruction.exact_format",
            "--repetitions",
            "1",
            "--warmup",
            "0",
        ]
    )
    capsys.readouterr()
    out_path = tmp_path / "r.json"
    assert main(["report", "--json", "-o", str(out_path)]) == 0
    payload = json.loads(out_path.read_text())
    assert payload["models"][0]["model"] == "mock-small:1b"


def test_report_without_runs_exits_nonzero(capsys):
    assert main(["report"]) == 2
    assert "no runs stored yet" in capsys.readouterr().err


def test_unknown_task_id_is_a_clean_error(capsys):
    assert main(["run", "--dry-run", "--model", "m", "--task", "nope"]) == 2
    assert "nope" in capsys.readouterr().err


def test_preflight_json(capsys, monkeypatch):
    monkeypatch.setattr(
        "caliper.safety.hardware.sample_gpus",
        lambda: [],
    )
    code = main(["preflight", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert "checks" in payload
    assert code in (0, 1)


def test_vllm_backend_is_reported_as_unimplemented(capsys):
    assert main(["models", "--engine", "vllm"]) == 2
    assert "not implemented" in capsys.readouterr().err


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "caliper" in capsys.readouterr().out


def test_serve_refuses_a_non_loopback_bind_without_the_flag(capsys, tmp_path, monkeypatch):
    """The viewer has no authentication, so binding outward must be explicit."""
    monkeypatch.setenv("CALIPER_HOME", str(tmp_path / "data"))
    main(
        [
            "run",
            "--dry-run",
            "--model",
            "mock-small:1b",
            "--task",
            "instruction.exact_format",
            "--repetitions",
            "1",
            "--warmup",
            "0",
        ]
    )
    capsys.readouterr()
    assert main(["serve", "--host", "0.0.0.0"]) == 2
    err = capsys.readouterr().err
    assert "refusing to bind" in err
    assert "ssh -L" in err


def test_serve_without_a_database_explains_rather_than_starting(capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("CALIPER_HOME", str(tmp_path / "empty"))
    assert main(["serve"]) == 2
    assert "no database" in capsys.readouterr().err


def test_regrade_cli_reports_changes(capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("CALIPER_HOME", str(tmp_path / "data"))
    main(
        [
            "run",
            "--dry-run",
            "--model",
            "mock-small:1b",
            "--category",
            "instruction_following",
            "--repetitions",
            "1",
            "--warmup",
            "0",
        ]
    )
    capsys.readouterr()
    assert main(["regrade", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "re-graded" in out
    assert "nothing was written" in out


def test_regrade_without_runs_exits_nonzero(capsys):
    assert main(["regrade"]) == 2
    assert "no runs stored yet" in capsys.readouterr().err
