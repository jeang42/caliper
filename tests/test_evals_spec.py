from __future__ import annotations

from pathlib import Path

import pytest

from caliper.evals import load_suite
from caliper.evals.longctx import PLACEHOLDER, make_filler, render
from caliper.evals.spec import CATEGORIES


def test_v1_suite_is_well_formed():
    suite = load_suite("v1")
    assert suite.suite_version == "1.1.0"
    assert len(suite.tasks) >= 20
    assert set(suite.categories) == set(CATEGORIES)
    for task in suite.tasks:
        assert task.prompt.strip()
        assert task.title.strip()
        assert task.id.startswith(task.category.split("_")[0][:4]) or "." in task.id


def test_task_ids_are_unique():
    ids = [t.id for t in load_suite("v1").tasks]
    assert len(ids) == len(set(ids))


def test_tool_tasks_declare_tools():
    for task in load_suite("v1").by_category(["tool_use"]):
        if task.validator["kind"] == "tool_call":
            assert task.tools, task.id


def test_long_context_tasks_use_the_filler_placeholder():
    tasks = load_suite("v1").by_category(["long_context"])
    assert tasks
    for task in tasks:
        assert PLACEHOLDER in task.prompt
        assert task.filler_tokens


def test_unknown_category_is_rejected(tmp_path: Path):
    suite_dir = tmp_path / "bad"
    suite_dir.mkdir()
    (suite_dir / "x.toml").write_text(
        'suite_version = "1"\n[[task]]\nid = "x"\ncategory = "nope"\n'
        'title = "t"\nprompt = "p"\n[task.validator]\nkind = "exact"\nexpected = "y"\n'
    )
    with pytest.raises(ValueError, match="unknown category"):
        load_suite("bad", suites_dir=tmp_path)


def test_missing_validator_kind_is_rejected(tmp_path: Path):
    suite_dir = tmp_path / "bad2"
    suite_dir.mkdir()
    (suite_dir / "x.toml").write_text(
        'suite_version = "1"\n[[task]]\nid = "x"\ncategory = "coding"\n'
        'title = "t"\nprompt = "p"\n[task.validator]\nexpected = "y"\n'
    )
    with pytest.raises(ValueError, match="missing 'kind'"):
        load_suite("bad2", suites_dir=tmp_path)


def test_unknown_task_key_is_rejected(tmp_path: Path):
    suite_dir = tmp_path / "bad3"
    suite_dir.mkdir()
    (suite_dir / "x.toml").write_text(
        'suite_version = "1"\n[[task]]\nid = "x"\ncategory = "coding"\n'
        'title = "t"\nprompt = "p"\ntypo_field = 1\n'
        '[task.validator]\nkind = "exact"\nexpected = "y"\n'
    )
    with pytest.raises(ValueError, match="unknown keys"):
        load_suite("bad3", suites_dir=tmp_path)


def test_missing_suite_directory_raises():
    with pytest.raises(FileNotFoundError):
        load_suite("does-not-exist")


def test_filler_is_deterministic_and_sized():
    a = make_filler(500, seed=7)
    b = make_filler(500, seed=7)
    c = make_filler(500, seed=8)
    assert a == b
    assert a != c
    assert 250 < len(a.split()) < 600


def test_render_substitutes_every_placeholder():
    prompt = f"A {PLACEHOLDER} B {PLACEHOLDER} C"
    out, changed = render(prompt, 100)
    assert changed is True
    assert PLACEHOLDER not in out
    assert out.startswith("A ") and out.endswith(" C")


def test_render_leaves_plain_prompts_alone():
    out, changed = render("no filler here", 100)
    assert out == "no filler here" and changed is False


def test_every_regex_pattern_in_the_suite_compiles():
    """Guards against Perl-only syntax such as \\z, which Python rejects."""
    import re

    for task in load_suite("v1").tasks:
        for pattern in task.validator.get("patterns", []):
            re.compile(pattern)
        if "answer_pattern" in task.validator:
            re.compile(task.validator["answer_pattern"])


def test_no_validator_errors_against_the_fixtures():
    """Every task must grade cleanly on its canned response."""
    from caliper.backends import mock_backend_for_suite
    from caliper.evals.validators import grade

    suite = load_suite("v1")
    backend = mock_backend_for_suite(suite, models=["m:1b"])
    for task in suite.tasks:
        spec = dict(task.validator)
        if spec["kind"] == "python_cases":
            spec["allow_code_execution"] = True
        score = grade(spec, backend.generate("m:1b", task.prompt))
        assert score.graded != "error", f"{task.id}: {score.detail}"
