from __future__ import annotations

import re
from pathlib import Path

import pytest

from caliper.evals import load_suite
from caliper.evals.longctx import INDEXED_PLACEHOLDER, PLACEHOLDER, make_filler, render
from caliper.evals.spec import CATEGORIES

#: Every installed suite. Suite-wide invariants are asserted for all of them,
#: not just v1; v2 inherits most of its tasks, but its own files and its
#: composition can break these just as easily.
SUITES = ("v1", "v2")

INDEX1 = "{{FILLER:1}}"
INDEX2 = "{{FILLER:2}}"
INDEX3 = "{{FILLER:3}}"


def test_v1_suite_is_well_formed():
    suite = load_suite("v1")
    assert suite.suite_version == "1.2.0"
    assert len(suite.tasks) >= 20
    assert set(suite.categories) == set(CATEGORIES)
    for task in suite.tasks:
        assert task.prompt.strip()
        assert task.title.strip()
        assert task.id.startswith(task.category.split("_")[0][:4]) or "." in task.id


@pytest.mark.parametrize("name", SUITES)
def test_task_ids_are_unique(name: str):
    ids = [t.id for t in load_suite(name).tasks]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("name", SUITES)
def test_tool_tasks_declare_tools(name: str):
    for task in load_suite(name).by_category(["tool_use"]):
        if task.validator["kind"] == "tool_call":
            assert task.tools, task.id


@pytest.mark.parametrize("name", SUITES)
def test_long_context_tasks_use_the_filler_placeholder(name: str):
    tasks = load_suite(name).by_category(["long_context"])
    assert tasks
    for task in tasks:
        assert PLACEHOLDER in task.prompt or INDEXED_PLACEHOLDER.search(task.prompt), task.id
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


def test_legacy_placeholder_repeats_identical_text():
    """v1's prompts were measured with this behaviour, so it must not change.

    Every ``{{FILLER}}`` site gets the same block, which is why a v1 document
    with two filler blocks repeats itself and repeats its entry numbers. The
    indexed form exists to avoid that; this one is frozen.
    """
    out, _ = render(f"A{PLACEHOLDER}B{PLACEHOLDER}C", 300)
    first, second = out[1:].split("B", 1)
    assert first == second[: -len("C")]


def test_indexed_placeholders_get_distinct_text_and_continuous_numbering():
    # The separator must be something the filler prose cannot contain; plain
    # letters appear inside it and split the blocks in the wrong places.
    sep = "\n=====\n"
    out, changed = render(f"{INDEX1}{sep}{INDEX2}{sep}{INDEX3}", 900)
    assert changed is True
    assert not INDEXED_PLACEHOLDER.search(out)
    blocks = out.split(sep)
    assert len(blocks) == 3
    assert len(set(blocks)) == 3, "each indexed segment must be different prose"
    entries = [int(n) for n in re.findall(r"^Entry (\d{4})\.", out, re.MULTILINE)]
    assert entries == sorted(entries), "entry numbers must increase down the document"
    assert len(entries) == len(set(entries)), "entry numbers must not repeat"


def test_indexed_filler_tokens_is_the_total_not_the_per_segment_size():
    """A task author sizes the document, not each needle gap.

    If filler_tokens were per segment, adding a needle would silently inflate
    the prompt and could push it past the context window.
    """
    one, _ = render(INDEX1, 2000)
    three, _ = render(f"{INDEX1}\n{INDEX2}\n{INDEX3}", 2000)
    assert 0.8 < len(three.split()) / len(one.split()) < 1.2


def test_indexed_filler_is_deterministic():
    prompt = f"{INDEX1} x {INDEX2}"
    assert render(prompt, 600) == render(prompt, 600)
    assert render(prompt, 600)[0] != render(prompt, 600, seed=99)[0]


@pytest.mark.parametrize("name", SUITES)
def test_every_regex_pattern_in_the_suite_compiles(name: str):
    """Guards against Perl-only syntax such as \\z, which Python rejects."""
    import re

    for task in load_suite(name).tasks:
        for pattern in task.validator.get("patterns", []):
            re.compile(pattern)
        if "answer_pattern" in task.validator:
            re.compile(task.validator["answer_pattern"])


@pytest.mark.parametrize("name", SUITES)
def test_no_validator_errors_against_the_fixtures(name: str):
    """Every task must grade cleanly on its canned response."""
    from caliper.backends import mock_backend_for_suite
    from caliper.evals.validators import grade

    suite = load_suite(name)
    backend = mock_backend_for_suite(suite, models=["m:1b"])
    for task in suite.tasks:
        spec = dict(task.validator)
        if spec["kind"] == "python_cases":
            spec["allow_code_execution"] = True
        score = grade(spec, backend.generate("m:1b", task.prompt))
        assert score.graded != "error", f"{task.id}: {score.detail}"
