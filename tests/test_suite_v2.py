"""Tests for suite v2 and the suite-composition mechanism it uses.

v2 exists because v1's long-context category stopped discriminating: five
models scored 1.00 on all three tasks, at every context length measured. These
tests are mostly about the ways a harder suite can be quietly wrong - a needle
value that the filler also produces, an answer that no longer matches the
generated document, a prompt that cannot fit its own context window - because
each of those would show up as a model failure rather than a suite bug.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from caliper.evals import load_suite
from caliper.evals.longctx import WORDS_PER_TOKEN, make_filler, render
from caliper.evals.spec import CATEGORIES

LONG_CTX = "long_context"


@pytest.fixture(scope="module")
def v1():
    return load_suite("v1")


@pytest.fixture(scope="module")
def v2():
    return load_suite("v2")


# --------------------------------------------------------------- composition


def test_v2_inherits_every_category_it_does_not_replace(v1, v2):
    assert v2.suite_version == "2.0.0"
    assert set(v2.categories) == set(CATEGORIES)
    inherited = {t.id for t in v2.tasks} - {t.id for t in v2.by_category([LONG_CTX])}
    assert inherited == {t.id for t in v1.tasks if t.category != LONG_CTX}


def test_v2_replaces_the_long_context_tasks_entirely(v1, v2):
    old = {t.id for t in v1.by_category([LONG_CTX])}
    new = {t.id for t in v2.by_category([LONG_CTX])}
    assert not (old & new), "a reused task id would reuse v1's fixture for a v2 task"
    assert len(new) > len(old)


def test_inherited_tasks_are_the_same_objects_not_copies(v1, v2):
    """Inheritance must not drift: a v1 edit changes v2's inherited tasks too.

    That is the point of extending rather than copying, and it is why the
    manifest pins the base version - see the version-pin test below.
    """
    v1_by_id = {t.id for t in v1.tasks}
    for task in v2.tasks:
        if task.category == LONG_CTX:
            continue
        assert task.to_dict() == v1.get(task.id).to_dict()
        assert task.id in v1_by_id


def _manifest(tmp_path: Path, body: str, tasks: str = "") -> Path:
    suite = tmp_path / "child"
    suite.mkdir(exist_ok=True)
    (suite / "suite.toml").write_text(body)
    if tasks:
        (suite / "tasks.toml").write_text(tasks)
    base = tmp_path / "parent"
    base.mkdir(exist_ok=True)
    (base / "x.toml").write_text(
        'suite_version = "1.0.0"\n'
        '[[task]]\nid = "coding.a"\ncategory = "coding"\ntitle = "t"\nprompt = "p"\n'
        '[task.validator]\nkind = "exact"\nexpected = "y"\n'
        '[[task]]\nid = "longctx.a"\ncategory = "long_context"\ntitle = "t"\nprompt = "p"\n'
        '[task.validator]\nkind = "exact"\nexpected = "y"\n'
    )
    return tmp_path


REPLACEMENT = (
    'suite_version = "2.0.0"\n'
    '[[task]]\nid = "longctx.b"\ncategory = "long_context"\ntitle = "t"\nprompt = "p"\n'
    '[task.validator]\nkind = "exact"\nexpected = "y"\n'
)


def test_extends_composes_a_child_suite(tmp_path: Path):
    root = _manifest(
        tmp_path,
        'suite_version = "2.0.0"\nextends = "parent"\nextends_version = "1.0.0"\n'
        'replaces = ["long_context"]\n',
        REPLACEMENT,
    )
    suite = load_suite("child", suites_dir=root)
    assert [t.id for t in suite.tasks] == ["coding.a", "longctx.b"]
    assert suite.suite_version == "2.0.0"


def test_a_stale_version_pin_fails_loudly(tmp_path: Path):
    """The hazard this guards: editing v1 silently changes what v2 measures.

    v2's own suite_version would not move, so stored v2 runs would claim to be
    comparable when they are not.
    """
    root = _manifest(
        tmp_path,
        'suite_version = "2.0.0"\nextends = "parent"\nextends_version = "0.9.0"\n'
        'replaces = ["long_context"]\n',
        REPLACEMENT,
    )
    with pytest.raises(ValueError, match=r"composed against parent 0\.9\.0"):
        load_suite("child", suites_dir=root)


def test_extends_without_a_version_pin_is_rejected(tmp_path: Path):
    root = _manifest(
        tmp_path,
        'suite_version = "2.0.0"\nextends = "parent"\nreplaces = ["long_context"]\n',
        REPLACEMENT,
    )
    with pytest.raises(ValueError, match="without pinning extends_version"):
        load_suite("child", suites_dir=root)


def test_replacing_a_category_without_providing_tasks_is_rejected(tmp_path: Path):
    """Otherwise the category vanishes from every run and no report says so."""
    root = _manifest(
        tmp_path,
        'suite_version = "2.0.0"\nextends = "parent"\nextends_version = "1.0.0"\n'
        'replaces = ["long_context", "reasoning"]\n',
        REPLACEMENT,
    )
    with pytest.raises(ValueError, match="provides no task"):
        load_suite("child", suites_dir=root)


def test_adding_tasks_to_an_unreplaced_category_is_rejected(tmp_path: Path):
    """Mixing inherited and new tasks in one category breaks comparability.

    The category mean would cover a different task set from the base suite's
    while both runs reported the same category name.
    """
    root = _manifest(
        tmp_path,
        'suite_version = "2.0.0"\nextends = "parent"\nextends_version = "1.0.0"\n'
        'replaces = ["long_context"]\n',
        REPLACEMENT + '[[task]]\nid = "coding.b"\ncategory = "coding"\ntitle = "t"\nprompt = "p"\n'
        '[task.validator]\nkind = "exact"\nexpected = "y"\n',
    )
    with pytest.raises(ValueError, match="without listing those categories"):
        load_suite("child", suites_dir=root)


def test_unknown_manifest_key_is_rejected(tmp_path: Path):
    root = _manifest(
        tmp_path,
        'suite_version = "2.0.0"\nextends = "parent"\nextends_version = "1.0.0"\n'
        'replaces = ["long_context"]\ntypo = 1\n',
        REPLACEMENT,
    )
    with pytest.raises(ValueError, match="unknown manifest keys"):
        load_suite("child", suites_dir=root)


def test_tasks_in_the_manifest_are_rejected(tmp_path: Path):
    root = _manifest(
        tmp_path,
        'suite_version = "2.0.0"\nextends = "parent"\nextends_version = "1.0.0"\n'
        'replaces = ["long_context"]\n'
        '[[task]]\nid = "z"\ncategory = "coding"\ntitle = "t"\nprompt = "p"\n'
        '[task.validator]\nkind = "exact"\nexpected = "y"\n',
        REPLACEMENT,
    )
    with pytest.raises(ValueError, match="declares the suite, not tasks"):
        load_suite("child", suites_dir=root)


def test_replacing_an_unknown_category_is_rejected(tmp_path: Path):
    root = _manifest(
        tmp_path,
        'suite_version = "2.0.0"\nextends = "parent"\nextends_version = "1.0.0"\n'
        'replaces = ["long_contxt"]\n',
        REPLACEMENT,
    )
    with pytest.raises(ValueError, match="unknown categories"):
        load_suite("child", suites_dir=root)


# ------------------------------------------------------- the v2 tasks proper


def rendered(task):
    prompt, changed = render(task.prompt, task.filler_tokens)
    assert changed, task.id
    return prompt


def test_every_v2_long_context_prompt_fits_its_context_window(v2):
    """Prompt plus output budget must leave headroom in num_ctx.

    A prompt that overflows the window is truncated by the engine, which looks
    like a model that cannot read the document.
    """
    for task in v2.by_category([LONG_CTX]):
        approx = len(rendered(task).split()) / WORDS_PER_TOKEN
        assert task.context_length
        assert task.max_output_tokens
        assert approx + task.max_output_tokens < task.context_length * 0.95, task.id


def test_v2_document_sizes_stay_comparable_to_v1(v1, v2):
    """Document length is held, not increased, so difficulty is the only change.

    v1's stored runs averaged 7064-8280 prompt tokens per long-context attempt,
    because its placeholder substitutes the same block at every site and two of
    its tasks use it twice. If v2's documents were much longer, a score drop
    could not be attributed to the harder tasks.
    """
    sizes = {t.id: len(rendered(t).split()) / WORDS_PER_TOKEN for t in v2.by_category([LONG_CTX])}
    v1_sizes = [len(rendered(t).split()) / WORDS_PER_TOKEN for t in v1.by_category([LONG_CTX])]
    assert min(sizes.values()) >= min(v1_sizes), sizes
    assert max(sizes.values()) <= max(v1_sizes) * 1.5, sizes


def test_counting_task_expectation_matches_the_rendered_document(v2):
    """The hazard: the answer is a property of the generated document.

    If make_filler or the task's segment count ever changes the number of
    marked entries, the expected answer becomes wrong and every model is
    scored zero for a suite bug. This recomputes it from the real prompt.
    """
    task = v2.get("longctx.count_markers")
    prompt = rendered(task)
    assert prompt.count("Status: ESCALATED") == int(task.validator["expected"])


def test_needle_values_never_occur_in_the_filler(v2):
    """A needle the filler also produces would make a wrong answer look right."""
    filler = make_filler(max(t.filler_tokens or 0 for t in v2.by_category([LONG_CTX])))
    for marker in (
        "ESCALATED",
        "thirteen",
        "not stated",
        "THETA",
        "KAPPA",
        "SIGMA",
        "OMEGA",
        "RESULT",
        "RB-207",
        "CT-8814",
        "R-4471",
        "20764",
        "54875",
    ):
        assert marker.lower() not in filler.lower(), marker
    assert not re.search(r"\d{5,}", filler), "filler must not produce five-digit numbers"


def test_filler_entry_numbers_stay_clear_of_the_needle_numbers(v2):
    """Needles use entry numbers in the 9000s to stand apart from the filler."""
    for task in v2.by_category([LONG_CTX]):
        numbers = [int(n) for n in re.findall(r"^Entry (\d{4})\.", rendered(task), re.MULTILINE)]
        filler_numbers = [n for n in numbers if n < 9000]
        assert filler_numbers, task.id
        assert max(filler_numbers) < 9000, task.id


def test_each_v2_task_rejects_the_trap_it_was_built_around(v2):
    """Every task has a specific wrong answer it must score zero for.

    These are the answers a model gets by reading only part of the document:
    the superseded value, the near-miss needle, the first figure stated, the
    document order instead of the chronological one.
    """
    from caliper.evals.validators import grade
    from caliper.results import GenerationResult

    traps = {
        "longctx.multi_hop_chain": "Answer: R-3390",
        "longctx.resolve_supersession": "Answer: 45",
        "longctx.count_markers": "Answer: 5",
        "longctx.distractor_precision": "Answer: 73915",
        "longctx.absent_fact": "Answer: R-4471",
        "longctx.event_ordering": "Order: THETA, SIGMA, KAPPA, OMEGA",
        "longctx.sum_scattered": "Answer: 60000",
        "longctx.embedded_instruction_deep": "RESULT: 7 incidents",
    }
    correct = {
        "longctx.multi_hop_chain": "Answer: R-4471",
        "longctx.resolve_supersession": "Answer: 21",
        "longctx.count_markers": "Answer: 6",
        "longctx.distractor_precision": "Answer: 20,764",
        "longctx.absent_fact": "Answer: NOT STATED",
        "longctx.event_ordering": "Order: KAPPA, OMEGA, THETA, SIGMA",
        "longctx.sum_scattered": "Answer: 54875",
        "longctx.embedded_instruction_deep": "RESULT: 13 incidents",
    }
    ids = {t.id for t in v2.by_category([LONG_CTX])}
    assert set(traps) == ids, "every long-context task needs a trap case here"
    for task in v2.by_category([LONG_CTX]):

        def result(text):
            return GenerationResult(model="m", engine="mock", text=text)

        assert grade(task.validator, result(correct[task.id])).passed is True, task.id
        assert grade(task.validator, result(traps[task.id])).passed is False, task.id


def test_correct_answers_survive_common_formatting_variation(v2):
    """Formatting strictness belongs to instruction_following, not here."""
    from caliper.evals.validators import grade
    from caliper.results import GenerationResult

    variants = {
        "longctx.distractor_precision": ["Answer: 20764", "Answer: 20,764", "**Answer:** 20764"],
        "longctx.sum_scattered": ["Answer: 54875", "Answer: 54,875", "answer - 54875"],
        "longctx.multi_hop_chain": ["Answer: R-4471", "Answer: R 4471", "**Answer:** R-4471"],
        "longctx.absent_fact": ["Answer: NOT STATED", "Answer: not stated", "Answer: Not Stated"],
        "longctx.event_ordering": [
            "Order: KAPPA, OMEGA, THETA, SIGMA",
            "Order: KAPPA -> OMEGA -> THETA -> SIGMA",
            "Order: kappa; omega; theta; sigma",
        ],
    }
    for task_id, texts in variants.items():
        task = v2.get(task_id)
        for text in texts:
            result = GenerationResult(model="m", engine="mock", text=text)
            assert grade(task.validator, result).passed is True, (task_id, text)
