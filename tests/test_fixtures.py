"""Tests for the dry-run fixtures.

These assert that the canned responses really do drive each validator, which
is what makes a dry run a meaningful smoke test of the grading pipeline.
"""

from __future__ import annotations

import pytest

from caliper.backends import mock_backend_for_suite
from caliper.evals import load_suite
from caliper.evals.fixtures import FIXTURES, missing_fixture_ids
from caliper.evals.longctx import render as longctx_render
from caliper.evals.validators import grade


@pytest.mark.parametrize("name", ["v1", "v2"])
def test_every_task_has_a_fixture(name: str):
    assert missing_fixture_ids(load_suite(name)) == []


def test_no_stale_fixture_ids():
    ids = {t.id for t in load_suite("v1").tasks} | {t.id for t in load_suite("v2").tasks}
    assert set(FIXTURES) <= ids


def test_fixtures_produce_a_mix_of_outcomes():
    """A dry run must exercise pass, partial and fail paths, not only zeros."""
    suite = load_suite("v1")
    backend = mock_backend_for_suite(suite, models=["m:1b"])
    passed = failed = pending = 0
    for task in suite.tasks:
        result = backend.generate("m:1b", task.prompt)
        spec = dict(task.validator)
        if spec["kind"] == "python_cases":
            spec["allow_code_execution"] = True
        score = grade(spec, result)
        if score.graded == "manual_rubric":
            pending += 1
        elif score.passed:
            passed += 1
        else:
            failed += 1
    assert passed >= 10, "fixtures should pass most automatic tasks"
    assert failed >= 2, "fixtures should also exercise failure paths"
    assert pending == 5


def test_v2_long_context_fixtures_exercise_pass_partial_and_fail():
    """A dry run of v2 must show the harder validators working, not just running.

    Each of these tasks has a trap, and a fixture that always answers
    correctly would leave the trap branches untested.
    """
    suite = load_suite("v2")
    backend = mock_backend_for_suite(suite, models=["m:1b"])
    scores = {}
    for task in suite.by_category(["long_context"]):
        prompt, changed = longctx_render(task.prompt, task.filler_tokens)
        assert changed, task.id
        scores[task.id] = grade(task.validator, backend.generate("m:1b", prompt)).score
    assert sorted(scores.values()) == [0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0], scores


def test_prompt_keys_never_returns_a_key_containing_a_placeholder():
    """A key with a placeholder in it can never match a rendered prompt.

    ``prompt_keys`` falls back to the whole prompt when neither the head nor
    the tail is unique, and for a long-context task that whole prompt still
    contains ``{{FILLER}}``. The mock backend would then never match, and the
    task would be scored on an empty response - a suite bug that looks exactly
    like a model that answered nothing.
    """
    from caliper.evals.fixtures import prompt_keys
    from caliper.evals.longctx import INDEXED_PLACEHOLDER, PLACEHOLDER

    for name in ("v1", "v2"):
        for key in prompt_keys(load_suite(name)).values():
            assert PLACEHOLDER not in key
            assert not INDEXED_PLACEHOLDER.search(key)


def test_tool_fixtures_emit_tool_calls():
    suite = load_suite("v1")
    backend = mock_backend_for_suite(suite, models=["m:1b"])
    task = suite.get("tool.select_single")
    result = backend.generate("m:1b", task.prompt)
    assert result.tool_calls
    assert grade(task.validator, result).passed is True


def test_abstain_fixture_emits_no_tool_call():
    suite = load_suite("v1")
    backend = mock_backend_for_suite(suite, models=["m:1b"])
    result = backend.generate("m:1b", suite.get("tool.abstain_when_unavailable").prompt)
    assert result.tool_calls == []


def test_long_context_fixture_matches_after_filler_substitution():
    from caliper.evals.longctx import render

    suite = load_suite("v1")
    backend = mock_backend_for_suite(suite, models=["m:1b"])
    task = suite.get("longctx.retrieve_fact")
    prompt, changed = render(task.prompt, task.filler_tokens)
    assert changed
    result = backend.generate("m:1b", prompt)
    assert grade(task.validator, result).passed is True


def test_prompt_keys_are_unique_per_task():
    from caliper.evals.fixtures import prompt_keys

    suite = load_suite("v1")
    keys = prompt_keys(suite)
    assert len(set(keys.values())) == len(suite.tasks)
    for task in suite.tasks:
        matches = [t.id for t in suite.tasks if keys[task.id] in t.prompt]
        assert matches == [task.id]
