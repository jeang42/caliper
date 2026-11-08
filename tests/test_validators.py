from __future__ import annotations

import pytest

from caliper.evals import validators
from caliper.results import GenerationResult


def res(text: str = "", tool_calls=None) -> GenerationResult:
    return GenerationResult(model="m", engine="mock", text=text, tool_calls=list(tool_calls or []))


def test_exact_requires_exact_match():
    spec = {"kind": "exact", "expected": "A=1;B=2"}
    assert validators.grade(spec, res("A=1;B=2")).passed is True
    assert validators.grade(spec, res("Sure! A=1;B=2")).passed is False


def test_contains_all_gives_partial_credit():
    spec = {"kind": "contains_all", "needles": ["alpha", "beta", "gamma"]}
    score = validators.grade(spec, res("alpha and beta only"))
    assert score.score == pytest.approx(2 / 3)
    assert score.passed is False


def test_contains_all_forbidden_zeroes_the_score():
    spec = {"kind": "contains_all", "needles": ["-"], "forbidden": ["proceed"]}
    score = validators.grade(spec, res("- item\nproceed"))
    assert score.score == 0.0


def test_choice_uses_the_last_answer_marker():
    spec = {"kind": "choice", "answer_pattern": r"answer\s*[:\-]\s*([A-D])\b", "expected": "c"}
    text = "First I thought Answer: A, but on reflection.\nAnswer: C"
    assert validators.grade(spec, res(text)).passed is True


def test_numeric_tolerance_and_last_number():
    spec = {"kind": "numeric", "expected": 21.0, "tolerance": 0.11}
    assert validators.grade(spec, res("4 + 12.5 + 4.5\nAnswer: 21.0")).passed is True
    assert validators.grade(spec, res("Answer: 25")).passed is False
    assert validators.grade(spec, res("no digits here")).passed is False


def test_regex_all_partial():
    spec = {"kind": "regex_all", "patterns": [r"^RESULT:", r"\d+"]}
    score = validators.grade(spec, res("RESULT: no number"))
    assert score.score == pytest.approx(0.5)


def test_extract_json_from_fenced_block():
    value = validators.extract_json('```json\n{"a": 1}\n```')
    assert value == {"a": 1}


def test_extract_json_from_surrounding_prose():
    value = validators.extract_json('Here you go: {"a": 1, "b": [2]} - done')
    assert value == {"a": 1, "b": [2]}


def test_json_schema_rejects_extra_keys_and_bad_enum():
    schema = {
        "type": "object",
        "required": ["status"],
        "additionalProperties": False,
        "properties": {"status": {"type": "string", "enum": ["ok", "partial"]}},
    }
    spec = {"kind": "json_schema", "schema": schema}
    assert validators.grade(spec, res('{"status": "partial"}')).passed is True
    assert validators.grade(spec, res('{"status": "nope"}')).passed is False
    assert validators.grade(spec, res('{"status": "ok", "extra": 1}')).passed is False


def test_json_schema_checks_expected_values():
    spec = {
        "kind": "json_schema",
        "schema": {"type": "object", "required": ["n"], "properties": {"n": {"type": "integer"}}},
        "expected_values": {"n": 12},
    }
    good = validators.grade(spec, res('{"n": 12}'))
    bad = validators.grade(spec, res('{"n": 13}'))
    assert good.passed is True
    assert bad.passed is False
    assert bad.sub_scores["schema_valid"] == 1.0


def test_schema_boolean_is_not_an_integer():
    errors = validators.check_schema(True, {"type": "integer"})
    assert errors


def test_tool_call_scores_name_and_arguments_separately():
    spec = {
        "kind": "tool_call",
        "expected_function": "get_gpu_metrics",
        "expected_arguments": {"host": "ada", "unit": "MiB"},
    }
    good = validators.grade(
        spec,
        res(
            tool_calls=[
                {
                    "function": {
                        "name": "get_gpu_metrics",
                        "arguments": {"host": "ada", "unit": "MiB"},
                    }
                }
            ]
        ),
    )
    assert good.passed is True and good.score == 1.0

    wrong_args = validators.grade(
        spec,
        res(
            tool_calls=[
                {
                    "function": {
                        "name": "get_gpu_metrics",
                        "arguments": {"host": "ada", "unit": "GiB"},
                    }
                }
            ]
        ),
    )
    assert wrong_args.score == pytest.approx(0.75)
    assert wrong_args.sub_scores["function_correct"] == 1.0

    none = validators.grade(spec, res("I think it is about 400 MiB"))
    assert none.score == 0.0


def test_tool_call_accepts_json_encoded_arguments():
    spec = {"kind": "tool_call", "expected_function": "f", "expected_arguments": {"n": 3}}
    score = validators.grade(
        spec, res(tool_calls=[{"function": {"name": "f", "arguments": '{"n": 3}'}}])
    )
    assert score.passed is True


def test_rubric_never_invents_a_score():
    spec = {"kind": "rubric", "criteria": ["a", "b"], "signal_phrases": ["expert"]}
    score = validators.grade(spec, res("the expert routing matters"))
    assert score.score is None
    assert score.graded == "manual_rubric"
    assert score.sub_scores["signal_phrases_found"] == 1.0


def test_failed_generation_scores_zero():
    spec = {"kind": "exact", "expected": "x"}
    bad = GenerationResult(model="m", engine="mock", error="timeout")
    score = validators.grade(spec, bad)
    assert score.score == 0.0 and score.passed is False


def test_unknown_validator_kind_is_reported_not_raised():
    score = validators.grade({"kind": "nope"}, res("x"))
    assert score.graded == "error" and score.score is None


def test_python_cases_skipped_unless_enabled():
    spec = {"kind": "python_cases", "cases": [{"name": "a", "expression": "f()", "expected": 1}]}
    score = validators.grade(spec, res("```python\ndef f(): return 1\n```"))
    assert score.graded == "skipped" and score.score is None


def test_python_cases_runs_candidate_code_when_enabled():
    spec = {
        "kind": "python_cases",
        "allow_code_execution": True,
        "timeout_s": 20.0,
        "cases": [
            {"name": "empty", "expression": "rle('')", "expected": []},
            {"name": "runs", "expression": "rle('aab')", "expected": [["a", 2], ["b", 1]]},
        ],
    }
    code = (
        "```python\n"
        "def rle(s):\n"
        "    out = []\n"
        "    for ch in s:\n"
        "        if out and out[-1][0] == ch:\n"
        "            out[-1][1] += 1\n"
        "        else:\n"
        "            out.append([ch, 1])\n"
        "    return out\n"
        "```"
    )
    score = validators.grade(spec, res(code))
    assert score.score == 1.0 and score.passed is True


def test_python_cases_reports_failing_cases():
    spec = {
        "kind": "python_cases",
        "allow_code_execution": True,
        "cases": [
            {"name": "ok", "expression": "f(1)", "expected": 1},
            {"name": "bad", "expression": "f(2)", "expected": 99},
        ],
    }
    score = validators.grade(spec, res("```python\ndef f(x): return x\n```"))
    assert score.score == 0.5
    assert "bad" in score.detail


def test_python_cases_handles_uncompilable_code():
    spec = {
        "kind": "python_cases",
        "allow_code_execution": True,
        "cases": [{"name": "a", "expression": "f()", "expected": 1}],
    }
    score = validators.grade(spec, res("```python\ndef f(: pass\n```"))
    assert score.score == 0.0


def test_python_cases_treats_tuples_and_lists_as_equal():
    """Expected values come from JSON/TOML, which cannot express tuples."""
    spec = {
        "kind": "python_cases",
        "allow_code_execution": True,
        "cases": [
            {"name": "tuples", "expression": "f()", "expected": [["a", 1], ["b", 2]]},
        ],
    }
    score = validators.grade(spec, res("```python\ndef f(): return [('a', 1), ('b', 2)]\n```"))
    assert score.score == 1.0


def test_python_cases_still_detects_wrong_values():
    spec = {
        "kind": "python_cases",
        "allow_code_execution": True,
        "cases": [{"name": "x", "expression": "f()", "expected": [["a", 1]]}],
    }
    score = validators.grade(spec, res("```python\ndef f(): return [('a', 2)]\n```"))
    assert score.score == 0.0


def test_empty_output_fails_even_a_negative_constraint():
    """Regression: `[^e]*` matches the empty string, so an empty response
    used to score full marks on a "do not use the letter e" task."""
    spec = {"kind": "regex_all", "patterns": [r"(?si)\A[^e]*\Z"]}
    score = validators.grade(spec, res(""))
    assert score.score == 0.0
    assert score.passed is False
    assert "no visible output" in score.detail


def test_truncated_empty_output_says_why():
    result = GenerationResult(
        model="m",
        engine="mock",
        text="",
        thinking="a long internal monologue that used up the whole budget",
        raw_final={"done_reason": "length"},
    )
    score = validators.grade({"kind": "exact", "expected": "x"}, result)
    assert score.score == 0.0
    assert "token limit" in score.detail


def test_thinking_alone_is_not_visible_output():
    result = GenerationResult(model="m", engine="mock", text="   ", thinking="reasoning")
    assert result.has_visible_output is False
    assert validators.grade({"kind": "contains_all", "needles": ["x"]}, result).score == 0.0


def test_a_tool_call_counts_as_visible_output():
    result = GenerationResult(
        model="m", engine="mock", text="", tool_calls=[{"function": {"name": "f"}}]
    )
    assert result.has_visible_output is True
    score = validators.grade({"kind": "tool_call", "expected_function": "f"}, result)
    assert score.passed is True
