"""Automated validators for benchmark tasks.

Every validator returns a :class:`~caliper.results.TaskScore`. Deterministic
validators set ``graded="automatic"``; the ``rubric`` validator returns
``score=None`` and preserves the output for manual grading, because an LLM
judge is not treated as ground truth here.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..results import GenerationResult, TaskScore

log = logging.getLogger(__name__)

Validator = Callable[[dict[str, Any], GenerationResult], TaskScore]

_REGISTRY: dict[str, Validator] = {}

CODE_FENCE = re.compile(r"```(?:[A-Za-z0-9_+-]*)\n(.*?)```", re.DOTALL)


def register(kind: str) -> Callable[[Validator], Validator]:
    def deco(fn: Validator) -> Validator:
        _REGISTRY[kind] = fn
        return fn

    return deco


def available_validators() -> list[str]:
    return sorted(_REGISTRY)


def extract_code(text: str) -> str:
    """Return the first fenced code block, or the whole text when unfenced."""
    match = CODE_FENCE.search(text)
    return match.group(1) if match else text


def extract_json(text: str) -> Any:
    """Best-effort extraction of a single JSON value from model output."""
    candidate = extract_code(text).strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = candidate.find(opener)
        end = candidate.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("no JSON value found in output")


def _norm(text: str, case_sensitive: bool) -> str:
    return text if case_sensitive else text.lower()


# ---------------------------------------------------------------------------
# Text validators
# ---------------------------------------------------------------------------


@register("exact")
def validate_exact(spec: dict[str, Any], result: GenerationResult) -> TaskScore:
    expected = spec["expected"]
    got = result.text.strip()
    if spec.get("strip_trailing_newlines", True):
        got = got.rstrip("\n")
    cs = spec.get("case_sensitive", True)
    passed = _norm(got, cs) == _norm(str(expected).strip(), cs)
    return TaskScore(
        score=1.0 if passed else 0.0,
        passed=passed,
        validator="exact",
        detail="exact match" if passed else f"expected {expected!r}, got {got[:200]!r}",
    )


@register("contains_all")
def validate_contains_all(spec: dict[str, Any], result: GenerationResult) -> TaskScore:
    """Partial credit for each required substring present."""
    needles: list[str] = spec["needles"]
    cs = spec.get("case_sensitive", False)
    hay = _norm(result.text, cs)
    hits = [n for n in needles if _norm(n, cs) in hay]
    forbidden: list[str] = spec.get("forbidden", [])
    bad = [f for f in forbidden if _norm(f, cs) in hay]
    score = len(hits) / len(needles) if needles else 0.0
    if bad:
        score = 0.0
    return TaskScore(
        score=score,
        passed=score >= float(spec.get("pass_threshold", 1.0)) and not bad,
        validator="contains_all",
        detail=f"{len(hits)}/{len(needles)} required phrases"
        + (f"; forbidden present: {bad}" if bad else ""),
        sub_scores={"found": float(len(hits)), "required": float(len(needles))},
    )


@register("regex_all")
def validate_regex_all(spec: dict[str, Any], result: GenerationResult) -> TaskScore:
    patterns: list[str] = spec["patterns"]
    flags = re.IGNORECASE if not spec.get("case_sensitive", False) else 0
    if spec.get("multiline", True):
        flags |= re.MULTILINE
    hits = [p for p in patterns if re.search(p, result.text, flags)]
    score = len(hits) / len(patterns) if patterns else 0.0
    return TaskScore(
        score=score,
        passed=score >= float(spec.get("pass_threshold", 1.0)),
        validator="regex_all",
        detail=f"{len(hits)}/{len(patterns)} patterns matched",
    )


@register("choice")
def validate_choice(spec: dict[str, Any], result: GenerationResult) -> TaskScore:
    """Extract a final answer label and compare it to the expected one.

    The answer is taken from the last occurrence of ``answer_pattern`` so that
    a model which reasons aloud before answering is still graded on its answer.
    """
    pattern = spec.get("answer_pattern", r"(?:final answer|answer)\s*[:\-]\s*([A-Za-z0-9_.\- ]+)")
    matches = re.findall(pattern, result.text, re.IGNORECASE)
    expected = str(spec["expected"]).strip().lower()
    if not matches:
        tail = result.text.strip().splitlines()[-1:] or [""]
        got = tail[0].strip().lower()
    else:
        got = str(matches[-1]).strip().lower()
    got = got.strip(" .*`\"'")
    passed = got == expected
    return TaskScore(
        score=1.0 if passed else 0.0,
        passed=passed,
        validator="choice",
        detail=f"expected {expected!r}, extracted {got[:120]!r}",
    )


@register("numeric")
def validate_numeric(spec: dict[str, Any], result: GenerationResult) -> TaskScore:
    expected = float(spec["expected"])
    tol = float(spec.get("tolerance", 0.0))
    numbers = re.findall(r"-?\d+(?:\.\d+)?", result.text.replace(",", ""))
    if not numbers:
        return TaskScore(0.0, passed=False, validator="numeric", detail="no number in output")
    got = float(numbers[-1] if spec.get("use", "last") == "last" else numbers[0])
    passed = abs(got - expected) <= tol
    return TaskScore(
        score=1.0 if passed else 0.0,
        passed=passed,
        validator="numeric",
        detail=f"expected {expected} +/-{tol}, got {got}",
    )


# ---------------------------------------------------------------------------
# Structured output
# ---------------------------------------------------------------------------


def check_schema(value: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """Validate ``value`` against a small JSON Schema subset.

    Supported: type, required, properties, additionalProperties, enum, items,
    minItems, maxItems, minimum, maximum, minLength, maxLength, pattern.
    Unsupported keywords are ignored rather than silently passing a wrong
    value, so keep suite schemas inside this subset.
    """
    errors: list[str] = []
    types = {
        "object": dict,
        "array": list,
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
        "null": type(None),
    }
    expected_type = schema.get("type")
    if expected_type:
        wanted = expected_type if isinstance(expected_type, list) else [expected_type]
        py = tuple(
            t
            for name in wanted
            for t in (types[name] if isinstance(types[name], tuple) else (types[name],))
        )
        if isinstance(value, bool) and "boolean" not in wanted:
            errors.append(f"{path}: expected {expected_type}, got boolean")
            return errors
        if not isinstance(value, py):
            errors.append(f"{path}: expected {expected_type}, got {type(value).__name__}")
            return errors
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} not in enum {schema['enum']}")
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}: missing required key {key!r}")
        props = schema.get("properties", {})
        for key, sub in props.items():
            if key in value:
                errors.extend(check_schema(value[key], sub, f"{path}.{key}"))
        if schema.get("additionalProperties") is False:
            extra = set(value) - set(props)
            if extra:
                errors.append(f"{path}: unexpected keys {sorted(extra)}")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{path}: expected at least {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: expected at most {schema['maxItems']} items")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for i, item in enumerate(value):
                errors.extend(check_schema(item, item_schema, f"{path}[{i}]"))
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: {value} < minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: {value} > maximum {schema['maximum']}")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(f"{path}: shorter than minLength {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path}: longer than maxLength {schema['maxLength']}")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            errors.append(f"{path}: does not match {schema['pattern']!r}")
    return errors


@register("json_schema")
def validate_json_schema(spec: dict[str, Any], result: GenerationResult) -> TaskScore:
    schema = spec["schema"]
    try:
        value = extract_json(result.text)
    except ValueError as exc:
        return TaskScore(0.0, passed=False, validator="json_schema", detail=str(exc))
    errors = check_schema(value, schema)
    sub: dict[str, float] = {"parsed": 1.0, "schema_valid": 0.0 if errors else 1.0}
    expected_values: dict[str, Any] = spec.get("expected_values", {})
    value_hits = 0
    for key, want in expected_values.items():
        got = value.get(key) if isinstance(value, dict) else None
        if isinstance(want, str) and isinstance(got, str):
            if want.strip().lower() == got.strip().lower():
                value_hits += 1
        elif got == want:
            value_hits += 1
    if expected_values:
        sub["values_correct"] = value_hits / len(expected_values)
    weights = [sub["schema_valid"], *([sub["values_correct"]] if expected_values else [])]
    score = sum(weights) / len(weights)
    return TaskScore(
        score=score,
        passed=not errors and value_hits == len(expected_values),
        validator="json_schema",
        detail="valid" if not errors else "; ".join(errors[:5]),
        sub_scores=sub,
    )


@register("tool_call")
def validate_tool_call(spec: dict[str, Any], result: GenerationResult) -> TaskScore:
    """Grade function selection and argument correctness separately."""
    expected_name = spec["expected_function"]
    expected_args: dict[str, Any] = spec.get("expected_arguments", {})
    calls = result.tool_calls
    if not calls:
        return TaskScore(0.0, passed=False, validator="tool_call", detail="no tool call emitted")
    first = calls[0].get("function", calls[0]) or {}
    name = first.get("name")
    args = first.get("arguments") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    name_ok = name == expected_name
    arg_hits = 0
    for key, want in expected_args.items():
        got = args.get(key)
        if isinstance(want, str) and isinstance(got, str):
            if want.strip().lower() == got.strip().lower():
                arg_hits += 1
        elif isinstance(want, (int, float)) and isinstance(got, (int, float, str)):
            try:
                if abs(float(got) - float(want)) <= float(spec.get("numeric_tolerance", 0.0)):
                    arg_hits += 1
            except (TypeError, ValueError):
                pass
        elif got == want:
            arg_hits += 1
    arg_score = arg_hits / len(expected_args) if expected_args else 1.0
    extra_penalty = 0.0
    if spec.get("forbid_extra_calls", True) and len(calls) > 1:
        extra_penalty = 0.25
    score = max(0.0, (0.5 * float(name_ok) + 0.5 * arg_score) - extra_penalty)
    return TaskScore(
        score=score,
        passed=name_ok and arg_hits == len(expected_args) and extra_penalty == 0.0,
        validator="tool_call",
        detail=f"function={name!r} (expected {expected_name!r}), "
        f"args {arg_hits}/{len(expected_args)} correct, {len(calls)} call(s)",
        sub_scores={"function_correct": float(name_ok), "arguments_correct": arg_score},
    )


# ---------------------------------------------------------------------------
# Code execution
# ---------------------------------------------------------------------------

_HARNESS = """
import json, sys, resource
resource.setrlimit(resource.RLIMIT_AS, ({mem}, {mem}))
resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))

def norm(value):
    # Expected values arrive as JSON, which cannot express tuples, so a tuple
    # and a list of the same items are treated as equal.
    if isinstance(value, (list, tuple)):
        return [norm(v) for v in value]
    if isinstance(value, dict):
        return {{k: norm(v) for k, v in value.items()}}
    return value

ns = {{}}
try:
    exec(compile(open({code!r}).read(), "candidate.py", "exec"), ns)
except Exception as exc:
    print(json.dumps({{"stage": "import", "ok": False, "error": repr(exc)}}))
    sys.exit(0)
results = []
for name, expr, expected in json.load(open({cases!r})):
    try:
        got = eval(expr, dict(ns))
        results.append(
            {{"name": name, "ok": norm(got) == norm(expected), "got": repr(got)}}
        )
    except Exception as exc:
        results.append({{"name": name, "ok": False, "got": "raised " + repr(exc)}})
print(json.dumps({{"stage": "run", "ok": True, "results": results}}))
"""


@register("python_cases")
def validate_python_cases(spec: dict[str, Any], result: GenerationResult) -> TaskScore:
    """Execute model-written Python against declared expression/value cases.

    Disabled unless the run enables code execution (``allow_code_execution``
    in the validator spec, set by the runner from ``--allow-code-execution``).
    Expected values come from the suite's TOML/JSON, which cannot express
    tuples, so tuples and lists of the same items compare equal.

    The candidate code runs in a separate interpreter with ``-I`` (isolated:
    no cwd/user site imports), an address-space cap, no subprocess budget and
    a wall-clock timeout. This is a guard against runaway test code, not a
    security sandbox: only run suites you wrote.
    """
    if not spec.get("allow_code_execution"):
        return TaskScore(
            score=None,
            passed=None,
            validator="python_cases",
            graded="skipped",
            detail="code execution disabled; rerun with --allow-code-execution",
        )
    cases = [[c["name"], c["expression"], c["expected"]] for c in spec["cases"]]
    code = extract_code(result.text)
    timeout = float(spec.get("timeout_s", 15.0))
    mem_bytes = int(spec.get("memory_limit_mib", 1024)) * 1024 * 1024
    with tempfile.TemporaryDirectory(prefix="caliper-exec-") as tmp:
        tmpdir = Path(tmp)
        code_path = tmpdir / "candidate_code.py"
        cases_path = tmpdir / "cases.json"
        runner_path = tmpdir / "runner.py"
        code_path.write_text(code)
        cases_path.write_text(json.dumps(cases))
        runner_path.write_text(
            _HARNESS.format(mem=mem_bytes, code=str(code_path), cases=str(cases_path))
        )
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-S", str(runner_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmp,
                check=False,
                env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"},
            )
        except subprocess.TimeoutExpired:
            return TaskScore(
                0.0, passed=False, validator="python_cases", detail=f"timed out after {timeout}s"
            )
    try:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return TaskScore(
            0.0,
            passed=False,
            validator="python_cases",
            detail=f"harness produced no result (stderr: {proc.stderr.strip()[:200]})",
        )
    if not payload.get("ok"):
        return TaskScore(
            0.0,
            passed=False,
            validator="python_cases",
            detail=f"code failed to load: {payload.get('error', '')[:200]}",
        )
    runs = payload["results"]
    hits = [r for r in runs if r["ok"]]
    score = len(hits) / len(runs) if runs else 0.0
    failed = [f"{r['name']}={r['got'][:60]}" for r in runs if not r["ok"]]
    return TaskScore(
        score=score,
        passed=score == 1.0,
        validator="python_cases",
        detail=f"{len(hits)}/{len(runs)} cases passed"
        + (f"; failed: {', '.join(failed[:4])}" if failed else ""),
        sub_scores={"cases_passed": float(len(hits)), "cases_total": float(len(runs))},
    )


# ---------------------------------------------------------------------------
# Manual grading
# ---------------------------------------------------------------------------


@register("rubric")
def validate_rubric(spec: dict[str, Any], result: GenerationResult) -> TaskScore:
    """Preserve the output for manual grading against a documented rubric.

    No automatic score is produced. Automated signals that happen to be cheap
    (length, required mentions) are reported as sub-scores but do not become
    the task score.
    """
    criteria: list[str] = spec.get("criteria", [])
    mentions: list[str] = spec.get("signal_phrases", [])
    hay = result.text.lower()
    found = [m for m in mentions if m.lower() in hay]
    sub = {"signal_phrases_found": float(len(found)), "signal_phrases_total": float(len(mentions))}
    return TaskScore(
        score=None,
        passed=None,
        validator="rubric",
        graded="manual_rubric",
        detail=f"manual grading required against {len(criteria)} criteria; output stored",
        sub_scores=sub,
    )


def grade(task_validator: dict[str, Any], result: GenerationResult) -> TaskScore:
    """Dispatch to the validator named by ``kind``."""
    kind = task_validator.get("kind", "")
    fn = _REGISTRY.get(kind)
    if fn is None:
        return TaskScore(
            None,
            passed=None,
            validator=kind or "missing",
            graded="error",
            detail=f"unknown validator kind {kind!r}; known: {', '.join(available_validators())}",
        )
    if not result.ok:
        return TaskScore(
            0.0, passed=False, validator=kind, detail=f"generation failed: {result.error}"
        )
    if not result.has_visible_output:
        # Several validators (negative constraints, lookahead patterns) match
        # the empty string, so an empty response would otherwise score full
        # marks. An answer that was never emitted is a failure, and the usual
        # cause is a thinking model spending its whole token budget before it
        # starts answering.
        detail = "no visible output to grade"
        if result.truncated:
            detail += " (generation hit the token limit; raise max_output_tokens)"
        return TaskScore(0.0, passed=False, validator=kind, detail=detail)
    try:
        return fn(task_validator, result)
    except Exception as exc:
        log.exception("validator %s raised", kind)
        return TaskScore(None, passed=None, validator=kind, graded="error", detail=repr(exc))
