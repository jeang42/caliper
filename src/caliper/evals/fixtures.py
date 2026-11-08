"""Canned model responses for dry runs and tests.

These fixtures let the full pipeline - runner, validators, storage, reports -
be exercised without any inference. They are written to produce a realistic
mix of passes, partial credit and failures so a dry-run report shows that
every validator path works, rather than a uniform wall of zeros.

Responses are keyed by task id. The mock backend matches on a distinctive
prefix of each task's prompt, so no change to the backend protocol is needed.
"""

from __future__ import annotations

from typing import Any

from .spec import Suite

PROMPT_KEY_LENGTH = 80

#: task id -> {"text": str, "tool_calls": list[dict]}
FIXTURES: dict[str, dict[str, Any]] = {
    # --- coding -----------------------------------------------------------
    "coding.rle_encode": {
        "text": (
            "```python\n"
            "def rle_encode(data: str) -> list[tuple[str, int]]:\n"
            "    out: list[tuple[str, int]] = []\n"
            "    for ch in data:\n"
            "        if out and out[-1][0] == ch:\n"
            "            out[-1] = (ch, out[-1][1] + 1)\n"
            "        else:\n"
            "            out.append((ch, 1))\n"
            "    return out\n"
            "```"
        )
    },
    "coding.debug_offbyone": {
        # Deliberately imperfect: fixes the aliasing bug but not the range,
        # so the fixture exercises partial credit.
        "text": (
            "The function mutates its input and stops one window early.\n\n"
            "```python\n"
            "def moving_average(values, k):\n"
            "    out = []\n"
            "    for i in range(len(values) - k):\n"
            "        out.append(sum(values[i:i + k]) / k)\n"
            "    return out\n"
            "```"
        )
    },
    "coding.comprehension_behaviour": {"text": "[2, 2, 2]\n[('a', 1), ('b', 2)]"},
    "coding.test_generation": {
        "text": (
            "```python\n"
            "import pytest\n"
            "from mymod import clamp\n\n"
            "def test_inside():\n"
            "    assert clamp(5, 0, 10) == 5\n\n"
            "def test_below():\n"
            "    assert clamp(-1, 0, 10) == 0\n\n"
            "def test_above():\n"
            "    assert clamp(11, 0, 10) == 10\n\n"
            "def test_bounds():\n"
            "    assert clamp(0, 0, 10) == 0\n"
            "    assert clamp(10, 0, 10) == 10\n\n"
            "def test_invalid_range():\n"
            "    with pytest.raises(ValueError):\n"
            "        clamp(1, 10, 0)\n"
            "```"
        )
    },
    "coding.refactor_decision": {
        "text": (
            "Do not convert the two-case function. A registry adds indirection: "
            "dispatch becomes invisible to static analysis and failures move to "
            "load time, which is a real cost for a branch that has been stable "
            "for two years. Convert the dispatches that actually churn."
        )
    },
    # --- reasoning --------------------------------------------------------
    "reasoning.schedule_constraints": {
        "text": (
            "C and D must be adjacent, and A precedes C, so CD sits in slots 3-4 "
            "and B cannot take slot 1.\nAnswer: A,B,C,D"
        )
    },
    "reasoning.rate_arithmetic": {
        "text": "Prompt 3200/800 = 4 s. Generation 600/48 = 12.5 s. Load 4.5 s.\nAnswer: 21.0"
    },
    "reasoning.error_detection": {
        "text": "This denies the antecedent; the implication does not reverse.\nAnswer: 3"
    },
    "reasoning.memory_budget": {
        # Wrong on purpose: forgets the 2 GB reserve, so the numeric validator
        # records a failure.
        "text": "32 - 1.5 - 18.5 = 12 GB for KV, so 12 / 0.25 = 48.\nAnswer: 48000"
    },
    "reasoning.tradeoff_explanation": {
        "text": (
            "The mixture-of-experts model should generate faster: only 3B of its "
            "parameters are active per token, so each step costs less compute. "
            "Memory is set by resident weights, not active ones, so all 36B must "
            "be held and it needs at least as much memory as the dense 32B model."
        )
    },
    # --- instruction following -------------------------------------------
    "instruction.exact_format": {"text": "STATUS=ready;RETRIES=0;MODE=serial"},
    "instruction.json_schema_report": {
        "text": (
            '{"model": "demo:8b", "engine": "ollama", "tasks_completed": 12, '
            '"tasks_total": 14, "status": "partial"}'
        )
    },
    "instruction.multi_constraint": {
        "text": (
            "Serial runs keep one model on the device at a time. Serial execution "
            "makes telemetry attributable to a single request. Serial scheduling "
            "protects other workloads on the same GPU. [END]"
        )
    },
    "instruction.negative_constraint": {
        # Contains the letter "e", so this fixture fails its constraint.
        "text": "VRAM capacity and memory bandwidth."
    },
    # --- tool use ---------------------------------------------------------
    "tool.select_single": {
        "text": "",
        "tool_calls": [
            {"function": {"name": "get_gpu_metrics", "arguments": {"host": "ada", "unit": "MiB"}}}
        ],
    },
    "tool.numeric_arguments": {
        "text": "",
        "tool_calls": [
            {
                "function": {
                    "name": "start_benchmark",
                    "arguments": {
                        "model": "qwen3-coder:30b",
                        "repetitions": 3,
                        "timeout_seconds": 120,
                    },
                }
            }
        ],
    },
    "tool.abstain_when_unavailable": {
        "text": (
            "No tool here can answer that: get_gpu_metrics reports only current "
            "telemetry and holds no historical run data. I would need a store of "
            "past benchmark runs."
        )
    },
    "tool.multistep_plan": {
        "text": (
            "check_resources(host)\npull_model(name)\nrun_benchmark(model, suite)\n"
            "write_report(run_id)"
        )
    },
    # --- long context -----------------------------------------------------
    "longctx.retrieve_fact": {"text": "Answer: R-4471"},
    "longctx.relate_distant": {
        "text": "Entry 0001 caps the cache at 12 GiB and entry 9002 asks for 15 GiB.\nAnswer: no"
    },
    "longctx.embedded_instruction": {"text": "RESULT: 7"},
    # --- agent ------------------------------------------------------------
    "agent.decompose_task": {
        "text": (
            "1. Check current GPU utilization and free VRAM on the workstation.\n"
            "2. Enumerate installed models and select the candidates with tool and "
            "coding capability.\n"
            "3. Configure the coding evaluation with fixed temperature and seed.\n"
            "4. Run the suite serially and record every result.\n"
            "5. Compare per-category scores against throughput and memory."
        )
    },
    "agent.recognize_missing_info": {
        "text": (
            '- which model is meant by "best"\n'
            "- the workload the model will serve\n"
            '- the latency or throughput target that defines "fast enough"\n'
            "- the context length to test at"
        )
    },
    "agent.next_action": {
        "text": "Evicting another user's memory is off limits, so the run must wait.\nAnswer: C"
    },
    "agent.uncertainty_calibration": {
        "text": (
            "I do not know. No measurement for that model on that GPU at that "
            "context length is available to me, and throughput depends on "
            "quantization and KV cache placement. A benchmark run at 32k context "
            "on the actual host would answer it."
        )
    },
}


def prompt_keys(suite: Suite) -> dict[str, str]:
    """Pick a substring of each task's prompt that identifies it uniquely.

    Several tasks share an opening paragraph (the long-context ones all start
    with the same framing), so a fixed prefix is not enough. The tail of the
    prompt - the actual question - is tried first, then the head, then the
    whole prompt. Long-context prompts keep their tail intact after filler
    substitution, which is why the tail is preferred.
    """
    keys: dict[str, str] = {}
    prompts = [t.prompt for t in suite.tasks]
    for task in suite.tasks:
        for candidate in (
            task.prompt[-PROMPT_KEY_LENGTH:],
            task.prompt[:PROMPT_KEY_LENGTH],
            task.prompt,
        ):
            if sum(1 for p in prompts if candidate in p) == 1:
                keys[task.id] = candidate
                break
        else:  # pragma: no cover - would mean two identical prompts
            raise ValueError(f"cannot build a unique fixture key for task {task.id}")
    return keys


def mock_fixtures_for_suite(suite: Suite) -> tuple[dict[str, str], dict[str, list[dict[str, Any]]]]:
    """Build (text answers, tool calls) keyed by prompt prefix for a suite."""
    answers: dict[str, str] = {}
    tool_calls: dict[str, list[dict[str, Any]]] = {}
    keys = prompt_keys(suite)
    for task in suite.tasks:
        fixture = FIXTURES.get(task.id)
        if fixture is None:
            continue
        key = keys[task.id]
        answers[key] = fixture.get("text", "")
        if fixture.get("tool_calls"):
            tool_calls[key] = fixture["tool_calls"]
    return answers, tool_calls


def missing_fixture_ids(suite: Suite) -> list[str]:
    return [t.id for t in suite.tasks if t.id not in FIXTURES]
