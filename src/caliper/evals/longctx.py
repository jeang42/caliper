"""Deterministic filler-document generation for long-context tasks.

The text is generated from an original word pool, so no third-party corpus is
involved and the same context is reproduced byte-for-byte on every run. The
document reads as a plausible internal engineering log, which keeps the task
legitimate: the model is asked to retrieve and relate facts in a real-looking
document, not to resist an injected instruction.

Two placeholder forms exist, and the difference matters for what a task can
measure:

``{{FILLER}}``
    The v1 form. Every occurrence is replaced with the *same* text, so a
    document with two filler blocks repeats itself word for word and repeats
    its entry numbers. Kept unchanged so v1 prompts render byte-for-byte as
    they did when v1 results were measured.

``{{FILLER:1}}``, ``{{FILLER:2}}``, ...
    The v2 form. Each occurrence gets distinct prose, and entry numbering
    continues across segments in document order, so the result reads as one
    log with unique entry numbers. ``filler_tokens`` is the approximate total
    across every segment, not per segment, which keeps a task's prompt size
    predictable however many needles are threaded through it.
"""

from __future__ import annotations

import random
import re

PLACEHOLDER = "{{FILLER}}"
INDEXED_PLACEHOLDER = re.compile(r"\{\{FILLER:(\d+)\}\}")

_SUBJECTS = (
    "the ingest worker",
    "the scheduler",
    "the cache layer",
    "the metrics agent",
    "the batch planner",
    "the retry queue",
    "the index compactor",
    "the audit log",
    "the config loader",
    "the shard router",
)
_VERBS = (
    "was restarted after",
    "recorded a regression in",
    "was reconfigured following",
    "absorbed extra load from",
    "deferred work because of",
    "flagged an anomaly in",
    "completed migration of",
    "throttled requests from",
)
_OBJECTS = (
    "the nightly compaction pass",
    "a malformed manifest entry",
    "the staging rollout",
    "an unexpected clock skew",
    "the quarterly retention sweep",
    "a partial disk fill",
    "the regional failover drill",
    "a duplicated job id",
)
_TAILS = (
    "No customer impact was observed.",
    "Follow-up was filed with the platform team.",
    "The change was reverted within the hour.",
    "Monitoring thresholds were left unchanged.",
    "The on-call engineer documented the sequence.",
    "Capacity headroom remained adequate.",
)

# ~1.4 tokens per word is a conservative rule of thumb for English prose with
# these models' tokenizers; the exact count is recorded per run from the
# engine's own prompt_eval_count, never estimated in results.
WORDS_PER_TOKEN = 0.72


def _segment(approx_tokens: int, seed: int, start_entry: int) -> tuple[str, int]:
    """Generate one filler block and return it with the next free entry number."""
    rng = random.Random(seed)
    target_words = max(int(approx_tokens * WORDS_PER_TOKEN), 20)
    lines: list[str] = []
    words = 0
    entry = start_entry
    while words < target_words:
        sentence = (
            f"Entry {entry:04d}. On day {rng.randint(1, 365)}, {rng.choice(_SUBJECTS)} "
            f"{rng.choice(_VERBS)} {rng.choice(_OBJECTS)}. {rng.choice(_TAILS)}"
        )
        lines.append(sentence)
        words += len(sentence.split())
        entry += 1
    return "\n".join(lines), entry


def make_filler(approx_tokens: int, seed: int = 1234) -> str:
    """Generate roughly ``approx_tokens`` tokens of deterministic filler prose."""
    return _segment(approx_tokens, seed=seed, start_entry=1)[0]


def render(prompt: str, filler_tokens: int | None, seed: int = 1234) -> tuple[str, bool]:
    """Substitute the filler placeholders in ``prompt``.

    Returns the rendered prompt and whether a substitution happened. Indexed
    placeholders are filled in document order so that entry numbers increase
    down the page no matter what indices a task author chose; the prose of a
    given segment is keyed to its index, so reordering a document changes the
    numbering but not which block of text carries which needle.
    """
    total = filler_tokens or 2000
    occurrences = INDEXED_PLACEHOLDER.findall(prompt)
    if occurrences:
        per = max(total // len(occurrences), 20)
        next_entry = 1

        def fill(match: re.Match[str]) -> str:
            nonlocal next_entry
            text, next_entry = _segment(
                per, seed=seed + int(match.group(1)), start_entry=next_entry
            )
            return text

        return INDEXED_PLACEHOLDER.sub(fill, prompt), True
    if PLACEHOLDER not in prompt:
        return prompt, False
    return prompt.replace(PLACEHOLDER, make_filler(total, seed=seed)), True
