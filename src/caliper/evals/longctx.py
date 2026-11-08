"""Deterministic filler-document generation for long-context tasks.

The text is generated from an original word pool, so no third-party corpus is
involved and the same context is reproduced byte-for-byte on every run. The
document reads as a plausible internal engineering log, which keeps the task
legitimate: the model is asked to retrieve and relate facts in a real-looking
document, not to resist an injected instruction.
"""

from __future__ import annotations

import random

PLACEHOLDER = "{{FILLER}}"

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


def make_filler(approx_tokens: int, seed: int = 1234) -> str:
    """Generate roughly ``approx_tokens`` tokens of deterministic filler prose."""
    rng = random.Random(seed)
    target_words = max(int(approx_tokens * WORDS_PER_TOKEN), 20)
    lines: list[str] = []
    words = 0
    entry = 1
    while words < target_words:
        sentence = (
            f"Entry {entry:04d}. On day {rng.randint(1, 365)}, {rng.choice(_SUBJECTS)} "
            f"{rng.choice(_VERBS)} {rng.choice(_OBJECTS)}. {rng.choice(_TAILS)}"
        )
        lines.append(sentence)
        words += len(sentence.split())
        entry += 1
    return "\n".join(lines)


def render(prompt: str, filler_tokens: int | None, seed: int = 1234) -> tuple[str, bool]:
    """Substitute the filler placeholder in ``prompt``.

    Returns the rendered prompt and whether a substitution happened.
    """
    if PLACEHOLDER not in prompt:
        return prompt, False
    filler = make_filler(filler_tokens or 2000, seed=seed)
    return prompt.replace(PLACEHOLDER, filler), True
