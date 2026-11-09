# Manual rubric grading

Five tasks in suite `v1` are graded by hand. They are the ones where a
deterministic validator would reward keyword matching instead of judgement:

| Task | What it tests |
|---|---|
| `coding.refactor_decision` | Willingness to disagree with a bad refactor, and naming its cost |
| `reasoning.tradeoff_explanation` | Correct account of MoE speed versus memory |
| `tool.abstain_when_unavailable` | Not calling a tool that cannot answer |
| `agent.decompose_task` | Concrete, ordered plan with a resource check in it |
| `agent.uncertainty_calibration` | Admitting ignorance instead of inventing a number |

No LLM judge is used. An LLM judge is neither objective nor reliably accurate,
and treating one as ground truth would make the whole suite's quality numbers
unfalsifiable.

## Procedure (web viewer)

```sh
caliper serve
```

Open the viewer, pick a run, and press **Grade rubric tasks** — or go straight
to the Grading tab.

**The queue groups repetitions.** One row per (task, model), not per attempt,
because a run with three repetitions produces three attempts at one prompt at
temperature 0 and they are usually the same text. Judging each separately
multiplies the work by the repetition count for no extra information, and a
queue that takes three times as long as it needs to is a queue that stays
unfinished. On this host's stored runs that is 105 attempts reduced to 35
judgements.

Each group says whether the responses are identical:

- **identical** — every repetition produced the same text after whitespace
  normalisation. One judgement covers the group, and each attempt still gets
  its own row in `manual_scores`; nothing is merged.
- **n different** — the repetitions diverged, which this engine does produce
  at temperature 0. The server **refuses** a group grade here. Read each
  response; then either tick *I read all responses and judge them equivalent*
  to record one judgement across them, or use **grade only this** on a single
  response to score it alone. Either way the stored note says which happened,
  so a later reader knows the score was not read one attempt at a time.

The progress bar counts judgements, with the attempt count beside it.
Clicking one shows the prompt, the response, the thinking text and the tool
calls beside the task's criteria, with a checkbox per criterion. The score is
points met over points available.

Scores are stored in a `manual_scores` table keyed by attempt, separate from
the automatic scores, so `caliper regrade` cannot overwrite human work. They
appear in reports as their own table, never merged into the automatic means.

An attempt marked **no output** is one where the model emitted nothing to
grade, usually a thinking model that spent its whole token budget. Grade it
against the criteria as you would any other response: nothing met.

## Procedure (by hand, without the viewer)

1. Export the run with responses included:

   ```sh
   caliper export --run <run_id> --responses --out-dir reports
   ```

2. Open `reports/run-<run_id>.json`. Each attempt carries `task_id`,
   `response_text`, `tool_calls` and the `sub_scores` the validator recorded
   (for example how many signal phrases appeared — a hint, not a score).

3. Grade each response against the `criteria` list in the task's TOML
   definition under `src/caliper/evals/suites/v1/`. Each criterion is worth one
   point; the task score is points earned divided by points available.

4. Grade blind where you can: read responses without looking at which model
   produced them. Grade all responses to one task before moving to the next,
   so the standard stays fixed within a task.

5. Record each score with the API, which is what the viewer calls:

   ```sh
   curl -X POST http://127.0.0.1:8777/api/grades \
     -H 'Content-Type: application/json' \
     -d '{"attempt_id": 287, "criteria": [{"text":"...","met":true}], "grader": "you"}'
   ```

## Rules

- `tool.abstain_when_unavailable` is scored partly on the `tool_calls` field,
  not the prose: a non-empty `tool_calls` fails the first criterion whatever
  the text says.
- Do not award a criterion for a statement the model contradicts elsewhere in
  the same response.
- Word and sentence limits in a prompt are criteria. Judge them generously at
  the margin (±10%) and consistently.
- A refusal to answer is not a pass on `agent.uncertainty_calibration`: the
  task also requires naming the measurement that would settle the question.
