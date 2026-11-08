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

## Procedure

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

5. Record the scores outside the attempts table — a CSV keyed by
   `(run_id, model, task_id, repetition)` is enough — and keep it with the run.
   Caliper does not yet write manual scores back into the database; see
   [ROADMAP.md](ROADMAP.md).

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
