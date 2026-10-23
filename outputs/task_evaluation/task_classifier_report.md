# TASK / SKILL / OTHER Classifier Gate: Status

**Date:** 2026-10-22
**Status: NOT RUN.** `ANTHROPIC_API_KEY` is not set.
- `.env` contains only `GEMINI_API_KEY` and `GROQ_API_KEY`.
- The key is also absent from the process, user and machine environment. Only its presence was checked; nothing was printed.

No classification was performed, so there are no metrics. `task_classifier_validation.csv` was deliberately **not** created. No tables were created or changed: `task_statements` (3,614) and `sprint3_task_input` (3,090) are untouched.

## What is ready

**Code:** `src/prototype/task_classifier.py`
- Uses the existing `AnthropicProvider` (`src/llm/anthropic_provider.py`): model `claude-opus-5-5`, effort `medium`, from `config/llm_config.json`.
- Structured JSON output is validated with pydantic.
- Responses are cached, so a rerun makes 0 calls.
- The key is read only from `.env`; `_scrub` redacts keys from any error message.

**Input per statement:** `statement_text`, `evidence_span` (first 1,200 characters), `source_section`.

**Output:** `task_class` (TASK, SKILL or OTHER) and `classification_reason`. Statements are sent in deterministic batches of 20.

**Blind validation:**
- The prompt is written from the label definitions only.
- `human_label` is dropped before prompting; a unit test checks this.
- Prompt version `task-class-v1` is frozen. Do not tune it on the audit rows and then report the audit score; that would be test-set leakage.
- **Caveat:** the user-supplied OTHER example "Learn from your peers" also appears as an audit row (labelled OTHER). That row's result is therefore not blind.

**Gate:** `apply` refuses to run unless validation recorded TASK precision ≥ 0.95 for the same prompt version. When it runs, it writes:
- `task_pipeline.sprint3_task_classification`: all 3,090 rows with their class.
- `task_pipeline.sprint3_task_input_clean`: TASK rows only, with `statement_id, posting_id, provider, statement_text, evidence_span, source_section, evidence_valid, task_class, classification_model, classification_reason, prompt_version`.

**Tests:** 3 new unit tests (metrics and gate, response-id checks, prompt never sees labels). Full suite: 35 passed.

## To run
1. Add `ANTHROPIC_API_KEY=...` to `.env`.
2. `.venv\Scripts\python.exe -m src.prototype.task_classifier validate`
   - Makes 5 API calls over the 100 audited rows.
   - Writes `task_classifier_validation.csv` and `task_classifier_metrics.json`.
   - Reports accuracy, TASK P/R/F1 and the confusion matrix for all 100 rows and for the 85 rows inside `sprint3_task_input`.
3. Only if TASK precision ≥ 95%: `.venv\Scripts\python.exe -m src.prototype.task_classifier apply` (about 155 calls).
