# Meeting checkpoint - Sprint 2 advanced prototype (paused 2026-09-25 ~18:05 UTC)

Nothing was deleted or reverted. All LLM responses are cached under `cache/llm/` (git-ignored);
all prototype tables live in the separate DuckDB schema `prototype` (Sprint 1 `main` tables untouched
except the documented bug fix below).

## COMPLETED AND VERIFIED
| Item | Evidence |
|---|---|
| Sprint 1 pipeline still passes | `run_pipeline.py --offline` 21/21 acceptance checks (after bug fix) |
| Sprint 1 unit tests | 5/5 passing |
| Secrets safety | `.env` git-ignored; keys present (length check only, never printed); literal key scan of repo = 0 hits; project is not a git repo (no history to leak) |
| **Sprint 1 bug fix BUG-001** | Entity-encoded HTML left literal tags in 249 Arbeitnow descriptions. Fixed + documented in `docs/sprint1_bugfix_log.md`. Raw 5,595 / canonical 5,497 unchanged; unique **4,037 -> 4,029**; duplicate rate **26.6% -> 26.7%**; en/de/fr 4,207/1,229/61 (76.5% / 22.4% / 1.1%). O*NET/ESCO counts unchanged. |
| Sprint 1 rebuild preserves `prototype` schema | `src/ingestion/build_db.py::preserve_prototype_schema` |
| Provider-agnostic LLM layer | `src/llm/` base, gemini_provider, groq_provider, router, cache, schemas, usage, embeddings; config in `config/llm_config.json` |
| Both providers work live | Gemini `gemini-3.1-flash-lite` and Groq `openai/gpt-oss-120b` returned schema-valid JSON |
| Model selection measured (not assumed) | model list queried from both APIs; `gemini-3.8-flash` 1/15 successful calls (503/429) -> not used; notes in config |
| Deterministic stratified sample | `prototype.prototype_extraction_sample` = 200 unique English postings (50 per source per phase), one per normalized title; rebuild gives identical IDs |
| Deterministic section detection | `prototype.prototype_sections` = 818 section spans; 173/200 postings rule-detected, 27 `llm_in_extraction` |
| End-to-end smoke test (4 postings) | extraction -> evidence gates -> matching -> consensus -> adjudication -> review queue ran without errors |
| Caching | identical request = cache hit (unit-tested; cache holds 230 extraction responses) |

## PARTIALLY IMPLEMENTED
| Item | State |
|---|---|
| Full 200-posting dual-model extraction | **Paused mid-run.** Cached: Gemini 190/200 extractions, Groq 35/100 dual-phase extractions, 4 adjudications. Re-running resumes from cache (0 repeat calls for cached items). |
| `prototype.prototype_statements` / `_model_outputs` / `_review_queue` / `_model_matching` / `_extraction_calls` | Exist but contain **only the 4-posting smoke test** (150 statements, pre tier-fix). Will be overwritten by the full run. **Do not report these numbers.** |
| Embeddings | Gemini embedding API measured rate-limited (429 after ~5 batches); switched to local `BAAI/bge-small-en-v1.5` (384-d, 87 texts/s). Reference pre-embedding interrupted (file lock from an orphan process, now stopped). |
| Unit tests for prototype | `tests/test_prototype.py`: 21/23 passing; 2 initiative tests fail on a test-fixture typing issue (`adjudication_decision` column typed INT in fixture) - code path not yet verified |

## CURRENTLY RUNNING / PAUSED
Nothing running. Extraction processes (2) stopped cleanly; DB is written only at the end of a run, so no partial writes occurred.

## IMPLEMENTED BUT NOT YET EXECUTED
- `src/prototype/alignment.py` (task->O*NET, skill->ESCO candidates, constrained two-model reranking, candidate gaps)
- `src/demo/initiative_to_work.py` (retrieval-grounded initiative -> tasks -> skills with explanation objects)
- `src/prototype/gold.py` (empty gold-100 builder + P/R/F1 evaluator); `docs/annotation_guidelines.md` written

## NOT STARTED
- `outputs/advanced/*` reports (usage, disagreement, candidate gaps, experiment report)
- `docs/ai_usage_log.md`
- Streamlit app, static HTML demo
- Full 12-slide Sprint 2 PPTX, presentation notes, engineering one-pager

## NEXT EXACT ACTIONS TO RESUME ("continue")
1. Fix test fixture: cast `adjudication_decision` to VARCHAR in `tests/test_prototype.py::_mini_db`; run `.\.venv\Scripts\python.exe -m pytest tests -q`.
2. Resume extraction (cache-backed): `.\.venv\Scripts\python.exe -m src.prototype.run_extraction`
   (remaining cost ~65 Groq dual extractions + triggered + <=30 adjudications; ~45-60 min at 7.2k TPM).
3. `.\.venv\Scripts\python.exe -m src.prototype.alignment`
4. `.\.venv\Scripts\python.exe -m src.prototype.gold build`
5. Initiative runs, reports, demo, PPTX as specified in prompt #2.
6. Re-run `.\.venv\Scripts\python.exe run_pipeline.py` and confirm 21/21 + main-schema counts.
