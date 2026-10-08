# Current status - Section A (task extraction) - verified 2026-10-08

Only facts verified in the 2026-10-08 session. Nothing committed to git.

## Sprint 1 (unchanged)
| Item | Verified value |
|---|---|
| `run_pipeline.py --offline` | **21/21** acceptance checks (log: `outputs/logs/sprint1_rerun_20261008.log`) |
| Corpus | raw 5,595 / canonical 5,497 / unique 4,029 |
| O\*NET 31.0 | 1,016 occupations, 18,838 task statements |
| Rebuild keeps Sprint 2 tables | `prototype` (7 tables) and `task_pipeline` (3 tables) preserved (`build_db.PRESERVED_SCHEMAS`) |
| Side effect of re-run | regenerated `outputs/tables/{locations,top_titles,nyc_job_category}.csv`, `sprint1_metrics.json`, notebook, contract/sources/summary timestamps. Values identical; only tie ordering and timestamps differ. |

## Frozen experiment corpus
* Sampler: `src/prototype/sample.py::build_sample` (existing; not re-implemented), seed `ie7945-sprint2-v1`
  (`config/llm_config.json`). Population: `job_postings_unique`, `language='en'`, description >= 80 words
  (2,816 postings: 1,404 NYC / 1,412 Arbeitnow).
* Frozen IDs: `data/experiment/task_extraction_sample_200.csv` (+ `posting_text_sha256`), mirrored to
  `task_pipeline.experiment_sample`. `python -m src.prototype.freeze` re-verifies: sampler reproduces the
  identical 200 IDs, 0 missing, 0 changed texts. Extraction now reads the frozen file and aborts on drift.
* Profile (`outputs/task_extraction/sample_profile/`): 100 NYC / 100 Arbeitnow; 200 distinct titles;
  13 keyword title groups (analysis grouping only, not an occupation taxonomy), largest 15%, unclassified 15%;
  seniority spread over 7 levels; description 120-1,761 words (median 629).

## Human gold annotation (blind, empty)
* `data/annotations/task_gold_100.csv` - Annotator A, 100 postings (existing `in_gold_100` flag; 50 NYC / 50 Arbeitnow).
* `data/annotations/task_gold_audit_20_annotator_b.csv` - Annotator B, 20 of the same postings (10 per source,
  spread over title groups), separate file so B labels without seeing A.
* `data/annotations/task_gold_100_assignment.csv` - assignment + selection rule.
* Gold columns empty; only posting text and a regex-based responsibilities excerpt (27/100 postings have no
  detected responsibilities heading, so annotators must read the full text). No model output, no O\*NET.
* Rules: `docs/annotation_guidelines.md` (rewritten for atomic tasks + mandatory verbatim evidence spans).

## Tests
`pytest tests` - **32/32 pass** (Sprint 1: 5, prototype: 18, new Section A: 9). The 2 earlier failures were a
test-fixture typing issue (an all-None `adjudication_decision` column inferred as INTEGER); the fixture now
casts it to VARCHAR. No production code or assertions changed for this fix.

## Extraction (baseline Gemini + Groq, frozen 200)
| Provider | Postings ok | New API calls this session | Task outputs (evidence-gate accepted) |
|---|---|---|---|
| Gemini `gemini-3.1-flash-lite` | 200/200 | 13 (187 from cache) | 3,284 (3,170) in 199 postings |
| Groq `openai/gpt-oss-120b` | 100/100 dual-model + 7 quality-triggered | 72 (35 from cache) | 1,018 (996) in 103 postings |

* AI adjudication of model disagreements (Groq) is **incomplete**: Groq's daily token quota ran out. 13 of
  the 30 capped postings were adjudicated (152 of 1,335 disagreement items). Unadjudicated items stay
  `review_required`; this does not affect evidence validity.
* One posting (`7b9b088dbccad49f1f52`, College Aide) yielded no task outputs from either model.

## `task_pipeline.task_statements` (production table) - 3,614 rows, 199 postings
| review_status | rows |
|---|---|
| accepted_cross_model_agreement | 683 |
| single_model_unconfirmed (dual phase, one model only) | 332 |
| single_model_not_compared (Gemini-primary phase) | 1,524 |
| review_required (boundary/kind disagreement, fuzzy evidence, adjudicator reject) | 943 |
| rejected_invalid_evidence (not traceable to posting) | 106 |
| rejected_unsupported_statement | 26 |
`is_valid_task` = 3,479. Evidence: 3,453 exact, 52 normalized, 3 fuzzy, 106 none. All rows have provider,
model, prompt version and `extraction_timestamp`; offsets for 3,508 rows.

## O\*NET alignment - `task_pipeline.task_onet_alignment` (34,790 rows = 3,479 tasks x top-10)
Local `BAAI/bge-small-en-v1.5`, cosine over all 18,838 O\*NET 31.0 task statements, no reranking
(Groq quota exhausted). Top-1 decisions: **accepted 286 / review 2,503 / unmatched 690**.
Thresholds (accept >= 0.86, review >= 0.75) are **provisional**, set from a manual spot check of 28 pairs
(0.80-0.83 mostly wrong). Embedding-only matches are often generic. Accepted = candidate, not ground truth.

## Claude / Anthropic provider
`src/llm/anthropic_provider.py` added (same schema/gate; model `claude-opus-5-5`, effort `medium`; separate
cache namespace `cache/llm/anthropic/`; no refusal fallback, so the comparison isn't confounded). Unit-tested with mocks.
**Not run: `ANTHROPIC_API_KEY` is not in `.env`.**

## Evaluation (ready, no results - no human labels exist)
`src/prototype/task_eval.py`: validate / evaluate / agreement. Normalized-exact -> lexical -> semantic (bge)
one-to-one matching, configurable thresholds, ambiguous pairs written for human review (not counted as TP),
per-posting metrics, FP/FN examples with heuristic error categories. Runs on the empty file and reports
`no_human_labels_yet`.

## Next actions
1. Teammate labels `task_gold_100.csv`; second annotator labels the audit file blind.
2. `python -m src.prototype.task_eval validate` -> `agreement` -> adjudicate -> `evaluate`.
3. Optional: when the Groq quota resets, re-run `python -m src.prototype.run_extraction` (cache-backed) to finish
   adjudication, then `python -m src.prototype.alignment tasks --rerank`.
4. Optional: add `ANTHROPIC_API_KEY` to `.env`, then `python -m src.prototype.run_extraction --provider anthropic`.
5. Calibrate O\*NET thresholds against a small human-labelled mapping sample.
