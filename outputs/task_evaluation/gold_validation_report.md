# Gold-100 Validation Report

**Date:** 2026-10-22
**File validated:** `data/annotations/task_gold_100.csv` (completed human gold; not modified by this run)
**Produced by:** `outputs/task_evaluation/evaluate_gold100.py` (reuses `src.prototype.task_eval.validate`; read-only DB, no LLM calls)
**Row-level detail:** `gold100_row_validation.csv`, `gold100_near_duplicates.csv`, `gold100/<mode>/gold_validation_problems.csv`

## Summary

| Check | Result | Status |
|---|---|---|
| Unique `posting_id`s | **100** | OK |
| Human task rows | **1,056** (10.56 per posting; min 3, median 10, max 15) | OK |
| `gold_task_statement` non-empty | 1,056 / 1,056 | OK |
| `evidence_span` non-empty | 1,056 / 1,056 | OK |
| Exact duplicate tasks in the same posting (normalized) | 0 | OK |
| Near-duplicate tasks in the same posting (stemmed Jaccard ≥ 0.75) | 0 pairs | OK |
| Every gold `posting_id` is in `job_postings` | 100 / 100 | OK |
| Every gold `posting_id` is in the frozen `in_gold_100` sample (`task_pipeline.experiment_sample`) | 100 / 100; no frozen gold posting is unlabelled | OK |
| Gold postings with a successful Gemini extraction call | 100 / 100 | OK |
| Gold postings with a successful Groq extraction call | **52 / 100** (50 dual_model + 2 gemini_primary) | **Gap** (see 5) |
| Physical rows in the file | **1,048,575** (1,047,519 padding rows) | **Problem** (see 1) |
| `adjudication_status` spelling | `labeled` (code expects `labelled`) | **Problem** (see 2) |
| Evidence is per task | **No.** 99 / 100 postings have one evidence block for all tasks | **Problem** (see 3) |
| Evidence block is verbatim in the posting | 806 / 1,056 rows (799 exact + 7 normalized); 250 rows in 25 postings are not | **Problem** (see 3) |
| One atomic task per row | 17 rows hold two actions; 25 are "verb and verb" + shared object | Minor (see 4) |
| Rows that are skills or qualifications, not tasks | about 12 | Minor (see 4) |

## Problems in detail

1. **Excel padding.** The file has 1,048,575 rows, Excel's maximum. 1,047,519 of them have a blank `posting_id` and only `annotator = human` filled in, most likely from a fill-down. The file is 28 MB, against 1.3 MB for the Sprint 2 working file.
   - What I did: evaluation uses a cleaned copy with those rows dropped, `outputs/task_evaluation/gold100_clean.csv`.
   - Recommendation: replace the annotation file with the cleaned version before committing. It is still uncommitted.

2. **Status spelling.** Every row says `labeled`. `src/prototype/task_gold.py` only recognises `labelled / no_tasks / adjudicated`. Run against the raw file, the existing evaluator would report **0 labelled postings**. The cleaned copy maps `labeled` to `labelled`; nothing else changed.

3. **Evidence spans are not per task.** For 99 postings the annotator pasted one block for every task row: the whole responsibilities section or the whole posting, median 2,273 characters. One posting has two blocks.
   - **Consequence:** evidence-level grounding of individual gold tasks cannot be scored. Only task-statement matching within the same posting is evaluated.
   - 25 postings (250 rows) have a block that `locate_evidence` cannot find verbatim. As in Sprint 2, this is consistent with blocks stitched together with headings or bullets dropped. It was not inspected character by character in this run.
   - 28 rows have under 50% of their content words inside their block: paraphrased, or taken from outside the pasted block.

4. **Atomicity and task purity of the gold rows.** The flags below are heuristic; review them in `gold100_row_validation.csv`.
   - **Two actions in one row (17 rows):** e.g. "Train staff and conduct mock hearings", "Order supplies and manage office stock", "Eject disorderly persons and issue summonses".
   - **Shared-object verb pairs (25 rows):** e.g. "Plan and execute strategic initiatives". Arguably acceptable.
   - **Skill or qualification rows (10 regex flags plus at least 2 seen by eye, about 12):** "Strong writing skills", "Interpersonal skills", "Organizational skills", "Time management skills", "Ability to work independently", "Windows 11 knowledge", "Experience with SaaS production", "Microsoft Excel", "AutoCAD", and others. "Certify agency payrolls" is a false regex flag. "Be available 24/7" is a work condition (OTHER).
   - **Vague rows:** "Oversee and manage", "Serve as a liason" (no object).
   - These rows can never be matched by a task extractor. They cost about 1% of recall.

5. **Groq coverage.** Groq ran on only 52 of the 100 gold postings. The frozen design only gave it 2 of the 50 gold postings in the `gemini_primary` phase. One covered posting (`2213f5426499a37cd6e2`) returned 0 accepted tasks.
   - Groq metrics are therefore measured on 52 postings (563 gold tasks), and so is the Gemini/Groq consensus.
   - Gemini is also reported on those same 52 postings for a like-for-like comparison.
   - Groq results for the other 48 postings **cannot be measured without new Groq calls**, which this run did not make.

## Is the gold exhaustive?

**Mostly, for responsibilities.** It is not exhaustive at the models' granularity.

- **Counts:** the gold has 10.6 tasks per posting, against 3 in the Sprint 2 gold. Gemini extracts 16.4 accepted tasks per posting and Groq 9.5.
- **Coverage of responsibility sentences:** 78.6% of Gemini's responsibility-section predictions, and 80.3% of Groq's, come from a source sentence that contains a human gold task (≥ 60% of the gold task's content words in the model's evidence sentence).
- **So:**
  - Annotators covered about 4 in 5 responsibility sentences the models used.
  - About 1 in 5 model predictions comes from a sentence the annotator did not label. These are candidate gold omissions.
  - Gemini also splits sentences more finely than the annotator, which one-to-one matching scores as false positives.
- **Precision against this gold is therefore a lower bound,** though a much tighter one than in Sprint 2.
