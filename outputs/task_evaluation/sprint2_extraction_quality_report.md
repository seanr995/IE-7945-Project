# Sprint 2 — Task Extraction Quality Report

**Date:** 2026-10-09
**Scope:** Section A (job postings → atomic task statements), measured against a human-labeled evaluation subset, plus O*NET reference alignment.
**Reproducibility:** no LLM/API calls were made for this report. All model outputs come from the cached tables in `database/taxonomy.duckdb` (`prototype.prototype_model_outputs`, `task_pipeline.task_statements`, `task_pipeline.task_onet_alignment`). Matching uses only the local embedding model (bge-small-en-v1.5, cached).

---

## 1. Headline

| | Gemini | Groq | Gemini+Groq consensus |
|---|---:|---:|---:|
| Evaluated postings | 43 | 43 | 43 |
| Human gold tasks | 129 | 129 | 129 |
| Predicted tasks | 698 | 413 | 267 |
| True positives (TP) | 63 | 49 | 38 |
| False positives (FP) | 635 | 364 | 229 |
| False negatives (FN) | 66 | 80 | 91 |
| **Precision** | **9.0%** | **11.9%** | **14.2%** |
| **Recall** | **48.8%** | **38.0%** | **29.5%** |
| **F1** | **15.2%** | **18.1%** | **19.2%** |
| Recall incl. near-miss pairs (cos 0.75–0.85) | 72.1% | 69.8% | 48.1% |
| Precision ceiling given 3 gold tasks/posting | 18.5% | 31.2% | 48.3% |

Micro-averaged over all evaluated postings, strict one-to-one matching. "Consensus" = tasks extracted by both models (`review_status = accepted_cross_model_agreement`). The union of both models (`baseline_valid`, 842 predictions) reaches the highest recall: **55.8%** (P 8.6%, F1 14.8%).

**How to read these numbers.** The human labeling protocol recorded the **three key tasks per posting** (129 tasks / 43 postings, exactly 3 each), while the models extract *every* task in the posting (Gemini ≈ 16 per posting, Groq ≈ 10). So:

- **Recall is the meaningful headline**: of the tasks a human picked out as central, Gemini recovers ~half under strict matching and ~72% once semantic near-misses are counted.
- **Precision is a lower bound.** Most "false positives" are real tasks that the human simply did not list (87% of Gemini FPs and 97% of Groq FPs are in the `extra_task_not_in_gold` category). Even a perfect extractor could reach at most 18.5% precision against this gold for Gemini's output volume.

---

## 2. Human-labeled evaluation subset

**Source file.** `data/annotations/task_gold_100.csv` is still the blank annotation template (0 labels). The completed labels are in the annotator's working copy, `data/annotations/task_gold_100_3rows 1.csv` (saved 2026-10-09). That file was used, unmodified. Its `adjudication_status` column had been left as `not_started`, so completed rows were identified by content instead: non-empty `gold_task_statement` and a `posting_id` that exists in `job_postings`. The normalized copy that the evaluator reads is `outputs/task_evaluation/gold_subset_labelled.csv`; it is rebuilt by `outputs/task_evaluation/prepare_gold_subset.py`.

**Composition.** 43 postings (23 NYC Jobs, 20 Arbeitnow), drawn from the frozen blind gold-100 set, with 129 gold tasks (3 per posting). Labeling was blind: annotators saw only the posting text and a rule-based responsibilities excerpt, with no model output and no O*NET tasks.

**Row validation** (`outputs/task_evaluation/gold_row_validation.csv`):

| Check | Result |
|---|---|
| Rows with a completed gold task | 129 / 129 |
| `posting_id` exists in the database | 129 / 129 |
| `gold_task_statement` non-empty | 129 / 129 |
| `evidence_span` present | 129 / 129 |
| Evidence span is a verbatim (exact/normalized) passage of the source | 98 / 129 |
| Remaining spans: every character found in the source, but stitched from non-contiguous passages (headings dropped) | 31 / 129 (11 postings; ≥99.8% character coverage) |

None of the evidence spans is fabricated. The 31 non-contiguous spans are a formatting issue: the annotator copied the responsibilities block without its sub-headings.

**Gold characteristics that affect the metrics:**
- Evidence was recorded **per posting block**, not per task (median 1 distinct span per posting), so evidence-level agreement cannot be scored. Only task-statement matching is evaluated.
- Gold tasks are short paraphrases (59 of 129 have ≤ 3 words, e.g. "Raise funding", "Hire engineers"). The matcher therefore relies on lexical and semantic matching, and only 15 Gemini matches are exact.
- About 10–14 gold rows are skills or tools rather than tasks ("Strong writing skills", "Windows 11 knowledge", "AutoCAD", "Microsoft Excel"). A task extractor is designed *not* to produce these, so they appear as unavoidable false negatives.

---

## 3. Matching method

Implemented in `src/prototype/task_eval.py`, unchanged. Each comparison stays within a single posting:
1. normalized exact match → 2. lexical (stemmed content-word Jaccard ≥ 0.60) → 3. semantic (cosine ≥ 0.85).
2. Pairs are assigned greedily and **one-to-one**, so no gold task can be credited twice.
3. Pairs with cosine 0.75–0.85 go to `ambiguous_review.csv`. They are **not** counted as TP in the strict metrics and appear only in the "near-miss" upper bound.
4. No LLM judge is used. Error categories are deterministic heuristics meant for triage.

Match methods among true positives: Gemini: 15 exact, 20 lexical, 28 semantic. Groq: 7 exact, 18 lexical, 24 semantic.

---

## 4. Error analysis

### Error categories

| Category | Gemini | Groq |
|---|---:|---:|
| **FP: extra task not in gold** (usually a valid task the human didn't list) | 555 | 352 |
| **FP: skill / qualification extracted as task** | 52 | 7 |
| **FP: weak or implied evidence** | 22 | 4 |
| **FP: compound over-split** (gold kept one task, model split it) | 6 | 1 |
| **FN: missed task** | 36 | 39 |
| **FN: near-miss** (semantic 0.75–0.85, pending human review) | 30 | 41 |

### Main error types

1. **Coverage mismatch (dominant).** The models list all tasks; the gold lists 3. This one cause accounts for most FPs and is a property of the evaluation protocol, not an extraction failure.
2. **Task-vs-skill confusion, in both directions.** Gemini pulls tool and skill lines from requirements/skills sections and phrases them as tasks: "Configure Microsoft Purview" and "Manage Microsoft Purview" (skills section), "Implement solutions" (preferred qualifications). Groq does the same, less often: "develop spreadsheets" and "develop reports" (preferred qualifications). On the gold side, skills were entered as tasks: "Strong writing skills", "Time management skills".
3. **Granularity / compound splitting.** The models split compound responsibilities more finely than the human did. For example, "Source relationships with equity investors" was extracted on its own, while the human kept one task, "Raise funding". In the other direction, the model merged gold tasks: the human's "Analyze revenue metrics" sits inside Groq's long "Contribute to revenue operations by analyzing metrics, improving CRM hygiene, and streamlining data flows" (cos 0.83, near-miss).
4. **Paraphrase distance (near-misses).** Some equivalent tasks fall just under the 0.85 threshold. "Lead high cost deals" ↔ "Close complex, high-value deals" scores 0.77, and "Process case files for probationers" ↔ "Process incoming cases" scores 0.77. A human decision on these 30–41 pairs would move recall toward the 70% upper bound.
5. **Genuine misses.** Examples are "Oversee design reviews" (Groq), "Respond to requests from other teams" and "Reduce dependence on manual workflows" (Gemini). These mostly come from narrative paragraphs rather than bulleted lists.

### Representative examples

**True positives (semantic matches)**
| Human gold | Model | cos |
|---|---|---:|
| Preparation of appraisal reports | Prepare appraisal reports (Gemini) | 0.97 |
| Manage communications between the department and program managers | Communicate between programs and the Department (Gemini) | 0.93 |
| Examine bid samples | Examine bid samples to determine compliance with specifications or bid standards (Groq) | 0.92 |
| Develop suicide prevention plans | Lead development and implementation of a citywide suicide prevention strategy (Groq) | 0.87 |

**False positives**
| Model output | Section | Category |
|---|---|---|
| Refine internal processes (Gemini) | responsibilities | extra task not in gold |
| Deploy application updates (Gemini) | responsibilities | extra task not in gold |
| Configure Microsoft Purview (Gemini) | skills | skill as task |
| develop spreadsheets (Groq) | preferred_qualifications | skill as task |
| Manage hearings (Gemini) | general_description | weak/implied evidence |
| Source relationships with equity investors (Gemini) | responsibilities | compound over-split |

**False negatives**
| Human gold | System | Category |
|---|---|---|
| Oversee design reviews | Groq | missed |
| Respond to requests from other teams | Gemini | missed |
| Lead high cost deals | Gemini | near-miss (0.77) |
| Strong writing skills | Groq | gold label is a skill |

All rows: `outputs/task_evaluation/{gemini,groq,baseline_agreed,baseline_valid}/{true_positives,false_positives,false_negatives,ambiguous_review,per_posting}.csv`.

---

## 5. Extraction pipeline output (full 200-posting frozen sample)

| | Count |
|---|---:|
| Frozen postings | 200 |
| Extracted task statements | 3,614 |
| Evidence-valid tasks | 3,479 |
| Rejected: invalid evidence | 106 |
| Rejected: unsupported statement | 26 |
| Cross-model agreed (Gemini ∩ Groq) | 683 |
| Unit tests | 32 / 32 passing |

Architecture: canonical postings → rule-based section parsing → Gemini / Groq extraction → evidence validation (each task must cite a span that appears verbatim in the posting) → `task_pipeline.task_statements`.

---

## 6. O*NET reference alignment (existing results, not rerun)

Each of the 3,479 evidence-valid tasks was compared with O*NET task statements by embedding similarity, using provisional thresholds (accept ≥ 0.86, review ≥ 0.75):

| Decision (top-1 candidate) | Tasks |
|---|---:|
| Accepted candidate match | 286 |
| Review candidate | 2,503 |
| Unmatched | 690 |
| **Total** | **3,479** |

**These are candidate mappings, not ground truth.** O*NET is used *after* extraction as a reference vocabulary. It never generates or filters the extracted tasks. The thresholds come from a 28-pair spot check and have not been validated against human judgments.

---

## 7. Limitations

- The metrics cover the completed human-labeled evaluation subset: 43 of the 100 gold postings, 129 tasks, one annotator. Inter-annotator agreement has not been measured yet, because the 20-posting audit file is unlabeled.
- The 3-tasks-per-posting gold makes precision a lower bound and compresses F1. Treat recall and the error categories as the primary signal.
- Near-miss pairs (cos 0.75–0.85) have not been human-adjudicated. Strict and upper-bound recall are both reported.
- Gold `adjudication_status` was not filled in. Completion was inferred from content, as described in §2.

## 8. Artifacts

- `outputs/task_evaluation/metrics.json`, `summary.csv`: all systems, micro and macro metrics, error categories
- `outputs/task_evaluation/gold_row_validation.csv`, `gold_validation_problems.csv`: gold checks
- `outputs/task_evaluation/prepare_gold_subset.py`: builds the evaluated subset from the annotator file
- Regenerate: `python outputs/task_evaluation/prepare_gold_subset.py` then
  `python -m src.prototype.task_eval evaluate --gold outputs/task_evaluation/gold_subset_labelled.csv --out outputs/task_evaluation`
