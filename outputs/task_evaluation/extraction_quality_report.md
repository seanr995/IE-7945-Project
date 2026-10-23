# Task Extraction Quality: Gold-100 Evaluation

**Date:** 2026-10-22
**Inputs:**
- Human gold `data/annotations/task_gold_100.csv`, via the cleaned copy `gold100_clean.csv`
- Cached model outputs in `database/taxonomy.duckdb`

**Code:** `outputs/task_evaluation/evaluate_gold100.py`, which drives the existing `src.prototype.task_eval.evaluate` and `match`.

**What was run:**
- No Gemini, Groq or Claude calls, and no re-extraction.
- No DB writes, no E5 and no Sprint 3 work.
- Similarity uses the project's cached local embedding model (`BAAI/bge-small-en-v1.5`). This is **not** E5.

**Artifacts:**
- `extraction_metrics.csv`: every system × matching mode, micro and macro
- `gold100/<mode>/<system>/`: TP, FP and FN lists, per-posting results, error summary
- `gold100_threshold_calibration.csv` and `gold100_threshold_band_samples.csv`
- `gold100_error_triage_sem_0.80.csv` / `_0.85.csv` and their summaries
- `task_purity_audit_100.csv`
- `gold100/summary.json`

---

## Final summary

**A. Gold dataset:** 100 unique postings, 1,056 human tasks (10.6 per posting). See `gold_validation_report.md`: Excel padding rows, status spelled `labeled`, evidence spans are per-posting blocks rather than per-task.

**B. Gemini** (100 postings, 1,636 predictions)

| Matching | P | R | F1 |
|---|---:|---:|---:|
| Strict (normalized exact) | 13.8% | 21.3% | 16.7% |
| **Semantic, cosine ≥ 0.85** | **40.1%** | **62.1%** | **48.7%** |
| Semantic, cosine ≥ 0.80 | 45.7% | 70.7% | 55.5% |

**C. Groq** (52 postings only, 495 predictions)

| Matching | P | R | F1 |
|---|---:|---:|---:|
| Strict (normalized exact) | 10.7% | 9.4% | 10.0% |
| **Semantic, cosine ≥ 0.85** | **57.2%** | **50.3%** | **53.5%** |
| Semantic, cosine ≥ 0.80 | 69.5% | 61.1% | 65.0% |

On the same 52 postings, Gemini at 0.85 scores P 40.4%, R 60.4%, F1 48.4%.

**D. Task purity**

What can be verified now:
- Only 9 of 3,479 evidence-valid `task_statements` use skill or qualification wording (regex).
- The pipeline's `looks_like_qualification` flag marks 0.
- **The main contamination risk is a different pool:** 389 of 3,479 evidence-valid statements (11.2%) were extracted from `requirements`, `preferred_qualifications` or `skills` sections.
  - Against human gold, only 3% of Gemini's (5 of 157) and 15% of Groq's (2 of 13) qualification-section predictions match a human task at 0.85.
  - Responsibilities and description predictions match at 43–73%.
  - Examples: "Meet deadlines", "Work with frequent interruptions", "Manage multiple assignments", "Deliver public presentations".
- No human-validated TASK/SKILL/OTHER classifier exists. The model's own `kind` flag only splits task from skill and has never been checked by a human.

**Manual audit is still needed.** `task_purity_audit_100.csv` is ready to label: 100 rows, 50 per provider, `human_label` blank.

**E. DECISION: "FIX TASK EXTRACTION FIRST"**

**Reason:** about 11% of evidence-valid `task_statements` (389 of 3,479) come from qualification sections. Human annotators almost never accepted these as tasks (3% match rate for Gemini). Unfiltered, they would go into E5 embeddings and Sprint 3 clustering as pseudo-tasks such as "Meet deadlines" and "Work with frequent interruptions".

**The fix is narrow and needs no re-extraction:**
1. Label `task_purity_audit_100.csv`.
2. If the audit confirms the pattern, exclude or flag `source_section IN ('requirements','preferred_qualifications','skills')` when building the Sprint 3 input set.
3. Then move to E5.

Recall is not the blocker. Only about 9–11% of gold tasks are truly missed (section 4). Most other gold-model differences are wording or granularity. If the audit finds ≥ 95% TASK including the qualification-section rows, this verdict flips to **SAFE TO MOVE TO E5 EMBEDDINGS** with no changes.

---

## 1. Model outputs found (actual schema)

| Source | Rows | Evidence-valid | Postings | Notes |
|---|---:|---:|---:|---|
| `task_pipeline.task_statements`, total | 3,614 | 3,479 | 199 | `experiment_arm = baseline_gemini_groq`; `is_valid_task` = evidence_valid (3,479) |
| `task_statements`, provider = gemini | 3,280 | 3,167 | 199 | 684 of these are Groq-confirmed (`confirming_provider = 'groq'`) |
| `task_statements`, provider = groq | 334 | 312 | 90 | **Groq-only** tasks. Agreements are merged into the Gemini row |
| `prototype.prototype_model_outputs`, gemini, kind = task, gate accepted | 3,170 | all evidence-located | 199 | raw evidence-gated output |
| `prototype.prototype_model_outputs`, groq, kind = task, gate accepted | 996 | all evidence-located | 103 | raw evidence-gated output |

**On the 100 gold postings:**
- Raw accepted tasks: Gemini 1,636 (100 postings), Groq 495 (51 postings with tasks, 52 with an OK call).
- `task_statements`: Gemini 1,668 rows (1,634 evidence-valid), Groq-only 166 (161 evidence-valid).

**Did both providers run on every gold posting? No.** Gemini covers 100 of 100. Groq covers 52 of 100: 50 `dual_model` postings plus 2 `gemini_primary` adjudications, because the frozen design ran Groq only in the dual-model phase.

**Systems evaluated.** These are the existing `task_eval.SYSTEMS` definitions, unchanged:
- `gemini` / `groq`: raw evidence-gated outputs.
- `consensus` (`baseline_agreed`): `review_status = accepted_cross_model_agreement`, on postings where both ran.
- `union` (`baseline_valid`): every `is_valid_task`.

## 2. Matching method

- **Scope:** only within the same `posting_id`. Pairs are assigned greedily and one-to-one, so a gold task can absorb at most one prediction and vice versa. Prediction strings are de-duplicated per posting.
- **Strict:** normalized exact only (`_norm`: lower case, unified quotes and dashes, whitespace collapsed, trailing punctuation dropped). Lexical and semantic stages are disabled.
- **Semantic @ t:** normalized exact, or bge-small cosine ≥ t. The lexical stage is disabled so the semantic results are not mixed with Jaccard.
- **Project-default cascade** (exact → Jaccard 0.6 → cosine 0.85) is in `extraction_metrics.csv` as `cascade` for continuity with Sprint 2. Its numbers are within 2 pt of sem_0.85.
- Micro-averaged. Macro per-posting values are in the CSV.

## 3. Model comparison

### Strict (normalized exact)

| System | Postings | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gemini | 100 | 225 | 1,411 | 831 | 13.8% | 21.3% | 16.7% |
| Groq | 52 | 53 | 442 | 510 | 10.7% | 9.4% | 10.0% |
| Gemini/Groq consensus | 52 | 82 | 252 | 481 | 24.6% | 14.6% | 18.3% |
| Union (all valid task_statements) | 100 | 233 | 1,562 | 823 | 13.0% | 22.1% | 16.3% |
| Gemini on Groq's 52 postings | 52 | 124 | 718 | 439 | 14.7% | 22.0% | 17.7% |

Strict scores are low because human gold tasks are short paraphrases (median 4 words), not copied text. Strict matching is reported for completeness and is not a useful quality estimate here.

### Semantic (exact or cosine ≥ t), reported separately

| System | t | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gemini (100) | 0.75 | 822 | 814 | 234 | 50.2% | 77.8% | 61.1% |
| | 0.80 | 747 | 889 | 309 | 45.7% | 70.7% | 55.5% |
| | **0.85** | 656 | 980 | 400 | **40.1%** | **62.1%** | **48.7%** |
| Groq (52) | 0.75 | 382 | 113 | 181 | 77.2% | 67.9% | 72.2% |
| | 0.80 | 344 | 151 | 219 | 69.5% | 61.1% | 65.0% |
| | **0.85** | 283 | 212 | 280 | **57.2%** | **50.3%** | **53.5%** |
| Consensus (52) | 0.75 | 279 | 55 | 284 | 83.5% | 49.6% | 62.2% |
| | 0.80 | 255 | 79 | 308 | 76.4% | 45.3% | 56.9% |
| | **0.85** | 227 | 107 | 336 | **68.0%** | **40.3%** | **50.6%** |
| Union (100) | 0.80 | 785 | 1,010 | 271 | 43.7% | 74.3% | 55.1% |
| | 0.85 | 687 | 1,108 | 369 | 38.3% | 65.1% | 48.2% |
| Gemini on Groq's 52 | 0.80 | 388 | 454 | 175 | 46.1% | 68.9% | 55.2% |
| | 0.85 | 340 | 502 | 223 | 40.4% | 60.4% | 48.4% |

**Reading:**
- Gemini has the higher recall: it finds more of the human tasks.
- Groq is more precise and has the higher F1 on the postings it covers. It extracts fewer, coarser tasks: 9.5 per posting, against 16.4 for Gemini and 10.6 in the gold.
- Consensus raises precision to 68% but loses about 20 pt of recall.

## 4. Threshold choice (not chosen silently)

**Calibration.** For every gold task, I took the best cosine to (a) its own posting's predictions and (b) the predictions of a deterministic, unrelated posting. The second is a false-match proxy (`gold100_threshold_calibration.csv`).

| t | Gemini: own posting ≥ t | Gemini: unrelated posting ≥ t | Groq: own | Groq: unrelated |
|---:|---:|---:|---:|---:|
| 0.75 | 81.6% | **9.9%** | 79.7% | 5.1% |
| 0.80 | 73.0% | 2.5% | 68.3% | 0.0% |
| 0.85 | 62.9% | 0.5% | 53.8% | 0.0% |

I also read the sampled matched pairs in each band (`gold100_threshold_band_samples.csv`). This was my reading, not a human label:
- **[0.75, 0.80): about half the pairs are wrong or doubtful.** Examples: "Maintain applicant tracking system" ↔ "Improve recruiting workflows", "Monitor administrative processes" ↔ "Develop administrative materials", "Windows 11 knowledge" ↔ "Maintain Windows 11 updates".
- **[0.80, 0.85): about 3 in 4 are clean.** Wrong examples: "Supervise a team" ↔ "Conduct team conferences", "Resolve procurement issues with vendors" ↔ "Implement procurement procedures".
- **[0.85, 0.90): about 9 in 10 are clean.**

**Recommendation:**
- **0.85 is the defensible headline.** It has a false-match proxy of about 0.5%, matches the project config (`semantic_match_min`), and matches the Sprint 2 setting.
- **Read 0.80 as an optimistic upper bound.** About a quarter of its extra matches are wrong.
- **Reject 0.75.** About 10% of gold tasks find a ≥ 0.75 "match" in an unrelated posting.

The pairs in [0.80, 0.85) are the ones a human should adjudicate if a single number is needed. The pairs are in `gold100/sem_0.80/<system>/true_positives.csv` with `score < 0.85`.

## 5. What the "errors" are (rule 6: model-only ≠ genuine FP)

These categories are deterministic triage, not ground truth (`gold100_error_triage_sem_0.85.csv`).

The rules, applied in order:
1. **Unsupported or non-task:** qualification or skill wording, or weak evidence.
2. **Qualification-section item:** extracted from a requirements, preferred-qualifications or skills section.
3. **Wording or splitting mismatch:** best cosine to some gold task ≥ 0.70, or ≥ 60% of a gold task's content words appear in the model's evidence sentence.
4. **Otherwise:** a possible legitimate task missing from gold (for model-only items) or a true miss (for gold-only items).

**Model-only (FP) at 0.85:**

| Category | Gemini | Groq | Consensus |
|---|---:|---:|---:|
| Unsupported / non-task (skill or qualification wording, weak evidence) | 36 | 3 | 5 |
| Qualification-section item (likely non-task) | 145 | 10 | 5 |
| Wording or splitting mismatch (same source sentence as a gold task) | 710 | 168 | 89 |
| Possible legitimate task missing from gold | 89 | 31 | 8 |
| **Total FP** | 980 | 212 | 107 |

**Gold-only (FN) at 0.85:**

| Category | Gemini | Groq | Consensus |
|---|---:|---:|---:|
| Gold row is a skill or qualification | 9 | 9 | 10 |
| Wording or splitting mismatch (model used the same sentence) | 292 | 207 | 197 |
| **True missed human task** | **99 (9.4% of 1,056)** | **64 (11.4% of 563)** | 129 (22.9%) |
| **Total FN** | 400 | 280 | 336 |

**What this means:**
- **Granularity drives most FPs.** About 72% of Gemini FPs come from a sentence that also produced a matched or near-matched gold task. Gemini splits one human task into several statements; one-to-one matching keeps one and scores the rest as FPs. This is a granularity difference, not a non-task problem.
- **Genuine non-task output** (the first two rows) is 181 of Gemini's 1,636 predictions (11%) and 13 of Groq's 495 (3%).
- **Possible gold omissions:** 89 Gemini predictions and 31 Groq predictions are task-like and come from sentences the annotator did not label.
- **True misses:** about 9–11% of human tasks for each single model. They are listed per posting in `gold100/sem_0.85/<system>/false_negatives.csv`.

## 6. Task purity (TASK vs SKILL vs OTHER)

**Existing flags:**
- `prototype_model_outputs.kind`: the model's own task/skill label at extraction. Accepted items: Gemini 3,170 tasks and 1,934 skills; Groq 996 tasks and 1,166 skills. `task_statements` contains only `kind = task`.
- `task_statements.looks_like_qualification` (regex): **0 of 3,614** flagged.
- There is no OTHER class and no human validation of either flag. **TASK/SKILL/OTHER counts therefore cannot be reported as measured.**

**Deterministic signals on 3,479 evidence-valid `task_statements`:**
- **Skill or qualification wording** (`QUALIFICATION_LIKE`, or a trailing "skills/knowledge/experience/ability/degree"): **9**.
- **Statements of 2 words or fewer: 579.** Inspected by eye, these are verb + object tasks ("Draft agendas", "Screen vendors", "Perform upgrades"), not bare skills.
- **From qualification sections: 389 (11.2%).** Gemini accounts for 375. Gold says these are mostly not tasks (section 3; match rate at 0.85):

| Source section | Gemini predictions | Matched to human task | Groq predictions | Matched |
|---|---:|---:|---:|---:|
| responsibilities | 1,239 | 44.3% | 386 | 54.7% |
| general_description | 238 | 42.9% | 96 | 72.9% |
| requirements | 87 | **1.1%** | 4 | 0% |
| preferred_qualifications | 70 | **5.7%** | 9 | 22.2% |

**Manual audit file.** `outputs/task_evaluation/task_purity_audit_100.csv` contains `statement_id, posting_id, provider, statement_text, evidence_span, human_label`, with `human_label` blank.
- **Selection:** 50 evidence-valid statements per provider, ordered by sha256(`gold100-eval-2026-10-22|purity|statement_id`).
- **Stratified, not proportional.** Groq is about 9% of `task_statements`, so weight by provider (Gemini 3,167 / Groq 312) when estimating overall purity.
- **Qualification-section rows:** the sample includes 15 (Gemini 12, Groq 3).

## 7. What could not be measured

- Groq and consensus quality on 48 of the 100 gold postings. Measuring them needs new Groq calls.
- Evidence-span grounding of individual gold tasks, because gold spans are per-posting blocks.
- TASK/SKILL/OTHER purity rates, until `task_purity_audit_100.csv` is labelled.
- Whether the 89 + 31 "possible gold omissions" are real tasks. That needs a human pass on the triage CSV.
- Inter-annotator agreement. `task_gold_audit_20_annotator_b.csv` was not part of this run.
