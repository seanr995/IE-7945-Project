# Task Purity Audit: Results

**Date:** 2026-10-22
**Audit file:** `outputs/task_evaluation/task_purity_audit_100.csv`, labelled by a human
**Code:** `outputs/task_evaluation/build_sprint3_task_input.py`
- No LLM or API calls.
- `task_pipeline.task_statements` is only read.

**Detail:** `outputs/task_evaluation/task_purity_results.csv`

## 1. Validation
- 100 rows, 100 unique `statement_id`s.
- Every `human_label` is one of TASK, SKILL or OTHER. There are no blank or invalid labels.
- All 100 rows still match `task_statements` on id, text and provider, and all are evidence-valid.

## 2. Counts

| | TASK | SKILL | OTHER | Purity | 95% CI (Wilson) |
|---|---:|---:|---:|---:|---|
| **All 100** | **84** | **9** | **7** | **84.0%** | 75.6–89.9% |
| Gemini (50) | 39 | 7 | 4 | 78.0% | 64.8–87.3% |
| Groq (50) | 45 | 2 | 3 | 90.0% | 78.6–95.7% |

The sample is stratified 50/50 by provider, but Gemini makes up 91% of the population. Re-weighted to the population by provider × section, purity of all 3,479 evidence-valid statements is **82.6%**.

## 3. By source section

| Section (as stored in `source_section`) | Audited | TASK | SKILL | OTHER | Purity | Population |
|---|---:|---:|---:|---:|---:|---:|
| responsibilities | 59 | 53 | 1 | 5 | 89.8% | 2,342 |
| general_description | 26 | 26 | 0 | 0 | 100% | 748 |
| requirements | 9 | 4 | 3 | 2 | **44.4%** | 215 |
| preferred_qualifications | 6 | 1 | 5 | 0 | **16.7%** | 172 |
| skills | 0 | – | – | – | not audited | 2 |

The section detector only produces these five labels. "Duties" is part of `responsibilities`, and "qualifications" is part of `requirements`. There is no separate `other` section.

## 4. Cause of contamination
1. **Qualification sections are the main source.** They contain 8 of the 9 SKILLs and 2 of the 7 OTHERs: 10 of 16 contaminated rows, with only 33% TASK.
   - The models turned requirement text into verb phrases: "Set priorities", "Carry out simple instructions", "Use GIS platforms", "Maintain working relationships".
   - This agrees with the gold-100 evaluation, where only 1–6% of Gemini's qualification-section predictions matched a human task.
2. **The rest (6 of 16) sit inside responsibilities, mostly OTHER.** Examples:
   - Team or context description: "milling and paving 1,150 lane miles yearly"
   - Aspirational phrasing: "Push for continuous improvement in execution", "Learn from your peers…"
   - One SKILL: "Navigate technical trade-offs"
   - These cannot be separated by section. Removing them would need keyword rules, which were not applied.
3. **By provider:** Gemini contributes 11 of the 16. It extracts far more from qualification sections: 375 statements, against 14 for Groq.

## 5. Decision: FILTER BEFORE E5 (purity 84.0% < 95%)

**Filter (validated by section behaviour):** remove the three unsafe sections.
- `requirements` and `preferred_qualifications`: audited TASK rate below 50%, and human-gold match rate 1–6%.
- `skills`: 2 statements. Not audited, but they match 0 of 2 gold tasks and are skill content by definition.

| | Statements |
|---|---:|
| Original evidence-valid | 3,479 |
| Removed (unsafe sections) | 389 |
| **Sprint 3 input** | **3,090** (Gemini 2,792, Groq 298) |

**Estimated purity after filtering:**
- **92.9%** on the 85 audited rows from the kept sections (79 TASK; 95% CI 85.4–96.7%).
- **88.2%** when re-weighted to the population. The difference comes from Gemini responsibilities at 20/24 TASK; that group has 2,099 statements.

**The filtered set is still below the 95% target.** The remaining contamination is about 7–12%, scattered inside responsibilities and mostly OTHER. Section filtering cannot remove it, and the 85-row sample cannot show it reaching 95%. Options before Sprint 3:
- (a) Accept it, and expect some noise in the clusters.
- (b) Audit more Gemini responsibility statements.
- (c) Add a validated second-stage filter.

**Table:** `task_pipeline.sprint3_task_input` in `database/taxonomy.duckdb`.
- **Columns:** `statement_id, posting_id, provider, statement_text, evidence_span, source_section, evidence_valid`, plus `confirming_provider, phase, filter_version`.
- **Why a table, not a view:** `build_db.preserve_prototype_schema` only copies tables across Sprint 1 rebuilds.
- **Rebuild:** run the script above. It uses `CREATE OR REPLACE` and is deterministic.
