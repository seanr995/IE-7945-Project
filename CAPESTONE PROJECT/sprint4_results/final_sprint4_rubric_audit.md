# Sprint 4 Final Rubric Audit

Based only on the REAL files produced by the completed Colab run (`sprint4_results/`). No mock values; no model/API call was made for this audit.
Status vocabulary: COMPLETE / PARTIALLY COMPLETE / REQUIRES HUMAN REVIEW / MISSING.

## A. Rubric requirements

| # | Rubric requirement | Status | Proving file(s) | Real evidence |
|---|---|---|---|---|
| 1a | Name and describe every leaf topic | COMPLETE | `leaf_topics_labeled.csv` | 80/80 rows with topic_name + description; final_status LABELED=80; mean label confidence 0.922 |
| 1b | Identify vague/mixed topics | COMPLETE (automated); REQUIRES HUMAN REVIEW (confirmation) | `leaf_topics_labeled.csv` (mixed_topic, coherent), `human_review_priority.csv` | Gemini flagged 1 mixed topic (LT_048 Designing Experiences and Curricula); the local validator rejects vague labels; 0 unresolved. Humans must confirm which labels are vague |
| 2 | Build parent levels above the leaves | COMPLETE | `task_hierarchy.csv`, `hierarchy_structure.json`, `leaf_topics_labeled.csv` | 8 Level-1 work domains, 20 Level-2 activity families, 80 Level-3 leaves, nested; every task has L1/L2/L3 |
| 3a | O*NET coverage | COMPLETE | `onet_summary.json`, `onet_evaluation.csv` | 0.975 (78/80 topics with meaningful alignment) |
| 3b | O*NET purity | COMPLETE | `onet_summary.json`, `onet_evaluation.csv` | mean 0.1935, median 0.1765 |
| 3c | O*NET nearest-neighbor agreement | COMPLETE | `onet_summary.json`, `onet_evaluation.csv` | top-10 agreement mean 0.6225; exact top-1 mean 0.2425 |
| 4 | Review hierarchy by hand; document over-/under-merging | REQUIRES HUMAN REVIEW | `human_review_template.csv`, `human_review_priority.csv`, `HUMAN_REVIEW_GUIDE.md`, `apply_human_review.py`, `human_review_summary.md` | Package ready; ALL human columns are blank -> HUMAN REVIEW PENDING. Automated over-/under-merge indicators are review hints only (section C) |
| 5 | Labeled hierarchical taxonomy + evaluation report | PARTIALLY COMPLETE (pending human review) | `task_hierarchy.csv`, `leaf_topics_labeled.csv`, `sprint4_evaluation_report.md`, `Sprint4_Labeling_Hierarchy_Presentation.pptx` | Taxonomy and report complete; the hand-review component is outstanding |

## B. Specific verifications

| Check | Result | Source |
|---|---|---|
| Labeled Leiden leaves | 80 | `leaf_topics_labeled.csv`, `validation.json` |
| Frozen task memberships | 3,015 tasks, 80 leaves, identical to `final_task_topic_membership_leiden80.csv` (re-verified by direct statement_id comparison) | `task_hierarchy.csv` vs frozen Sprint 3 deliverable |
| Every leaf has name + description | Yes (80/80) | `leaf_topics_labeled.csv` |
| Hierarchy contains L1, L2, L3, task membership | Yes: level1/level2/leaf ids and names per task | `task_hierarchy.csv` |
| Level-1 count | 8 | `run_summary.json`, `task_hierarchy.csv` |
| Level-2 count | 20 | `run_summary.json`, `task_hierarchy.csv` |
| Hierarchy method actually selected | Ward linkage on the 80 leaf centroids, 20 L2 / 8 L1 (score 0.9026); then ONE Gemini critic round (1 leaf move, no merges/splits) | `hierarchy_diagnostics.csv`, `run_summary.json` |
| Average-linkage diagnostics | 0 of 9 candidates pass the gates. 20/8: L1 dominant share 0.866 (2612 of 3,015 tasks in one domain), 6 tiny L2 parents, 4 tiny L1 parents, score 0.4212 | `hierarchy_diagnostics.csv` |
| Ward diagnostics | 9 of 9 candidates pass the gates. 20/8: L1 dominant share 0.213 (max 643 tasks), L2 dominant share 0.105, 0 tiny L1 parents, score 0.9026 (selected) | `hierarchy_diagnostics.csv` |
| O*NET coverage / purity / NN agreement | 0.975 / 0.1935 / 0.6225 (top-10) | `onet_summary.json` |
| Gemini O*NET judgments | STRONG 58, PARTIAL 20, NO_MATCH 2 (LT_060, LT_076) | `onet_evaluation.csv` |
| Hierarchy stability | 5 seeds, 80% sampling; mean L1 persistence 0.7150, L2 0.7925; pairwise ARI L1 0.4843, L2 0.6501; ARI vs final L1 0.5504, L2 0.6469; leaves with persistence < 0.60: L1 24, L2 16 | `hierarchy_stability.csv`, `hierarchy_stability_summary.json` |
| O*NET coverage-gap analysis | WELL_COVERED 33, PARTIALLY_COVERED 44, POSSIBLE_EMERGING_WORK 3 (LT_041, LT_060, LT_062) | `onet_gap_analysis.csv`, `onet_gap_summary.json` |
| Explainability cards | 80 (`topic_explainability_cards.csv` 80 rows + `topic_explainability_cards/LT_001..LT_080.md`) | `topic_explainability_cards*` |
| Critic / refinement | 1 round, 31 cases: 30 ACCEPT, 1 MOVE_PARENT (LT_048), 0 renames, 0 merges, 0 splits, 0 flagged; 1.25% leaf-to-parent assignments changed; stop: converged | `run_summary.json`, `agent_decisions.jsonl` |
| Unresolved cases | None (`unresolved` is empty) | `run_summary.json` |
| Actual Gemini real calls | 10 (all successful; leaf x4, hierarchy x1, O*NET x4, critic x1); 0 repair / 0 retry / 0 split; hard cap 12; model gemini-3.1-flash-lite; tokens 36,351 in / 26,236 out; Sprint 4 Claude calls 0 | `api_usage.csv`, `api_ledger.jsonl`, `run_summary.json` |
| Actual Gemini cache usage | 5 cache hits (leaf x4 + hierarchy x1 re-used when the run was resumed, so the first five calls were not repeated) | `api_usage.csv` (cached=True rows) |

## C. Observations for the human reviewers (hints, not verdicts)
- Singleton Level-2 families (possible under-merging, or genuinely distinct work): LT_041 Developing AI Application Roadmaps (34 tasks), LT_080 Represent Company Externally (6 tasks).
- Level-1 domains containing a single Level-2 family: D1_06, D1_07.
- 24 leaves have Level-1 persistence < 0.60 and 16 have Level-2 persistence < 0.60: their parent assignment is sensitive to perturbation.
- O*NET purity is low in absolute terms (mean 0.194) because top-10 O*NET neighbours span many occupations; read it against the P25/P50 thresholds in `onet_gap_summary.json`, not as an accuracy score.
- LT_048 (the one Gemini-flagged mixed topic) combines design of experiences/curricula with engineering design tasks; the critic moved it once.

## D. Provenance note
`onet_gap_analysis.csv` and the explainability cards originally carried the word "Claude" in their evidence strings (leftover template wording). Gemini made those judgments, so wording-only corrections ("Claude" -> "Gemini") were applied to those saved files and to `sprint4_src/extras.py`. No numeric value or membership changed.

## E. Remaining items
1. Manual hierarchy review by teammates (`HUMAN_REVIEW_GUIDE.md`), then `python apply_human_review.py`. Current status: **HUMAN REVIEW PENDING**.
