# Sprint 4 Human Review Guide

**Status: HUMAN REVIEW COMPLETE (80/80).** Do not edit leaf memberships.

## Files
- `human_review_priority.csv` - the 80 topics, riskiest first (`review_priority` 1 = review first; `risk_reason` says why).
- `human_review_template.csv` - the sheet to fill (join on `topic_id`).
- `topic_explainability_cards/LT_xxx.md` (and `topic_explainability_cards.csv`) - one evidence card per topic: description, representative tasks, boundary tasks, keywords, hierarchy path, O*NET matches.
- `task_hierarchy.csv`, `hierarchy_stability.csv`, `onet_gap_analysis.csv` - supporting evidence.

## Efficient process
1. **Split the work.** Two or more reviewers take alternating rows of `human_review_priority.csv`. Priority 1-20 are reviewed independently by TWO people (they carry most of the risk); 21-80 by one.
2. **Per topic (about 3-4 minutes; 8-10 for the top 20).** Open the card. Read name and description, then representative and boundary tasks. Treat the O*NET matches as a cross-check only.
3. **Answer five questions in the template** (Y or N):
   - `label_correct` - is the leaf label correct for the representative tasks?
   - `coherent` - do the representative tasks describe ONE coherent work concept?
   - `boundary_ok` - do the boundary tasks reasonably belong in this topic?
   - `over_merged` - Y if the topic mixes two or more distinct concepts that should be separate.
   - `under_merged` - Y if another topic describes the same concept and they should be one; name that topic in `reviewer_notes`.
4. **Free text only when needed:** `suggested_name` when `label_correct` = N; `reviewer_notes` for boundary examples, the other topic for under-merging, or a parent concern.
5. **Check parents once, not per leaf.** After the top-priority topics, read the 20 Level-2 families and 8 Level-1 domains (`task_hierarchy.csv`) and note any over-/under-merged parent in `reviewer_notes` of one child topic, prefixed `PARENT:`.
6. **Reconcile.** Compare the two independent reviews of priority 1-20, agree on one answer, record it. Leave a cell blank rather than guess.
7. **Ingest.** Save the completed sheet in place and run `python apply_human_review.py` from `CAPESTONE PROJECT/`. It writes `human_review_summary.md` and `human_review_issues.csv` and never changes memberships. Renames are only reported; apply them later as a deliberate step.

## Review first (current top 10)
| Priority | Topic | Why |
|---|---|---|
| 1 | LT_048 Designing Experiences and Curricula | Gemini flagged mixed_topic; label marked not coherent; low label confidence (0.70); Level-1 persistence 0.0; Level-2 persistence 0.0; PARTIAL_MATCH to O*NET; low O*NET purity (0.032); critic action MOVE_PARENT |
| 2 | LT_043 Owning End-to-End Product Commercialization | Level-1 persistence 0.2; Level-2 persistence 0.4; PARTIAL_MATCH to O*NET; low O*NET purity (0.029) |
| 3 | LT_041 Developing AI Application Roadmaps | Level-1 persistence 0.4; PARTIAL_MATCH to O*NET; possible emerging work (limited O*NET alignment); low O*NET purity (0.088) |
| 4 | LT_062 Define Product Roadmaps | Level-2 persistence 0.4; PARTIAL_MATCH to O*NET; possible emerging work (limited O*NET alignment); low O*NET purity (0.074) |
| 5 | LT_076 Drive Organizational Growth | Level-1 persistence 0.4; Gemini O*NET judgment NO_MATCH; low O*NET purity (0.067) |
| 6 | LT_023 Coordinate Organizational Communications | Level-1 persistence 0.4; Level-2 persistence 0.4; PARTIAL_MATCH to O*NET |
| 7 | LT_018 Supervise Staff And Personnel | Level-1 persistence 0.4; Level-2 persistence 0.2; weak boundary tasks (mean cos 0.842) |
| 8 | LT_005 Conduct Quality Assurance Testing | low label confidence (0.85); Level-1 persistence 0.4; low O*NET purity (0.119); weak boundary tasks (mean cos 0.848) |
| 9 | LT_040 Prepare Technical Specifications | Level-1 persistence 0.0; Level-2 persistence 0.0 |
| 10 | LT_060 Maintaining Data Pipelines | Gemini O*NET judgment NO_MATCH; possible emerging work (limited O*NET alignment) |

## How risk was ranked (from saved evidence)
Points for: Gemini mixed_topic flag; not coherent; label confidence below 0.90; Level-1/Level-2 persistence below 0.6 (plus 1 - persistence); Gemini O*NET NO_MATCH/PARTIAL; POSSIBLE_EMERGING_WORK; O*NET purity in the lowest quartile; weak boundary tasks (lowest-quartile boundary similarity); critic action other than ACCEPT. The score only orders the work; it is not a quality judgment (`risk_score` is the last column of the priority file).

## What counts as an issue
Over-merged = one leaf, several concepts. Under-merged = one concept split over leaves. Boundary issue = boundary tasks belong elsewhere. Label issue = the name misleads.
