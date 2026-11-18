import json, pandas as pd
import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sprint4_results'))
R = json.load(open('run_summary.json')); O = R['onet']; S = R['stability']; G = R['gap']; LG = R['ledger']; C = R['convergence']['rounds'][0]
D = pd.read_csv('hierarchy_diagnostics.csv'); Lf = pd.read_csv('leaf_topics_labeled.csv'); P = pd.read_csv('human_review_priority.csv')
avg = D[(D.stage == 'search') & (D.linkage == 'average')]; wd = D[(D.stage == 'search') & (D.linkage == 'ward')]
a20 = avg[(avg.level2_count == 20) & (avg.level1_count == 8)].iloc[0]; w20 = wd[(wd.level2_count == 20) & (wd.level1_count == 8)].iloc[0]
sing = Lf.groupby('level2_id').topic_id.count(); single_f = Lf[Lf.level2_id.isin(sing[sing == 1].index)][['topic_id', 'topic_name', 'topic_size']]
fam1 = Lf.groupby('level1_id').level2_id.nunique()


def tab(hdr, rows):
    return '\n'.join(['| ' + ' | '.join(hdr) + ' |', '|' + '---|' * len(hdr)] + ['| ' + ' | '.join(map(str, r)) + ' |' for r in rows])


NOMATCH = ', '.join(Lf[Lf.onet_alignment == 'NO_MATCH'].topic_id)
EMERG = 'LT_041, LT_060, LT_062'
audit = f"""# Sprint 4 Final Rubric Audit

Based only on the REAL files produced by the completed Colab run (`sprint4_results/`). No mock values; no model/API call was made for this audit.
Status vocabulary: COMPLETE / PARTIALLY COMPLETE / REQUIRES HUMAN REVIEW / MISSING.

## A. Rubric requirements

{tab(['#', 'Rubric requirement', 'Status', 'Proving file(s)', 'Real evidence'], [
['1a', 'Name and describe every leaf topic', 'COMPLETE', '`leaf_topics_labeled.csv`', f'80/80 rows with topic_name + description; final_status LABELED=80; mean label confidence {Lf.label_confidence.mean():.3f}'],
['1b', 'Identify vague/mixed topics', 'COMPLETE (automated); REQUIRES HUMAN REVIEW (confirmation)', '`leaf_topics_labeled.csv` (mixed_topic, coherent), `human_review_priority.csv`', f'Gemini flagged {int(Lf.mixed_topic.sum())} mixed topic (LT_048 Designing Experiences and Curricula); the local validator rejects vague labels; 0 unresolved. Humans must confirm which labels are vague'],
['2', 'Build parent levels above the leaves', 'COMPLETE', '`task_hierarchy.csv`, `hierarchy_structure.json`, `leaf_topics_labeled.csv`', '8 Level-1 work domains, 20 Level-2 activity families, 80 Level-3 leaves, nested; every task has L1/L2/L3'],
['3a', 'O*NET coverage', 'COMPLETE', '`onet_summary.json`, `onet_evaluation.csv`', f'{O["coverage_meaningful_alignment"]:.3f} (78/80 topics with meaningful alignment)'],
['3b', 'O*NET purity', 'COMPLETE', '`onet_summary.json`, `onet_evaluation.csv`', f'mean {O["purity_mean"]:.4f}, median {O["purity_median"]:.4f}'],
['3c', 'O*NET nearest-neighbor agreement', 'COMPLETE', '`onet_summary.json`, `onet_evaluation.csv`', f'top-10 agreement mean {O["nn_agreement_top10_mean"]:.4f}; exact top-1 mean {O["nn_agreement_top1_exact_mean"]:.4f}'],
['4', 'Review hierarchy by hand; document over-/under-merging', 'REQUIRES HUMAN REVIEW', '`human_review_template.csv`, `human_review_priority.csv`, `HUMAN_REVIEW_GUIDE.md`, `apply_human_review.py`, `human_review_summary.md`', 'Package ready; ALL human columns are blank -> HUMAN REVIEW PENDING. Automated over-/under-merge indicators are review hints only (section C)'],
['5', 'Labeled hierarchical taxonomy + evaluation report', 'PARTIALLY COMPLETE (pending human review)', '`task_hierarchy.csv`, `leaf_topics_labeled.csv`, `sprint4_evaluation_report.md`, `Sprint4_Labeling_Hierarchy_Presentation.pptx`', 'Taxonomy and report complete; the hand-review component is outstanding'],
])}

## B. Specific verifications

{tab(['Check', 'Result', 'Source'], [
['Labeled Leiden leaves', '80', '`leaf_topics_labeled.csv`, `validation.json`'],
['Frozen task memberships', '3,015 tasks, 80 leaves, identical to `final_task_topic_membership_leiden80.csv` (re-verified by direct statement_id comparison)', '`task_hierarchy.csv` vs frozen Sprint 3 deliverable'],
['Every leaf has name + description', 'Yes (80/80)', '`leaf_topics_labeled.csv`'],
['Hierarchy contains L1, L2, L3, task membership', 'Yes: level1/level2/leaf ids and names per task', '`task_hierarchy.csv`'],
['Level-1 count', R['level1'], '`run_summary.json`, `task_hierarchy.csv`'],
['Level-2 count', R['level2'], '`run_summary.json`, `task_hierarchy.csv`'],
['Hierarchy method actually selected', f'Ward linkage on the 80 leaf centroids, 20 L2 / 8 L1 (score {w20.score:.4f}); then ONE Gemini critic round (1 leaf move, no merges/splits)', '`hierarchy_diagnostics.csv`, `run_summary.json`'],
['Average-linkage diagnostics', f'0 of 9 candidates pass the gates. 20/8: L1 dominant share {a20.L1_dominant_share:.3f} ({int(a20.L1_max_tasks)} of 3,015 tasks in one domain), {int(a20.L2_tiny_parents)} tiny L2 parents, {int(a20.L1_tiny_parents)} tiny L1 parents, score {a20.score:.4f}', '`hierarchy_diagnostics.csv`'],
['Ward diagnostics', f'9 of 9 candidates pass the gates. 20/8: L1 dominant share {w20.L1_dominant_share:.3f} (max {int(w20.L1_max_tasks)} tasks), L2 dominant share {w20.L2_dominant_share:.3f}, {int(w20.L1_tiny_parents)} tiny L1 parents, score {w20.score:.4f} (selected)', '`hierarchy_diagnostics.csv`'],
['O*NET coverage / purity / NN agreement', f'{O["coverage_meaningful_alignment"]:.3f} / {O["purity_mean"]:.4f} / {O["nn_agreement_top10_mean"]:.4f} (top-10)', '`onet_summary.json`'],
['Gemini O*NET judgments', f'STRONG {O["alignment_counts"]["STRONG_MATCH"]}, PARTIAL {O["alignment_counts"]["PARTIAL_MATCH"]}, NO_MATCH {O["alignment_counts"]["NO_MATCH"]} ({NOMATCH})', '`onet_evaluation.csv`'],
['Hierarchy stability', f'5 seeds, 80% sampling; mean L1 persistence {S["mean_level1_persistence"]:.4f}, L2 {S["mean_level2_persistence"]:.4f}; pairwise ARI L1 {S["level1_ari_pairwise_mean"]:.4f}, L2 {S["level2_ari_pairwise_mean"]:.4f}; ARI vs final L1 {S["level1_ari_vs_final_mean"]:.4f}, L2 {S["level2_ari_vs_final_mean"]:.4f}; leaves with persistence < 0.60: L1 {S["leaves_level1_persistence_below_0.60"]}, L2 {S["leaves_level2_persistence_below_0.60"]}', '`hierarchy_stability.csv`, `hierarchy_stability_summary.json`'],
['O*NET coverage-gap analysis', f'WELL_COVERED {G["counts"]["WELL_COVERED"]}, PARTIALLY_COVERED {G["counts"]["PARTIALLY_COVERED"]}, POSSIBLE_EMERGING_WORK {G["counts"]["POSSIBLE_EMERGING_WORK"]} ({EMERG})', '`onet_gap_analysis.csv`, `onet_gap_summary.json`'],
['Explainability cards', '80 (`topic_explainability_cards.csv` 80 rows + `topic_explainability_cards/LT_001..LT_080.md`)', '`topic_explainability_cards*`'],
['Critic / refinement', f'1 round, {C["cases"]} cases: {C["accepts"]} ACCEPT, {C["moves"]} MOVE_PARENT (LT_048), 0 renames, 0 merges, 0 splits, 0 flagged; {C["leaf_assignment_change_share"]*100:.2f}% leaf-to-parent assignments changed; stop: converged', '`run_summary.json`, `agent_decisions.jsonl`'],
['Unresolved cases', 'None (`unresolved` is empty)', '`run_summary.json`'],
['Actual Gemini real calls', f'{LG["gemini_real_calls"]} (all successful; leaf x4, hierarchy x1, O*NET x4, critic x1); 0 repair / 0 retry / 0 split; hard cap {LG["hard_cap"]}; model {LG["model"]}; tokens {LG["input_tokens"]:,} in / {LG["output_tokens"]:,} out; Sprint 4 Claude calls {LG["claude_sprint4_calls"]}', '`api_usage.csv`, `api_ledger.jsonl`, `run_summary.json`'],
['Actual Gemini cache usage', f'{LG["cache_hits"]} cache hits (leaf x4 + hierarchy x1 re-used when the run was resumed, so the first five calls were not repeated)', '`api_usage.csv` (cached=True rows)'],
])}

## C. Observations for the human reviewers (hints, not verdicts)
- Singleton Level-2 families (possible under-merging, or genuinely distinct work): {', '.join(f'{r.topic_id} {r.topic_name} ({r.topic_size} tasks)' for r in single_f.itertuples())}.
- Level-1 domains containing a single Level-2 family: {', '.join(fam1[fam1 == 1].index)}.
- {S["leaves_level1_persistence_below_0.60"]} leaves have Level-1 persistence < 0.60 and {S["leaves_level2_persistence_below_0.60"]} have Level-2 persistence < 0.60: their parent assignment is sensitive to perturbation.
- O*NET purity is low in absolute terms (mean {O["purity_mean"]:.3f}) because top-10 O*NET neighbours span many occupations; read it against the P25/P50 thresholds in `onet_gap_summary.json`, not as an accuracy score.
- LT_048 (the one Gemini-flagged mixed topic) combines design of experiences/curricula with engineering design tasks; the critic moved it once.

## D. Provenance note
`onet_gap_analysis.csv` and the explainability cards originally carried the word "Claude" in their evidence strings (leftover template wording). Gemini made those judgments, so wording-only corrections ("Claude" -> "Gemini") were applied to those saved files and to `sprint4_src/extras.py`. No numeric value or membership changed.

## E. Remaining items
1. Manual hierarchy review by teammates (`HUMAN_REVIEW_GUIDE.md`), then `python apply_human_review.py`. Current status: **HUMAN REVIEW PENDING**.
"""
open('final_sprint4_rubric_audit.md', 'w', encoding='utf-8').write(audit)

metrics = f"""# Sprint 4 Key Metrics (real values and sources)

| Metric | Value | Source file |
|---|---|---|
| Validated tasks | 3,015 | `task_hierarchy.csv` |
| Frozen leaf topics (Sentence-T5 + Leiden) | 80 | `leaf_topics_labeled.csv` |
| Level 1 work domains | {R['level1']} | `run_summary.json` |
| Level 2 activity families | {R['level2']} | `run_summary.json` |
| Hierarchy method | Ward linkage, 20 L2 / 8 L1, composite score {w20.score:.4f}, + 1 critic round | `hierarchy_diagnostics.csv` |
| Candidates passing gates | Average 0/9; Ward 9/9 | `hierarchy_diagnostics.csv` |
| Average 20/8: L1 dominant share | {a20.L1_dominant_share:.3f} ({int(a20.L1_max_tasks)} tasks) | `hierarchy_diagnostics.csv` |
| Ward 20/8: L1 dominant share | {w20.L1_dominant_share:.3f} ({int(w20.L1_max_tasks)} tasks) | `hierarchy_diagnostics.csv` |
| Average 20/8: L2 tiny / L1 tiny parents | {int(a20.L2_tiny_parents)} / {int(a20.L1_tiny_parents)} | `hierarchy_diagnostics.csv` |
| Ward 20/8: L2 tiny / L1 tiny parents | {int(w20.L2_tiny_parents)} / {int(w20.L1_tiny_parents)} | `hierarchy_diagnostics.csv` |
| O*NET task statements | {R['onet_tasks']:,} (O*NET 31.0) | `run_summary.json` |
| O*NET coverage | {O['coverage_meaningful_alignment']:.3f} | `onet_summary.json` |
| O*NET purity (mean / median) | {O['purity_mean']:.4f} / {O['purity_median']:.4f} | `onet_summary.json` |
| NN agreement (top-10 mean / exact top-1 mean) | {O['nn_agreement_top10_mean']:.4f} / {O['nn_agreement_top1_exact_mean']:.4f} | `onet_summary.json` |
| Top-1 cosine (mean / median) | {O['top1_mean']:.4f} / {O['top1_median']:.4f} | `onet_summary.json` |
| Noise threshold (p99) / Otsu top-1 | {O['threshold_noise_p99_background']:.4f} / {O['threshold_otsu_top1']:.4f} | `onet_summary.json` |
| Gemini O*NET judgments | STRONG {O['alignment_counts']['STRONG_MATCH']} / PARTIAL {O['alignment_counts']['PARTIAL_MATCH']} / NO_MATCH {O['alignment_counts']['NO_MATCH']} | `onet_evaluation.csv` |
| Stability: mean L1 / L2 persistence | {S['mean_level1_persistence']:.4f} / {S['mean_level2_persistence']:.4f} | `hierarchy_stability_summary.json` |
| Stability: pairwise ARI L1 / L2 | {S['level1_ari_pairwise_mean']:.4f} / {S['level2_ari_pairwise_mean']:.4f} | `hierarchy_stability_summary.json` |
| Stability: ARI vs final L1 / L2 | {S['level1_ari_vs_final_mean']:.4f} / {S['level2_ari_vs_final_mean']:.4f} | `hierarchy_stability_summary.json` |
| Leaves with persistence < 0.60 (L1 / L2) | {S['leaves_level1_persistence_below_0.60']} / {S['leaves_level2_persistence_below_0.60']} | `hierarchy_stability_summary.json` |
| Stability design | seeds 101/202/303/404/505; 80% per-leaf subsample ({S['tasks_sampled_per_trial'][0]:,} tasks/trial) | `hierarchy_stability_summary.json` |
| Coverage gap classes | WELL {G['counts']['WELL_COVERED']} / PARTIAL {G['counts']['PARTIALLY_COVERED']} / POSSIBLE_EMERGING {G['counts']['POSSIBLE_EMERGING_WORK']} | `onet_gap_summary.json` |
| Explainability cards | 80 | `topic_explainability_cards.csv` |
| Critic | {C['cases']} cases, {C['accepts']} accept, {C['moves']} move, 0 unresolved | `run_summary.json` |
| Gemini real calls / cap | {LG['gemini_real_calls']} / {LG['hard_cap']} (model {LG['model']}) | `api_usage.csv` |
| Gemini cache hits | {LG['cache_hits']} | `api_usage.csv` |
| Gemini tokens in / out | {LG['input_tokens']:,} / {LG['output_tokens']:,} | `api_usage.csv` |
| Sprint 4 Claude calls | 0 | `run_summary.json` |
| Human review | PENDING (all judgment columns blank) | `human_review_template.csv` |

## Definitions
- **Coverage**: share of the 80 leaf topics with Gemini STRONG/PARTIAL alignment AND centroid top-1 cosine above the data-derived noise threshold (p99 of all centroid-O*NET cosines) - here {O['coverage_meaningful_alignment']:.3f}.
- **Purity**: the share of a topic's member tasks whose own nearest O*NET task carries the topic's dominant O*NET (DWA) concept. Low values mean members map to many different O*NET activities; it is not an accuracy rate.
- **Nearest-neighbor agreement**: the share of the 5 representative tasks whose own top-1 O*NET task lies inside the centroid's top-10 (mean over topics); the exact version requires the same single top-1 task.
- **Hierarchy persistence**: for a leaf, the fraction of the 5 perturbation trials in which its Level-1 (or Level-2) parent was recovered (one-to-one Jaccard matching of child sets). Values are multiples of 0.2.
- **ARI (Adjusted Rand Index)**: agreement between two groupings corrected for chance; 1 = identical, 0 = chance level.
- **Cosine similarity**: angle-based similarity of two normalised Sentence-T5 vectors (1 = same direction).
- **Core tasks**: the 10 member tasks closest to the topic centroid (the most typical tasks).
- **Boundary tasks**: the 5 member tasks farthest from the centroid (least typical; used to test fit).

## Likely professor questions
**Why did you freeze the 80 leaves?** So every Sprint 4 change (labels, parents, O*NET evaluation) is attributable to one thing and Sprint 3 stays comparable. Sprint 4 never moves a task; all 3,015 memberships were re-verified identical.

**Why was O*NET not used to create the taxonomy?** O*NET is an external reference; building from it would make the taxonomy mirror O*NET instead of current job postings and make the evaluation circular. O*NET is used only to evaluate and to point out possible gaps.

**Why was Ward chosen over average linkage?** Both were evaluated with the same gates and score. No average-linkage candidate passed (20/8 puts {a20.L1_dominant_share*100:.0f}% of tasks in one domain); all 9 Ward candidates passed and Ward 20/8 had the highest score ({w20.score:.3f}). The choice followed the numbers, not manual preference.

**What role did Gemini play?** Semantic work only: naming/describing leaves, naming parents and flagging misplaced children, judging how well already-retrieved O*NET tasks match a topic, and one critic pass. Output was schema-constrained and locally validated.

**What did Gemini NOT do?** It did not assign tasks to topics, create clusters, choose the hierarchy cut, retrieve O*NET tasks, compute any metric, or write any file. Retrieval (exact cosine top-10), metrics, stability and gaps are deterministic Python.

**How do you know the hierarchy is stable?** A perturbation test (5 seeds, 80% subsampling inside each leaf, recomputed centroids, re-clustered): mean persistence {S['mean_level1_persistence']:.2f} (L1) and {S['mean_level2_persistence']:.2f} (L2); ARI vs final {S['level1_ari_vs_final_mean']:.2f} / {S['level2_ari_vs_final_mean']:.2f}. That is moderate: Level 2 is more stable than Level 1 and {S['leaves_level1_persistence_below_0.60']} leaves are unstable at Level 1. We do not claim the hierarchy is ground truth.

**What does POSSIBLE_EMERGING_WORK mean?** The topic's current job-posting evidence has limited alignment with O*NET by our thresholds ({G['counts']['POSSIBLE_EMERGING_WORK']} topics). It prompts human interpretation; it is not proof that O*NET is outdated.

**How was / will human review be performed?** It has not yet been performed. Teammates review the 80 topics in risk order (`human_review_priority.csv`), record five judgments in `human_review_template.csv`, and `apply_human_review.py` summarises the issues. Until then: HUMAN REVIEW PENDING.
"""
open('../Sprint4_Presentation_Key_Metrics.md', 'w', encoding='utf-8').write(metrics)

top = P.head(10)
guide = f"""# Sprint 4 Human Review Guide

**Status: HUMAN REVIEW PENDING.** No judgments have been entered. Do not edit leaf memberships.

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
{tab(['Priority', 'Topic', 'Why'], [[r.review_priority, f'{r.topic_id} {r.topic_name}', r.risk_reason] for r in top.itertuples()])}

## How risk was ranked (from saved evidence)
Points for: Gemini mixed_topic flag; not coherent; label confidence below 0.90; Level-1/Level-2 persistence below 0.6 (plus 1 - persistence); Gemini O*NET NO_MATCH/PARTIAL; POSSIBLE_EMERGING_WORK; O*NET purity in the lowest quartile; weak boundary tasks (lowest-quartile boundary similarity); critic action other than ACCEPT. The score only orders the work; it is not a quality judgment (`risk_score` is the last column of the priority file).

## What counts as an issue
Over-merged = one leaf, several concepts. Under-merged = one concept split over leaves. Boundary issue = boundary tasks belong elsewhere. Label issue = the name misleads.
"""
open('HUMAN_REVIEW_GUIDE.md', 'w', encoding='utf-8').write(guide)
print('docs ok')
