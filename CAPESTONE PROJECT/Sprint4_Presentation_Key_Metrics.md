# Sprint 4 Key Metrics (real values and sources)

| Metric | Value | Source file |
|---|---|---|
| Validated tasks | 3,015 | `task_hierarchy.csv` |
| Frozen leaf topics (Sentence-T5 + Leiden) | 80 | `leaf_topics_labeled.csv` |
| Level 1 work domains | 8 | `run_summary.json` |
| Level 2 activity families | 20 | `run_summary.json` |
| Hierarchy method | Ward linkage, 20 L2 / 8 L1, composite score 0.9026, + 1 critic round | `hierarchy_diagnostics.csv` |
| Candidates passing gates | Average 0/9; Ward 9/9 | `hierarchy_diagnostics.csv` |
| Average 20/8: L1 dominant share | 0.866 (2612 tasks) | `hierarchy_diagnostics.csv` |
| Ward 20/8: L1 dominant share | 0.213 (643 tasks) | `hierarchy_diagnostics.csv` |
| Average 20/8: L2 tiny / L1 tiny parents | 6 / 4 | `hierarchy_diagnostics.csv` |
| Ward 20/8: L2 tiny / L1 tiny parents | 2 / 0 | `hierarchy_diagnostics.csv` |
| O*NET task statements | 18,838 (O*NET 31.0) | `run_summary.json` |
| O*NET coverage | 0.975 | `onet_summary.json` |
| O*NET purity (mean / median) | 0.1935 / 0.1765 | `onet_summary.json` |
| NN agreement (top-10 mean / exact top-1 mean) | 0.6225 / 0.2425 | `onet_summary.json` |
| Top-1 cosine (mean / median) | 0.9224 / 0.9238 | `onet_summary.json` |
| Noise threshold (p99) / Otsu top-1 | 0.8847 / 0.9168 | `onet_summary.json` |
| Gemini O*NET judgments | STRONG 58 / PARTIAL 20 / NO_MATCH 2 | `onet_evaluation.csv` |
| Stability: mean L1 / L2 persistence | 0.7150 / 0.7925 | `hierarchy_stability_summary.json` |
| Stability: pairwise ARI L1 / L2 | 0.4843 / 0.6501 | `hierarchy_stability_summary.json` |
| Stability: ARI vs final L1 / L2 | 0.5504 / 0.6469 | `hierarchy_stability_summary.json` |
| Leaves with persistence < 0.60 (L1 / L2) | 24 / 16 | `hierarchy_stability_summary.json` |
| Stability design | seeds 101/202/303/404/505; 80% per-leaf subsample (2,414 tasks/trial) | `hierarchy_stability_summary.json` |
| Coverage gap classes | WELL 33 / PARTIAL 44 / POSSIBLE_EMERGING 3 | `onet_gap_summary.json` |
| Explainability cards | 80 | `topic_explainability_cards.csv` |
| Critic | 31 cases, 30 accept, 1 move, 0 unresolved | `run_summary.json` |
| Gemini real calls / cap | 10 / 12 (model gemini-3.1-flash-lite) | `api_usage.csv` |
| Gemini cache hits | 5 | `api_usage.csv` |
| Gemini tokens in / out | 36,351 / 26,236 | `api_usage.csv` |
| Sprint 4 Claude calls | 0 | `run_summary.json` |
| Human review | PENDING (all judgment columns blank) | `human_review_template.csv` |

## Definitions
- **Coverage**: share of the 80 leaf topics with Gemini STRONG/PARTIAL alignment AND centroid top-1 cosine above the data-derived noise threshold (p99 of all centroid-O*NET cosines) - here 0.975.
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

**Why was Ward chosen over average linkage?** Both were evaluated with the same gates and score. No average-linkage candidate passed (20/8 puts 87% of tasks in one domain); all 9 Ward candidates passed and Ward 20/8 had the highest score (0.903). The choice followed the numbers, not manual preference.

**What role did Gemini play?** Semantic work only: naming/describing leaves, naming parents and flagging misplaced children, judging how well already-retrieved O*NET tasks match a topic, and one critic pass. Output was schema-constrained and locally validated.

**What did Gemini NOT do?** It did not assign tasks to topics, create clusters, choose the hierarchy cut, retrieve O*NET tasks, compute any metric, or write any file. Retrieval (exact cosine top-10), metrics, stability and gaps are deterministic Python.

**How do you know the hierarchy is stable?** A perturbation test (5 seeds, 80% subsampling inside each leaf, recomputed centroids, re-clustered): mean persistence 0.71 (L1) and 0.79 (L2); ARI vs final 0.55 / 0.65. That is moderate: Level 2 is more stable than Level 1 and 24 leaves are unstable at Level 1. We do not claim the hierarchy is ground truth.

**What does POSSIBLE_EMERGING_WORK mean?** The topic's current job-posting evidence has limited alignment with O*NET by our thresholds (3 topics). It prompts human interpretation; it is not proof that O*NET is outdated.

**How was / will human review be performed?** It has not yet been performed. Teammates review the 80 topics in risk order (`human_review_priority.csv`), record five judgments in `human_review_template.csv`, and `apply_human_review.py` summarises the issues. Until then: HUMAN REVIEW PENDING.
