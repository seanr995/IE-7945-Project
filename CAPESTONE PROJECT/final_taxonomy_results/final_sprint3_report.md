# Final Sprint 3 Report - agentic taxonomy induction + matched-granularity benchmark

All numbers are generated from the result tables; every superlative below was re-verified programmatically (`audit/report_consistency_check.json`: all_consistent=True). No metric here is an accuracy: no topic-level gold labels exist. O*NET was not used (Sprint 4).

> **Known limitation.** The Sprint 3 input was section-filtered following a manually labelled task-purity audit. Post-filter audited task purity was 92.9%. Residual non-task contamination may therefore remain and is treated as a known limitation of the clustering experiment.

## Decision summary
- **Outcome: LEIDEN SELECTED** - `LEIDEN_T5@80` (Leiden k_nn=10 res=5.537 (target 80))
- Recommended topic count: **80** (coverage of work statements 100.0%)
- Freeze status: **NOT READY TO FREEZE** - blockers: human review sample not yet labelled (criterion 1: human semantic coherence)
- Agentic Claude calls: 19 of hard limit 20; benchmark: 0

## Candidate comparison (final_method_comparison.csv)
| label | n_clusters | coverage_work | stability_ari | persistence_share_j50 | paraphrase_kappa | centroid_inconsistency_rate | granularity_score | low_confidence_rate | provider_nn_gap_abs | semantic_cohesion_lift_incl_unassigned | silhouette | complexity_rank | gate_failures |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Agentic taxonomy (Claude concepts + decisions) | 80 | 0.991 | 0.679 | 0.975 | 0.614 | 0.241 | 1.000 | 0.408 | 0.029 | 0.184 | -0.041 | 4 | tiny_cluster_share=0.412 > 0.25 |
| Hybrid: agentic concepts + deterministic T5 assignment | 80 | 1.000 | 0.676 | 0.975 | 0.612 | 0.236 | 1.000 | 0.410 | 0.026 | 0.184 | -0.042 | 3 | tiny_cluster_share=0.412 > 0.25 |
| K-Means (T5) k=64 | 64 | 1.000 | 0.400 | 0.328 | 0.677 | 0.024 | 1.000 | 0.137 | 0.007 | 0.257 | 0.068 | 1 | stability_ari=0.400 < 0.5;persistence_share_j50=0.328 < 0.5 |
| K-Means (E5) k=64 | 64 | 1.000 | 0.456 | 0.500 | 0.569 | 0.020 | 1.000 | 0.121 | 0.045 | 0.250 | 0.069 | 1 | stability_ari=0.456 < 0.5 |
| GMM (diag) n=80 | 80 | 1.000 | 0.396 | 0.388 | 0.728 | 0.072 | 1.000 | 0.174 | 0.024 | 0.268 | 0.058 | 1 | stability_ari=0.396 < 0.5;persistence_share_j50=0.388 < 0.5 |
| Leiden k_nn=10 res=5.537 (target 80) | 80 | 1.000 | 0.567 | 0.700 | 0.800 | 0.116 | 1.000 | 0.210 | 0.067 | 0.256 | 0.046 | 2 |  |
| HDBSCAN mcs=10 ms=1 eom (target 64) | 3 | 0.787 | 0.149 | 0.000 | -8.962 | 0.013 | 0.000 | 0.021 | 0.004 | 0.005 | -0.024 | 2 | coverage_work=0.787 < 0.8;stability_ari=0.149 < 0.5;persistence_share_j50=0.000 < 0.5;dominant_share=0.990 > 0.2 |
| BERTopic natural (70 topics) | 70 | 0.729 | 0.341 | 0.586 | 0.653 | 0.068 | 1.000 | 0.130 | 0.171 | 0.253 | 0.062 | 3 | coverage_work=0.729 < 0.8;stability_ari=0.341 < 0.5 |
| Historical winner: K-Means (T5) k=100 | 100 | 1.000 | 0.385 | 0.250 | 0.684 | 0.016 | 1.000 | 0.100 | 0.003 | 0.290 | 0.069 | 1 | stability_ari=0.385 < 0.5;persistence_share_j50=0.250 < 0.5 |
| Historical: GMM (diag) n=140 | 140 | 1.000 | 0.357 | 0.229 | 0.706 | 0.038 | 1.000 | 0.110 | 0.020 | 0.315 | 0.064 | 1 | stability_ari=0.357 < 0.5;persistence_share_j50=0.229 < 0.5 |
| Historical: Leiden k_nn=20 res=2.0 | 31 | 1.000 | 0.562 | 0.613 | 0.782 | 0.146 | 0.368 | 0.286 | 0.003 | 0.180 | 0.045 | 2 |  |

Decision trace (gates, then lexicographic hierarchy with tolerances):
- 1 human semantic coherence [human_preference_share]: not available for any candidate -> PENDING / no effect
- 2 coverage [coverage_work]: best=1.0000 tol=0.05 -> kept LEIDEN_T5@80, HIST_LEIDEN_T5_k20_r2.0
- 3 stability [stability_ari]: best=0.5675 tol=0.05 -> kept LEIDEN_T5@80, HIST_LEIDEN_T5_k20_r2.0
- 4 topic persistence [persistence_share_j50]: best=0.7000 tol=0.05 -> kept LEIDEN_T5@80

## Answers

**1. Why was k=100 not considered ground truth?** (1) every one of the 26 Sprint 3 configurations carried a pathology flag; the winner was chosen 'among the least-flagged' (all K-Means configurations are flagged unstable: subsample ARI 0.385-0.466 < 0.50). (2) intfloat/e5-base-v2: cohesion lift rises monotonically with k (k=25->100: 0.174->0.280) while subsample stability falls (0.428->0.404); cohesion-first selection therefore drifts to the largest k tested. (3) sentence-transformers/sentence-t5-base: cohesion lift rises monotonically with k (k=25->100: 0.185->0.290) while subsample stability falls (0.454->0.385); cohesion-first selection therefore drifts to the largest k tested. (4) the winner k=100 lies on the BOUNDARY of the tested grid [25, 40, 60, 80, 100]: k=100 is an artefact of the grid edge, not evidence of 100 natural task concepts. K-Means needs k before fitting, so k=100 was an input, not a finding.
**2. How many topics did the agentic system discover naturally?** N_AGENTIC = **80** active topics with members (registry: 105 IDs created; status counts {'active': 80, 'merged': 12, 'split': 10, 'rejected': 3}). No topic count was given to Claude. Discovery calls flagged saturated: 3 of 3 - the per-call output limit may have bound discovery; N may be an underestimate.
**3. Why did it converge at that number?** refinement round limit (2) reached; unmet: topic-count change <= 1. Convergence trace:

| round | topics_before | topics_after | assignment_change_rate | assignment_noise_floor | merges | splits | new_topics | duplicate_flags |
|---|---|---|---|---|---|---|---|---|
| 1 | 72 | 76 | 0.484 | 0.172 | 2 | 6 | 12 | 0 |
| 2 | 76 | 80 | 0.117 | 0.165 | 0 | 4 | 8 | 0 |

**4. N_LOW / N_AGENTIC / N_HIGH:** 64 / 80 / 96 (delta = max(2, round(0.2 * N)) = 16 (+-20%)). This is a reference granularity for a controlled comparison, not ground truth.
**5. K-Means around that range:**

| target_count | stability_ari | persistence_share_j50 | paraphrase_kappa | semantic_cohesion_lift_incl_unassigned | low_confidence_rate | silhouette |
|---|---|---|---|---|---|---|
| 64.000 | 0.400 | 0.328 | 0.677 | 0.257 | 0.137 | 0.068 |
| 80.000 | 0.391 | 0.312 | 0.622 | 0.275 | 0.112 | 0.069 |
| 96.000 | 0.389 | 0.219 | 0.661 | 0.289 | 0.109 | 0.068 |

Best K-Means count by the hierarchy: **k=64** (decided by 5 paraphrase consistency) - algorithmic quality is better at a nearby count than at N_AGENTIC.
**6. Did E5/T5 conclusions change after controlling granularity?** no material difference (within tolerances) (votes {'t5': 2, 'e5': 3, 'tie': 7}). k=64: T5 stability 0.400 vs E5 0.456; persistence(J>=.5) 32.8% vs 50.0%; paraphrase kappa 0.677 vs 0.569; cohesion lift 0.257 vs 0.250 | k=80: T5 stability 0.391 vs E5 0.420; persistence(J>=.5) 31.2% vs 36.2%; paraphrase kappa 0.622 vs 0.575; cohesion lift 0.275 vs 0.267 | k=96: T5 stability 0.389 vs E5 0.423; persistence(J>=.5) 21.9% vs 33.3%; paraphrase kappa 0.661 vs 0.579; cohesion lift 0.289 vs 0.279 Historically T5 won only on the 4th-level robustness tie-break.
**7. GMM:**

| partition_id | actual_count | empty_components | bic | aic | stability_ari | persistence_share_j50 | paraphrase_kappa | posterior_entropy_norm | low_confidence_rate | semantic_cohesion_lift_incl_unassigned |
|---|---|---|---|---|---|---|---|---|---|---|
| GMM_T5@64 | 64.000 | 0.000 | -474300.294 | -513151.682 | 0.414 | 0.328 | 0.734 | 0.012 | 0.184 | 0.253 |
| GMM_T5@80 | 80.000 | 0.000 | -466767.701 | -515333.439 | 0.396 | 0.388 | 0.728 | 0.007 | 0.174 | 0.268 |
| GMM_T5@96 | 96.000 | 0.000 | -458614.169 | -516894.256 | 0.386 | 0.312 | 0.719 | 0.005 | 0.144 | 0.285 |
| HIST_GMM_T5_140 | 140.000 | n/a | n/a | n/a | 0.357 | 0.229 | 0.706 | n/a | 0.110 | 0.315 |

**8. Could Leiden naturally reproduce similar granularity?**

| target_count | actual_count | abs_difference | resolution | k_neighbors | close_match | modularity |
|---|---|---|---|---|---|---|
| 64.000 | 64.000 | 0.000 | 4.938 | 20.000 | True | 0.459 |
| 80.000 | 80.000 | 0.000 | 5.537 | 10.000 | True | 0.556 |
| 96.000 | 96.000 | 0.000 | 7.517 | 10.000 | True | 0.544 |

**9. Leiden stability at comparable granularity:** target 64: Leiden 0.599 vs K-Means 0.400; target 80: Leiden 0.567 vs K-Means 0.391; target 96: Leiden 0.566 vs K-Means 0.389.
**10. BERTopic natural vs reduced:**

| label | n_clusters | noise_share | coverage_work | stability_ari | persistence_share_j50 | paraphrase_kappa | semantic_cohesion_lift_incl_unassigned |
|---|---|---|---|---|---|---|---|
| BERTopic natural (70 topics) | 70 | 0.272 | 0.729 | 0.341 | 0.586 | 0.653 | 0.253 |

Reduction: {"variant": "POST_HOC_REDUCED_TO_AGENTIC_GRANULARITY", "applicable": false, "natural_topics": 70, "target_count": 80, "reason": "natural topic count is not above the target - reduction can only merge topics"}. The reduced variant is a post-hoc reduction, not BERTopic's natural discovery.
**11. Could HDBSCAN obtain comparable granularity without excessive noise?** target 64: nearest feasible 3 clusters at 21.3% noise (NOT close); target 80: nearest feasible 3 clusters at 21.3% noise (NOT close); target 96: nearest feasible 3 clusters at 21.3% noise (NOT close) (configurations above 50% noise rejected).
**12. Strongest global stability:** Agentic taxonomy (Claude concepts + decisions) (stability_ari 0.679).
**13. Strongest topic-level persistence:** Agentic taxonomy (Claude concepts + decisions) = Hybrid: agentic concepts + deterministic T5 assignment (persistence_share_j50 0.975); median best-Jaccard: Agentic taxonomy (Claude concepts + decisions) (persistence_median_jaccard 0.778).
**14. Best paraphrase preservation:** Leiden k_nn=10 res=5.537 (target 80) (paraphrase_kappa 0.800) (chance-corrected kappa).
**15. Clearest human-interpretable concepts:** cannot be determined numerically - the human review sample (`evaluation/human_review_sample.csv`) is not yet labelled. Structurally, only the agentic/hybrid topics carry explicit names, definitions and inclusion/exclusion rules; algorithmic topics have keywords only.
**16. Where did methods agree?** Highest pairwise agreement: AGENTIC_FULL vs HYBRID_PROTOTYPE ARI 0.998 / matched Jaccard 0.993; LEIDEN_T5@80 vs BERTOPIC_NATURAL ARI 0.685 / matched Jaccard 0.498; GMM_T5@80 vs BERTOPIC_NATURAL ARI 0.461 / matched Jaccard 0.364. 8.0% of statements are cross-method CORE members; topics with the largest core share: Law Enforcement Operations (100.0%), Database Optimization (100.0%), Data Retrieval and Reporting (100.0%), Product Release Gatekeeping (100.0%), Code Review (100.0%).
**17. Where did methods disagree?** Lowest agreement: LEIDEN_T5@80 vs HDBSCAN_T5@80 ARI 0.000; KMEANS_T5@80 vs HDBSCAN_T5@80 ARI 0.000; GMM_T5@80 vs HDBSCAN_T5@80 ARI 0.000. Topics with the most method disagreement: Climate Impact Assessment and Reduction (100.0%), Records, Budget and Compliance Reporting (75.0%), Growth Initiative and Scale-up Driving (55.9%), Software Architecture and Technical Decision-Making (44.6%), Investment Assumption Challenge (44.1%).
**18. Did agentic reasoning improve topic boundaries?** Versus K-Means at the same count: centroid inconsistency 0.241 vs 0.020, low-confidence rate 0.408 vs 0.112, paraphrase kappa 0.614 vs 0.622. Semantic boundary quality itself needs the human review.
**19. Did agentic reasoning introduce instability?** Agentic stability 0.679 (Claude decisions frozen), hybrid 0.676, K-Means at k=80 0.391. Run-to-run variance of Claude discovery itself was NOT measured (one budgeted run); repeated runs would be needed.
**20. Is a hybrid system better?** Hybrid vs agentic: stability_ari 0.676 vs 0.679, persistence_share_j50 0.975 vs 0.975, paraphrase_kappa 0.612 vs 0.614, semantic_cohesion_lift_incl_unassigned 0.184 vs 0.184, low_confidence_rate 0.410 vs 0.408, coverage_work 1.000 vs 0.991. In the evidence hierarchy the hybrid ranks behind LEIDEN_T5@80.
**21. Recommended final taxonomy method:** LEIDEN SELECTED - Leiden k_nn=10 res=5.537 (target 80).
**22. Recommended final topic count:** 80.
**23. Why?** Decided at: 4 topic persistence. Candidates passing all gates: ['LEIDEN_T5@80', 'HIST_LEIDEN_T5_k20_r2.0']. Gate failures: AGENTIC_FULL: tiny_cluster_share=0.412 > 0.25; HYBRID_PROTOTYPE: tiny_cluster_share=0.412 > 0.25; KMEANS_T5@64: stability_ari=0.400 < 0.5;persistence_share_j50=0.328 < 0.5; KMEANS_E5@64: stability_ari=0.456 < 0.5; GMM_T5@80: stability_ari=0.396 < 0.5;persistence_share_j50=0.388 < 0.5; HDBSCAN_T5@64: coverage_work=0.787 < 0.8;stability_ari=0.149 < 0.5;persistence_share_j50=0.000 < 0.5;dominant_share=0.990 > 0.2; BERTOPIC_NATURAL: coverage_work=0.729 < 0.8;stability_ari=0.341 < 0.5; HIST_KMEANS_T5_k100: stability_ari=0.385 < 0.5;persistence_share_j50=0.250 < 0.5; HIST_GMM_T5_140: stability_ari=0.357 < 0.5;persistence_share_j50=0.229 < 0.5.
**24. Limitations:** (a) The Sprint 3 input was section-filtered following a manually labelled task-purity audit. Post-filter audited task purity was 92.9%. Residual non-task contamination may therefore remain and is treated as a known limitation of the clustering experiment. (b) No topic-level gold labels; nothing here is accuracy. (c) Agentic discovery ran once under a 20-call budget; its run-to-run variance is unmeasured and its stability figure freezes Claude decisions. (d) 3 discovery call(s) hit the output ceiling. (e) Ambiguous queue: 3006 statements, 166 reviewed by Claude, 32 propagated, 2808 left as low-confidence automatic assignments. (f) Paraphrase pairs come from Sprint 2 cross-model agreement and favour easier paraphrases. (g) Human review criterion is PENDING.
**25. Is Sprint 3 ready to freeze before Sprint 4?** **NO** - human review sample not yet labelled (criterion 1: human semantic coherence). Label `evaluation/human_review_sample.csv`, then re-run the benchmark with `--human-review` (no Claude calls; checkpoints make it fast).

## Files
- `evaluation/final_method_comparison.csv`, `matched_granularity_comparison.csv`, `cross_method_agreement.csv`, `topic_persistence.csv`, `paraphrase_consistency.csv`, `provider_robustness.csv`, `human_review_sample.csv`, `api_usage.csv`
- `consensus/` co-association core / boundary members
- `audit/existing_results_audit.md` historical audit; `audit/report_consistency_check.json`
- figures: 01_matched_granularity_curves.png, 02_leiden_resolution_search.png, 03_agentic_convergence.png, 04_cross_method_ari.png, 05_coassociation_distribution.png, 06_final_candidates.png