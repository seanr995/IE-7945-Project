# Sprint 3 Advanced Challenger Report - does complexity earn its place?

Generated 2026-10-22 21:52 UTC by `sprint3_advanced_challengers.py`. All numbers are computed by the script; none are hand-entered. Representation frozen: `sentence-transformers/sentence-t5-base` (embeddings reused, not regenerated).

> **Known limitation.** The Sprint 3 input was section-filtered following a manually labelled task-purity audit. Post-filter audited task purity was 92.9%. Residual non-task contamination may therefore remain and is treated as a known limitation of the clustering experiment.

## Setup
- Baseline: `EXP_3_T5_KMEANS__k=100` (Sprint 3 overall winner (run_manifest.json chosen_configuration)), refit with identical parameters; reproduction check (`baseline/baseline_reproduction_check.json`): n_clusters diff 0.000000, semantic_cohesion diff 0.000000, semantic_cohesion_lift diff 0.000000, stability_ari diff 0.000000, silhouette diff 0.000000, provider_pair_kappa diff 0.000000, low_confidence_rate diff 0.000000; labels ARI vs recorded Sprint 3 membership 1.000
- Validation: {"input_sha256_matches_manifest_and_frozen": true, "input_rows": 3015, "embedding_shape": [3015, 768], "mapping_aligned": true, "embedding_meta_matches": true, "embeddings_normalised": true, "paired_wordings": 432, "pca_matches_sprint3": true}
- One evaluator for all methods (imported from the Sprint 3 script): semantic cohesion in the ORIGINAL T5 space with a size-preserving random-partition baseline (lift); silhouette in the shared Sprint 3 PCA space; the same 5 x 80% subsamples; the same provider nearest-neighbour test.
- Outlier-producing methods are compared on **coverage-adjusted** cohesion lift (outliers attached to their nearest centroid for the metric only) so that discarding hard statements is not rewarded; assigned-only values are also shown.
- Common low-confidence definition: primary-vs-secondary centroid margin below 0.0066 (P10 of the baseline's margins); each method's native confidence is reported separately.
- Replacement rule: a challenger replaces K-Means only with >= 1 gain beyond tolerance {"semantic_cohesion_lift_incl_noise": [0.02, "higher"], "stability_ari": [0.05, "higher"], "paraphrase_kappa": [0.05, "higher"], "provider_nn_gap_abs": [0.05, "lower"], "low_confidence_rate_common": [0.05, "lower"], "granularity_score": [0.25, "higher"]} and no cost beyond {"semantic_cohesion_lift_incl_noise": 0.02, "stability_ari": 0.05, "paraphrase_kappa": 0.05, "coverage": 0.1} or pathology flag.

## Comparison (best configuration per method)
| pipeline | parameters | n_clusters | coverage | semantic_cohesion_lift_incl_noise | kmeans_same_k_cohesion_lift | stability_ari | stability_ari_sd | paraphrase_consistency | paraphrase_kappa | provider_nn_gap_abs | silhouette | granularity_score | low_confidence_rate_common | secondary_membership_rate | pathology_flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Sentence-T5 + K-Means | k=100 | 100 | 1.000 | 0.290 | 0.290 | 0.385 | 0.017 | 0.688 | 0.684 | 0.003 | 0.069 | 1.000 | 0.100 | 0.051 | unstable |
| Sentence-T5 + BERTopic | n_neighbors=30,n_components=5,min_dist=0.0,metric=cosine,min_cluster_size=10,min_samples=5 | 70 | 0.728 | 0.253 | 0.264 | 0.341 | 0.047 | 0.660 | 0.653 | 0.171 | 0.062 | 1.000 | 0.130 | 0.029 | unstable |
| Sentence-T5 + Leiden | k_neighbors=20,resolution=2.0 | 31 | 1.000 | 0.180 | 0.201 | 0.562 | 0.028 | 0.789 | 0.782 | 0.003 | 0.045 | 0.368 | 0.286 | 0.117 |  |
| Sentence-T5 + GMM | n_components=140,covariance_type=diag | 140 | 1.000 | 0.315 | 0.316 | 0.357 | 0.016 | 0.708 | 0.706 | 0.020 | 0.064 | 1.000 | 0.110 | 0.008 | unstable |

`kmeans_same_k_cohesion_lift` = K-Means refit at the challenger's topic count: it shows whether a cohesion difference is method quality or just granularity.

## Bootstrap 95% CIs (posting level, 100 resamples)
| method | metric | point_estimate | ci95_low | ci95_high |
|---|---|---|---|---|
| kmeans | semantic_cohesion | 0.914 | 0.913 | 0.915 |
| kmeans | semantic_cohesion_incl_noise | 0.914 | 0.913 | 0.915 |
| kmeans | silhouette | 0.069 | 0.066 | 0.071 |
| kmeans | paraphrase_consistency | 0.688 | 0.648 | 0.730 |
| bertopic | semantic_cohesion | 0.915 | 0.913 | 0.916 |
| bertopic | semantic_cohesion_incl_noise | 0.909 | 0.908 | 0.910 |
| bertopic | silhouette | 0.062 | 0.058 | 0.066 |
| bertopic | paraphrase_consistency | 0.660 | 0.613 | 0.715 |
| leiden | semantic_cohesion | 0.898 | 0.897 | 0.900 |
| leiden | semantic_cohesion_incl_noise | 0.898 | 0.897 | 0.900 |
| leiden | silhouette | 0.045 | 0.042 | 0.048 |
| leiden | paraphrase_consistency | 0.789 | 0.752 | 0.828 |
| gmm | semantic_cohesion | 0.918 | 0.917 | 0.920 |
| gmm | semantic_cohesion_incl_noise | 0.918 | 0.917 | 0.920 |
| gmm | silhouette | 0.064 | 0.061 | 0.067 |
| gmm | paraphrase_consistency | 0.708 | 0.662 | 0.742 |

## Answers
1. **Did BERTopic beat K-Means?** **No** - no gain beyond tolerance; costs: semantic_cohesion_lift_incl_noise 0.290 -> 0.253; coverage 1.000 -> 0.728; pathology: unstable.
2. **Did Leiden beat K-Means?** **No** - gains (stability_ari 0.385 -> 0.562; paraphrase_kappa 0.684 -> 0.782) come with unacceptable costs (semantic_cohesion_lift_incl_noise 0.290 -> 0.180).
3. **Did GMM beat K-Means?** **No** - gains (semantic_cohesion_lift_incl_noise 0.290 -> 0.315) come with unacceptable costs (pathology: unstable).
4. **Highest semantic cohesion:** Sentence-T5 + Leiden (coverage-adjusted lift 0.180; assigned-only highest: Sentence-T5 + Leiden). Compare with `kmeans_same_k_cohesion_lift` before attributing it to the method.
5. **Most stable:** Sentence-T5 + Leiden (subsample ARI 0.562).
6. **Best at preserving Gemini/Groq paraphrases:** Sentence-T5 + Leiden (chance-corrected kappa 0.782; raw rates favour coarse partitions, so kappa is the fair comparison).
7. **Handles ambiguous tasks best:** Sentence-T5 + Leiden (lowest common low-confidence rate 28.6%); secondary-membership rates: Sentence-T5 + K-Means 5.1%, Sentence-T5 + BERTopic 2.9%, Sentence-T5 + Leiden 11.7%, Sentence-T5 + GMM 0.8%.
8. **Most useful topic granularity:** Sentence-T5 + Leiden (granularity score 0.368, 31 topics, median size 93.000).
9. **Adds complexity without enough value:** Sentence-T5 + BERTopic (4 tuned hyper-parameters, extra deps: bertopic, umap-learn, hdbscan), Sentence-T5 + Leiden (2 tuned hyper-parameters, extra deps: python-igraph, leidenalg), Sentence-T5 + GMM (1 tuned hyper-parameters, extra deps: none (scikit-learn)).

## 10. FINAL RECOMMENDATION

**KEEP T5 + KMEANS**

Key reason: no challenger delivered a gain beyond the pre-declared tolerances without an unacceptable cost, so the simpler baseline is kept (complexity did not earn its place).

## Multi-topic membership rules (derived from each method's observed distributions)
- **Sentence-T5 + K-Means**: Sprint 3 centroid rule (secondary similarity >= P25 own-centroid similarity AND gap <= P10 gap) - thresholds {"secondary_min_similarity": 0.8988989293575287, "secondary_max_gap": 0.007227468490600586, "noise_post_hoc_min_similarity": 0.9141106605529785}
- **Sentence-T5 + BERTopic**: Sprint 3 centroid rule AND HDBSCAN soft membership of the secondary topic >= P75 of non-primary soft memberships; outliers assigned post hoc only by the Sprint 3 outlier rule - thresholds {"soft_membership_P75_non_primary": 0.007429887991235363, "secondary_min_similarity": 0.8974924236536026, "secondary_max_gap": 0.009571027755737305, "noise_post_hoc_min_similarity": 0.9151618480682373}
- **Sentence-T5 + Leiden**: Sprint 3 centroid rule AND edge-weight share to the secondary community >= P75 of positive non-own community shares - thresholds {"edge_share_P75": 0.08837477476797292, "secondary_min_similarity": 0.8827358782291412, "secondary_max_gap": 0.004312396049499512, "noise_post_hoc_min_similarity": 0.8988345265388489}
- **Sentence-T5 + GMM**: secondary = 2nd-highest posterior iff p2 >= max(P90 of p2, numerical floor) AND p1 - p2 <= P10 of (p1 - p2) - thresholds {"p2_P90": 0.001939270455879969, "numerical_floor": 0.05, "p2_threshold_used": 0.05, "gap_P10": 0.9957677045009885, "share_p2_below_floor": 0.9917081260364843}

## Per-method selection traces
**Sentence-T5 + BERTopic** -> `ADV_1_T5_BERTOPIC__nn=30_nc=5_mcs=10_ms=5`
- no non-pathological configuration - choosing among the least-flagged
- semantic_cohesion_lift_incl_noise: best=0.265 tol=0.02 -> kept ADV_1_T5_BERTOPIC__nn=15_nc=5_mcs=10_ms=5, ADV_1_T5_BERTOPIC__nn=15_nc=5_mcs=10_ms=10, ADV_1_T5_BERTOPIC__nn=15_nc=10_mcs=10_ms=5, ADV_1_T5_BERTOPIC__nn=15_nc=10_mcs=10_ms=10, ADV_1_T5_BERTOPIC__nn=30_nc=5_mcs=10_ms=5, ADV_1_T5_BERTOPIC__nn=30_nc=10_mcs=10_ms=5
- stability_ari: best=0.341 tol=0.05 -> kept ADV_1_T5_BERTOPIC__nn=30_nc=5_mcs=10_ms=5, ADV_1_T5_BERTOPIC__nn=30_nc=10_mcs=10_ms=5
- granularity_score: best=1.000 tol=0.0 -> kept ADV_1_T5_BERTOPIC__nn=30_nc=5_mcs=10_ms=5, ADV_1_T5_BERTOPIC__nn=30_nc=10_mcs=10_ms=5
- robustness: best=0.653 tol=0.05 -> kept ADV_1_T5_BERTOPIC__nn=30_nc=5_mcs=10_ms=5
- decided by robustness

**Sentence-T5 + Leiden** -> `ADV_2_T5_LEIDEN__k_neighbors=20_resolution=2.0`
- semantic_cohesion_lift_incl_noise: best=0.180 tol=0.02 -> kept ADV_2_T5_LEIDEN__k_neighbors=20_resolution=2.0, ADV_2_T5_LEIDEN__k_neighbors=30_resolution=2.0
- stability_ari: best=0.564 tol=0.05 -> kept ADV_2_T5_LEIDEN__k_neighbors=20_resolution=2.0, ADV_2_T5_LEIDEN__k_neighbors=30_resolution=2.0
- granularity_score: best=0.368 tol=0.0 -> kept ADV_2_T5_LEIDEN__k_neighbors=20_resolution=2.0
- decided by granularity_score

**Sentence-T5 + GMM** -> `ADV_3_T5_GMM__n_components=140_cov=diag`
- no non-pathological configuration - choosing among the least-flagged
- semantic_cohesion_lift_incl_noise: best=0.315 tol=0.02 -> kept ADV_3_T5_GMM__n_components=120_cov=diag, ADV_3_T5_GMM__n_components=140_cov=diag
- stability_ari: best=0.360 tol=0.05 -> kept ADV_3_T5_GMM__n_components=120_cov=diag, ADV_3_T5_GMM__n_components=140_cov=diag
- granularity_score: best=1.000 tol=0.0 -> kept ADV_3_T5_GMM__n_components=120_cov=diag, ADV_3_T5_GMM__n_components=140_cov=diag
- robustness: best=0.706 tol=0.05 -> kept ADV_3_T5_GMM__n_components=120_cov=diag, ADV_3_T5_GMM__n_components=140_cov=diag
- silhouette: best=0.065 tol=0.01 -> kept ADV_3_T5_GMM__n_components=120_cov=diag, ADV_3_T5_GMM__n_components=140_cov=diag
- decided by tie_break

## Limitations
- The Sprint 3 input was section-filtered following a manually labelled task-purity audit. Post-filter audited task purity was 92.9%. Residual non-task contamination may therefore remain and is treated as a known limitation of the clustering experiment.
- No ground-truth topics: all metrics are internal or robustness measures, not accuracy.
- BERTopic keywords (`bertopic/bertopic_keywords_all_configs.csv`) are for inspection only, not taxonomy names.
- GMM BIC/AIC are supporting evidence only. Bootstrap CIs are conditional on the fitted partition.
- Paraphrase pairs come from Sprint 2 cross-model agreement and skew towards easier paraphrases.

Runtime: {"gmm_seconds": 84.5, "leiden_seconds": 167.8, "bertopic_seconds": 575.8, "total_seconds": 924.1}