# Sprint 3 - Representation & Clustering: Stability Report

Generated 2026-10-22 21:13 UTC by `sprint3_colab_experiment.py`. All numbers below are computed by the script from the frozen input; nothing is hand-entered.

> **Known limitation.** The Sprint 3 input was section-filtered following a manually labelled task-purity audit. Post-filter audited task purity was 92.9%. Residual non-task contamination may therefore remain and is treated as a known limitation of the clustering experiment.

## 1. Objective
Choose a semantic representation for extracted work-task statements and a clustering method that yields stable, semantically cohesive leaf topics, via a controlled 2x2 experiment (E5 / Sentence-T5 x K-Means / HDBSCAN). No labels exist for clusters: every metric below is an internal or robustness measure, **not accuracy**.

## 2. Frozen input
- file `sprint3_task_input.csv`, SHA-256 `010795fed2864f9963f159204a857f2198a2246206009ab0dbbbb786799f4551` (matches the frozen version)
- 3,015 statements (after removing exact normalised duplicates in export), 188 postings, providers {'gemini': 2719, 'groq': 296}, sections {'responsibilities': 2289, 'general_description': 726}
- 432 Gemini statements carry a non-identical Groq wording of the same task (Sprint 2 cross-model agreement); these are used only for robustness/representation tests, never clustered.

## 3. Representation methodology
- **E5**: `intfloat/e5-base-v2`, input `"passage: <statement>"` for every statement (no query/passage mixing), attention-masked mean pooling, L2 normalisation.
- **Sentence-T5**: `sentence-transformers/sentence-t5-base`, raw statement text, L2 normalisation.
- Original normalised embeddings are kept for cosine similarity, nearest neighbours, centroid cohesion and memberships.
- Clustering space: identical controlled PCA for both models (max 50 components, full SVD, no whitening): E5 50 components, cumulative explained variance 0.580; T5 50 components, cumulative explained variance 0.607. UMAP is used only for one figure.

## 4. Clustering methodology
- K-Means: k in [25, 40, 60, 80, 100], k-means++, n_init=10, random_state=42.
- HDBSCAN (hdbscan package): min_cluster_size in [10, 20, 30, 40] x min_samples in [5, 10], Euclidean on PCA space, EOM selection; noise is never forced.
- Stability: 5 deterministic 80% subsamples, identical for every configuration; pairwise ARI on shared statements (noise counted as its own label; ARI on rows assigned in both runs also reported). K-Means additionally: 5 seed restarts on the full data.
- Semantic cohesion: mean cosine of members to their cluster centroid in the original embedding space; chance baseline from 5 size-preserving random partitions; **cohesion lift** = (cohesion - chance)/(1 - chance).

## 5. 2x2 experimental comparison (one candidate per cell)
| experiment_id | parameters | n_clusters | noise_share | semantic_cohesion | semantic_cohesion_lift | stability_ari | silhouette | granularity_score | robustness | pathology_flags | final_rank |
|---|---|---|---|---|---|---|---|---|---|---|---|
| EXP_1_E5_KMEANS | k=100 | 100 | 0.000 | 0.911 | 0.280 | 0.404 | 0.069 | 1.000 | 0.410 | unstable | 2 (second) |
| EXP_2_E5_HDBSCAN | min_cluster_size=30,min_samples=5 | 4 | 0.938 | 0.912 | 0.204 | 0.586 | 0.174 | 0.000 | -0.030 | too_few_clusters;dominant_cluster;excessive_noise |  |
| EXP_3_T5_KMEANS | k=100 | 100 | 0.000 | 0.914 | 0.290 | 0.385 | 0.069 | 1.000 | 0.464 | unstable | 1 (winner) |
| EXP_4_T5_HDBSCAN | min_cluster_size=20,min_samples=10 | 3 | 0.958 | 0.925 | 0.189 | 0.620 | 0.171 | 0.000 | -0.069 | too_few_clusters;dominant_cluster;excessive_noise |  |

## 6. Parameter sweep
### K-Means
| embedding_model | param_k | silhouette | davies_bouldin | calinski_harabasz | min_size | median_size | max_size | very_small_clusters | semantic_cohesion_lift | stability_ari | seed_ari_mean | pathology_flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| intfloat/e5-base-v2 | 25.000 | 0.058 | 2.838 | 48.760 | 66.000 | 120.000 | 178.000 | 0.000 | 0.174 | 0.428 | 0.450 | unstable |
| intfloat/e5-base-v2 | 40.000 | 0.067 | 2.625 | 39.132 | 44.000 | 75.500 | 108.000 | 0.000 | 0.213 | 0.466 | 0.503 | unstable |
| intfloat/e5-base-v2 | 60.000 | 0.068 | 2.521 | 31.196 | 23.000 | 48.500 | 79.000 | 0.000 | 0.243 | 0.458 | 0.520 | unstable |
| intfloat/e5-base-v2 | 80.000 | 0.071 | 2.449 | 26.560 | 15.000 | 37.000 | 74.000 | 0.000 | 0.267 | 0.420 | 0.483 | unstable |
| intfloat/e5-base-v2 | 100.000 | 0.069 | 2.453 | 22.904 | 12.000 | 29.000 | 56.000 | 0.000 | 0.280 | 0.404 | 0.431 | unstable |
| sentence-transformers/sentence-t5-base | 25.000 | 0.060 | 2.749 | 49.889 | 62.000 | 118.000 | 234.000 | 0.000 | 0.185 | 0.454 | 0.452 | unstable |
| sentence-transformers/sentence-t5-base | 40.000 | 0.065 | 2.598 | 39.186 | 38.000 | 75.500 | 128.000 | 0.000 | 0.221 | 0.437 | 0.500 | unstable |
| sentence-transformers/sentence-t5-base | 60.000 | 0.066 | 2.570 | 30.906 | 24.000 | 49.500 | 86.000 | 0.000 | 0.251 | 0.430 | 0.468 | unstable |
| sentence-transformers/sentence-t5-base | 80.000 | 0.069 | 2.480 | 26.197 | 18.000 | 37.000 | 69.000 | 0.000 | 0.275 | 0.391 | 0.432 | unstable |
| sentence-transformers/sentence-t5-base | 100.000 | 0.069 | 2.488 | 22.793 | 13.000 | 28.500 | 65.000 | 0.000 | 0.290 | 0.385 | 0.418 | unstable |

### HDBSCAN
| embedding_model | param_min_cluster_size | param_min_samples | n_clusters | noise_count | noise_share | min_size | median_size | max_size | hdbscan_prob_mean | silhouette | semantic_cohesion_lift | semantic_cohesion_lift_incl_noise | stability_ari | stability_ari_nonnoise | pathology_flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| intfloat/e5-base-v2 | 10.000 | 5.000 | 3 | 2115 | 0.701 | 11.000 | 22.000 | 867.000 | 0.993 | 0.026 | 0.017 | 0.011 | 0.417 | 0.239 | too_few_clusters;unstable;dominant_cluster;excessive_noise;poor_cohesion |
| intfloat/e5-base-v2 | 10.000 | 10.000 | 2 | 2533 | 0.840 | 16.000 | 241.000 | 466.000 | 0.979 | 0.032 | 0.012 | 0.007 | 0.552 | 0.589 | too_few_clusters;dominant_cluster;excessive_noise;poor_cohesion |
| intfloat/e5-base-v2 | 20.000 | 5.000 | 2 | 1888 | 0.626 | 22.000 | 563.500 | 1105.000 | 0.979 | 0.070 | 0.008 | 0.007 | 0.384 | 0.115 | too_few_clusters;unstable;dominant_cluster;excessive_noise;poor_cohesion |
| intfloat/e5-base-v2 | 20.000 | 10.000 | 3 | 2874 | 0.953 | 34.000 | 47.000 | 60.000 | 0.996 | 0.143 | 0.144 | 0.037 | 0.629 | 0.720 | too_few_clusters;dominant_cluster;excessive_noise |
| intfloat/e5-base-v2 | 30.000 | 5.000 | 4 | 2829 | 0.938 | 36.000 | 40.500 | 69.000 | 0.999 | 0.174 | 0.204 | 0.054 | 0.586 | 0.398 | too_few_clusters;dominant_cluster;excessive_noise |
| intfloat/e5-base-v2 | 30.000 | 10.000 | 3 | 2874 | 0.953 | 34.000 | 47.000 | 60.000 | 1.000 | 0.143 | 0.144 | 0.037 | 0.370 | 0.279 | too_few_clusters;unstable;dominant_cluster;excessive_noise |
| intfloat/e5-base-v2 | 40.000 | 5.000 | 2 | 2855 | 0.947 | 45.000 | 80.000 | 115.000 | 0.995 | 0.158 | 0.096 | 0.017 | 0.322 | 0.459 | too_few_clusters;unstable;dominant_cluster;excessive_noise |
| intfloat/e5-base-v2 | 40.000 | 10.000 | 2 | 2871 | 0.952 | 60.000 | 72.000 | 84.000 | 0.998 | 0.094 | 0.059 | 0.023 | 1.000 | n/a | too_few_clusters;dominant_cluster;excessive_noise |
| sentence-transformers/sentence-t5-base | 10.000 | 5.000 | 3 | 2128 | 0.706 | 11.000 | 15.000 | 861.000 | 0.998 | -0.000 | 0.013 | 0.012 | 0.477 | 0.186 | too_few_clusters;unstable;dominant_cluster;excessive_noise;poor_cohesion |
| sentence-transformers/sentence-t5-base | 10.000 | 10.000 | 2 | 1794 | 0.595 | 14.000 | 610.500 | 1207.000 | 0.969 | 0.050 | 0.005 | 0.006 | 0.461 | 0.043 | too_few_clusters;unstable;dominant_cluster;excessive_noise;poor_cohesion |
| sentence-transformers/sentence-t5-base | 20.000 | 5.000 | 2 | 2633 | 0.873 | 35.000 | 191.000 | 347.000 | 0.993 | 0.083 | 0.040 | 0.016 | 0.356 | 0.164 | too_few_clusters;unstable;dominant_cluster;excessive_noise;poor_cohesion |
| sentence-transformers/sentence-t5-base | 20.000 | 10.000 | 3 | 2888 | 0.958 | 24.000 | 27.000 | 76.000 | 0.986 | 0.171 | 0.189 | 0.033 | 0.620 | 0.408 | too_few_clusters;dominant_cluster;excessive_noise |
| sentence-transformers/sentence-t5-base | 30.000 | 5.000 | 2 | 2633 | 0.873 | 35.000 | 191.000 | 347.000 | 0.994 | 0.083 | 0.040 | 0.016 | 0.553 | 0.146 | too_few_clusters;dominant_cluster;excessive_noise;poor_cohesion |
| sentence-transformers/sentence-t5-base | 30.000 | 10.000 | 0 | 3015 | 1.000 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.600 | n/a | too_few_clusters;dominant_cluster;excessive_noise;poor_cohesion |
| sentence-transformers/sentence-t5-base | 40.000 | 5.000 | 0 | 3015 | 1.000 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.285 | 0.144 | too_few_clusters;unstable;dominant_cluster;excessive_noise;poor_cohesion |
| sentence-transformers/sentence-t5-base | 40.000 | 10.000 | 0 | 3015 | 1.000 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.600 | n/a | too_few_clusters;dominant_cluster;excessive_noise;poor_cohesion |

## 7. Stability
| config_id | stability_ari | stability_ari_sd | stability_ari_nonnoise | stability_noise_flip_rate | seed_ari_mean |
|---|---|---|---|---|---|
| EXP_2_E5_HDBSCAN__min_cluster_size=40_min_samples=10 | 1.000 | 0.000 | n/a | 0.000 | n/a |
| EXP_2_E5_HDBSCAN__min_cluster_size=20_min_samples=10 | 0.629 | 0.107 | 0.720 | 0.038 | n/a |
| EXP_4_T5_HDBSCAN__min_cluster_size=20_min_samples=10 | 0.620 | 0.154 | 0.408 | 0.028 | n/a |
| EXP_4_T5_HDBSCAN__min_cluster_size=30_min_samples=10 | 0.600 | 0.516 | n/a | 0.026 | n/a |
| EXP_4_T5_HDBSCAN__min_cluster_size=40_min_samples=10 | 0.600 | 0.516 | n/a | 0.026 | n/a |
| EXP_2_E5_HDBSCAN__min_cluster_size=30_min_samples=5 | 0.586 | 0.100 | 0.398 | 0.058 | n/a |
| EXP_4_T5_HDBSCAN__min_cluster_size=30_min_samples=5 | 0.553 | 0.157 | 0.146 | 0.069 | n/a |
| EXP_2_E5_HDBSCAN__min_cluster_size=10_min_samples=10 | 0.552 | 0.208 | 0.589 | 0.079 | n/a |
| EXP_4_T5_HDBSCAN__min_cluster_size=10_min_samples=5 | 0.477 | 0.186 | 0.186 | 0.158 | n/a |
| EXP_1_E5_KMEANS__k=40 | 0.466 | 0.024 | 0.466 | 0.000 | 0.503 |

Configurations flagged unstable (subsample ARI < 0.5): 18 of 26.

## 8. Semantic cohesion (95% posting-bootstrap CIs, conditional on the fitted partition)
| experiment_id | semantic_cohesion | ci | cohesion_random_baseline | semantic_cohesion_lift |
|---|---|---|---|---|
| EXP_1_E5_KMEANS | 0.911 | [0.909, 0.912] | 0.876 | 0.280 |
| EXP_2_E5_HDBSCAN | 0.912 | [0.909, 0.915] | 0.889 | 0.204 |
| EXP_3_T5_KMEANS | 0.914 | [0.913, 0.915] | 0.879 | 0.290 |
| EXP_4_T5_HDBSCAN | 0.925 | [0.921, 0.930] | 0.908 | 0.189 |

Raw cohesion is not comparable across embedding models (different cosine scales); compare the lift.

## 9. Nearest-neighbour analysis
`evaluation/nearest_neighbor_examples.csv` lists the top-5 neighbours of 25 deterministic sample statements under each model for human inspection. Quantitative proxy - cross-provider paraphrase retrieval (rank of the Gemini statement when its Groq wording is the query, over the full corpus):
| model | paraphrase_queries | paraphrase_recall_at_1 | paraphrase_recall_at_5 | paraphrase_mrr | paraphrase_median_rank | random_pair_cosine_mean | neighbor_contrast_z |
|---|---|---|---|---|---|---|---|
| E5 | 432 | 0.808 | 0.995 | 0.895 | 1.000 | 0.760 | 4.950 |
| T5 | 432 | 0.808 | 0.993 | 0.891 | 1.000 | 0.765 | 4.541 |

Top-5 neighbour-set Jaccard E5 vs T5: 0.441. Caveat: Sprint 2 accepted these pairs by lexical/semantic agreement, so they skew towards easy paraphrases.

## 10. E5 vs Sentence-T5: **inconclusive**
- semantic_cohesion_lift: E5 mean 0.156 vs T5 0.151; E5 higher in 3/10 matched configurations -> tie
- stability_ari: E5 mean 0.495 vs T5 0.465; E5 higher in 10/13 matched configurations -> e5
- provider_pair_kappa: E5 mean -2.984 vs T5 -3.008; E5 higher in 4/10 matched configurations -> tie
- paraphrase MRR E5 0.895 vs T5 0.891 (diff 95% CI [-0.015, 0.022]) -> tie

## 11. K-Means vs HDBSCAN
See `evaluation/model_selection.md` section 2 for the within-representation comparison; the overall winner uses **kmeans**.

## 12. Cross-representation agreement (E5 vs Sentence-T5, same method and parameters)
| comparison | ari_both_assigned | nmi_both_assigned | mean_best_jaccard | e5_robust_groups | e5_representation_sensitive_groups | t5_robust_groups | t5_representation_sensitive_groups |
|---|---|---|---|---|---|---|---|
| same_parameters:kmeans:k=25 | 0.256 | 0.429 | 0.311 | 4 | 13 | 4 | 13 |
| same_parameters:kmeans:k=40 | 0.267 | 0.516 | 0.342 | 8 | 18 | 8 | 18 |
| same_parameters:kmeans:k=60 | 0.259 | 0.550 | 0.303 | 9 | 32 | 9 | 32 |
| same_parameters:kmeans:k=80 | 0.241 | 0.573 | 0.278 | 7 | 47 | 7 | 47 |
| same_parameters:kmeans:k=100 | 0.245 | 0.608 | 0.274 | 11 | 61 | 11 | 61 |
| same_parameters:hdbscan:min_cluster_size=10,min_samples=5 | -0.028 | 0.006 | 0.144 | 0 | 2 | 0 | 2 |
| same_parameters:hdbscan:min_cluster_size=10,min_samples=10 | 0.000 | 0.000 | 0.135 | 0 | 2 | 0 | 2 |
| same_parameters:hdbscan:min_cluster_size=20,min_samples=5 | 0.000 | 0.000 | 0.134 | 0 | 2 | 0 | 2 |
| same_parameters:hdbscan:min_cluster_size=20,min_samples=10 | 0.330 | 0.535 | 0.187 | 0 | 3 | 0 | 3 |
| same_parameters:hdbscan:min_cluster_size=30,min_samples=5 | 0.000 | 0.000 | 0.063 | 0 | 4 | 0 | 2 |
| same_parameters:hdbscan:min_cluster_size=30,min_samples=10 | n/a | n/a | n/a | 0 | 0 | 0 | 0 |
| same_parameters:hdbscan:min_cluster_size=40,min_samples=5 | n/a | n/a | n/a | 0 | 0 | 0 | 0 |
| same_parameters:hdbscan:min_cluster_size=40,min_samples=10 | n/a | n/a | n/a | 0 | 0 | 0 | 0 |
| final_candidates:kmeans | 0.245 | 0.608 | 0.274 | 11 | 61 | 11 | 61 |
| final_candidates:hdbscan | -0.028 | 0.197 | 0.117 | 0 | 4 | 0 | 3 |

Robust group: best Jaccard >= 0.5; representation-sensitive: < 0.3. Cluster-level rows: `evaluation/cross_representation_agreement.csv`.

## 13. Gemini/Groq provider robustness
Primary statistic: for Gemini statements whose task Groq also extracted (different wording, same evidence), the Groq wording is assigned out-of-sample by the fitted model (K-Means predict / HDBSCAN approximate_predict) and we measure how often it lands in the Gemini statement's topic, chance-corrected (kappa-style: (rate - chance)/(1 - chance)). Secondary: same-topic rate of nearest cross-provider vs nearest same-provider neighbours within matched cosine bins (a value near 0 means topics follow meaning, not provider wording). Provider equality is not treated as ground truth.
| experiment_id | provider_pair_kappa | nn_cross_minus_same_provider_same_topic |
|---|---|---|
| EXP_1_E5_KMEANS | 0.576 | -0.044 |
| EXP_2_E5_HDBSCAN | -0.060 | 0.005 |
| EXP_3_T5_KMEANS | 0.684 | -0.003 |
| EXP_4_T5_HDBSCAN | -0.468 | 0.000 |

## 14. Contamination stress test (winner only)
Winner refit with identical parameters on the responsibilities-only subset (2,289 of 3,015 statements) vs the full run restricted to the same statements: ARI (both assigned) 0.358, NMI 0.703, mean best Jaccard 0.360, major topics (size >= 30) preserved at Jaccard >= 0.5: 14/45 (31.1%).
Caveat: audit purity by section: responsibilities 53/59 TASK (89.8%), general_description 26/26 TASK (100%) - the responsibilities-only subset is the requested 'duty-type' subset, not one the audit proves cleaner.

## 15. Bootstrap uncertainty
| experiment_id | metric | point_estimate | ci95_low | ci95_high | n_resamples |
|---|---|---|---|---|---|
| EXP_1_E5_KMEANS | semantic_cohesion | 0.911 | 0.909 | 0.912 | 100 |
| EXP_1_E5_KMEANS | silhouette | 0.069 | 0.066 | 0.071 | 100 |
| EXP_2_E5_HDBSCAN | semantic_cohesion | 0.912 | 0.909 | 0.915 | 100 |
| EXP_2_E5_HDBSCAN | silhouette | 0.174 | 0.162 | 0.185 | 100 |
| EXP_2_E5_HDBSCAN | noise_share | 0.938 | 0.924 | 0.949 | 100 |
| EXP_3_T5_KMEANS | semantic_cohesion | 0.914 | 0.913 | 0.915 | 100 |
| EXP_3_T5_KMEANS | silhouette | 0.069 | 0.066 | 0.071 | 100 |
| EXP_4_T5_HDBSCAN | semantic_cohesion | 0.925 | 0.921 | 0.930 | 100 |
| EXP_4_T5_HDBSCAN | silhouette | 0.171 | 0.146 | 0.200 | 100 |
| EXP_4_T5_HDBSCAN | noise_share | 0.958 | 0.950 | 0.965 | 100 |

## 16. Multi-topic membership
Rule (derived from the winner's own distributions; quantiles in `evaluation/membership_threshold_derivation.json`): secondary topic iff similarity to the best other centroid >= P25 of members' own-centroid similarity AND (primary - secondary) <= P10 of non-negative primary/secondary gaps; HDBSCAN noise is assigned post hoc only if its best centroid similarity >= P50 of members' own-centroid similarity AND its top1-top2 gap >= the same P10 gap; otherwise it stays UNASSIGNED; max 2 memberships.
- secondary similarity floor 0.899, maximum primary-secondary gap 0.007, noise post-hoc floor 0.914
- statements with a secondary topic: 153 (5.1%); HDBSCAN noise assigned post hoc: 0; left UNASSIGNED: 0.
- Decision: one statement MAY belong to two topics (primary + at most one secondary) when the rule fires; unresolved statements stay unassigned rather than being forced.

## 17. Error analysis
`evaluation/cluster_error_analysis.csv` holds 710 rows (largest / medium / smallest meaningful clusters, lowest-confidence assignments, noise examples, ambiguous top-1 vs top-2) for the 4 candidates, with `error_review_label` left blank for human review (GOOD, MIXED_TOPIC, TOO_BROAD, OVER_SPLIT, RESIDUAL_NON_TASK, AMBIGUOUS_TASK, OUTLIER). No LLM was used to label them.
| experiment_id | low_confidence_rate | centroid_inconsistency_rate | secondary_membership_rate |
|---|---|---|---|
| EXP_1_E5_KMEANS | 0.347 | 0.018 | 0.049 |
| EXP_2_E5_HDBSCAN | 0.000 | 0.043 | 0.001 |
| EXP_3_T5_KMEANS | 0.356 | 0.016 | 0.051 |
| EXP_4_T5_HDBSCAN | 0.000 | 0.024 | 0.001 |

## 18. Final chosen configuration
**EXP_3_T5_KMEANS__k=100** - sentence-transformers/sentence-t5-base + kmeans (k=100); 100 leaf topics (`final/task_topics.csv`, IDs TASK_TOPIC_001..100, unnamed - naming is Sprint 4); memberships in `final/task_topic_membership.csv`. Decided by: robustness. Second-best: EXP_1_E5_KMEANS__k=100. Full reasoning: `evaluation/model_selection.md`.

## 19. Limitations
- The Sprint 3 input was section-filtered following a manually labelled task-purity audit. Post-filter audited task purity was 92.9%. Residual non-task contamination may therefore remain and is treated as a known limitation of the clustering experiment.
- No ground-truth topics exist; cohesion, silhouette, stability and agreement are internal/robustness measures, not accuracy.
- Raw cosine scales differ between models; cross-model comparisons use chance-adjusted lift and retrieval ranks.
- HDBSCAN cohesion/silhouette are computed on non-noise points (noise-inclusive cohesion lift reported alongside).
- Bootstrap CIs are conditional on the fitted partition; refit variability is captured by the stability ARI.
- Paraphrase pairs come from Sprint 2 cross-model agreement and are biased towards easier paraphrases.
- Gemini contributes ~90% of statements; provider-composition statistics reflect genuine content differences too.

## 20. Sprint 3 conclusion
Representation verdict: inconclusive. Selected pipeline: sentence-transformers/sentence-t5-base + kmeans (k=100) with 100 leaf topics, semantic cohesion 0.914 (lift 0.290), subsample stability ARI 0.385, 0.0% noise before the post-hoc rule. These topics are the input to Sprint 4 (naming, hierarchy, O*NET evaluation), which was not started here.

Runtime: 313.3 s.