# Sprint 3 Model Selection

Decision procedure (pre-declared in the script): (1) reject pathological configurations ({"min_stability_ari": 0.5, "max_dominant_share": 0.2, "max_tiny_cluster_share": 0.25, "max_noise_share": 0.5, "min_clusters": 10, "max_clusters": 300, "min_cohesion_lift": 0.05}); (2) lexicographic hierarchy with tolerances semantic_cohesion_lift (tol 0.02), stability_ari (tol 0.05), granularity_score (tol 0.0), robustness (tol 0.05), silhouette (tol 0.01). A level only breaks ties among configurations that are within tolerance of the best on all higher levels. No single metric decides.

Cosine-based cohesion is compared across embedding models only as a chance-adjusted **lift** ((cohesion - random-partition cohesion) / (1 - random-partition cohesion)) because raw cosine scales differ between E5 and Sentence-T5. Silhouette is measured in each model's own PCA space and is the lowest-priority criterion.

## 1. E5 vs Sentence-T5

**Verdict: INCONCLUSIVE - the evidence does not clearly favour either representation**

- semantic_cohesion_lift: E5 mean 0.156 vs T5 0.151; E5 higher in 3/10 matched configurations -> tie
- stability_ari: E5 mean 0.495 vs T5 0.465; E5 higher in 10/13 matched configurations -> e5
- provider_pair_kappa: E5 mean -2.984 vs T5 -3.008; E5 higher in 4/10 matched configurations -> tie
- paraphrase MRR E5 0.895 vs T5 0.891 (diff 95% CI [-0.015, 0.022]) -> tie

| criterion | matched configs | E5 | T5 | E5 better in |
|---|---:|---:|---:|---:|
| semantic_cohesion_lift | 10.0 | 0.156 | 0.151 | 3.0 |
| stability_ari | 13.0 | 0.495 | 0.465 | 10.0 |
| provider_pair_kappa | 10.0 | -2.984 | -3.008 | 4.0 |
| silhouette | 10.0 | 0.078 | 0.072 | 6.0 |
| low_confidence_rate | 13.0 | 0.161 | 0.166 | 4.0 |
| paraphrase_mrr | nan | 0.895 | 0.891 | nan |

Top-5 neighbour sets agree between E5 and T5 with mean Jaccard 0.441. Inspect `nearest_neighbor_examples.csv` before accepting the verdict.

## 2. K-Means vs HDBSCAN (within Sentence-T5)

Candidates: K-Means `EXP_3_T5_KMEANS__k=100` vs HDBSCAN `EXP_4_T5_HDBSCAN__min_cluster_size=20_min_samples=10` -> **kmeans** (decided by semantic_cohesion_lift).

- no non-pathological configuration - choosing among the least-flagged
- semantic_cohesion_lift: best=0.290 tol=0.02 -> kept EXP_3_T5_KMEANS__k=100

| config_id | n_clusters | noise_share | semantic_cohesion_lift | stability_ari | granularity_score | robustness | silhouette | low_confidence_rate | pathology_flags |
|---|---|---|---|---|---|---|---|---|---|
| EXP_3_T5_KMEANS__k=100 | 100 | 0.000 | 0.290 | 0.385 | 1.000 | 0.464 | 0.069 | 0.356 | unstable |
| EXP_4_T5_HDBSCAN__min_cluster_size=20_min_samples=10 | 3 | 0.958 | 0.189 | 0.620 | 0.000 | -0.069 | 0.171 | 0.000 | too_few_clusters;dominant_cluster;excessive_noise |

## 3. Winning pipeline

- configuration: `EXP_3_T5_KMEANS__k=100`
- embedding model: sentence-transformers/sentence-t5-base
- clustering method: kmeans
- exact parameters: k=100 (PCA 50 components, cumulative variance 0.607; seed 42)
- leaf topic count: 100
- noise / unassigned (before post-hoc rule): 0 (0.0%)
- semantic cohesion: 0.914 (chance 0.879, lift 0.290)
- stability: subsample ARI 0.385 ± 0.017; seed-restart ARI 0.418
- robustness: provider-pair kappa 0.684, cross-representation ARI 0.245, contamination stress-test major-topic preservation 31.1% (ARI 0.358)
- decided by: robustness

  - no non-pathological configuration - choosing among the least-flagged
  - semantic_cohesion_lift: best=0.290 tol=0.02 -> kept EXP_1_E5_KMEANS__k=100, EXP_3_T5_KMEANS__k=100
  - stability_ari: best=0.404 tol=0.05 -> kept EXP_1_E5_KMEANS__k=100, EXP_3_T5_KMEANS__k=100
  - granularity_score: best=1.000 tol=0.0 -> kept EXP_1_E5_KMEANS__k=100, EXP_3_T5_KMEANS__k=100
  - robustness: best=0.464 tol=0.05 -> kept EXP_3_T5_KMEANS__k=100

## 4. Second-best pipeline

`EXP_1_E5_KMEANS__k=100` - intfloat/e5-base-v2, kmeans, k=100; 100 topics, noise 0.0%, cohesion lift 0.280, stability ARI 0.404.

## 5. Why the winner beat the second-best

| criterion | tolerance | winner | second | difference |
|---|---:|---:|---:|---:|
| semantic_cohesion_lift | 0.02 | 0.290 | 0.280 | 0.010 |
| stability_ari | 0.05 | 0.385 | 0.404 | -0.019 |
| granularity_score | 0.0 | 1.000 | 1.000 | 0.000 |
| robustness | 0.05 | 0.464 | 0.410 | 0.054 |
| silhouette | 0.01 | 0.069 | 0.069 | -0.000 |

First criterion (in priority order) where they differ beyond tolerance: **robustness** (winner 0.464 vs 0.410).
Within tolerance (effectively tied) on: semantic_cohesion_lift, stability_ari, granularity_score, silhouette.
Because the decision falls to a low-priority criterion, the margin is **small**; do not over-interpret it.

## Per-experiment candidate selection

**EXP_1_E5_KMEANS** -> `EXP_1_E5_KMEANS__k=100` (decided by tie_break(semantic_cohesion_lift, stability_ari))
- no non-pathological configuration - choosing among the least-flagged
- semantic_cohesion_lift: best=0.280 tol=0.02 -> kept EXP_1_E5_KMEANS__k=80, EXP_1_E5_KMEANS__k=100
- stability_ari: best=0.420 tol=0.05 -> kept EXP_1_E5_KMEANS__k=80, EXP_1_E5_KMEANS__k=100
- granularity_score: best=1.000 tol=0.0 -> kept EXP_1_E5_KMEANS__k=80, EXP_1_E5_KMEANS__k=100
- robustness: best=0.410 tol=0.05 -> kept EXP_1_E5_KMEANS__k=80, EXP_1_E5_KMEANS__k=100
- silhouette: best=0.071 tol=0.01 -> kept EXP_1_E5_KMEANS__k=80, EXP_1_E5_KMEANS__k=100
- still tied -> tie-break on raw cohesion lift then stability -> EXP_1_E5_KMEANS__k=100

**EXP_2_E5_HDBSCAN** -> `EXP_2_E5_HDBSCAN__min_cluster_size=30_min_samples=5` (decided by semantic_cohesion_lift)
- no non-pathological configuration - choosing among the least-flagged
- semantic_cohesion_lift: best=0.204 tol=0.02 -> kept EXP_2_E5_HDBSCAN__min_cluster_size=30_min_samples=5

**EXP_3_T5_KMEANS** -> `EXP_3_T5_KMEANS__k=100` (decided by tie_break(semantic_cohesion_lift, stability_ari))
- no non-pathological configuration - choosing among the least-flagged
- semantic_cohesion_lift: best=0.290 tol=0.02 -> kept EXP_3_T5_KMEANS__k=80, EXP_3_T5_KMEANS__k=100
- stability_ari: best=0.391 tol=0.05 -> kept EXP_3_T5_KMEANS__k=80, EXP_3_T5_KMEANS__k=100
- granularity_score: best=1.000 tol=0.0 -> kept EXP_3_T5_KMEANS__k=80, EXP_3_T5_KMEANS__k=100
- robustness: best=0.464 tol=0.05 -> kept EXP_3_T5_KMEANS__k=80, EXP_3_T5_KMEANS__k=100
- silhouette: best=0.069 tol=0.01 -> kept EXP_3_T5_KMEANS__k=80, EXP_3_T5_KMEANS__k=100
- still tied -> tie-break on raw cohesion lift then stability -> EXP_3_T5_KMEANS__k=100

**EXP_4_T5_HDBSCAN** -> `EXP_4_T5_HDBSCAN__min_cluster_size=20_min_samples=10` (decided by semantic_cohesion_lift)
- no non-pathological configuration - choosing among the least-flagged
- semantic_cohesion_lift: best=0.189 tol=0.02 -> kept EXP_4_T5_HDBSCAN__min_cluster_size=20_min_samples=10
