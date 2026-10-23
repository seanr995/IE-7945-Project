# Audit of existing Sprint 3 results

Machine-readable tables are treated as the source of truth; prose was re-checked claim by claim. No historical file was modified (SHA-256 inventory: `historical_file_inventory.csv`).

## 1. Primary Sprint 3 experiment (`sprint3_results/`)
Chosen: `EXP_3_T5_KMEANS__k=100` (decided by `robustness`).

- every one of the 26 Sprint 3 configurations carried a pathology flag; the winner was chosen 'among the least-flagged' (all K-Means configurations are flagged unstable: subsample ARI 0.385-0.466 < 0.50)
- intfloat/e5-base-v2: cohesion lift rises monotonically with k (k=25->100: 0.174->0.280) while subsample stability falls (0.428->0.404); cohesion-first selection therefore drifts to the largest k tested
- sentence-transformers/sentence-t5-base: cohesion lift rises monotonically with k (k=25->100: 0.185->0.290) while subsample stability falls (0.454->0.385); cohesion-first selection therefore drifts to the largest k tested
- the winner k=100 lies on the BOUNDARY of the tested grid [25, 40, 60, 80, 100]: k=100 is an artefact of the grid edge, not evidence of 100 natural task concepts
- HDBSCAN on the 50-d PCA space collapsed: 0-4 clusters with 60%-100% noise for every configuration
- 7 HDBSCAN configurations have chance-corrected paraphrase kappa < -1 (chance near 1 under a dominant cluster); the report's E5-vs-T5 'provider_pair_kappa' means average these degenerate values and are not interpretable
- E5 vs Sentence-T5 (both k=100) were tied on cohesion lift, stability and granularity; T5 won only at the 4th hierarchy level (robustness). Trace: no non-pathological configuration - choosing among the least-flagged | semantic_cohesion_lift: best=0.290 tol=0.02 -> kept EXP_1_E5_KMEANS__k=100, EXP_3_T5_KMEANS__k=100 | stability_ari: best=0.404 tol=0.05 -> kept EXP_1_E5_KMEANS__k=100, EXP_3_T5_KMEANS__k=100 | granularity_score: best=1.000 tol=0.0 -> kept EXP_1_E5_KMEANS__k=100, EXP_3_T5_KMEANS__k=100 | robustness: best=0.464 tol=0.05 -> kept EXP_3_T5_KMEANS__k=100
- figure 04_stability_comparison.png was not produced (pandas 'config_id is both an index level and a column label' error in the Sprint 3 script) - see sprint3_results/figures/figure_errors.txt

Key facts:
```
all_configurations_pathological: True
intfloat/e5-base-v2: cohesion_lift_monotone_increasing_in_k: True
intfloat/e5-base-v2: stability_by_k: {25: 0.428, 40: 0.466, 60: 0.458, 80: 0.42, 100: 0.404}
sentence-transformers/sentence-t5-base: cohesion_lift_monotone_increasing_in_k: True
sentence-transformers/sentence-t5-base: stability_by_k: {25: 0.454, 40: 0.437, 60: 0.43, 80: 0.391, 100: 0.385}
paraphrase_mrr: {'intfloat/e5-base-v2': 0.8947, 'sentence-transformers/sentence-t5-base': 0.8915}
```

**Conclusion:** k=100 is not evidence of 100 natural task concepts. K-Means needs k before fitting, the cohesion criterion rises mechanically with k, every K-Means configuration failed the stability threshold, and the selected k sits on the grid boundary.

## 2. Advanced challenger experiment (`sprint3_advanced_results/`)

### 2a. Report prose vs numeric table

| claim in report | prose says | metric checked | table says | value | consistent |
|---|---|---|---|---:|---|
| Highest semantic cohesion: | Sentence-T5 + Leiden | highest_cohesion_lift_coverage_adjusted | Sentence-T5 + GMM | 0.3155 | **NO** |
| Highest semantic cohesion: | Sentence-T5 + Leiden | highest_semantic_cohesion_raw | Sentence-T5 + GMM | 0.9183 | **NO** |
| Most stable: | Sentence-T5 + Leiden | highest_stability | Sentence-T5 + Leiden | 0.5619 | yes |
| Best at preserving Gemini/Groq paraphrases: | Sentence-T5 + Leiden | best_paraphrase_kappa | Sentence-T5 + Leiden | 0.7815 | yes |
| Handles ambiguous tasks best: | Sentence-T5 + Leiden | lowest_low_confidence_rate_common | Sentence-T5 + K-Means | 0.1002 | **NO** |
| Most useful topic granularity: | Sentence-T5 + Leiden | highest_granularity_score | Sentence-T5 + K-Means, Sentence-T5 + BERTopic, Sentence-T5 + GMM | 1.0000 | **NO** |
| Adds complexity without enough value | includes GMM | extra_dependencies / tuned_hyperparameters | GMM: no extra dependencies, 1 tuned hyper-parameter (same as K-Means) |  | **NO** |

**5 prose inconsistencies.** Root cause: advanced_challengers.write_report computed every 'highest/best/lowest' answer over the NON-PATHOLOGICAL pool only (['leiden']); with a single non-pathological method every superlative named it regardless of the numbers.

### 2b. Numeric ground truth (from `advanced_challenger_comparison.csv`)

| quantity | method | value |
|---|---|---:|
| highest_semantic_cohesion_raw | gmm | 0.9183 |
| highest_cohesion_lift_assigned | gmm | 0.3155 |
| highest_cohesion_lift_coverage_adjusted | gmm | 0.3155 |
| highest_stability | leiden | 0.5619 |
| lowest_low_confidence_rate_common | kmeans | 0.1002 |
| lowest_low_confidence_rate_native | gmm | 0.0003 |
| best_paraphrase_kappa | leiden | 0.7815 |
| best_provider_robustness (lowest |gap|) | kmeans | 0.0029 |
| highest_coverage | kmeans=leiden=gmm | 1.0000 |
| lowest_noise | kmeans=leiden=gmm | 0.0000 |
| highest_granularity_score | kmeans=bertopic=gmm | 1.0000 |

### 2c. Decision-rule issues

- bertopic: disqualified for 'pathology: unstable' although the K-Means baseline carries the same flag
- leiden: cohesion-lift cost (0.290 -> 0.180) compares 31 communities with 100 K-Means clusters; K-Means at k=31 reaches only 0.201, so most of the gap is granularity, not method quality
- gmm: disqualified for 'pathology: unstable' although the K-Means baseline carries the same flag

The historical recommendation was `KEEP T5 + KMEANS`. Because the replacement rule was asymmetric and compared partitions of different granularity, that recommendation is NOT carried forward; the final benchmark re-evaluates all methods at matched granularity with one symmetric rule.

## 3. Consequences for the final design
- The topic count is investigated independently (agentic induction, no count given) instead of inherited.
- Methods are compared at matched granularity (N_LOW / N_AGENTIC / N_HIGH) with one symmetric rule.
- Every superlative in the final report is generated from, and re-verified against, the numeric tables.
- Topic-level persistence is added because global ARI hides which topics survive resampling.