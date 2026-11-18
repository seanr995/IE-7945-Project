# Final Leiden@80 Stability Report

Selected final model: **Sentence-T5 + Leiden @ 80 topics** (`LEIDEN_T5@80`). All values below are read from the existing Sprint 3 final results (`matched_benchmark/leiden/leiden_partitions.csv`, `matched_benchmark/leiden/resolution_search.csv`, `evaluation/final_method_comparison.csv`, `evaluation/topic_persistence.csv`, `final_sprint3_report.md`); nothing was re-run (no clustering, embeddings, Claude calls, or benchmarks).

## Verified final parameters and metrics

| Item | Value |
|---|---|
| Representation | Sentence-T5 embeddings (`t5`) |
| Leiden k_neighbors (kNN graph) | 10 |
| Leiden resolution | 5.537317 (resolution search: 80 communities exactly, |diff| = 0 to target 80) |
| Topics | 80 |
| Tasks | 3015 (no noise / unassigned) |
| Coverage | 1.000 (coverage_work 1.000) |
| Stability ARI | 0.567 (SD 0.029) |
| Topic persistence >= 50% (share of topics with median best-Jaccard >= 0.5) | 0.700 (56 of 80 topics; median Jaccard 0.567) |
| Paraphrase kappa (chance-corrected) | 0.800 |
| Cohesion lift (semantic_cohesion_lift) | 0.256 |
| Modularity | 0.556 |
| Topic size min / median / max | 6 / 34.5 / 88 |
| Tiny-cluster share | 0.025 |

Gate failures: none. It is the only target-granularity candidate that passes every gate (`stability_ari` >= 0.5 and `persistence_share_j50` >= 0.5, plus coverage and tiny-cluster gates).

## Why Leiden@80 over the alternatives

| Method | Topics | Coverage (work) | Stability ARI | Persistence >=50% | Paraphrase kappa | Cohesion lift | Gate failures |
|---|---|---|---|---|---|---|---|
| **Leiden (T5) @80** | 80 | 1.000 | 0.567 | 0.700 | 0.800 | 0.256 | none |
| K-Means (T5) k=64 | 64 | 1.000 | 0.400 | 0.328 | 0.677 | 0.257 | stability_ari=0.400 < 0.5;persistence_share_j50=0.328 < 0.5 |
| GMM (diag) n=80 | 80 | 1.000 | 0.396 | 0.388 | 0.728 | 0.268 | stability_ari=0.396 < 0.5;persistence_share_j50=0.388 < 0.5 |
| BERTopic natural | 70 | 0.729 | 0.341 | 0.586 | 0.653 | 0.253 | coverage_work=0.729 < 0.8;stability_ari=0.341 < 0.5 |
| HDBSCAN (mcs=10, ms=1) | 3 | 0.787 | 0.149 | 0.000 | -8.962 | 0.005 | coverage_work=0.787 < 0.8;stability_ari=0.149 < 0.5;persistence_share_j50=0.000 < 0.5;dominant_share=0.990 > 0.2 |
| Agentic (Claude) | 80 | 0.991 | 0.679 | 0.975 | 0.614 | 0.184 | tiny_cluster_share=0.412 > 0.25 |

- **K-Means:** stability ARI 0.400 and persistence 0.328 fail the 0.5 gates. Its partitions are not reproducible across resamples. At matched granularity (target 80) Leiden's ARI is 0.567 vs 0.391 for K-Means (final report Q9). K-Means has slightly higher cohesion lift, but cohesion is lower priority than stability in the decision hierarchy.
- **GMM:** also fails stability (0.396) and persistence (0.388), with lower paraphrase kappa (0.728 vs 0.800), despite the highest cohesion lift among the 80-topic methods.
- **BERTopic:** assigns only 72.9% of work statements to topics (fails the 0.8 coverage gate) and has stability ARI 0.341.
- **HDBSCAN:** collapses to 3 clusters with a 99% dominant cluster, 78.7% coverage, ARI 0.149 and persistence 0.000. It cannot produce a usable 80-topic taxonomy.
- **Agentic (Claude):** the most stable scores (ARI 0.679, persistence 0.975), but 41% of its topics are tiny (fails the tiny_cluster_share <= 0.25 gate), 19 of 20 Claude calls were consumed, and it is more complex (complexity rank 4 vs 2). The final report's hierarchy ranks the hybrid variant behind Leiden@80.
- **Leiden@80:** the only candidate at the target granularity that passes all gates, with the best paraphrase kappa of all candidates (0.800), full coverage, and a modest tiny-cluster share (0.025).

## Caveats
- Selection was decided at hierarchy step 4 (topic persistence). The human-coherence criterion was pending, and the final report flags the freeze status as "NOT READY TO FREEZE" until the human review sample is labelled.
- No topic-level gold labels exist, so none of these metrics is an accuracy.
- Topic names are not assigned. `LT_001`..`LT_080` are deterministic IDs ordered by topic size (descending), ties broken by raw Leiden label.
- `membership_strength` is the cosine similarity of a task's T5 embedding to its own topic centroid (`own_sim` from the stored checkpoint). Representative tasks are the 3 highest-strength tasks per topic (ties by statement_id).
- Known limitation: residual non-task contamination may remain (post-filter audited task purity 92.9%).
