# Sprint 4 Evaluation Report - Agentic Labeling, Hierarchy and O*NET Alignment

Generated 2026-11-05T01:33:29.102214+00:00. Selected Sprint 3 model: Sentence-T5 + Leiden @ 80 topics (frozen).

## 1. Frozen leaf layer
- **80 frozen leaf topics**, 3,015 tasks. Leaf membership was loaded from `final_task_topic_membership_leiden80.csv` and is never
  modified (validated byte-for-byte on `topic_id` for every task). The LLM had no ability to move tasks.
- Embeddings: sentence-transformers/sentence-t5-base (frozen Sprint 3 embeddings).

## 2. Leaf evidence
- **Centroid**: mean of the L2-normalised Sentence-T5 vectors of the topic's tasks, re-normalised.
- **Core tasks**: the 10 member tasks with highest cosine to the centroid (ties by `statement_id`). **Boundary tasks**: the 5 lowest.
  (For topics with fewer than 15 tasks, boundary = bottom third and core = the rest, up to 10, so the two sets never overlap.)
- **c-TF-IDF keywords**: one pseudo-document per topic (all member task texts); unigrams+bigrams, English stop words;
  weight = (term count / topic word count) x ln(1 + A / f_term), A = mean words per topic, f_term = term count over all 80 topics.
  Top 10 per topic (`leaf_evidence.csv`).

## 3. Gemini leaf labeler
- 80 topics in batches of 20 (4 calls). Each topic received its size, its 7 most typical tasks, its 3 least typical tasks and 8 c-TF-IDF keywords (evidence files keep the full 10/5/10).
- Native structured output (Pydantic schema) with local validation and word limits; fields: topic_name, description, core_action, object_or_outcome, include_if, exclude_if, coherent, mixed_topic, confidence, reason.
  Local validation rejects vague/one-word labels; invalid records get ONE batched repair request (max 2 per run), otherwise FLAG_UNRESOLVED. Responses cached.
- Mixed topics flagged by Gemini: 1; mean label confidence
  0.922.

## 4. Hierarchy search (deterministic, no LLM)
- Average-linkage agglomerative clustering (SciPy `linkage(method="average")`, equivalent to scikit-learn
  `AgglomerativeClustering(linkage="average", metric="precomputed")`) on the **cosine distance matrix of the 80 leaf centroids**.
  Level-2 counts tested: 12, 16, 20; level-1 counts tested: 4, 6, 8 (9 candidates per linkage; both levels are cuts of one dendrogram, so levels are nested).
  **Added candidate family:** the same grid with Ward linkage (Euclidean distance on the unit-norm centroids, monotone in cosine distance), because
  average linkage chains on this embedding space (see the candidate table: one level-1 domain absorbs most tasks). All 18 candidates use the identical metrics and gates and the
  selection is automatic.
- Each candidate is scored on silhouette, size-weighted within-parent cosine cohesion, min/median/max parent size, dominant-parent share and
  tiny-parent count. Gates: no level-1 domain with >50% of tasks, no single-child level-1 domain, at most 10% single-leaf level-2 families.
  Among passing candidates the score is {"silhouette_L2": 0.2, "silhouette_L1": 0.2, "cohesion_L2": 0.2, "cohesion_L1": 0.2, "dominant_share_penalty": 0.1, "tiny_parent_penalty": 0.1} on min-max normalised metrics.

| linkage | level2_count | level1_count | sil_L2 | sil_L1 | coh_L2 | coh_L1 | L2_min | L2_med | L2_max | dom_L2 | dom_L1 | tiny_L2 | tiny_L1 | passes_gates | score | selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| average | 12 | 4 | 0.0858 | 0.1072 | 0.9729 | 0.9630 | 1 | 3.0000 | 25 | 0.3459 | 0.9841 | 4 | 3 | False | 0.2292 | False |
| average | 12 | 6 | 0.0858 | 0.0330 | 0.9729 | 0.9655 | 1 | 3.0000 | 25 | 0.3459 | 0.9025 | 4 | 3 | False | 0.1314 | False |
| average | 12 | 8 | 0.0858 | 0.0122 | 0.9729 | 0.9668 | 1 | 3.0000 | 25 | 0.3459 | 0.8663 | 4 | 4 | False | 0.1057 | False |
| average | 16 | 4 | 0.1229 | 0.1072 | 0.9769 | 0.9630 | 1 | 3.0000 | 15 | 0.1937 | 0.9841 | 4 | 3 | False | 0.5309 | False |
| average | 16 | 6 | 0.1229 | 0.0330 | 0.9769 | 0.9655 | 1 | 3.0000 | 15 | 0.1937 | 0.9025 | 4 | 3 | False | 0.4331 | False |
| average | 16 | 8 | 0.1229 | 0.0122 | 0.9769 | 0.9668 | 1 | 3.0000 | 15 | 0.1937 | 0.8663 | 4 | 4 | False | 0.4074 | False |
| average | 20 | 4 | 0.1171 | 0.1072 | 0.9789 | 0.9630 | 1 | 3.0000 | 14 | 0.1774 | 0.9841 | 6 | 3 | False | 0.5447 | False |
| average | 20 | 6 | 0.1171 | 0.0330 | 0.9789 | 0.9655 | 1 | 3.0000 | 14 | 0.1774 | 0.9025 | 6 | 3 | False | 0.4469 | False |
| average | 20 | 8 | 0.1171 | 0.0122 | 0.9789 | 0.9668 | 1 | 3.0000 | 14 | 0.1774 | 0.8663 | 6 | 4 | False | 0.4212 | False |
| ward | 12 | 4 | 0.1141 | 0.0856 | 0.9757 | 0.9682 | 2 | 5.5000 | 14 | 0.2030 | 0.2687 | 0 | 0 | True | 0.6431 | False |
| ward | 12 | 6 | 0.1141 | 0.0878 | 0.9757 | 0.9707 | 2 | 5.5000 | 14 | 0.2030 | 0.2624 | 0 | 0 | True | 0.7017 | False |
| ward | 12 | 8 | 0.1141 | 0.0941 | 0.9757 | 0.9724 | 2 | 5.5000 | 14 | 0.2030 | 0.2133 | 0 | 0 | True | 0.7525 | False |
| ward | 16 | 4 | 0.1310 | 0.0856 | 0.9784 | 0.9682 | 1 | 4.5000 | 14 | 0.1559 | 0.2687 | 1 | 0 | True | 0.7899 | False |
| ward | 16 | 6 | 0.1310 | 0.0878 | 0.9784 | 0.9707 | 1 | 4.5000 | 14 | 0.1559 | 0.2624 | 1 | 0 | True | 0.8485 | False |
| ward | 16 | 8 | 0.1310 | 0.0941 | 0.9784 | 0.9724 | 1 | 4.5000 | 14 | 0.1559 | 0.2133 | 1 | 0 | True | 0.8993 | False |
| ward | 20 | 4 | 0.1190 | 0.0856 | 0.9804 | 0.9682 | 1 | 3.5000 | 7 | 0.1051 | 0.2687 | 2 | 0 | True | 0.7933 | False |
| ward | 20 | 6 | 0.1190 | 0.0878 | 0.9804 | 0.9707 | 1 | 3.5000 | 7 | 0.1051 | 0.2624 | 2 | 0 | True | 0.8518 | False |
| ward | 20 | 8 | 0.1190 | 0.0941 | 0.9804 | 0.9724 | 1 | 3.5000 | 7 | 0.1051 | 0.2133 | 2 | 0 | True | 0.9026 | True |

- **Why average linkage was rejected:** All 9 average-linkage candidates failed the balance gates (passing: 0/9); level-1 dominant-domain task share ranged 0.87-0.98 (gate <= 0.50), level-1 single-child domains 3-4 (gate 0), level-2 single-leaf families 4-6. Ward linkage: 9/9 passed.
- **Chosen: ward linkage, 8 level-1 domains / 20 level-2 families / 80 leaves.**
  Final structure after critic: 8 / 20 / 80.

**Final hierarchy metrics (after critic; Level-1 = broad work domains, Level-2 = work-activity families):**

| level | parents | silhouette | cohesion | min/median/max children | min/median/max tasks | dominant share | tiny parents |
|---|---|---|---|---|---|---|---|
| L1 | 8 | 0.0872 | 0.9722 | 5/8.0/19 | 148/287/622 | 0.206 | 0 |
| L2 | 20 | 0.1180 | 0.9803 | 1/4.0/7 | 6/142/317 | 0.105 | 2 |

## 5. Gemini hierarchy builder
- ONE call: names every level-2 family and level-1 domain of the numerically selected hierarchy from compact leaf evidence (id, name, core action,
  short description) and lists clearly misplaced children. Grouping is fixed by Python; Gemini only recommends (`misplaced_child_ids`).

## 6. O*NET retrieval
- **O*NET tasks embedded: 18,838** (O*NET 31.0 Task Statements, unique Task IDs), with the same
  `sentence-transformers/sentence-t5-base` model, embedded once and cached. A re-embedding check against the frozen Sprint 3
  vectors confirms the model identity (real runs).
- For each leaf centroid the cosine similarity to every O*NET task is computed and the **top 10 are selected by similarity alone**
  (`onet_top10_matches.csv`, 800 rows). Gemini never chose candidates.

## 7. Gemini O*NET evaluator and quantitative results
- 4 batched calls (20 topics each); Gemini saw 3 representative tasks and the top 5 of the 10 deterministically retrieved candidates and could only choose among them. Alignment counts: STRONG 58, PARTIAL 20, NO_MATCH 2.
- **Thresholds derived from the observed distribution (nothing hand-set)**: noise floor tau_noise = 99th percentile of all
  80 x 18,838 centroid-O*NET cosines = 0.884721; weak-match split tau_split = Otsu threshold on the
  80 observed top-1 scores = 0.916811.
- **Coverage** (leaf topics with Gemini STRONG/PARTIAL alignment and top-1 above tau_noise): **97.5%**.
- **Purity** (share of a topic's member tasks whose own nearest O*NET task carries the topic's dominant O*NET concept - same task or
  a shared Detailed Work Activity): mean **0.194**, median 0.176.
- **Nearest-neighbour agreement** (share of the 5 representative tasks whose own top-1 O*NET task is inside the centroid's top-10):
  mean **0.623**; exact top-1 agreement 0.242.
- Top-1 cosine: mean **0.9224**, median **0.9238**.

| rank | mean | median | min | max |
|---|---|---|---|---|
| rank1 | 0.9224 | 0.9238 | 0.8868 | 0.9529 |
| rank2 | 0.9173 | 0.9213 | 0.8854 | 0.9397 |
| rank3 | 0.9148 | 0.9172 | 0.8845 | 0.9376 |
| rank4 | 0.9131 | 0.9152 | 0.8843 | 0.9345 |
| rank5 | 0.9118 | 0.9143 | 0.8836 | 0.9344 |

## 8. Critic and convergence
- Critic cases: mixed topics, low label confidence, weak O*NET alignment, low-purity topics, low-cohesion parents, overlapping parents,
  misplaced children and inconsistent boundary tasks; quartile cut-offs are computed from the observed distributions. ONE batched critic call (plus at most the shared batched repair); persistence from a preliminary stability run is one of the flags.
- Round 1: 31 cases; accepts 30, leaf renames 0, parent renames 0, refinements 0, moves 1, merges 0, splits 0, flagged 0, rejected 0; leaf-to-parent change 1.2%; delta L2 cohesion -0.0001
- Stop reason: **converged after the critic round: <5% leaf-to-parent assignments changed, no merge/split requested, labels stable or metrics not materially improving**.
- Provisional names (from SPLIT_PARENT, awaiting rename): none.

### Unresolved cases
- None.

## 9. Hierarchy stability (validation only; hierarchy not changed by it)
- 5 deterministic trials, seeds [101, 202, 303, 404, 505]. In every trial, **within each of the 80 leaves independently** 80% of member task embeddings
  were sampled without replacement (no leaf removed; the 3,015 frozen memberships were not touched), leaf centroids recomputed, and
  the hierarchy rebuilt with the final method (ward linkage, 20 level-2 / 8 level-1 targets).
- Parents are compared as sets of child leaf IDs; perturbed parents are matched to final parents by one-to-one Hungarian matching on
  maximum Jaccard, independently per level. (The final hierarchy after critic has 20 level-2 families and
  8 level-1 domains; perturbed runs have [20, 20, 20, 20, 20] / [8, 8, 8, 8, 8].)

| metric | Level-1 | Level-2 |
|---|---|---|
| mean leaf parent persistence | **0.715** | **0.7925** |
| ARI between perturbation runs (mean of 10 pairs) | 0.4843 | 0.6501 |
| ARI vs final hierarchy (mean of 5) | 0.5504 | 0.6469 |
| leaves with persistence < 0.60 | 24 | 16 |

Files: `hierarchy_stability.csv`, `hierarchy_stability_summary.json`.

## 10. O*NET coverage-gap analysis (possible gaps for interpretation; not a claim that O*NET is outdated)
- Classes use quantiles of the observed distributions of top-1 cosine, top-10 profile, purity, nearest-neighbour agreement and Gemini
  alignment confidence. POSSIBLE_EMERGING_WORK requires: coherent topic with cohesive core tasks, consistently weak similarity profile,
  weak purity or agreement, **and** limited Gemini-judged coverage.
- **WELL_COVERED 33 / PARTIALLY_COVERED 44 / POSSIBLE_EMERGING_WORK 3** (of 80 leaves).
- Cut-offs: {"top1_P25": 0.915237, "top1_P50": 0.923822, "mean_top10_P25": 0.901706, "purity_P25": 0.1215, "purity_P50": 0.1765, "nn_agreement_P25": 0.4, "nn_agreement_P50": 0.6, "alignment_confidence_P50": 0.95, "core_cohesion_P50": 0.909473}. File: `onet_gap_analysis.csv`.

## 11. Explainability cards and human review
- 80 explainability cards: `topic_explainability_cards.csv` and `topic_explainability_cards/LT_001.md ... LT_080.md`.
- `human_review_template.csv` (80 rows, one per leaf) has the human columns left **blank**; no human review has been performed or simulated.

## 12. LLM usage (Gemini) and historical Claude spend
| item | value |
|---|---|
| provider / model | gemini / gemini-3.1-flash-lite |
| **Sprint 4 Claude calls** | **0** |
| **Sprint 4 Gemini real generation calls** | **10** (normal 10, repair 0, retry 0, split 0; hard cap 12) |
| cache hits (no call made) | 5 |
| **Sprint 4 Gemini tokens** | **36,351 input / 26,236 output** |
| Sprint 3 Claude historical spend (estimate) | $1.6195 (19 calls; rebuilt from the Sprint 3 token logs at $2/$10 per MTok) |

No dollar cost is claimed for Gemini because the configured billing tier is not known to this pipeline. Per-call tokens, status and timing are in
`api_usage.csv` / `api_ledger.jsonl`. 
