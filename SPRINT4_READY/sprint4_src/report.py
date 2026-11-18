"""sprint4_evaluation_report.md"""
from __future__ import annotations

import json

import pandas as pd

from .config import utc
from .hierarchy import SCORE_WEIGHTS


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)


def write_report(s, fz, st, diag, onet_sum, onet_df, summary, preflight, diag_all=None):
    led, conv, hist = summary["ledger"], summary["convergence"], summary["historical_claude"]
    sel = diag[diag.selected].iloc[0]
    mock_note = ("\n> **MOCK RUN** - Gemini responses and O*NET embeddings in this report come from the offline test double, "
                 "not from the real model. Numbers are for pipeline validation only.\n") if s.mock else ""
    cand = diag[["linkage", "level2_count", "level1_count", "L2_silhouette", "L1_silhouette", "L2_cohesion", "L1_cohesion", "L2_min_children",
                 "L2_median_children", "L2_max_children", "L2_dominant_share", "L1_dominant_share", "L2_tiny_parents",
                 "L1_tiny_parents", "passes_gates", "score", "selected"]].copy()
    cand = cand.rename(columns={"L2_silhouette": "sil_L2", "L1_silhouette": "sil_L1", "L2_cohesion": "coh_L2", "L1_cohesion": "coh_L1",
                                "L2_min_children": "L2_min", "L2_median_children": "L2_med", "L2_max_children": "L2_max",
                                "L2_dominant_share": "dom_L2", "L1_dominant_share": "dom_L1", "L2_tiny_parents": "tiny_L2",
                                "L1_tiny_parents": "tiny_L1"})
    ac = onet_sum["alignment_counts"]
    rounds = conv["rounds"]
    unresolved = summary.get("unresolved", {})
    top5 = "\n".join(f"| {k} | {v['mean']} | {v['median']} | {v['min']} | {v['max']} |" for k, v in onet_sum["top5_similarity_distribution"].items())
    r_lines = "\n".join(f"- Round {r['round']}: {r['cases']} cases; accepts {r['accepts']}, leaf renames {r['leaf_renames']}, parent renames "
                        f"{r['parent_renames']}, refinements {r['refinements']}, moves {r['moves']}, merges {r['merges']}, splits {r['splits']}, "
                        f"flagged {r['flagged']}, rejected {r['rejected']}; leaf-to-parent change {r['leaf_assignment_change_share']:.1%}; "
                        f"delta L2 cohesion {r['delta_L2_cohesion']:+.4f}" for r in rounds) or "- No problematic cases; critic not called."
    unres = "\n".join(f"- `{k}`: {v}" for k, v in unresolved.items()) or "- None."
    prov = ", ".join(sorted(st.provisional)) or "none"
    avg = diag[diag.linkage == "average"]
    wrd = diag[diag.linkage == "ward"]
    avg_txt = (f"All {len(avg)} average-linkage candidates {'failed' if not avg.passes_gates.any() else 'did not all pass'} the balance gates "
               f"(passing: {int(avg.passes_gates.sum())}/{len(avg)}); level-1 dominant-domain task share ranged "
               f"{avg.L1_dominant_share.min():.2f}-{avg.L1_dominant_share.max():.2f} (gate <= 0.50), level-1 single-child domains "
               f"{int(avg.L1_tiny_parents.min())}-{int(avg.L1_tiny_parents.max())} (gate 0), level-2 single-leaf families "
               f"{int(avg.L2_tiny_parents.min())}-{int(avg.L2_tiny_parents.max())}. Ward linkage: {int(wrd.passes_gates.sum())}/{len(wrd)} passed.")
    fin = (diag_all[diag_all.stage != "search"].iloc[-1] if diag_all is not None and (diag_all.stage != "search").any() else None)
    NL = chr(10)
    if fin is not None:
        head = ("| level | parents | silhouette | cohesion | min/median/max children | min/median/max tasks | dominant share | tiny parents |"
                + NL + "|---|---|---|---|---|---|---|---|" + NL)
        body = NL.join(f"| {lv} | {int(fin[f'{lv}_k'])} | {fin[f'{lv}_silhouette']:.4f} | {fin[f'{lv}_cohesion']:.4f} | "
                       f"{int(fin[f'{lv}_min_children'])}/{fin[f'{lv}_median_children']:.1f}/{int(fin[f'{lv}_max_children'])} | "
                       f"{int(fin[f'{lv}_min_tasks'])}/{fin[f'{lv}_median_tasks']:.0f}/{int(fin[f'{lv}_max_tasks'])} | "
                       f"{fin[f'{lv}_dominant_share']:.3f} | {int(fin[f'{lv}_tiny_parents'])} |" for lv in ("L1", "L2"))
        fin_txt = head + body
    else:
        fin_txt = "n/a"
    sb, gp = summary["stability"], summary["gap"]
    gc = gp["counts"]
    sel_info = summary["selected"]
    mock_prefix = ""
    md = f"""# Sprint 4 Evaluation Report - Agentic Labeling, Hierarchy and O*NET Alignment

Generated {utc()}. Selected Sprint 3 model: Sentence-T5 + Leiden @ 80 topics (frozen).{mock_note}

## 1. Frozen leaf layer
- **80 frozen leaf topics**, 3,015 tasks. Leaf membership was loaded from `final_task_topic_membership_leiden80.csv` and is never
  modified (validated byte-for-byte on `topic_id` for every task). The LLM had no ability to move tasks.
- Embeddings: {summary['embedding_source']}.

## 2. Leaf evidence
- **Centroid**: mean of the L2-normalised Sentence-T5 vectors of the topic's tasks, re-normalised.
- **Core tasks**: the 10 member tasks with highest cosine to the centroid (ties by `statement_id`). **Boundary tasks**: the 5 lowest.
  (For topics with fewer than 15 tasks, boundary = bottom third and core = the rest, up to 10, so the two sets never overlap.)
- **c-TF-IDF keywords**: one pseudo-document per topic (all member task texts); unigrams+bigrams, English stop words;
  weight = (term count / topic word count) x ln(1 + A / f_term), A = mean words per topic, f_term = term count over all 80 topics.
  Top 10 per topic (`leaf_evidence.csv`).

## 3. Gemini leaf labeler
- {len(fz.topic_ids)} topics in batches of 20 (4 calls). Each topic received its size, its 7 most typical tasks, its 3 least typical tasks and 8 c-TF-IDF keywords (evidence files keep the full 10/5/10).
- Native structured output (Pydantic schema) with local validation and word limits; fields: topic_name, description, core_action, object_or_outcome, include_if, exclude_if, coherent, mixed_topic, confidence, reason.
  Local validation rejects vague/one-word labels; invalid records get ONE batched repair request (max 2 per run), otherwise FLAG_UNRESOLVED. Responses cached.
- Mixed topics flagged by Gemini: {int(sum(l['mixed_topic'] for l in st.labels.values()))}; mean label confidence
  {sum(l['confidence'] for l in st.labels.values()) / len(st.labels):.3f}.

## 4. Hierarchy search (deterministic, no LLM)
- Average-linkage agglomerative clustering (SciPy `linkage(method="average")`, equivalent to scikit-learn
  `AgglomerativeClustering(linkage="average", metric="precomputed")`) on the **cosine distance matrix of the 80 leaf centroids**.
  Level-2 counts tested: 12, 16, 20; level-1 counts tested: 4, 6, 8 (9 candidates per linkage; both levels are cuts of one dendrogram, so levels are nested).
  **Added candidate family:** the same grid with Ward linkage (Euclidean distance on the unit-norm centroids, monotone in cosine distance), because
  average linkage chains on this embedding space (see the candidate table: one level-1 domain absorbs most tasks). All 18 candidates use the identical metrics and gates and the
  selection is automatic.
- Each candidate is scored on silhouette, size-weighted within-parent cosine cohesion, min/median/max parent size, dominant-parent share and
  tiny-parent count. Gates: no level-1 domain with >50% of tasks, no single-child level-1 domain, at most 10% single-leaf level-2 families.
  Among passing candidates the score is {json.dumps(SCORE_WEIGHTS)} on min-max normalised metrics.

{_md_table(cand)}

- **Why average linkage was rejected:** {avg_txt}
- **Chosen: {sel.linkage} linkage, {int(sel.level1_count)} level-1 domains / {int(sel.level2_count)} level-2 families / 80 leaves.**
  Final structure after critic: {summary['level1']} / {summary['level2']} / 80.

**Final hierarchy metrics (after critic; Level-1 = broad work domains, Level-2 = work-activity families):**

{fin_txt}

## 5. Gemini hierarchy builder
- ONE call: names every level-2 family and level-1 domain of the numerically selected hierarchy from compact leaf evidence (id, name, core action,
  short description) and lists clearly misplaced children. Grouping is fixed by Python; Gemini only recommends (`misplaced_child_ids`).

## 6. O*NET retrieval
- **O*NET tasks embedded: {summary['onet_tasks']:,}** (O*NET 31.0 Task Statements, unique Task IDs), with the same
  `sentence-transformers/sentence-t5-base` model, embedded once and cached. A re-embedding check against the frozen Sprint 3
  vectors confirms the model identity (real runs).
- For each leaf centroid the cosine similarity to every O*NET task is computed and the **top 10 are selected by similarity alone**
  (`onet_top10_matches.csv`, 800 rows). Gemini never chose candidates.

## 7. Gemini O*NET evaluator and quantitative results
- 4 batched calls (20 topics each); Gemini saw 3 representative tasks and the top 5 of the 10 deterministically retrieved candidates and could only choose among them. Alignment counts: STRONG {ac['STRONG_MATCH']}, PARTIAL {ac['PARTIAL_MATCH']}, NO_MATCH {ac['NO_MATCH']}.
- **Thresholds derived from the observed distribution (nothing hand-set)**: noise floor tau_noise = 99th percentile of all
  80 x {summary['onet_tasks']:,} centroid-O*NET cosines = {onet_sum['threshold_noise_p99_background']}; weak-match split tau_split = Otsu threshold on the
  80 observed top-1 scores = {onet_sum['threshold_otsu_top1']}.
- **Coverage** (leaf topics with Gemini STRONG/PARTIAL alignment and top-1 above tau_noise): **{onet_sum['coverage_meaningful_alignment']:.1%}**.
- **Purity** (share of a topic's member tasks whose own nearest O*NET task carries the topic's dominant O*NET concept - same task or
  a shared Detailed Work Activity): mean **{onet_sum['purity_mean']:.3f}**, median {onet_sum['purity_median']:.3f}.
- **Nearest-neighbour agreement** (share of the 5 representative tasks whose own top-1 O*NET task is inside the centroid's top-10):
  mean **{onet_sum['nn_agreement_top10_mean']:.3f}**; exact top-1 agreement {onet_sum['nn_agreement_top1_exact_mean']:.3f}.
- Top-1 cosine: mean **{onet_sum['top1_mean']}**, median **{onet_sum['top1_median']}**.

| rank | mean | median | min | max |
|---|---|---|---|---|
{top5}

## 8. Critic and convergence
- Critic cases: mixed topics, low label confidence, weak O*NET alignment, low-purity topics, low-cohesion parents, overlapping parents,
  misplaced children and inconsistent boundary tasks; quartile cut-offs are computed from the observed distributions. ONE batched critic call (plus at most the shared batched repair); persistence from a preliminary stability run is one of the flags.
{r_lines}
- Stop reason: **{conv['stop_reason']}**.
- Provisional names (from SPLIT_PARENT, awaiting rename): {prov}.

### Unresolved cases
{unres}

## 9. Hierarchy stability (validation only; hierarchy not changed by it)
- 5 deterministic trials, seeds {sb['seeds']}. In every trial, **within each of the 80 leaves independently** 80% of member task embeddings
  were sampled without replacement (no leaf removed; the 3,015 frozen memberships were not touched), leaf centroids recomputed, and
  the hierarchy rebuilt with the final method ({sel_info['linkage']} linkage, {sel_info['level2']} level-2 / {sel_info['level1']} level-1 targets).
- Parents are compared as sets of child leaf IDs; perturbed parents are matched to final parents by one-to-one Hungarian matching on
  maximum Jaccard, independently per level. (The final hierarchy after critic has {sb['final_level2_count']} level-2 families and
  {sb['final_level1_count']} level-1 domains; perturbed runs have {sb['perturbed_level2_counts']} / {sb['perturbed_level1_counts']}.)

| metric | Level-1 | Level-2 |
|---|---|---|
| mean leaf parent persistence | **{sb['mean_level1_persistence']}** | **{sb['mean_level2_persistence']}** |
| ARI between perturbation runs (mean of 10 pairs) | {sb['level1_ari_pairwise_mean']} | {sb['level2_ari_pairwise_mean']} |
| ARI vs final hierarchy (mean of 5) | {sb['level1_ari_vs_final_mean']} | {sb['level2_ari_vs_final_mean']} |
| leaves with persistence < 0.60 | {sb['leaves_level1_persistence_below_0.60']} | {sb['leaves_level2_persistence_below_0.60']} |

Files: `hierarchy_stability.csv`, `hierarchy_stability_summary.json`.

## 10. O*NET coverage-gap analysis (possible gaps for interpretation; not a claim that O*NET is outdated)
- Classes use quantiles of the observed distributions of top-1 cosine, top-10 profile, purity, nearest-neighbour agreement and Gemini
  alignment confidence. POSSIBLE_EMERGING_WORK requires: coherent topic with cohesive core tasks, consistently weak similarity profile,
  weak purity or agreement, **and** limited Gemini-judged coverage.
- **WELL_COVERED {gc['WELL_COVERED']} / PARTIALLY_COVERED {gc['PARTIALLY_COVERED']} / POSSIBLE_EMERGING_WORK {gc['POSSIBLE_EMERGING_WORK']}** (of 80 leaves).
- Cut-offs: {json.dumps(gp['thresholds_from_observed_distributions'])}. File: `onet_gap_analysis.csv`.

## 11. Explainability cards and human review
- 80 explainability cards: `topic_explainability_cards.csv` and `topic_explainability_cards/LT_001.md ... LT_080.md`.
- `human_review_template.csv` (80 rows, one per leaf) has the human columns left **blank**; no human review has been performed or simulated.

## 12. LLM usage (Gemini) and historical Claude spend
| item | value |
|---|---|
| provider / model | {led['provider']} / {led['model']} |
| **Sprint 4 Claude calls** | **0** |
| **Sprint 4 Gemini real generation calls** | **{led['gemini_real_calls']}** (normal {led['normal_calls']}, repair {led['repair_calls']}, retry {led['retry_calls']}, split {led['split_calls']}; hard cap {led['hard_cap']}) |
| cache hits (no call made) | {led['cache_hits']} |
| **Sprint 4 Gemini tokens** | **{led['input_tokens']:,} input / {led['output_tokens']:,} output** |
| Sprint 3 Claude historical spend (estimate) | ${hist['estimated_cost_usd']:.4f} ({hist.get('claude_calls_made', 0)} calls; rebuilt from the Sprint 3 token logs at $2/$10 per MTok) |

No dollar cost is claimed for Gemini because the configured billing tier is not known to this pipeline. Per-call tokens, status and timing are in
`api_usage.csv` / `api_ledger.jsonl`. {'Mock run: no real Gemini call was made and the real ledger was not touched.' if s.mock else ''}
"""
    (s.results / "sprint4_evaluation_report.md").write_text(md, encoding="utf-8")
