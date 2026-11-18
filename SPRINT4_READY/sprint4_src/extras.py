"""Sprint 4 advancements (all deterministic, no Gemini): hierarchy stability, O*NET coverage-gap analysis,
explainability cards, human-review template, API usage export."""
from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from . import hierarchy as H
from .evidence import Frozen, l2

SEEDS = (101, 202, 303, 404, 505)
FRACTION = 0.8


def jl(x):
    return json.loads(x) if isinstance(x, str) else x


# ------------------------------------------------------------------ 1. hierarchy stability
def _match_persistence(final_lab, pert_lab):
    """One-to-one Hungarian matching of perturbed parents to final parents on max Jaccard of child-leaf sets.
    Returns a boolean array: leaf stays in its matched equivalent parent."""
    fk, pk = sorted(set(final_lab)), sorted(set(pert_lab))
    fl, pl = np.asarray(final_lab), np.asarray(pert_lab)
    J = np.zeros((len(pk), len(fk)))
    for i, p in enumerate(pk):
        sp = set(np.where(pl == p)[0])
        for j, f in enumerate(fk):
            sf = set(np.where(fl == f)[0])
            J[i, j] = len(sp & sf) / len(sp | sf)
    r, c = linear_sum_assignment(-J)
    mapping = {pk[i]: fk[j] for i, j in zip(r, c)}
    return np.array([mapping.get(pl[i]) == fl[i] for i in range(len(fl))])


def hierarchy_stability(fz: Frozen, st, linkage_name: str, k2: int, k1: int, seeds=SEEDS, frac=FRACTION):
    n = len(fz.topic_ids)
    final2 = [st.l2[t] for t in fz.topic_ids]
    final1 = [st.l1_of_l2[st.l2[t]] for t in fz.topic_ids]
    p2_runs, p1_runs, pers2, pers1, sizes_log = [], [], [], [], []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        cent = np.zeros_like(fz.centroids)
        tot = 0
        for i, t in enumerate(fz.topic_ids):        # every leaf is sampled independently; none is ever dropped
            rows = fz.tid_to_rows[t]
            m = max(1, int(round(frac * len(rows))))
            pick = rng.choice(rows, size=m, replace=False)
            cent[i] = l2(fz.emb[pick].mean(0))
            tot += m
        sizes_log.append(tot)
        Z = linkage(cent, method="ward") if linkage_name == "ward" else H.build_linkage(H.cosine_distance(cent))
        a2, a1 = H.cut(Z, k2), H.cut(Z, k1)
        p2_runs.append(a2)
        p1_runs.append(a1)
        pers2.append(_match_persistence(final2, a2))
        pers1.append(_match_persistence(final1, a1))
    pers2, pers1 = np.array(pers2), np.array(pers1)      # trials x leaves
    l2p, l1p = pers2.mean(0), pers1.mean(0)
    pair = [(i, j) for i in range(len(seeds)) for j in range(i + 1, len(seeds))]
    ari = lambda runs, ref=None: ([adjusted_rand_score(runs[i], runs[j]) for i, j in pair] if ref is None
                                  else [adjusted_rand_score(ref, r) for r in runs])
    a2p, a1p = ari(p2_runs), ari(p1_runs)
    a2f, a1f = ari(p2_runs, final2), ari(p1_runs, final1)
    df = pd.DataFrame({"topic_id": fz.topic_ids,
                       "topic_name": [st.labels[t]["topic_name"] for t in fz.topic_ids],
                       "level2_id": final2, "level2_name": [st.l2_info[p]["name"] for p in final2],
                       "level1_id": final1, "level1_name": [st.l1_info[d]["name"] for d in final1],
                       "level2_trials_persisted": pers2.sum(0).astype(int), "level2_parent_persistence": np.round(l2p, 4),
                       "level1_trials_persisted": pers1.sum(0).astype(int), "level1_parent_persistence": np.round(l1p, 4)})
    for k, sd in enumerate(seeds):
        df[f"l2_persisted_seed_{sd}"] = pers2[k].astype(int)
        df[f"l1_persisted_seed_{sd}"] = pers1[k].astype(int)
    summ = {"seeds": list(seeds), "sample_fraction_per_leaf": frac, "sampling": "without replacement, independently inside every leaf",
            "leaves_kept": n, "tasks_sampled_per_trial": sizes_log, "method": linkage_name,
            "target_level2_count": k2, "target_level1_count": k1,
            "final_level2_count": len(set(final2)), "final_level1_count": len(set(final1)),
            "perturbed_level2_counts": [len(set(x)) for x in p2_runs], "perturbed_level1_counts": [len(set(x)) for x in p1_runs],
            "matching": "one-to-one Hungarian on maximum Jaccard of child-leaf sets, independently per level",
            "mean_level1_persistence": round(float(l1p.mean()), 4), "mean_level2_persistence": round(float(l2p.mean()), 4),
            "level1_ari_pairwise_mean": round(float(np.mean(a1p)), 4), "level2_ari_pairwise_mean": round(float(np.mean(a2p)), 4),
            "level1_ari_vs_final_mean": round(float(np.mean(a1f)), 4), "level2_ari_vs_final_mean": round(float(np.mean(a2f)), 4),
            "leaves_level1_persistence_below_0.60": int((l1p < 0.60).sum()),
            "leaves_level2_persistence_below_0.60": int((l2p < 0.60).sum()),
            "note": "validation only - the final hierarchy was not changed because of this test"}
    return df, summ


# ------------------------------------------------------------------ 2. O*NET coverage-gap analysis
def gap_analysis(fz: Frozen, st, onet_df: pd.DataFrame, top10: pd.DataFrame):
    """Classes from quantiles of the observed distributions (no hand-set cut-offs).
    sim_weak   : top-1 <= P25(top-1) AND mean top-10 <= P25(mean top-10)         (consistently weak similarity profile)
    agree_weak : purity <= P25(purity) OR nn-agreement <= P25(nn-agreement)      (weak purity/agreement)
    coherent   : Gemini coherent & not mixed AND core-task cohesion >= P50       (representative tasks support ONE concept)
    limited    : Gemini NO_MATCH, or PARTIAL_MATCH with confidence <= P50(confidence)
    POSSIBLE_EMERGING_WORK = sim_weak & agree_weak & coherent & limited.
    WELL_COVERED = Gemini STRONG_MATCH & top-1 >= P50 & (purity >= P50 or agreement >= P50); everything else PARTIALLY_COVERED."""
    o = onet_df.set_index("topic_id").loc[fz.topic_ids]
    mean10 = top10.groupby("topic_id").cosine_similarity.mean().loc[fz.topic_ids]
    coh = fz.ev.set_index("topic_id").mean_sim_to_centroid.loc[fz.topic_ids]
    conf = o.semantic_alignment_confidence.astype(float)
    P = lambda s, q: float(np.percentile(s, q))
    th = {"top1_P25": P(o.top1_similarity, 25), "top1_P50": P(o.top1_similarity, 50), "mean_top10_P25": P(mean10, 25),
          "purity_P25": P(o.purity, 25), "purity_P50": P(o.purity, 50), "nn_agreement_P25": P(o.nn_agreement_top10, 25),
          "nn_agreement_P50": P(o.nn_agreement_top10, 50), "alignment_confidence_P50": P(conf, 50),
          "core_cohesion_P50": P(coh, 50)}
    rows = []
    for t in fz.topic_ids:
        r, L = o.loc[t], st.labels[t]
        sim_weak = r.top1_similarity <= th["top1_P25"] and mean10[t] <= th["mean_top10_P25"]
        agree_weak = r.purity <= th["purity_P25"] or r.nn_agreement_top10 <= th["nn_agreement_P25"]
        coherent = bool(L["coherent"]) and not L["mixed_topic"] and coh[t] >= th["core_cohesion_P50"]
        limited = r.alignment == "NO_MATCH" or (r.alignment == "PARTIAL_MATCH" and conf[t] <= th["alignment_confidence_P50"])
        well = (r.alignment == "STRONG_MATCH" and r.top1_similarity >= th["top1_P50"]
                and (r.purity >= th["purity_P50"] or r.nn_agreement_top10 >= th["nn_agreement_P50"]))
        cls = "POSSIBLE_EMERGING_WORK" if (sim_weak and agree_weak and coherent and limited) else "WELL_COVERED" if well else "PARTIALLY_COVERED"
        ev = (f"top1={r.top1_similarity:.3f} (P25 {th['top1_P25']:.3f}); mean_top10={mean10[t]:.3f} (P25 {th['mean_top10_P25']:.3f}); "
              f"purity={r.purity:.3f} (P25 {th['purity_P25']:.3f}); nn_agreement={r.nn_agreement_top10:.3f} (P25 {th['nn_agreement_P25']:.3f}); "
              f"core_cohesion={coh[t]:.3f} (P50 {th['core_cohesion_P50']:.3f}); gemini={r.alignment} conf={conf[t]:.2f}; "
              f"flags: sim_weak={sim_weak}, agree_weak={agree_weak}, coherent={coherent}, limited_gemini_coverage={limited}")
        base = r.reason if isinstance(r.reason, str) else ""
        if cls == "POSSIBLE_EMERGING_WORK":
            reason = ("Possible coverage gap requiring interpretation (coherent topic, consistently weak O*NET similarity, weak purity/agreement, "
                      "limited Gemini-judged coverage). This does not mean O*NET is outdated. Gemini: " + base)
        elif cls == "WELL_COVERED":
            reason = "Strong retrieved O*NET alignment on similarity, purity/agreement and Gemini judgment. Gemini: " + base
        else:
            reason = "Partial or mixed evidence of O*NET coverage. Gemini: " + base
        rows.append({"topic_id": t, "topic_name": st.labels[t]["topic_name"], "coverage_class": cls,
                     "top1_similarity": round(float(r.top1_similarity), 6), "mean_top5_similarity": round(float(r.top5_mean_similarity), 6),
                     "purity": round(float(r.purity), 4), "nearest_neighbor_agreement": round(float(r.nn_agreement_top10), 4),
                     "semantic_alignment": r.alignment, "semantic_alignment_confidence": round(float(conf[t]), 4),
                     "evidence": ev, "reason": reason})
    df = pd.DataFrame(rows)
    return df, {"thresholds_from_observed_distributions": {k: round(v, 6) for k, v in th.items()},
                "counts": {c: int((df.coverage_class == c).sum()) for c in ["WELL_COVERED", "PARTIALLY_COVERED", "POSSIBLE_EMERGING_WORK"]},
                "interpretation": "POSSIBLE_EMERGING_WORK marks possible coverage gaps that need human interpretation; it is not a claim that O*NET is outdated."}


# ------------------------------------------------------------------ 3. explainability cards
def _onet_matches(t, top10, onet_row, n=3):
    rows = top10[top10.topic_id == t]
    best = [int(x) for x in json.loads(onet_row.best_onet_task_ids)]
    order = list(rows[rows.onet_task_id.isin(best)].onet_task_id) + list(rows[~rows.onet_task_id.isin(best)].onet_task_id)
    out = []
    for tid in order[:n]:
        r = rows[rows.onet_task_id == tid].iloc[0]
        out.append(f"{r.onet_task} [{r.occupation_title}; O*NET task {r.onet_task_id}; cosine {r.cosine_similarity:.3f}"
                   f"{'; Gemini-selected' if tid in best else ''}]")
    return out + [""] * (n - len(out))


def explainability_cards(s, fz: Frozen, st, onet_df, top10, gap_df, stab_df):
    R = s.results
    ev = fz.ev.set_index("topic_id")
    o, g, sb = onet_df.set_index("topic_id"), gap_df.set_index("topic_id"), stab_df.set_index("topic_id")
    cdir = R / "topic_explainability_cards"
    if cdir.exists():
        shutil.rmtree(cdir)
    cdir.mkdir(parents=True)
    rows = []
    for t in fz.topic_ids:
        L, p = st.labels[t], st.l2[t]
        d = st.l1_of_l2[p]
        core, bound = jl(ev.loc[t].core_tasks), jl(ev.loc[t].boundary_tasks)
        core5 = (core + [""] * 5)[:5]
        bound3 = (bound + [""] * 3)[:3]
        kw = jl(ev.loc[t].ctfidf_keywords)
        m3 = _onet_matches(t, top10, o.loc[t])
        row = {"topic_id": t, "topic_name": L["topic_name"], "description": L["description"], "core_action": L["core_action"],
               "object_or_outcome": L["object_or_outcome"], "level2_family": st.l2_info[p]["name"], "level1_domain": st.l1_info[d]["name"],
               "topic_size": int(fz.sizes[fz.topic_index[t]]),
               **{f"representative_task_{i + 1}": core5[i] for i in range(5)}, **{f"boundary_task_{i + 1}": bound3[i] for i in range(3)},
               "top_keywords": "; ".join(kw), **{f"best_onet_match_{i + 1}": m3[i] for i in range(3)},
               "onet_coverage_class": g.loc[t, "coverage_class"], "label_confidence": L["confidence"],
               "onet_alignment_confidence": o.loc[t, "semantic_alignment_confidence"],
               "level1_parent_persistence": sb.loc[t, "level1_parent_persistence"],
               "level2_parent_persistence": sb.loc[t, "level2_parent_persistence"]}
        rows.append(row)
        md = f"""# {t} - {L['topic_name']}

**Description:** {L['description']}

**Work activity represented:** *{L['core_action']}* -> {L['object_or_outcome']}
- Include if: {L['include_if']}
- Exclude if: {L['exclude_if']}
- Size: {row['topic_size']} tasks | Keywords: {row['top_keywords']}

## Hierarchy
- **Level-1 domain:** {row['level1_domain']} (`{d}`)
- **Level-2 family:** {row['level2_family']} (`{p}`)
- Hierarchy persistence under 5 perturbation trials: Level-1 **{row['level1_parent_persistence']:.2f}**, Level-2 **{row['level2_parent_persistence']:.2f}**

## Representative task evidence (closest to the topic centroid)
""" + "\n".join(f"{i + 1}. {x}" for i, x in enumerate(core5) if x) + """

## Boundary task evidence (least typical members)
""" + "\n".join(f"{i + 1}. {x}" for i, x in enumerate(bound3) if x) + f"""

## Top O*NET matches (deterministic retrieval; Gemini may only choose among the retrieved top 10)
""" + "\n".join(f"{i + 1}. {x}" for i, x in enumerate(m3) if x) + f"""

- **O*NET coverage class:** {row['onet_coverage_class']} (Gemini alignment: {o.loc[t, 'alignment']}; dominant concept: {o.loc[t, 'dominant_onet_concept'] or 'n/a'})
- O*NET top-1 cosine {o.loc[t, 'top1_similarity']:.3f}; purity {o.loc[t, 'purity']:.3f}; nearest-neighbour agreement {o.loc[t, 'nn_agreement_top10']:.3f}
- **Label confidence (Gemini):** {L['confidence']:.2f}  |  **O*NET alignment confidence (Gemini):** {row['onet_alignment_confidence']:.2f}
- Label coherent: {L['coherent']}; mixed topic: {L['mixed_topic']}; critic action: {st.last_action.get(t, 'NOT_REVIEWED')}{'; UNRESOLVED: ' + st.unresolved[t] if t in st.unresolved else ''}

*Coverage classes describe possible gaps in O*NET coverage for interpretation; they do not imply O*NET is outdated.*
"""
        (cdir / f"{t}.md").write_text(md, encoding="utf-8")
    pd.DataFrame(rows).to_csv(R / "topic_explainability_cards.csv", index=False)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 4. human review template
HUMAN_COLS = ["label_correct", "coherent", "boundary_ok", "over_merged", "under_merged", "suggested_name", "reviewer_notes"]


def human_review_template(s, fz: Frozen, st):
    ev = fz.ev.set_index("topic_id")
    rows = []
    for t in fz.topic_ids:
        p = st.l2[t]
        rows.append({"topic_id": t, "topic_name": st.labels[t]["topic_name"], "description": st.labels[t]["description"],
                     "level1_domain": st.l1_info[st.l1_of_l2[p]]["name"], "level2_family": st.l2_info[p]["name"],
                     "core_tasks": " || ".join(jl(ev.loc[t].core_tasks)), "boundary_tasks": " || ".join(jl(ev.loc[t].boundary_tasks)),
                     **{c: "" for c in HUMAN_COLS}})
    df = pd.DataFrame(rows)
    df.to_csv(s.results / "human_review_template.csv", index=False)
    return df


# ------------------------------------------------------------------ 5. API usage export (Gemini ledger; no dollar claims)
LEDGER_COLS = ["call_number", "provider", "model", "agent_role", "batch_id", "topic_count", "input_tokens", "output_tokens", "status",
               "cached", "error_type", "timestamp", "kind"]


def export_api_usage(s, led):
    rows = [{c: r.get(c) for c in LEDGER_COLS} for r in led.d["rows"]]
    pd.DataFrame(rows, columns=LEDGER_COLS).to_csv(s.results / "api_usage.csv", index=False)
    with open(s.results / "api_ledger.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + chr(10))
    return rows
