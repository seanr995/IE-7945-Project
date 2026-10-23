"""One shared evaluator for every partition (agentic, hybrid, algorithmic).

Definitions are taken from the Sprint 3 script (`s3`): semantic cohesion in the ORIGINAL embedding space with a
size-preserving random baseline (lift), silhouette in the Sprint 3 PCA space, size statistics, granularity score,
provider nearest-neighbour test, centroid membership rule. New here: topic-level persistence, Hungarian alignment,
co-association. None of these is an accuracy measure.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


def to_int_labels(values, unassigned=("", None, "UNASSIGNED", "OUTLIER", -1)):
    """Map arbitrary topic labels to 0..K-1 ints; unassigned -> -1. Returns (labels, id_list)."""
    vals = list(values)
    uniq = sorted({v for v in vals if v not in unassigned and not (isinstance(v, float) and np.isnan(v))}, key=str)
    m = {v: i for i, v in enumerate(uniq)}
    return np.array([m.get(v, -1) for v in vals], dtype=int), uniq


def paraphrase(labels, pred, paired_rows):
    own = labels[paired_rows]
    valid = own >= 0
    same = (pred == own) & valid
    lab = labels[labels >= 0]
    shares = np.bincount(lab) / len(lab) if len(lab) else np.array([1.0])
    chance = float((shares ** 2).sum())
    rate = float(same[valid].mean()) if valid.any() else None
    return {"paraphrase_consistency": rate, "paraphrase_pairs": int(valid.sum()),
            "paraphrase_wording_unassigned": int((valid & (pred < 0)).sum()), "paraphrase_chance": chance,
            "paraphrase_kappa": (rate - chance) / (1 - chance) if rate is not None and chance < 1 else None}, same, valid


def evaluate_partition(s3, labels, X, Z, P, paired_rows, nn_pairs, providers, T_common, native_conf=None,
                       pred_native=None, method="other"):
    labels = np.asarray(labels, dtype=int)
    r = dict(s3.size_stats(labels))
    r["coverage"] = r["assigned"] / len(labels)
    im, sil = s3.internal_metrics(Z, labels)
    r.update(im)
    coh, own, ids, C = s3.cohesion(X, labels)
    r.update(coh)
    base = s3.cohesion_baseline(X, labels)
    r["cohesion_random_baseline"] = base
    r["semantic_cohesion_lift"] = s3.lift(coh["semantic_cohesion"], base)
    own_all = own.copy()
    if (labels < 0).any() and len(ids):
        la = labels.copy()
        noise = la < 0
        la[noise] = ids[np.argmax(s3.sims_to_centroids(X[noise], C), axis=1)]
        c2, own_all, _, _ = s3.cohesion(X, la)
        r["semantic_cohesion_incl_unassigned"] = c2["semantic_cohesion"]
        r["semantic_cohesion_lift_incl_unassigned"] = s3.lift(c2["semantic_cohesion"], s3.cohesion_baseline(X, la))
    else:
        r["semantic_cohesion_incl_unassigned"] = coh["semantic_cohesion"]
        r["semantic_cohesion_lift_incl_unassigned"] = r["semantic_cohesion_lift"]
    empty = pd.DataFrame(columns=["row"])
    r.update(s3.provider_robustness(labels, method, None, None, X, None, empty, nn_pairs, ids, C, providers))
    g = r.get("nn_cross_minus_same_provider_same_topic")
    r["provider_nn_gap_abs"] = abs(g) if g is not None else None
    nc = np.asarray(native_conf, dtype=float) if native_conf is not None else np.full(len(labels), np.nan)
    mem, thr = s3.membership(X, labels, ids, C, nc, "other", nc)
    m = labels >= 0
    r["low_confidence_rate"] = float((mem.gap[m] < T_common).mean()) if m.any() else None
    r["centroid_inconsistency_rate"] = float(mem.centroid_inconsistent[m].mean()) if m.any() else None
    r["secondary_capability_rate"] = float(mem.has_secondary.mean())
    r["granularity_score"] = s3.granularity_score(r.get("n_clusters"), r.get("median_size"))
    if len(P) and len(ids):
        pc = ids[np.argmax(s3.sims_to_centroids(P, C), axis=1)]
        pr, _, _ = paraphrase(labels, pc, paired_rows)
        r["paraphrase_consistency_centroid"], r["paraphrase_kappa_centroid"] = pr["paraphrase_consistency"], pr["paraphrase_kappa"]
    arrays = {"labels": labels, "own_sim": own, "own_sim_incl": own_all, "sil": sil, "mem": mem}
    if pred_native is not None and len(P):
        pn, same, valid = paraphrase(labels, np.asarray(pred_native, dtype=int), paired_rows)
        r.update(pn)
        arrays["para_same"], arrays["para_valid"] = same, valid
    return r, arrays


# ---------------------------------------------------------------- stability + topic persistence
def subsample_runs(fit_labels, n, subs):
    runs = []
    for idx in subs:
        full = np.full(n, -2, dtype=int)
        full[idx] = np.asarray(fit_labels(idx), dtype=int)
        runs.append(full)
    return runs


def stability_from_runs(s3, runs):
    a_all, a_nn = [], []
    for i, j in combinations(range(len(runs)), 2):
        sh = (runs[i] != -2) & (runs[j] != -2)
        li, lj = runs[i][sh], runs[j][sh]
        a_all.append(s3.ari_pair(li, lj))
        both = (li >= 0) & (lj >= 0)
        if both.sum() > 1:
            a_nn.append(s3.ari_pair(li[both], lj[both]))
    return {"stability_ari": float(np.mean(a_all)), "stability_ari_sd": float(np.std(a_all, ddof=1)) if len(a_all) > 1 else None,
            "stability_ari_assigned": float(np.mean(a_nn)) if a_nn else None, "stability_pairs": len(a_all)}


def topic_persistence(full_labels, runs):
    """For every full-data topic: best Jaccard with any topic of each subsample run (on shared statements);
    per topic the median over runs. Returns (per-topic frame, summary)."""
    rows = []
    for t in sorted(set(full_labels[full_labels >= 0].tolist())):
        js = []
        for run in runs:
            inside = run != -2
            A = set(np.where((full_labels == t) & inside)[0])
            if not A:
                continue
            best = 0.0
            for b in set(run[inside & (run >= 0)].tolist()):
                B = set(np.where(run == b)[0])
                inter = len(A & B)
                if inter:
                    best = max(best, inter / len(A | B))
            js.append(best)
        rows.append({"topic": int(t), "size": int((full_labels == t).sum()),
                     "median_best_jaccard": float(np.median(js)) if js else 0.0,
                     "min_best_jaccard": float(np.min(js)) if js else 0.0})
    D = pd.DataFrame(rows)
    if D.empty:
        return D, {"persistence_median_jaccard": None, "persistence_share_j50": None, "persistence_share_j70": None}
    return D, {"persistence_median_jaccard": float(D.median_best_jaccard.median()),
               "persistence_share_j50": float((D.median_best_jaccard >= 0.5).mean()),
               "persistence_share_j70": float((D.median_best_jaccard >= 0.7).mean())}


def seed_ari(s3, runs):
    a = [s3.ari_pair(runs[i], runs[j]) for i, j in combinations(range(len(runs)), 2)]
    return {"seed_ari_mean": float(np.mean(a)) if a else None, "seed_ari_sd": float(np.std(a, ddof=1)) if len(a) > 1 else None}


# ---------------------------------------------------------------- alignment + co-association
def jaccard_matrix(la, lb):
    A = sorted(set(la[la >= 0].tolist()))
    B = sorted(set(lb[lb >= 0].tolist()))
    both = (la >= 0) & (lb >= 0)
    ct = pd.crosstab(la[both], lb[both]).reindex(index=A, columns=B, fill_value=0).values.astype(float)
    sa = np.array([(la == a).sum() for a in A], dtype=float)[:, None]
    sb = np.array([(lb == b).sum() for b in B], dtype=float)[None, :]
    return A, B, ct / np.clip(sa + sb - ct, 1, None), ct


def align(la, lb):
    """ARI / NMI on statements assigned in both + Hungarian one-to-one topic matching on Jaccard."""
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    both = (la >= 0) & (lb >= 0)
    out = {"rows_both_assigned": int(both.sum()),
           "ari": float(adjusted_rand_score(la[both], lb[both])) if both.sum() > 1 else None,
           "nmi": float(normalized_mutual_info_score(la[both], lb[both])) if both.sum() > 1 else None}
    A, B, J, ct = jaccard_matrix(la, lb)
    if len(A) and len(B):
        r, c = linear_sum_assignment(-J)
        mj = J[r, c]
        sizes = np.array([(la == A[i]).sum() for i in r], dtype=float)
        out.update({"matched_pairs": int(len(r)), "matched_jaccard_mean": float(mj.mean()),
                    "matched_jaccard_size_weighted": float((mj * sizes).sum() / sizes.sum()),
                    "matched_pairs_j50": int((mj >= 0.5).sum())})
        match = {A[i]: (B[j], float(J[i, j])) for i, j in zip(r, c)}
    else:
        match = {}
    return out, match, (A, B, J)


def coassociation(partitions: dict, n: int):
    """Mean co-membership over methods for every statement pair (methods where either is unassigned are skipped)."""
    num = np.zeros((n, n), dtype=np.float32)
    den = np.zeros((n, n), dtype=np.float32)
    for lab in partitions.values():
        lab = np.asarray(lab)
        ok = (lab >= 0).astype(np.float32)
        den += np.outer(ok, ok)
        ids = sorted(set(lab[lab >= 0].tolist()))
        if ids:
            O = np.zeros((n, len(ids)), dtype=np.float32)          # one-hot membership -> O O^T = co-membership
            pos = {t: k for k, t in enumerate(ids)}
            rows = np.where(lab >= 0)[0]
            O[rows, [pos[t] for t in lab[rows]]] = 1.0
            num += O @ O.T
    with np.errstate(invalid="ignore", divide="ignore"):
        A = np.where(den > 0, num / np.clip(den, 1, None), np.nan)
    np.fill_diagonal(A, np.nan)
    return A


def core_boundary(A, ref_labels, core=2 / 3, boundary=1 / 3):
    """Per statement: mean co-association with its reference-topic mates. CORE >= 2/3 of methods agree on average,
    BOUNDARY between 1/3 and 2/3, METHOD_DISAGREEMENT below 1/3."""
    rows = []
    for i in range(len(ref_labels)):
        t = ref_labels[i]
        if t < 0:
            rows.append({"row": i, "topic": -1, "mean_coassociation": None, "member_class": "UNASSIGNED"})
            continue
        mates = np.where((ref_labels == t) & (np.arange(len(ref_labels)) != i))[0]
        v = float(np.nanmean(A[i, mates])) if len(mates) else np.nan
        cls = "CORE" if v >= core else ("BOUNDARY" if v >= boundary else "METHOD_DISAGREEMENT")
        rows.append({"row": i, "topic": int(t), "mean_coassociation": v, "member_class": cls if not np.isnan(v) else "SINGLETON"})
    return pd.DataFrame(rows)
