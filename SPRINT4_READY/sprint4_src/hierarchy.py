"""Stage 3 & 9 helpers: average-linkage agglomerative hierarchy over the 80 leaf centroids + diagnostics."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.spatial.distance import squareform

from .evidence import Frozen, l2

L2_COUNTS = [12, 16, 20]
L1_COUNTS = [4, 6, 8]
SCORE_WEIGHTS = {"silhouette_L2": 0.2, "silhouette_L1": 0.2, "cohesion_L2": 0.2, "cohesion_L1": 0.2,
                 "dominant_share_penalty": 0.1, "tiny_parent_penalty": 0.1}


def cosine_distance(cent):
    D = 1.0 - cent @ cent.T
    D = np.clip((D + D.T) / 2, 0, 2)
    np.fill_diagonal(D, 0.0)
    return D


def build_linkage(D):
    return linkage(squareform(D, checks=False), method="average")


def cut(Z, k):
    return cut_tree(Z, n_clusters=k).ravel()


def parent_centroids(cent, sizes, labels):
    out = {}
    for p in np.unique(labels):
        m = labels == p
        out[int(p)] = l2((cent[m] * sizes[m, None]).sum(0))
    return out


def cohesion(cent, sizes, labels):
    """Size-weighted mean cosine of each leaf centroid to its parent's task-weighted centroid."""
    pc = parent_centroids(cent, sizes, labels)
    sims = np.array([cent[i] @ pc[int(labels[i])] for i in range(len(labels))])
    return float((sims * sizes).sum() / sizes.sum()), sims


def level_stats(cent, sizes, D, labels, min_children):
    from sklearn.metrics import silhouette_score
    k = len(np.unique(labels))
    sil = float(silhouette_score(D, labels, metric="precomputed")) if 2 <= k < len(labels) else float("nan")
    coh, _ = cohesion(cent, sizes, labels)
    kids = pd.Series(labels).value_counts()
    tasks = pd.Series(sizes).groupby(labels).sum()
    return {"k": k, "silhouette": sil, "cohesion": coh, "min_children": int(kids.min()), "median_children": float(kids.median()),
            "max_children": int(kids.max()), "min_tasks": int(tasks.min()), "median_tasks": float(tasks.median()),
            "max_tasks": int(tasks.max()), "dominant_share": float(tasks.max() / tasks.sum()),
            "tiny_parents": int((kids < min_children).sum())}


def search(fz: Frozen):
    """Candidate grid = {average linkage (required), ward linkage (added, see note)} x L2 {12,16,20} x L1 {4,6,8}.
    Note: on this data average linkage chains (one level-1 domain holds 87-98% of the tasks and several level-2 families are
    single leaves), so a Ward-linkage family (Euclidean on unit-norm centroids, monotone in cosine distance) is evaluated
    with the identical metrics and gates. Selection is automatic over all candidates."""
    cent, sizes = fz.centroids, fz.sizes.astype(float)
    D = cosine_distance(cent)
    Zs = {"average": build_linkage(D), "ward": linkage(cent, method="ward")}
    rows = []
    for fam, Z in Zs.items():
        for k2, k1 in itertools.product(L2_COUNTS, L1_COUNTS):
            l2lab, l1lab = cut(Z, k2), cut(Z, k1)
            a, b = level_stats(cent, sizes, D, l2lab, 2), level_stats(cent, sizes, D, l1lab, 2)
            row = {"stage": "search", "linkage": fam, "level2_count": k2, "level1_count": k1}
            row.update({f"L2_{k}": v for k, v in a.items()})
            row.update({f"L1_{k}": v for k, v in b.items()})
            rows.append(row)
    df = pd.DataFrame(rows)

    def mm(c, invert=False):
        v = df[c].astype(float)
        r = (v - v.min()) / (v.max() - v.min()) if v.max() > v.min() else v * 0 + 0.5
        return 1 - r if invert else r
    df["score"] = (SCORE_WEIGHTS["silhouette_L2"] * mm("L2_silhouette") + SCORE_WEIGHTS["silhouette_L1"] * mm("L1_silhouette")
                   + SCORE_WEIGHTS["cohesion_L2"] * mm("L2_cohesion") + SCORE_WEIGHTS["cohesion_L1"] * mm("L1_cohesion")
                   + SCORE_WEIGHTS["dominant_share_penalty"] * (mm("L1_dominant_share", True) + mm("L2_dominant_share", True)) / 2
                   + SCORE_WEIGHTS["tiny_parent_penalty"] * (mm("L2_tiny_parents", True) + mm("L1_tiny_parents", True)) / 2)
    # hard gates: no level-1 domain may hold more than half the tasks, L1 has no single-child domains, and at most 10% of
    # L2 families may be single-leaf
    df["passes_gates"] = ((df.L1_dominant_share <= 0.5) & (df.L1_tiny_parents == 0) & (df.L2_tiny_parents <= 0.1 * df.level2_count))
    pool = df[df.passes_gates] if df.passes_gates.any() else df
    best = pool.assign(_f=(pool.linkage != "average")).sort_values(["score", "_f", "level2_count"], ascending=[False, True, True]).index[0]
    df["selected"] = False
    df.loc[best, "selected"] = True
    sel = df.loc[best]
    return df, Zs[sel.linkage], int(sel.level2_count), int(sel.level1_count)


def order_labels(labels, sizes):
    """Relabel clusters 0..k-1 by descending task count (deterministic ids)."""
    tasks = pd.Series(sizes).groupby(labels).sum().sort_values(ascending=False, kind="stable")
    m = {old: new for new, old in enumerate(tasks.index)}
    return np.array([m[x] for x in labels])


def diagnostics_row(fz: Frozen, l2_assign: dict, l1_of_l2: dict, stage: str) -> dict:
    """Diagnostics of the CURRENT hierarchy. l2_assign: leaf topic_id -> L2 parent id; l1_of_l2: L2 id -> L1 id."""
    cent, sizes = fz.centroids, fz.sizes.astype(float)
    D = cosine_distance(cent)
    ids2 = sorted(set(l2_assign.values()))
    lab2 = np.array([ids2.index(l2_assign[t]) for t in fz.topic_ids])
    l1_leaf = {t: l1_of_l2[l2_assign[t]] for t in fz.topic_ids}
    ids1 = sorted(set(l1_leaf.values()))
    lab1 = np.array([ids1.index(l1_leaf[t]) for t in fz.topic_ids])
    a, b = level_stats(cent, sizes, D, lab2, 2), level_stats(cent, sizes, D, lab1, 2)
    row = {"stage": stage, "level2_count": a["k"], "level1_count": b["k"]}
    row.update({f"L2_{k}": v for k, v in a.items()})
    row.update({f"L1_{k}": v for k, v in b.items()})
    return row


def parent_table(fz: Frozen, l2_assign: dict) -> pd.DataFrame:
    """Per-L2-parent cohesion and nearest-other-parent similarity (used by the critic case selector)."""
    cent, sizes = fz.centroids, fz.sizes.astype(float)
    ids2 = sorted(set(l2_assign.values()))
    lab = np.array([ids2.index(l2_assign[t]) for t in fz.topic_ids])
    pc = parent_centroids(cent, sizes, lab)
    P = np.stack([pc[i] for i in range(len(ids2))])
    PS = P @ P.T
    np.fill_diagonal(PS, -1)
    rows = []
    for i, pid in enumerate(ids2):
        m = lab == i
        sims = cent[m] @ pc[i]
        j = int(np.argmax(PS[i]))
        rows.append({"parent_id": pid, "n_children": int(m.sum()), "n_tasks": int(sizes[m].sum()),
                     "cohesion": float((sims * sizes[m]).sum() / sizes[m].sum()), "min_child_sim": float(sims.min()),
                     "nearest_parent_id": ids2[j], "nearest_parent_sim": float(PS[i, j])})
    return pd.DataFrame(rows)


def leaf_parent_sims(fz: Frozen, l2_assign: dict) -> dict:
    cent, sizes = fz.centroids, fz.sizes.astype(float)
    ids2 = sorted(set(l2_assign.values()))
    lab = np.array([ids2.index(l2_assign[t]) for t in fz.topic_ids])
    pc = parent_centroids(cent, sizes, lab)
    return {t: float(cent[i] @ pc[lab[i]]) for i, t in enumerate(fz.topic_ids)}


def best_alternative_parent(fz: Frozen, l2_assign: dict, leaf: str):
    cent, sizes = fz.centroids, fz.sizes.astype(float)
    ids2 = sorted(set(l2_assign.values()))
    lab = np.array([ids2.index(l2_assign[t]) if t != leaf else -1 for t in fz.topic_ids])
    out = {}
    for i, pid in enumerate(ids2):
        m = lab == i
        if m.any():
            out[pid] = float(cent[fz.topic_index[leaf]] @ l2((cent[m] * sizes[m, None]).sum(0)))
    return out
