"""Algorithmic fitters for the matched-granularity benchmark (no Claude, no re-embedding).

K-Means / GMM accept the target count directly. Leiden and HDBSCAN do NOT: Leiden gets an adaptive resolution search
whose ACTUAL community count is reported next to the target; HDBSCAN gets a small focused grid with noise rejection.
BERTopic is run naturally (its own pipeline) and, separately, POST-HOC reduced with BERTopic's own reduce_topics().
Hyper-parameters reuse the Sprint 3 / advanced choices (k-means++ n_init=10, diag GMM n_init=3, cosine kNN graph,
RBConfiguration Leiden, UMAP min_dist=0 cosine).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import SEED

GMM_N_INIT, GMM_MAX_ITER, GMM_REG_COVAR = 3, 500, 1e-6
LEIDEN_EDGE_MIN_COSINE = 0.0
LEIDEN_K_NEIGHBORS = [10, 20, 30]
LEIDEN_RES_BOUNDS = (0.02, 50.0)
LEIDEN_MAX_EVALS = 14
HDB_GRID = {"min_cluster_size": [5, 8, 10, 15, 20], "min_samples": [1, 3, 5], "selection": ["eom", "leaf"]}
HDB_MAX_NOISE = 0.50          # Sprint 3 pathology threshold: more noise than this is rejected, never "matched"
POST_HOC_LABEL = "POST_HOC_REDUCED_TO_AGENTIC_GRANULARITY"
BERTOPIC_NATURAL = {"n_neighbors": 30, "n_components": 5, "min_dist": 0.0, "metric": "cosine",
                    "min_cluster_size": 10, "min_samples": 5}   # advanced-experiment best natural configuration


# ---------------------------------------------------------------- K-Means / GMM (exact counts)
def fit_kmeans(Z, k, seed=SEED):
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=int(k), init="k-means++", n_init=10, random_state=seed)
    return km, km.fit_predict(Z).astype(int)


def fit_gmm(Z, n, seed=SEED):
    from sklearn.mixture import GaussianMixture
    gm = GaussianMixture(n_components=int(n), covariance_type="diag", random_state=seed, n_init=GMM_N_INIT,
                         max_iter=GMM_MAX_ITER, reg_covar=GMM_REG_COVAR)
    gm.fit(Z)
    post = gm.predict_proba(Z)
    return gm, post.argmax(1).astype(int), post


# ---------------------------------------------------------------- Leiden (no direct count parameter)
def knn_graph(X, k):
    import igraph as ig
    from scipy.sparse import csr_matrix
    S = X @ X.T
    np.fill_diagonal(S, -np.inf)
    nbr = np.argpartition(-S, k, axis=1)[:, :k]
    rows = np.repeat(np.arange(len(X)), k)
    cols = nbr.ravel()
    w = S[rows, cols]
    keep = w > LEIDEN_EDGE_MIN_COSINE
    a, b, w = np.minimum(rows, cols)[keep], np.maximum(rows, cols)[keep], w[keep]
    e = pd.DataFrame({"a": a, "b": b, "w": w}).groupby(["a", "b"], as_index=False).w.max()
    g = ig.Graph(n=len(X), edges=list(zip(e.a.tolist(), e.b.tolist())), directed=False)
    g.es["weight"] = e.w.tolist()
    W = csr_matrix((np.r_[e.w, e.w], (np.r_[e.a, e.b], np.r_[e.b, e.a])), shape=(len(X), len(X)))
    return g, W


def leiden(g, res, seed=SEED):
    import leidenalg
    part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, weights="weight",
                                    resolution_parameter=float(res), seed=seed, n_iterations=-1)
    lab = np.asarray(part.membership, dtype=int)
    return lab, float(g.modularity(part.membership, weights="weight"))


def leiden_nearest_count(g, target, seed=SEED, bounds=LEIDEN_RES_BOUNDS, max_evals=LEIDEN_MAX_EVALS, cache=None):
    """Bisection on log(resolution) toward `target` communities. Community count is only roughly monotone in the
    resolution, so every evaluated point is kept and the nearest ACTUAL count is returned (never forced)."""
    cache = {} if cache is None else cache
    lo, hi = math.log(bounds[0]), math.log(bounds[1])
    trace = []

    def ev(logr):
        r = round(math.exp(logr), 6)
        if r not in cache:
            lab, mod = leiden(g, r, seed)
            cache[r] = (lab, mod)
        lab, mod = cache[r]
        n = int(len(set(lab.tolist())))
        trace.append({"resolution": r, "communities": n, "modularity": mod})
        return n
    for _ in range(max_evals):
        mid = (lo + hi) / 2
        n = ev(mid)
        if n == target:
            break
        if n < target:
            lo = mid
        else:
            hi = mid
    T = pd.DataFrame(trace).drop_duplicates("resolution")
    T["abs_diff"] = (T.communities - target).abs()
    best = T.sort_values(["abs_diff", "resolution"]).iloc[0]
    lab, mod = cache[float(best.resolution)]
    return {"target_count": int(target), "actual_count": int(best.communities), "abs_difference": int(best.abs_diff),
            "resolution": float(best.resolution), "modularity": float(mod), "evaluations": len(T),
            "close_match": bool(best.abs_diff <= max(1, round(0.05 * target)))}, lab, T


def community_shares(W, labels):
    from scipy.sparse import csr_matrix
    n, K = len(labels), int(labels.max()) + 1
    O = csr_matrix((np.ones(n), (np.arange(n), labels)), shape=(n, K))
    Sh = (W @ O).toarray()
    tot = Sh.sum(1)
    return np.where(tot > 0, Sh[np.arange(n), labels] / np.where(tot > 0, tot, 1), np.nan)


def knn_vote(P, X, labels, k):
    S = P @ X.T
    nbr = np.argpartition(-S, k, axis=1)[:, :k]
    out = np.full(len(P), -1, dtype=int)
    for i in range(len(P)):
        votes = {}
        for j in nbr[i]:
            if labels[j] >= 0:
                votes[labels[j]] = votes.get(labels[j], 0.0) + float(S[i, j])
        if votes:
            out[i] = max(votes, key=votes.get)
    return out


# ---------------------------------------------------------------- HDBSCAN (no count parameter)
def fit_hdbscan(Z, mcs, ms, selection="eom"):
    import hdbscan
    cl = hdbscan.HDBSCAN(min_cluster_size=int(mcs), min_samples=int(ms), metric="euclidean",
                         cluster_selection_method=selection, prediction_data=True, core_dist_n_jobs=1)
    cl.fit(Z)
    return cl, cl.labels_.astype(int), np.asarray(cl.probabilities_, dtype=float)


def hdbscan_search(Z, targets, grid=HDB_GRID, max_noise=HDB_MAX_NOISE, fit=fit_hdbscan):
    """Fit a focused grid once; for each target return the nearest-count configuration whose noise <= max_noise.
    Configurations above the noise ceiling are REJECTED (reported, never selected)."""
    rows, labs = [], {}
    for mcs in grid["min_cluster_size"]:
        for ms in grid["min_samples"]:
            for sel in grid["selection"]:
                _, lab, _ = fit(Z, mcs, ms, sel)
                key = (mcs, ms, sel)
                labs[key] = lab
                rows.append({"min_cluster_size": mcs, "min_samples": ms, "selection": sel,
                             "clusters": int(len(set(lab[lab >= 0].tolist()))), "noise_share": float((lab < 0).mean())})
    G = pd.DataFrame(rows)
    G["rejected_excessive_noise"] = G.noise_share > max_noise
    picks = {}
    for t in targets:
        ok = G[~G.rejected_excessive_noise].assign(abs_diff=lambda d: (d.clusters - t).abs())
        if ok.empty:
            picks[t] = {"target_count": int(t), "feasible": False, "reason": f"every configuration exceeds {max_noise:.0%} noise"}
            continue
        b = ok.sort_values(["abs_diff", "noise_share"]).iloc[0]
        picks[t] = {"target_count": int(t), "feasible": True, "actual_count": int(b.clusters),
                    "abs_difference": int(b.abs_diff), "noise_share": float(b.noise_share),
                    "min_cluster_size": int(b.min_cluster_size), "min_samples": int(b.min_samples), "selection": b.selection,
                    "close_match": bool(b.abs_diff <= max(1, round(0.10 * t)))}
    return G, picks, labs


# ---------------------------------------------------------------- BERTopic (natural vs post-hoc reduced)
def umap_embed(X, nn, nc, seed=SEED):
    import umap
    u = umap.UMAP(n_neighbors=nn, n_components=nc, min_dist=BERTOPIC_NATURAL["min_dist"],
                  metric=BERTOPIC_NATURAL["metric"], random_state=seed)
    return u, np.asarray(u.fit_transform(X), dtype=np.float64)


def bertopic_natural(X, docs, cfg=BERTOPIC_NATURAL, seed=SEED, umap_fn=umap_embed, hdb_fn=fit_hdbscan):
    u, U = umap_fn(X, cfg["n_neighbors"], cfg["n_components"], seed)
    cl, lab, prob = hdb_fn(U, cfg["min_cluster_size"], cfg["min_samples"], "eom")
    return {"umap": u, "U": U, "hdbscan": cl, "labels": lab, "prob": prob, "variant": "NATURAL"}


def bertopic_reduce(docs, X, natural_labels, n_target, model_factory=None):
    """BERTopic's own reduce_topics() on the natural topics (manual BERTopic: precomputed labels + real T5 embeddings).
    Returns labels clearly marked POST_HOC_REDUCED_TO_AGENTIC_GRANULARITY, or a not-applicable record."""
    nat = int(len(set(natural_labels[natural_labels >= 0].tolist())))
    if nat <= n_target:
        return None, {"variant": POST_HOC_LABEL, "applicable": False, "natural_topics": nat, "target_count": int(n_target),
                      "reason": "natural topic count is not above the target - reduction can only merge topics"}
    if model_factory is None:
        def model_factory():
            from bertopic import BERTopic
            from bertopic.cluster import BaseCluster
            from bertopic.dimensionality import BaseDimensionalityReduction
            from sklearn.feature_extraction.text import CountVectorizer
            return BERTopic(embedding_model=None, umap_model=BaseDimensionalityReduction(), hdbscan_model=BaseCluster(),
                            vectorizer_model=CountVectorizer(stop_words="english", ngram_range=(1, 2)), verbose=False)
    tm = model_factory()
    tm.fit(list(docs), embeddings=np.asarray(X), y=np.asarray(natural_labels))
    has_out = bool((natural_labels < 0).any())
    tm.reduce_topics(list(docs), nr_topics=int(n_target) + (1 if has_out else 0))   # nr_topics counts the -1 topic
    red = np.asarray(tm.topics_, dtype=int)
    actual = int(len(set(red[red >= 0].tolist())))
    mapping = pd.crosstab(natural_labels, red).idxmax(axis=1).to_dict()
    return red, {"variant": POST_HOC_LABEL, "applicable": True, "natural_topics": nat, "target_count": int(n_target),
                 "actual_count": actual, "abs_difference": abs(actual - int(n_target)), "natural_to_reduced": mapping,
                 "note": "post-hoc reduction of BERTopic's natural topics; NOT BERTopic's natural discovery"}


def bertopic_keywords(docs, labels, top_n=8):
    """c-TF-IDF keywords for inspection only (never used as taxonomy names)."""
    from sklearn.feature_extraction.text import CountVectorizer
    ids = sorted(set(labels[labels >= 0].tolist()))
    if not ids:
        return {}
    cv = CountVectorizer(stop_words="english", ngram_range=(1, 2))
    M = cv.fit_transform(list(docs))
    vocab = np.array(cv.get_feature_names_out())
    tf = np.vstack([np.asarray(M[labels == c].sum(0)).ravel() for c in ids]).astype(float)
    tf = tf / np.clip(tf.sum(1, keepdims=True), 1, None)
    sc = tf * np.log(1 + len(ids) / np.clip((tf > 0).sum(0), 1, None))
    return {c: ", ".join(vocab[np.argsort(-sc[i])[:top_n]]) for i, c in enumerate(ids)}
