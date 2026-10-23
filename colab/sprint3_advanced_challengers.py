#!/usr/bin/env python
"""Sprint 3 - advanced challenger experiment on the FROZEN Sentence-T5 representation. Google Colab only.

Question: does a more advanced clustering / topic-discovery method materially outperform the existing
Sentence-T5 + K-Means baseline?  Principle: COMPLEXITY MUST EARN ITS PLACE.

    BASELINE      Sentence-T5 + K-Means         (configuration read from the Sprint 3 result files)
    CHALLENGER 1  Sentence-T5 + BERTopic        (T5 embeddings -> UMAP -> HDBSCAN -> BERTopic c-TF-IDF representation)
    CHALLENGER 2  Sentence-T5 + Leiden          (cosine kNN graph on the original T5 embeddings)
    CHALLENGER 3  Sentence-T5 + GMM             (diagonal Gaussian mixture on the Sprint 3 PCA space)

    !python "/content/drive/MyDrive/CAPESTONE PROJECT/sprint3_advanced_challengers.py" \
      --project-dir "/content/drive/MyDrive/CAPESTONE PROJECT" \
      --output "/content/drive/MyDrive/CAPESTONE PROJECT/sprint3_advanced_results"

Nothing is re-embedded and no API is called. The Sentence-T5 embeddings, their row mapping, the cross-provider
paired-wording embeddings and run_manifest.json are read from <project-dir>/sprint3_results and validated against the
frozen input; any mismatch stops the run.

Fairness: every metric definition (semantic cohesion + chance baseline + lift, silhouette in the Sprint 3 PCA space,
cluster-size statistics, the provider nearest-neighbour test, the centroid membership rule, the 5 x 80% subsamples) is
IMPORTED from the Sprint 3 script (<project-dir>/sprint3_colab_experiment.py), so baseline and challengers are scored by
identical code. The baseline is refit with its recorded parameters and checked against the recorded Sprint 3 metrics.

All stages are checkpointed in <output>/checkpoints; re-running the same command resumes. Figures and reports are
best-effort and never discard completed work. ZIP: <project-dir>/Sprint3_Advanced_Challenger_Results.zip
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import hashlib
import importlib.metadata as md
import importlib.util
import json
import math
import pickle
import platform
import subprocess
import sys
import time
import traceback
import warnings
import zipfile
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ====================================================================== configuration (pre-declared, not tuned)
SEED = 42
T5_MODEL = "sentence-transformers/sentence-t5-base"
T5_KEY = "sentence_t5"
BASELINE_EXPERIMENT = "EXP_3_T5_KMEANS"
SPRINT3_SCRIPT_SHA256 = "c2cbc1bf58126876106b19538769927e2f6cc6b687e9d014818ffb6e7c747a12"

BERTOPIC_GRID = {"n_neighbors": [15, 30], "n_components": [5, 10], "min_dist": 0.0, "metric": "cosine",
                 "min_cluster_size": [10, 20, 30], "min_samples": [5, 10]}
LEIDEN_GRID = {"k_neighbors": [10, 20, 30], "resolution": [0.5, 1.0, 1.5, 2.0]}
LEIDEN_EDGE_MIN_COSINE = 0.0          # kNN edges with cosine <= 0 are dropped (documented weak-edge threshold)
GMM_GRID = {"n_components": [60, 80, 100, 120, 140], "covariance_type": "diag"}
GMM_N_INIT, GMM_MAX_ITER, GMM_REG_COVAR = 3, 500, 1e-6
STABILITY_SEEDS = [101, 202, 303, 404, 505]
BERTOPIC_SEED_RUNS = 3                # UMAP seed restarts (costly) - first 3 of STABILITY_SEEDS
N_BOOTSTRAP = 100
NATIVE_LOW_CONF = {"kmeans": ("ratio confidence 1 - d_own/d_second", 0.10), "bertopic": ("HDBSCAN probability", 0.50),
                   "leiden": ("edge-weight share to own community", 0.50), "gmm": ("max posterior", 0.50)}
GMM_SECONDARY_FLOOR = 0.05            # numerical floor: posteriors below this are treated as no membership
# Selection inside each method: Sprint 3 hierarchy, but on COVERAGE-ADJUSTED cohesion lift (outliers assigned to their
# nearest centroid for the metric only) so that dropping hard statements is never rewarded.
HIERARCHY = [("semantic_cohesion_lift_incl_noise", 0.02), ("stability_ari", 0.05), ("granularity_score", 0.0),
             ("robustness", 0.05), ("silhouette", 0.01)]
# Replacement rule (complexity must earn its place): a challenger replaces K-Means only with >= 1 meaningful gain and
# no unacceptable cost. Gains/costs are measured against the refit baseline under the shared evaluator.
GAIN_TOL = {"semantic_cohesion_lift_incl_noise": (0.02, "higher"), "stability_ari": (0.05, "higher"),
            "paraphrase_kappa": (0.05, "higher"), "provider_nn_gap_abs": (0.05, "lower"),
            "low_confidence_rate_common": (0.05, "lower"), "granularity_score": (0.25, "higher")}
COST_TOL = {"semantic_cohesion_lift_incl_noise": 0.02, "stability_ari": 0.05, "paraphrase_kappa": 0.05,
            "coverage": 0.10}
COMPLEXITY = {"kmeans": {"tuned_hyperparameters": 1, "extra_dependencies": "none (scikit-learn)", "stochastic_stages": 1},
              "bertopic": {"tuned_hyperparameters": 4, "extra_dependencies": "bertopic, umap-learn, hdbscan", "stochastic_stages": 1},
              "leiden": {"tuned_hyperparameters": 2, "extra_dependencies": "python-igraph, leidenalg", "stochastic_stages": 1},
              "gmm": {"tuned_hyperparameters": 1, "extra_dependencies": "none (scikit-learn)", "stochastic_stages": 1}}
METHOD_LABEL = {"kmeans": "Sentence-T5 + K-Means", "bertopic": "Sentence-T5 + BERTopic", "leiden": "Sentence-T5 + Leiden",
                "gmm": "Sentence-T5 + GMM"}
RECOMMENDATION = {"kmeans": "KEEP T5 + KMEANS", "bertopic": "USE T5 + BERTOPIC", "leiden": "USE T5 + LEIDEN", "gmm": "USE T5 + GMM"}
DEPENDENCIES = [("numpy", "numpy"), ("pandas", "pandas"), ("scipy", "scipy"), ("sklearn", "scikit-learn"),
                ("matplotlib", "matplotlib"), ("umap", "umap-learn"), ("hdbscan", "hdbscan"), ("bertopic", "bertopic"),
                ("igraph", "python-igraph"), ("leidenalg", "leidenalg")]
CONFIG_FINGERPRINT = hashlib.sha256(json.dumps([SEED, BERTOPIC_GRID, LEIDEN_GRID, LEIDEN_EDGE_MIN_COSINE, GMM_GRID,
                                                GMM_N_INIT, STABILITY_SEEDS, BERTOPIC_SEED_RUNS, "adv-v1"]).encode()).hexdigest()[:16]
T0 = time.time()
np = pd = s3 = None


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:6.0f}s] {msg}", flush=True)


def stop(msg):
    print(f"\nSTOP: {msg}\n", file=sys.stderr, flush=True)
    sys.exit(1)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dependencies(skip):
    missing = [pip for mod, pip in DEPENDENCIES if importlib.util.find_spec(mod) is None]
    if missing and not skip:
        log(f"installing missing packages: {missing}")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing])
        importlib.invalidate_caches()
    for mod in ("numpy", "pandas", "scipy", "sklearn", "umap", "hdbscan", "igraph", "leidenalg"):
        if importlib.util.find_spec(mod) is None:
            stop(f"required package '{mod}' is not installed. In Colab run: !pip install {dict(DEPENDENCIES)[mod]}")


def pkg_version(name):
    try:
        return md.version(name)
    except Exception:
        return None


def fnum(x, nd=3):
    try:
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return "n/a"
        if isinstance(x, (bool, np.bool_)):
            return str(bool(x))
        if isinstance(x, (int, np.integer)):
            return f"{int(x):,}"
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def pct(x):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * float(x):.1f}%"


class Checkpoints:
    def __init__(self, root, fingerprint):
        self.dir = root / "checkpoints"
        self.dir.mkdir(parents=True, exist_ok=True)
        meta, want = self.dir / "meta.json", {"fingerprint": fingerprint}
        if meta.exists() and json.loads(meta.read_text() or "{}") != want:
            log("checkpoint fingerprint changed -> clearing checkpoints")
            for f in list(self.dir.glob("*.pkl")) + list(self.dir.glob("*.npy")):
                f.unlink()
        meta.write_text(json.dumps(want))

    def get(self, name):
        p = self.dir / f"{name}.pkl"
        if p.exists():
            try:
                with open(p, "rb") as f:
                    return pickle.load(f)
            except Exception:
                log(f"checkpoint {name} unreadable - recomputing")
        return None

    def put(self, name, obj):
        p = self.dir / f"{name}.pkl"
        with open(p.with_suffix(".tmp"), "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(p.with_suffix(".tmp"), p)
        return obj

    def array(self, name, fn):
        p = self.dir / f"{name}.npy"
        if p.exists():
            return np.load(p)
        a = fn()
        np.save(p, a)
        return a


# ====================================================================== PART 1 - load + validate existing artifacts
def load_sprint3_module(project):
    path = project / "sprint3_colab_experiment.py"
    if not path.exists():
        stop(f"{path} not found. The challenger experiment imports the Sprint 3 metric definitions from it so that the "
             f"baseline and challengers are scored identically. Upload sprint3_colab_experiment.py to {project}.")
    sha = sha256_file(path)
    spec = importlib.util.spec_from_file_location("sprint3_primary", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.lazy_imports()
    if sha != SPRINT3_SCRIPT_SHA256:
        log(f"WARNING: sprint3_colab_experiment.py SHA-256 {sha[:16]}... differs from the validated version "
            f"{SPRINT3_SCRIPT_SHA256[:16]}... - metric definitions are taken from the file on Drive (recorded in manifest)")
    return mod, sha


def validate_inputs(project):
    s3r = project / "sprint3_results"
    files = {"input": project / "sprint3_task_input.csv",
             "embeddings": s3r / "embeddings" / "sentence_t5_embeddings.npy",
             "mapping": s3r / "embeddings" / "sentence_t5_embedding_mapping.csv",
             "embedding_meta": s3r / "embeddings" / "sentence_t5_meta.json",
             "paired_embeddings": s3r / "embeddings" / "sentence_t5_paired_text_embeddings.npy",
             "paired_mapping": s3r / "embeddings" / "sentence_t5_paired_text_mapping.csv",
             "manifest": s3r / "run_manifest.json",
             "clustering_experiments": s3r / "evaluation" / "clustering_experiments.csv"}
    missing = [f"{k}: {v}" for k, v in files.items() if not v.exists()]
    if missing:
        stop("required Sprint 3 artifacts are missing (embeddings are NOT regenerated):\n  " + "\n  ".join(missing))
    checks = {}
    input_sha = sha256_file(files["input"])
    manifest = json.loads(files["manifest"].read_text())
    if input_sha != manifest.get("input_sha256"):
        stop(f"input SHA-256 {input_sha} != Sprint 3 manifest input_sha256 {manifest.get('input_sha256')}")
    if input_sha != s3.EXPECTED_INPUT_SHA256:
        stop(f"input SHA-256 {input_sha} is not the frozen Sprint 3 input {s3.EXPECTED_INPUT_SHA256}")
    checks["input_sha256_matches_manifest_and_frozen"] = True
    df = pd.read_csv(files["input"], dtype=str, keep_default_na=False, encoding="utf-8")
    if not df.statement_id.is_unique or (df.statement_text.str.strip() == "").any():
        stop("input has duplicate statement_id or blank statement_text")
    if len(df) != manifest.get("row_count"):
        stop(f"input rows {len(df)} != manifest row_count {manifest.get('row_count')}")
    checks["input_rows"] = len(df)
    X = np.load(files["embeddings"]).astype(np.float32)
    mp = pd.read_csv(files["mapping"], dtype=str, keep_default_na=False)
    meta = json.loads(files["embedding_meta"].read_text())
    if X.shape[0] != len(df) or len(mp) != len(df):
        stop(f"embedding rows {X.shape[0]} / mapping rows {len(mp)} != input rows {len(df)}")
    if mp.embedding_row.astype(int).tolist() != list(range(len(df))):
        stop("embedding mapping rows are not 0..n-1 in order")
    if mp.statement_id.tolist() != df.statement_id.tolist():
        bad = int((mp.statement_id.values != df.statement_id.values).sum())
        stop(f"embedding mapping is misaligned with the input ({bad} rows differ in statement_id order)")
    if meta.get("model") != T5_MODEL or meta.get("input_sha256") != input_sha or meta.get("ids_hash") != s3.h(*df.statement_id):
        stop(f"sentence_t5_meta.json does not describe these embeddings (model/input/ids hash mismatch): {meta}")
    norms = np.linalg.norm(X, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        stop(f"Sentence-T5 embeddings are not L2-normalised (norm range {norms.min():.4f}-{norms.max():.4f})")
    checks.update({"embedding_shape": list(X.shape), "mapping_aligned": True, "embedding_meta_matches": True,
                   "embeddings_normalised": True})
    # paired Groq wordings - recomputed from the input exactly as in Sprint 3, then checked against the saved mapping
    pt = df.cross_provider_paired_text.fillna("").str.strip()
    norm_col = df.normalized_text if "normalized_text" in df.columns else df.statement_text.map(s3.norm_text)
    pm = (pt != "") & (pt.map(s3.norm_text) != norm_col)
    paired = pd.DataFrame({"row": np.where(pm)[0], "statement_id": df.statement_id[pm].values, "paired_text": pt[pm].values})
    P = np.load(files["paired_embeddings"]).astype(np.float32)
    pmp = pd.read_csv(files["paired_mapping"], dtype=str, keep_default_na=False)
    if len(pmp) != len(paired) or P.shape[0] != len(paired) or pmp.statement_id.tolist() != paired.statement_id.tolist():
        stop("paired-wording embeddings/mapping do not match the paired rows derived from the input")
    checks["paired_wordings"] = len(paired)
    hashes = {k: sha256_file(files[k]) for k in ("embeddings", "mapping", "paired_embeddings", "manifest")}
    return df, X, P, paired, manifest, files, input_sha, hashes, checks


# ====================================================================== PART 2 - baseline from the Sprint 3 results
def read_baseline(manifest, files):
    R = pd.read_csv(files["clustering_experiments"])
    chosen = manifest.get("chosen_configuration", {}) or {}
    cands = manifest.get("experiment_candidates", {}) or {}
    if str(chosen.get("config_id", "")).startswith(BASELINE_EXPERIMENT):
        cid, why = chosen["config_id"], "Sprint 3 overall winner (run_manifest.json chosen_configuration)"
    elif BASELINE_EXPERIMENT in cands:
        cid, why = cands[BASELINE_EXPERIMENT], (f"Sprint 3 {BASELINE_EXPERIMENT} candidate (the overall winner was "
                                                f"{chosen.get('config_id')} - not Sentence-T5 + K-Means)")
    else:
        stop(f"run_manifest.json has neither a {BASELINE_EXPERIMENT} winner nor candidate")
    row = R[R.config_id == cid]
    if len(row) != 1:
        stop(f"baseline config {cid} not found in clustering_experiments.csv")
    row = row.iloc[0].to_dict()
    if row.get("clustering_method") != "kmeans" or row.get("embedding_model") != T5_MODEL:
        stop(f"baseline {cid} is not Sentence-T5 + K-Means: {row.get('embedding_model')} / {row.get('clustering_method')}")
    k = int(row["param_k"])
    keys = ["n_clusters", "semantic_cohesion", "semantic_cohesion_lift", "stability_ari", "stability_ari_sd", "silhouette",
            "provider_pair_kappa", "paired_same_topic_rate", "nn_cross_minus_same_provider_same_topic",
            "low_confidence_rate", "min_size", "median_size", "max_size", "size_distribution", "seed_ari_mean",
            "cross_rep_ari", "robustness"]
    recorded = {kk: row.get(kk) for kk in keys}
    log(f"baseline: {cid} (k={k}) - {why}")
    return cid, k, why, recorded, chosen


# ====================================================================== shared evaluator
def stability(fit_labels, n, subs):
    """Sprint 3 protocol: fit on each 80% subsample, pairwise ARI on shared statements (noise = own label)."""
    labs = []
    for idx in subs:
        full = np.full(n, -2, dtype=int)
        full[idx] = fit_labels(idx)
        labs.append(full)
    a_all, a_nn = [], []
    for i, j in combinations(range(len(labs)), 2):
        sh = (labs[i] != -2) & (labs[j] != -2)
        li, lj = labs[i][sh], labs[j][sh]
        a_all.append(s3.ari_pair(li, lj))
        both = (li >= 0) & (lj >= 0)
        if both.sum() > 1:
            a_nn.append(s3.ari_pair(li[both], lj[both]))
    return {"stability_ari": float(np.mean(a_all)), "stability_ari_sd": float(np.std(a_all, ddof=1)),
            "stability_ari_nonnoise": float(np.mean(a_nn)) if a_nn else None, "stability_pairs": len(a_all)}


def seed_ari(label_runs):
    a = [s3.ari_pair(label_runs[i], label_runs[j]) for i, j in combinations(range(len(label_runs)), 2)]
    return {"seed_ari_mean": float(np.mean(a)), "seed_ari_sd": float(np.std(a, ddof=1)) if len(a) > 1 else None,
            "seed_runs": len(label_runs)}


def paraphrase(labels, pred, paired):
    """same-topic rate of (Gemini statement, Groq wording of the same task); outlier wording counts as a miss."""
    gi = paired.row.values
    own = labels[gi]
    valid = own >= 0
    same = (pred == own) & valid
    lab = labels[labels >= 0]
    shares = np.bincount(lab) / len(lab) if len(lab) else np.array([1.0])
    chance = float((shares ** 2).sum())
    rate = float(same[valid].mean()) if valid.any() else None
    return {"paraphrase_consistency": rate, "paraphrase_pairs_evaluated": int(valid.sum()),
            "paraphrase_wording_outlier": int((valid & (pred < 0)).sum()), "paraphrase_chance": chance,
            "paraphrase_kappa": (rate - chance) / (1 - chance) if rate is not None and chance < 1 else None}, same, valid


def evaluate(method, labels, native_conf, X, Z, P, paired, nn_pairs, providers, T_common):
    """Identical scoring for every method. Returns (metrics, per-statement arrays)."""
    r = {k: v for k, v in s3.size_stats(labels).items()}
    r["coverage"] = r["assigned"] / len(labels)
    im, sil = s3.internal_metrics(Z, labels)                     # silhouette etc. in the shared Sprint 3 PCA space
    r.update(im)
    coh, own, ids, C = s3.cohesion(X, labels)
    r.update(coh)
    base = s3.cohesion_baseline(X, labels)
    r["cohesion_random_baseline"] = base
    r["semantic_cohesion_lift"] = s3.lift(coh["semantic_cohesion"], base)
    own_all = own.copy()
    if (labels < 0).any() and len(ids):
        lab_all = labels.copy()
        noise = lab_all < 0
        lab_all[noise] = ids[np.argmax(s3.sims_to_centroids(X[noise], C), axis=1)]
        c2, own_all, _, _ = s3.cohesion(X, lab_all)
        r["semantic_cohesion_incl_noise"] = c2["semantic_cohesion"]
        r["semantic_cohesion_lift_incl_noise"] = s3.lift(c2["semantic_cohesion"], s3.cohesion_baseline(X, lab_all))
    else:
        r["semantic_cohesion_incl_noise"] = coh["semantic_cohesion"]
        r["semantic_cohesion_lift_incl_noise"] = r["semantic_cohesion_lift"]
    pr = s3.provider_robustness(labels, method, None, None, X, None, paired.iloc[0:0], nn_pairs, ids, C, providers)
    r.update(pr)
    g = r.get("nn_cross_minus_same_provider_same_topic")
    r["provider_nn_gap_abs"] = abs(g) if g is not None else None
    mem, thr = s3.membership(X, labels, ids, C, native_conf, method, native_conf)
    m = labels >= 0
    r["low_confidence_rate_common"] = float((mem.gap[m] < T_common).mean()) if m.any() and T_common is not None else None
    cut = NATIVE_LOW_CONF[method][1]
    nc = np.asarray(native_conf, dtype=float) if native_conf is not None else np.full(len(labels), np.nan)
    r["low_confidence_rate_native"] = float((nc[m] < cut).mean()) if m.any() else None
    r["centroid_inconsistency_rate"] = float(mem.centroid_inconsistent[m].mean()) if m.any() else None
    r["granularity_score"] = s3.granularity_score(r.get("n_clusters"), r.get("median_size"))
    pc_centroid = ids[np.argmax(s3.sims_to_centroids(P, C), axis=1)] if len(ids) and len(P) else np.full(len(P), -1)
    pcc, _, _ = paraphrase(labels, pc_centroid, paired)
    r["paraphrase_consistency_centroid"] = pcc["paraphrase_consistency"]
    r["paraphrase_kappa_centroid"] = pcc["paraphrase_kappa"]
    arrays = {"labels": labels, "own_sim": own, "own_sim_incl_noise": own_all, "sil": sil, "native_conf": nc,
              "mem": mem, "mem_thr": thr, "ids": ids, "C": C}
    return r, arrays


def add_paraphrase(r, arrays, labels, pred_native, paired):
    pn, same, valid = paraphrase(labels, np.asarray(pred_native, dtype=int), paired)
    r.update(pn)
    r["robustness"] = pn["paraphrase_kappa"]
    arrays["para_same"], arrays["para_valid"] = same, valid
    return r


def pathology(r):
    return ";".join(s3.pathology_flags(r))


def select(rows):
    """Sprint 3 lexicographic-with-tolerance selection, using HIERARCHY above."""
    ok = rows[rows.pathology_flags.fillna("") == ""]
    pool = ok if len(ok) else rows.assign(_nf=rows.pathology_flags.str.count(";") + 1).nsmallest(1, "_nf", keep="all")
    trace = [] if len(ok) else ["no non-pathological configuration - choosing among the least-flagged"]
    decided = None
    for col, tol in HIERARCHY:
        v = pd.to_numeric(pool[col], errors="coerce").fillna(-np.inf)
        keep = (v >= v.max() - tol).values
        trace.append(f"{col}: best={fnum(v.max())} tol={tol} -> kept {', '.join(pool.index[keep].astype(str))}")
        pool = pool[keep]
        if len(pool) == 1:
            decided = col
            break
    if len(pool) > 1:
        pool = pool.sort_values(["semantic_cohesion_lift_incl_noise", "stability_ari"], ascending=False)
        decided = "tie_break"
    return pool.index[0], trace, decided


# ====================================================================== method implementations
def fit_gmm(Zs, n, seed):
    from sklearn.mixture import GaussianMixture
    gm = GaussianMixture(n_components=n, covariance_type="diag", random_state=seed, n_init=GMM_N_INIT,
                         max_iter=GMM_MAX_ITER, reg_covar=GMM_REG_COVAR)
    gm.fit(Zs)
    return gm


def umap_embed(Xs, nn, nc, seed):
    import umap
    u = umap.UMAP(n_neighbors=nn, n_components=nc, min_dist=BERTOPIC_GRID["min_dist"], metric=BERTOPIC_GRID["metric"],
                  random_state=seed)
    U = u.fit_transform(Xs)
    return u, np.asarray(U, dtype=np.float64)


def fit_hdb(U, mcs, ms):
    import hdbscan
    cl = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms, metric="euclidean", cluster_selection_method="eom",
                         prediction_data=True, core_dist_n_jobs=1)
    cl.fit(U)
    return cl


def knn_graph(Xs, k):
    import igraph as ig
    from scipy.sparse import csr_matrix
    S = Xs @ Xs.T
    np.fill_diagonal(S, -np.inf)
    nbr = np.argpartition(-S, k, axis=1)[:, :k]
    rows = np.repeat(np.arange(len(Xs)), k)
    cols = nbr.ravel()
    w = S[rows, cols]
    keep = w > LEIDEN_EDGE_MIN_COSINE
    a, b, w = np.minimum(rows, cols)[keep], np.maximum(rows, cols)[keep], w[keep]
    e = pd.DataFrame({"a": a, "b": b, "w": w}).groupby(["a", "b"], as_index=False).w.max()   # union of kNN lists
    g = ig.Graph(n=len(Xs), edges=list(zip(e.a.tolist(), e.b.tolist())), directed=False)
    g.es["weight"] = e.w.tolist()
    W = csr_matrix((np.r_[e.w, e.w], (np.r_[e.a, e.b], np.r_[e.b, e.a])), shape=(len(Xs), len(Xs)))
    return g, W, {"edges": int(len(e)), "isolated_nodes": int((np.asarray(W.sum(1)).ravel() == 0).sum())}


def leiden(g, res, seed):
    import leidenalg
    part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, weights="weight",
                                    resolution_parameter=res, seed=seed, n_iterations=-1)
    memb = np.asarray(part.membership, dtype=int)
    return memb, float(part.quality()), float(g.modularity(part.membership, weights="weight"))


def community_shares(W, labels):
    from scipy.sparse import csr_matrix
    n, K = len(labels), int(labels.max()) + 1
    O = csr_matrix((np.ones(n), (np.arange(n), labels)), shape=(n, K))
    Sh = (W @ O).toarray()
    tot = Sh.sum(1)
    own = np.where(tot > 0, Sh[np.arange(n), labels] / np.where(tot > 0, tot, 1), np.nan)
    return Sh / np.where(tot[:, None] > 0, tot[:, None], 1), own


def knn_vote(P, X, labels, k):
    S = P @ X.T
    nbr = np.argpartition(-S, k, axis=1)[:, :k]
    out = np.empty(len(P), dtype=int)
    for i in range(len(P)):
        votes = {}
        for j in nbr[i]:
            if labels[j] >= 0:
                votes[labels[j]] = votes.get(labels[j], 0.0) + float(S[i, j])
        out[i] = max(votes, key=votes.get) if votes else -1
    return out


def ctfidf_keywords(docs, labels, top_n=10):
    """BERTopic topic representation (manual BERTopic: our UMAP+HDBSCAN labels -> BERTopic c-TF-IDF).
    Falls back to a plain c-TF-IDF only if the bertopic package cannot be used; the backend is recorded."""
    from sklearn.feature_extraction.text import CountVectorizer
    try:
        from bertopic import BERTopic
        from bertopic.cluster import BaseCluster
        from bertopic.dimensionality import BaseDimensionalityReduction
        tm = BERTopic(embedding_model=None, umap_model=BaseDimensionalityReduction(), hdbscan_model=BaseCluster(),
                      vectorizer_model=CountVectorizer(stop_words="english", ngram_range=(1, 2)),
                      calculate_probabilities=False, verbose=False)
        dummy = np.zeros((len(docs), 2))
        tm.fit(list(docs), embeddings=dummy, y=np.asarray(labels))
        bt = np.asarray(tm.topics_)
        out = {}
        for c in sorted(set(labels[labels >= 0].tolist())):
            vals, cnt = np.unique(bt[labels == c], return_counts=True)
            t = int(vals[np.argmax(cnt)])
            out[c] = ", ".join(w for w, _ in (tm.get_topic(t) or [])[:top_n])
        return out, f"bertopic {pkg_version('bertopic')} (manual mode: precomputed labels, c-TF-IDF)"
    except Exception as e:
        cv = CountVectorizer(stop_words="english", ngram_range=(1, 2))
        M = cv.fit_transform(list(docs))
        vocab = np.array(cv.get_feature_names_out())
        cs = sorted(set(labels[labels >= 0].tolist()))
        tf = np.vstack([np.asarray(M[labels == c].sum(0)).ravel() for c in cs]).astype(float)
        tf = tf / np.clip(tf.sum(1, keepdims=True), 1, None)
        idf = np.log(1 + tf.shape[0] / np.clip((tf > 0).sum(0), 1, None))
        sc = tf * idf
        return ({c: ", ".join(vocab[np.argsort(-sc[i])[:top_n]]) for i, c in enumerate(cs)},
                f"fallback c-TF-IDF (bertopic unavailable: {type(e).__name__}: {str(e)[:120]})")


# ====================================================================== secondary membership per method
def secondary_membership(method, arrays, extra):
    """max 2 memberships. Thresholds are quantiles of each method's own observed distributions."""
    mem = arrays["mem"].copy()
    labels = arrays["labels"]
    m = labels >= 0
    info = {"rule": "", "thresholds": {}}
    if method == "kmeans":
        info["rule"] = "Sprint 3 centroid rule (secondary similarity >= P25 own-centroid similarity AND gap <= P10 gap)"
        info["thresholds"] = {k: v for k, v in arrays["mem_thr"].items() if k != "rule"}
    elif method == "gmm":
        post = extra["posterior"]
        srt = np.sort(post, axis=1)
        p1, p2 = srt[:, -1], srt[:, -2]
        t2, tg = float(np.quantile(p2, 0.90)), float(np.quantile(p1 - p2, 0.10))
        t2_eff = max(t2, GMM_SECONDARY_FLOOR)
        sec = np.argsort(-post, axis=1)[:, 1]
        mem["has_secondary"] = (p2 >= t2_eff) & ((p1 - p2) <= tg)
        mem["second_id"] = sec
        mem["second_post"] = p2
        info["rule"] = ("secondary = 2nd-highest posterior iff p2 >= max(P90 of p2, numerical floor) AND p1 - p2 <= P10 of "
                        "(p1 - p2)")
        info["thresholds"] = {"p2_P90": t2, "numerical_floor": GMM_SECONDARY_FLOOR, "p2_threshold_used": t2_eff,
                              "gap_P10": tg, "share_p2_below_floor": float((p2 < GMM_SECONDARY_FLOOR).mean())}
    elif method == "bertopic":
        soft = extra.get("soft")
        base_rule = mem.has_secondary.values.copy()
        if soft is not None and soft.shape[1] > 1:
            nonprim = soft.copy()
            rows = np.where(m)[0]
            nonprim[rows, labels[rows]] = np.nan
            q = float(np.nanquantile(nonprim[rows], 0.75)) if len(rows) else np.nan
            sid = mem.second_id.values
            sp = np.array([soft[i, sid[i]] if sid[i] >= 0 and sid[i] < soft.shape[1] else np.nan for i in range(len(labels))])
            mem["has_secondary"] = base_rule & (sp >= q)
            mem["second_native"] = sp
            info["thresholds"]["soft_membership_P75_non_primary"] = q
        info["rule"] = ("Sprint 3 centroid rule AND HDBSCAN soft membership of the secondary topic >= P75 of non-primary "
                        "soft memberships; outliers assigned post hoc only by the Sprint 3 outlier rule")
        info["thresholds"].update({k: v for k, v in arrays["mem_thr"].items() if k != "rule"})
    elif method == "leiden":
        shares = extra["shares"]
        rows = np.where(m)[0]
        nonown = shares.copy()
        nonown[rows, labels[rows]] = np.nan
        vals = nonown[rows][nonown[rows] > 0]
        q = float(np.quantile(vals, 0.75)) if len(vals) else np.nan
        sid = mem.second_id.values
        sp = np.array([shares[i, sid[i]] if 0 <= sid[i] < shares.shape[1] else np.nan for i in range(len(labels))])
        mem["has_secondary"] = mem.has_secondary.values & (sp >= q)
        mem["second_native"] = sp
        info["rule"] = ("Sprint 3 centroid rule AND edge-weight share to the secondary community >= P75 of positive "
                        "non-own community shares")
        info["thresholds"] = {"edge_share_P75": q, **{k: v for k, v in arrays["mem_thr"].items() if k != "rule"}}
    info["secondary_rate"] = float(mem.has_secondary.mean())
    return mem, info


# ====================================================================== run
def run(args):
    global np, pd, s3
    project, out = Path(args.project_dir), Path(args.output)
    for d in ("bertopic", "leiden", "gmm", "baseline", "evaluation", "figures"):
        (out / d).mkdir(parents=True, exist_ok=True)
    ensure_dependencies(args.skip_install)
    import numpy as _np
    import pandas as _pd
    np, pd = _np, _pd
    np.random.seed(SEED)
    s3, s3_sha = load_sprint3_module(project)
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        gpu = torch.cuda.get_device_name(0) if dev == "cuda" else None
    except Exception:
        dev, gpu = "cpu (torch not importable)", None
    log(f"device={dev} gpu={gpu or 'none'} (this experiment is CPU-bound; no model is loaded)")

    df, X, P, paired, manifest, files, input_sha, hashes, checks = validate_inputs(project)
    N = len(df)
    log(f"validated: rows={N} postings={df.posting_id.nunique()} T5 embeddings {X.shape} paired wordings={len(paired)}")
    base_cid, base_k, base_why, recorded, chosen = read_baseline(manifest, files)
    pca, Z, pinfo = s3.controlled_pca(X)
    rec_p = manifest.get("pca", {}).get(T5_KEY, {})
    pca_match = (rec_p.get("n_components") == pinfo["n_components"] and
                 abs(rec_p.get("cumulative_explained_variance", -1) - pinfo["cumulative_explained_variance"]) < 1e-4)
    if not pca_match:
        log(f"WARNING: recomputed PCA {pinfo['n_components']} comps / {pinfo['cumulative_explained_variance']:.5f} differs "
            f"from Sprint 3 manifest {rec_p}")
    checks["pca_matches_sprint3"] = bool(pca_match)
    ck = Checkpoints(out, CONFIG_FINGERPRINT + input_sha[:12] + hashes["embeddings"][:12])
    providers = df.provider.values
    subs = s3.subsample_indices(N)
    nn_pairs = s3.provider_nn_pairs(X, providers)
    docs = df.statement_text.values
    results, store, timings = {}, {}, {}

    # ------------------------------------------------------------------ baseline (refit with the recorded parameters)
    t = time.time()
    km, lab_km = s3.fit_kmeans(Z, base_k)
    km_conf = s3.kmeans_ratio_confidence(km, Z)
    ids0, C0 = s3.centroids(X, lab_km)
    mem0, _ = s3.membership(X, lab_km, ids0, C0, km_conf, "kmeans", km_conf)
    T_common = float(np.nanquantile(mem0.gap[lab_km >= 0], 0.10))
    log(f"common low-confidence margin (P10 of baseline primary-secondary centroid gap) = {T_common:.4f}")
    cached = ck.get("baseline")
    if cached is None:
        r, arr = evaluate("kmeans", lab_km, km_conf, X, Z, P, paired, nn_pairs, providers, T_common)
        add_paraphrase(r, arr, lab_km, km.predict(pca.transform(P).astype(np.float64)), paired)
        r.update(stability(lambda idx: s3.fit_kmeans(Z[idx], base_k)[1], N, subs))
        r.update(seed_ari([s3.fit_kmeans(Z, base_k, s)[1] for s in STABILITY_SEEDS]))
        cached = ck.put("baseline", (r, arr))
    r, arr = cached
    r.update({"config_id": f"ADV_0_T5_KMEANS_BASELINE__k={base_k}", "method": "kmeans", "parameters": f"k={base_k}",
              "param_k": base_k, "sprint3_config_id": base_cid})
    r["pathology_flags"] = pathology(r)
    r["fit_seconds"] = round(time.time() - t, 1)
    # reproduction check against the recorded Sprint 3 values
    repro = {}
    for kk in ("n_clusters", "semantic_cohesion", "semantic_cohesion_lift", "stability_ari", "silhouette",
               "provider_pair_kappa", "low_confidence_rate"):
        rec = recorded.get(kk)
        new = r.get("paraphrase_kappa") if kk == "provider_pair_kappa" else (
            r.get("low_confidence_rate_native") if kk == "low_confidence_rate" else r.get(kk))
        try:
            repro[kk] = {"recorded": float(rec), "recomputed": float(new), "abs_diff": abs(float(rec) - float(new))}
        except Exception:
            repro[kk] = {"recorded": rec, "recomputed": new, "abs_diff": None}
    memb_file = project / "sprint3_results" / "final" / "task_topic_membership.csv"
    if str(chosen.get("config_id")) == base_cid and memb_file.exists():
        fm = pd.read_csv(memb_file, dtype=str)
        fm = fm[fm.membership_rank == "1"].drop_duplicates("statement_id").set_index("statement_id").topic_id
        rec_lab = df.statement_id.map(fm).fillna("NA").values
        repro["labels_vs_recorded_final_membership_ARI"] = s3.ari_pair(rec_lab, lab_km)
    worst = max((v["abs_diff"] for v in repro.values() if isinstance(v, dict) and v.get("abs_diff") is not None), default=0)
    r["reproduces_sprint3"] = bool(worst < 1e-3 and repro.get("labels_vs_recorded_final_membership_ARI", 1.0) > 0.999)
    log(f"baseline reproduction vs Sprint 3 records: max abs diff {worst:.2e}; "
        f"labels ARI {fnum(repro.get('labels_vs_recorded_final_membership_ARI'))} -> reproduces={r['reproduces_sprint3']}")
    results[r["config_id"]] = r
    store[r["config_id"]] = {**arr, "extra": {}}
    pd.DataFrame([r]).to_csv(out / "baseline" / "baseline_kmeans_recomputed.csv", index=False)
    (out / "baseline" / "baseline_reproduction_check.json").write_text(json.dumps(
        {"sprint3_config": base_cid, "selection_reason": base_why, "recorded": recorded, "comparison": repro}, indent=2, default=str))

    # ------------------------------------------------------------------ GMM
    t = time.time()
    for n in GMM_GRID["n_components"]:
        cid = f"ADV_3_T5_GMM__n_components={n}_cov=diag"
        c = ck.get(cid)
        if c is None:
            log(f"GMM n_components={n}")
            gm = fit_gmm(Z, n, SEED)
            post = gm.predict_proba(Z)
            lab = post.argmax(1).astype(int)
            r, arr = evaluate("gmm", lab, post.max(1), X, Z, P, paired, nn_pairs, providers, T_common)
            add_paraphrase(r, arr, lab, gm.predict(pca.transform(P).astype(np.float64)), paired)
            ent = -(post * np.log(np.clip(post, 1e-300, None))).sum(1)
            r.update({"bic": float(gm.bic(Z)), "aic": float(gm.aic(Z)), "converged": bool(gm.converged_),
                      "posterior_entropy_mean": float(ent.mean()), "posterior_entropy_norm_mean": float((ent / np.log(n)).mean()),
                      "empty_components": int(n - len(set(lab.tolist())))})
            r.update(stability(lambda idx: fit_gmm(Z[idx], n, SEED).predict(Z[idx]), N, subs))
            r.update(seed_ari([fit_gmm(Z, n, s).predict(Z) for s in STABILITY_SEEDS]))
            c = ck.put(cid, (r, arr, {"posterior": post, "entropy": ent}))
        r, arr, extra = c
        r.update({"config_id": cid, "method": "gmm", "parameters": f"n_components={n},covariance_type=diag",
                  "param_n_components": n})
        r["pathology_flags"] = pathology(r)
        results[cid], store[cid] = r, {**arr, "extra": extra}
        log(f"  {cid}: comps used={r['n_clusters']} lift={fnum(r['semantic_cohesion_lift_incl_noise'])} "
            f"stab={fnum(r['stability_ari'])} BIC={r['bic']:.0f}")
    timings["gmm_seconds"] = round(time.time() - t, 1)

    # ------------------------------------------------------------------ Leiden
    t = time.time()
    graphs = {}
    for k in LEIDEN_GRID["k_neighbors"]:
        graphs[k] = knn_graph(X, k)
        log(f"Leiden kNN graph k={k}: {graphs[k][2]}")
    for k in LEIDEN_GRID["k_neighbors"]:
        g, W, ginfo = graphs[k]
        for res in LEIDEN_GRID["resolution"]:
            cid = f"ADV_2_T5_LEIDEN__k_neighbors={k}_resolution={res}"
            c = ck.get(cid)
            if c is None:
                log(f"Leiden k={k} resolution={res}")
                lab, quality, modq = leiden(g, res, SEED)
                shares, own_share = community_shares(W, lab)
                r, arr = evaluate("leiden", lab, own_share, X, Z, P, paired, nn_pairs, providers, T_common)
                add_paraphrase(r, arr, lab, knn_vote(P, X, lab, k), paired)
                r.update({"leiden_quality": quality, "modularity": modq, **{f"graph_{a}": b for a, b in ginfo.items()},
                          "boundary_node_rate": float(np.nanmean(own_share < 0.5)),
                          "singleton_communities": int((np.bincount(lab) == 1).sum())})

                def sub_fit(idx, k=k, res=res):
                    gs, _, _ = knn_graph(X[idx], k)
                    return leiden(gs, res, SEED)[0]
                r.update(stability(sub_fit, N, subs))
                r.update(seed_ari([leiden(g, res, s)[0] for s in STABILITY_SEEDS]))
                c = ck.put(cid, (r, arr, {"shares": shares, "own_share": own_share}))
            r, arr, extra = c
            r.update({"config_id": cid, "method": "leiden", "parameters": f"k_neighbors={k},resolution={res}",
                      "param_k_neighbors": k, "param_resolution": res, "edge_min_cosine": LEIDEN_EDGE_MIN_COSINE})
            r["pathology_flags"] = pathology(r)
            results[cid], store[cid] = r, {**arr, "extra": extra}
            log(f"  {cid}: communities={r['n_clusters']} modularity={fnum(r['modularity'])} "
                f"lift={fnum(r['semantic_cohesion_lift_incl_noise'])} stab={fnum(r['stability_ari'])}")
    timings["leiden_seconds"] = round(time.time() - t, 1)

    # ------------------------------------------------------------------ BERTopic (UMAP -> HDBSCAN -> c-TF-IDF)
    t = time.time()
    umaps = {}
    for nn in BERTOPIC_GRID["n_neighbors"]:
        for nc in BERTOPIC_GRID["n_components"]:
            def full_umap(nn=nn, nc=nc):
                u, U = umap_embed(X, nn, nc, SEED)
                umaps[(nn, nc)] = u
                return U
            log(f"UMAP n_neighbors={nn} n_components={nc} (full data)")
            umaps[("U", nn, nc)] = ck.array(f"umap_full_nn{nn}_nc{nc}", full_umap)
    keyword_rows = []
    for nn in BERTOPIC_GRID["n_neighbors"]:
        for nc in BERTOPIC_GRID["n_components"]:
            U = umaps[("U", nn, nc)]
            for mcs in BERTOPIC_GRID["min_cluster_size"]:
                for ms in BERTOPIC_GRID["min_samples"]:
                    cid = f"ADV_1_T5_BERTOPIC__nn={nn}_nc={nc}_mcs={mcs}_ms={ms}"
                    c = ck.get(cid)
                    if c is None:
                        log(f"BERTopic {cid}")
                        cl = fit_hdb(U, mcs, ms)
                        lab = cl.labels_.astype(int)
                        prob = np.asarray(cl.probabilities_, dtype=float)
                        r, arr = evaluate("bertopic", lab, prob, X, Z, P, paired, nn_pairs, providers, T_common)
                        if (nn, nc) not in umaps:                      # resumed run: refit UMAP for paraphrase transform
                            umaps[(nn, nc)] = umap_embed(X, nn, nc, SEED)[0]
                        import hdbscan
                        if lab.max() >= 0:
                            pred, _ = hdbscan.approximate_predict(cl, np.asarray(umaps[(nn, nc)].transform(P), dtype=np.float64))
                            soft = hdbscan.all_points_membership_vectors(cl)
                            soft = soft.reshape(len(lab), -1) if soft.ndim == 1 else soft
                        else:
                            pred, soft = np.full(len(P), -1), None
                        add_paraphrase(r, arr, lab, pred, paired)
                        if lab.max() >= 0:
                            from sklearn.metrics import silhouette_score
                            mm = lab >= 0
                            r["silhouette_umap_space"] = float(silhouette_score(U[mm], lab[mm])) if len(set(lab[mm])) > 1 else None
                        kw, backend = ctfidf_keywords(docs, lab) if lab.max() >= 0 else ({}, "none")
                        r["bertopic_representation_backend"] = backend

                        def sub_fit(idx, nn=nn, nc=nc, mcs=mcs, ms=ms):
                            j = [i for i, s in enumerate(subs) if len(s) == len(idx) and np.array_equal(s, idx)][0]
                            Us = ck.array(f"umap_sub{j}_nn{nn}_nc{nc}", lambda: umap_embed(X[idx], nn, nc, SEED)[1])
                            return fit_hdb(Us, mcs, ms).labels_.astype(int)
                        r.update(stability(sub_fit, N, subs))
                        c = ck.put(cid, (r, arr, {"soft": soft, "keywords": kw, "prob": prob}))
                    r, arr, extra = c
                    r.update({"config_id": cid, "method": "bertopic",
                              "parameters": f"n_neighbors={nn},n_components={nc},min_dist=0.0,metric=cosine,"
                                            f"min_cluster_size={mcs},min_samples={ms}",
                              "param_n_neighbors": nn, "param_n_components": nc, "param_min_cluster_size": mcs,
                              "param_min_samples": ms})
                    r["pathology_flags"] = pathology(r)
                    kws = extra.get("keywords", {})
                    r["keywords_largest_5_topics"] = " | ".join(
                        kws.get(cc, "") for cc in pd.Series(arr["labels"][arr["labels"] >= 0]).value_counts().index[:5])
                    for cc, words in kws.items():
                        keyword_rows.append({"config_id": cid, "topic": int(cc), "size": int((arr["labels"] == cc).sum()),
                                             "keywords_for_inspection_only": words})
                    results[cid], store[cid] = r, {**arr, "extra": extra}
                    log(f"  {cid}: topics={r['n_clusters']} outliers={pct(r['noise_share'])} "
                        f"lift_incl_noise={fnum(r['semantic_cohesion_lift_incl_noise'])} stab={fnum(r['stability_ari'])}")
    pd.DataFrame(keyword_rows).to_csv(out / "bertopic" / "bertopic_keywords_all_configs.csv", index=False)
    timings["bertopic_seconds"] = round(time.time() - t, 1)

    # ------------------------------------------------------------------ best configuration per method
    R = pd.DataFrame(results.values()).set_index("config_id", drop=False)
    best, traces = {"kmeans": R[R.method == "kmeans"].index[0]}, {"kmeans": ["baseline - fixed by Sprint 3"]}
    for method in ("bertopic", "leiden", "gmm"):
        cid, tr, dec = select(R[R.method == method])
        best[method], traces[method] = cid, tr + [f"decided by {dec}"]
        log(f"best {method}: {cid}")
    # BERTopic UMAP-seed stability for the chosen configuration only (costly)
    cid = best["bertopic"]
    sr = ck.get("bertopic_seed_stability")
    if sr is None:
        rr = R.loc[cid]
        nn, nc, mcs, ms = (int(rr[f"param_{x}"]) for x in ("n_neighbors", "n_components", "min_cluster_size", "min_samples"))
        runs = [store[cid]["labels"]] + [fit_hdb(ck.array(f"umap_seed{s}_nn{nn}_nc{nc}", lambda s=s: umap_embed(X, nn, nc, s)[1]),
                                                 mcs, ms).labels_.astype(int) for s in STABILITY_SEEDS[:BERTOPIC_SEED_RUNS]]
        sr = ck.put("bertopic_seed_stability", seed_ari(runs))
    for kk, v in sr.items():
        R.loc[cid, kk] = v
        results[cid][kk] = v
    # granularity-matched K-Means reference (same PCA space, k = challenger topic count)
    for method, cid in best.items():
        k = int(R.loc[cid, "n_clusters"])
        if method == "kmeans" or k < 2:
            R.loc[cid, "kmeans_same_k_cohesion_lift"] = R.loc[cid, "semantic_cohesion_lift"] if method == "kmeans" else None
            continue
        ref = ck.get(f"kmeans_ref_k{k}")
        if ref is None:
            lab = s3.fit_kmeans(Z, k)[1]
            c_, _, _, _ = s3.cohesion(X, lab)
            ref = ck.put(f"kmeans_ref_k{k}", s3.lift(c_["semantic_cohesion"], s3.cohesion_baseline(X, lab)))
        R.loc[cid, "kmeans_same_k_cohesion_lift"] = ref
    for method in ("bertopic", "leiden", "gmm"):
        R[R.method == method].to_csv(out / method / f"{method}_all_configurations.csv", index=False)
    R.to_csv(out / "evaluation" / "all_configurations.csv", index=False)

    # ------------------------------------------------------------------ multi-topic membership for the best configs
    memberships, mem_info = {}, {}
    for method, cid in best.items():
        mem, info = secondary_membership(method, store[cid], store[cid]["extra"])
        memberships[method], mem_info[method] = mem, info
        R.loc[cid, "secondary_membership_rate"] = info["secondary_rate"]
        out_rows = pd.DataFrame({"statement_id": df.statement_id, "posting_id": df.posting_id, "provider": df.provider,
                                 "primary_topic": mem.primary, "primary_centroid_similarity": mem.own_sim,
                                 "assignment_confidence": store[cid]["native_conf"],
                                 "secondary_topic": np.where(mem.has_secondary, mem.second_id, -1),
                                 "secondary_centroid_similarity": np.where(mem.has_secondary, mem.second_sim, np.nan),
                                 "outlier_post_hoc_topic": np.where(mem.post_hoc, mem.top1_id, -1) if method == "bertopic" else -1})
        sub = "baseline" if method == "kmeans" else method
        out_rows.to_csv(out / sub / f"{method}_best_membership.csv", index=False)
    (out / "evaluation" / "multi_topic_membership_rules.json").write_text(json.dumps(mem_info, indent=2, default=str))

    # ------------------------------------------------------------------ bootstrap (posting level) for the 4 best configs
    boot = ck.get("bootstrap")
    if boot is None:
        boot = []
        posts = df.posting_id.values
        up = np.array(sorted(set(posts)))
        by_post = {p: np.where(posts == p)[0] for p in up}
        pair_post = posts[paired.row.values]
        pair_by_post = {p: np.where(pair_post == p)[0] for p in up}
        rng = np.random.default_rng(SEED)
        draws = [rng.choice(up, size=len(up), replace=True) for _ in range(N_BOOTSTRAP)]
        for method, cid in best.items():
            a = store[cid]
            lab = a["labels"]
            vals = {"semantic_cohesion": [], "semantic_cohesion_incl_noise": [], "silhouette": [], "paraphrase_consistency": []}
            for dr in draws:
                idx = np.concatenate([by_post[p] for p in dr])
                mm = lab[idx] >= 0
                vals["semantic_cohesion"].append(np.nanmean(a["own_sim"][idx][mm]) if mm.any() else np.nan)
                vals["semantic_cohesion_incl_noise"].append(np.nanmean(a["own_sim_incl_noise"][idx]))
                vals["silhouette"].append(np.nanmean(a["sil"][idx][mm]) if mm.any() and not np.all(np.isnan(a["sil"])) else np.nan)
                pj = np.concatenate([pair_by_post[p] for p in dr])
                v = a["para_valid"][pj]
                vals["paraphrase_consistency"].append(a["para_same"][pj][v].mean() if v.any() else np.nan)
            for metric, v in vals.items():
                v = np.asarray(v, dtype=float)
                if np.all(np.isnan(v)):
                    continue
                boot.append({"method": method, "config_id": cid, "metric": metric, "point_estimate": results[cid].get(metric),
                             "ci95_low": float(np.nanpercentile(v, 2.5)), "ci95_high": float(np.nanpercentile(v, 97.5)),
                             "n_resamples": N_BOOTSTRAP, "resampling_unit": "posting",
                             "note": "conditional on the fitted partition; refit variability = stability ARI"})
        ck.put("bootstrap", boot)
    B = pd.DataFrame(boot)
    B.to_csv(out / "evaluation" / "bootstrap_confidence_intervals.csv", index=False)

    # ------------------------------------------------------------------ error analysis
    ea = []
    for method, cid in best.items():
        a, mem = store[cid], memberships[method]
        lab = a["labels"]
        sz = pd.Series(lab[lab >= 0]).value_counts()
        amb = (mem.second_sim - mem.own_sim).values                  # >= 0 means closer to another topic centroid

        def add(i, item):
            p = int(lab[i])
            ea.append({"method": method, "experiment_id": cid, "review_item": item, "statement_id": df.statement_id[i],
                       "posting_id": df.posting_id[i], "provider": df.provider[i], "statement_text": df.statement_text[i],
                       "topic_id": f"T{p}" if p >= 0 else "OUTLIER", "cluster_size": int(sz.get(p, 0)) if p >= 0 else None,
                       "assignment_confidence": None if np.isnan(a["native_conf"][i]) else float(a["native_conf"][i]),
                       "centroid_similarity": None if np.isnan(mem.own_sim.iloc[i]) else float(mem.own_sim.iloc[i]),
                       "ambiguity_score": None if np.isnan(amb[i]) else float(amb[i]),
                       "secondary_topic_id": f"T{int(mem.second_id.iloc[i])}" if mem.has_secondary.iloc[i] else "",
                       "source_section": df.source_section[i], "error_review_label": ""})

        def sample(c, item):
            members = np.where(lab == c)[0]
            o = members[np.argsort(-mem.own_sim.values[members])]
            for i in list(o[:5]) + [x for x in o[-5:] if x not in o[:5]]:
                add(i, f"{item} (T{c}, n={len(members)})")
        order = list(sz.sort_values(ascending=False).index)
        for c in order[:5]:
            sample(c, "largest")
        mid = len(order) // 2
        for c in order[max(0, mid - 2): mid + 3]:
            sample(c, "medium")
        for c in [c for c in sz.sort_values().index if sz[c] >= s3.MEANINGFUL_CLUSTER][:5]:
            sample(c, "smallest_meaningful")
        assigned = np.where(lab >= 0)[0]
        conf = np.nan_to_num(a["native_conf"], nan=np.inf)
        for i in assigned[np.argsort(conf[assigned])][:20]:
            add(i, f"lowest_confidence ({NATIVE_LOW_CONF[method][0]})")
        sec = np.where(mem.has_secondary.values)[0]
        pool = sec if len(sec) >= 20 else assigned
        for i in pool[np.argsort(-np.nan_to_num(amb[pool], nan=-np.inf))][:20]:
            add(i, "most_ambiguous_multi_topic" if len(sec) >= 20 else "most_ambiguous (few secondary memberships)")
        if method == "bertopic":
            for i in np.where(lab < 0)[0][:20]:
                add(i, "bertopic_outlier")
        if method == "leiden":
            osh = np.nan_to_num(store[cid]["extra"]["own_share"], nan=-1)
            for i in np.argsort(osh)[:20]:
                add(i, "leiden_boundary_or_weak_node")
        if method == "gmm":
            for i in np.argsort(-store[cid]["extra"]["entropy"])[:20]:
                add(i, "gmm_high_entropy")
    EA = pd.DataFrame(ea)
    EA.to_csv(out / "evaluation" / "challenger_error_analysis.csv", index=False)

    # ------------------------------------------------------------------ final comparison + decision
    base_id = best["kmeans"]
    cols = ["method", "config_id", "parameters", "n_clusters", "coverage", "noise_share", "semantic_cohesion",
            "semantic_cohesion_lift", "semantic_cohesion_incl_noise", "semantic_cohesion_lift_incl_noise",
            "kmeans_same_k_cohesion_lift", "stability_ari", "stability_ari_sd", "stability_ari_nonnoise", "seed_ari_mean",
            "paraphrase_consistency", "paraphrase_kappa", "paraphrase_consistency_centroid", "paraphrase_kappa_centroid",
            "nn_cross_minus_same_provider_same_topic", "provider_nn_gap_abs", "provider_cramers_v", "silhouette",
            "min_size", "median_size", "max_size", "dominant_share", "tiny_cluster_share", "very_small_clusters",
            "granularity_score", "low_confidence_rate_common", "low_confidence_rate_native", "centroid_inconsistency_rate",
            "secondary_membership_rate", "bic", "aic", "posterior_entropy_norm_mean", "modularity", "boundary_node_rate",
            "pathology_flags"]
    CMP = R.loc[[best[m] for m in ("kmeans", "bertopic", "leiden", "gmm")]].reindex(columns=cols).copy()
    CMP.insert(0, "pipeline", [METHOD_LABEL[m] for m in CMP.method])
    for m in CMP.method:
        CMP.loc[best[m], "tuned_hyperparameters"] = COMPLEXITY[m]["tuned_hyperparameters"]
        CMP.loc[best[m], "extra_dependencies"] = COMPLEXITY[m]["extra_dependencies"]
        CMP.loc[best[m], "configs_tested"] = int((R.method == m).sum())
    CMP.loc[base_id, "method_runtime_seconds"] = results[base_id].get("fit_seconds")
    for m, key in (("bertopic", "bertopic_seconds"), ("leiden", "leiden_seconds"), ("gmm", "gmm_seconds")):
        CMP.loc[best[m], "method_runtime_seconds"] = timings.get(key)
    for metric in ("semantic_cohesion", "silhouette", "paraphrase_consistency"):
        sub = B[B.metric == metric].set_index("config_id") if len(B) else pd.DataFrame()
        CMP[f"{metric}_ci95"] = [f"[{fnum(sub.loc[c, 'ci95_low'])}, {fnum(sub.loc[c, 'ci95_high'])}]" if c in sub.index else ""
                                 for c in CMP.index]
    CMP["error_analysis_rows"] = [int((EA.experiment_id == c).sum()) for c in CMP.index]
    base = CMP.loc[base_id]
    decisions = {}
    for m in ("bertopic", "leiden", "gmm"):
        row = CMP.loc[best[m]]
        gains, costs = [], []
        for col, (tol, direction) in GAIN_TOL.items():
            a, b = row.get(col), base.get(col)
            if pd.isna(a) or pd.isna(b):
                continue
            d = (a - b) if direction == "higher" else (b - a)
            if d > tol:
                gains.append(f"{col} {fnum(b)} -> {fnum(a)}")
        for col, tol in COST_TOL.items():
            a, b = row.get(col), base.get(col)
            if pd.isna(a) or pd.isna(b):
                continue
            if b - a > tol:
                costs.append(f"{col} {fnum(b)} -> {fnum(a)}")
        if row.pathology_flags:
            costs.append(f"pathology: {row.pathology_flags}")
        decisions[m] = {"gains": gains, "costs": costs, "qualifies": bool(gains) and not costs}
        CMP.loc[best[m], "meaningful_gains_vs_kmeans"] = "; ".join(gains)
        CMP.loc[best[m], "unacceptable_costs_vs_kmeans"] = "; ".join(costs)
        CMP.loc[best[m], "replaces_kmeans"] = decisions[m]["qualifies"]
    qual = [m for m, d in decisions.items() if d["qualifies"]]
    if qual:
        pick, ptrace, _ = select(R.loc[[best[m] for m in qual]])
        final = R.loc[pick, "method"]
    else:
        final, ptrace = "kmeans", ["no challenger shows a meaningful gain without an unacceptable cost"]
    CMP["final_recommendation"] = [RECOMMENDATION[final] if m == final else "" for m in CMP.method]
    CMP.to_csv(out / "evaluation" / "advanced_challenger_comparison.csv", index=False)
    log(f"FINAL RECOMMENDATION: {RECOMMENDATION[final]}")

    figs = make_figures(out, R, CMP, B, best)
    timings["total_seconds"] = round(time.time() - T0, 1)
    manifest_out = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "script": Path(__file__).name,
        "config_fingerprint": CONFIG_FINGERPRINT, "project_dir": str(project),
        "input_sha256": input_sha, "row_count": N, "unique_postings": int(df.posting_id.nunique()),
        "sentence_t5_embeddings_sha256": hashes["embeddings"], "sentence_t5_mapping_sha256": hashes["mapping"],
        "paired_embeddings_sha256": hashes["paired_embeddings"], "sprint3_manifest_sha256": hashes["manifest"],
        "sprint3_script_sha256": s3_sha, "sprint3_script_is_validated_version": s3_sha == SPRINT3_SCRIPT_SHA256,
        "validation_checks": checks, "embedding_model": T5_MODEL, "embeddings_regenerated": False,
        "pca": {k: v for k, v in pinfo.items() if k != "explained_variance_ratio"},
        "random_seeds": {"global": SEED, "stability_seeds": STABILITY_SEEDS, "bertopic_umap_seed_runs": BERTOPIC_SEED_RUNS,
                         "subsample_protocol": "sprint3 subsample_indices (rng seed 42, 5 x 80%)", "bootstrap_rng": SEED},
        "package_versions": {p: pkg_version(p) for p in ("numpy", "pandas", "scipy", "scikit-learn", "umap-learn", "hdbscan",
                                                         "bertopic", "python-igraph", "leidenalg", "matplotlib")},
        "python": sys.version.split()[0], "platform": platform.platform(), "device": dev, "gpu": gpu,
        "grids": {"bertopic": BERTOPIC_GRID, "leiden": {**LEIDEN_GRID, "edge_min_cosine": LEIDEN_EDGE_MIN_COSINE,
                                                         "partition": "RBConfigurationVertexPartition", "weights": "cosine"},
                  "gmm": {**GMM_GRID, "n_init": GMM_N_INIT, "max_iter": GMM_MAX_ITER, "reg_covar": GMM_REG_COVAR,
                          "space": "Sprint 3 PCA of Sentence-T5"},
                  "kmeans_baseline": {"k": base_k, "sprint3_config_id": base_cid, "reason": base_why}},
        "common_low_confidence_margin": T_common, "selection_hierarchy": HIERARCHY, "gain_tolerances": GAIN_TOL,
        "cost_tolerances": COST_TOL, "best_configurations": best, "selection_traces": traces,
        "replacement_decisions": decisions, "final_method": final, "final_recommendation": RECOMMENDATION[final],
        "final_trace": ptrace, "baseline_reproduces_sprint3": bool(results[base_id].get("reproduces_sprint3")),
        "membership_rules": mem_info, "runtime_seconds": timings, "figures": figs,
        "known_input_quality_limitation": s3.LIMITATION}
    (out / "run_manifest.json").write_text(json.dumps(manifest_out, indent=2, default=str))
    try:
        write_report(out, CMP, R, B, best, decisions, final, base_cid, base_why, repro, mem_info, T_common, timings, checks,
                     traces)
    except Exception as e:
        log(f"report failed: {e}\n{traceback.format_exc()}")
    zip_results(out, project / "Sprint3_Advanced_Challenger_Results.zip")
    log("DONE")


# ====================================================================== figures
def make_figures(out, R, CMP, B, best):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fd, done, errors = out / "figures", [], []
    names = [METHOD_LABEL[m].replace("Sentence-T5 + ", "") for m in CMP.method]

    def fig(name, fn):
        try:
            fn()
            plt.tight_layout()
            plt.savefig(fd / name, dpi=150)
            done.append(name)
        except Exception as e:
            errors.append(f"{name}: {e}\n{traceback.format_exc()}")
            log(f"figure {name} failed: {e} (continuing)")
        finally:
            plt.close("all")

    def bars(col, title, err=None, ylabel=None):
        _, ax = plt.subplots(figsize=(7, 4))
        y = pd.to_numeric(CMP[col], errors="coerce").values
        ax.bar(names, y, yerr=err, capsize=4, color=["tab:gray", "tab:blue", "tab:green", "tab:purple"])
        ax.set_title(title)
        if ylabel:
            ax.set_ylabel(ylabel)

    def f1():
        bars("semantic_cohesion_lift_incl_noise", "Coverage-adjusted semantic cohesion lift (best config per method)")
    fig("01_semantic_cohesion_comparison.png", f1)

    def f2():
        bars("stability_ari", "Subsample stability (5 x 80%, mean pairwise ARI ± sd)", err=pd.to_numeric(CMP.stability_ari_sd, errors="coerce").fillna(0).values)
    fig("02_stability_comparison.png", f2)

    def f3():
        _, ax = plt.subplots(figsize=(7, 5))
        for m, g in R.groupby("method"):
            ax.scatter(g.n_clusters, g.semantic_cohesion_lift_incl_noise, label=METHOD_LABEL[m], s=35)
        ax.set_xlabel("topics / communities / components")
        ax.set_ylabel("coverage-adjusted cohesion lift")
        ax.set_title("Granularity vs cohesion (all configurations)")
        ax.legend(fontsize=8)
    fig("03_granularity_comparison.png", f3)

    def f4():
        bars("paraphrase_consistency", "Paraphrase consistency (Gemini statement vs Groq wording, method-native)")
    fig("04_paraphrase_consistency.png", f4)

    def f5():
        _, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].bar(names, pd.to_numeric(CMP.paraphrase_kappa, errors="coerce"))
        ax[0].set_title("Paraphrase kappa (chance-corrected)")
        ax[1].bar(names, pd.to_numeric(CMP.provider_nn_gap_abs, errors="coerce"))
        ax[1].set_title("|cross - same provider| neighbour same-topic gap (lower = better)")
    fig("05_provider_robustness.png", f5)

    def f6():
        g = R[R.method == "gmm"].sort_values("param_n_components")
        _, ax = plt.subplots(figsize=(7, 4))
        ax.plot(g.param_n_components, g.bic, "o-", label="BIC")
        ax.plot(g.param_n_components, g.aic, "s--", label="AIC")
        ax.set_xlabel("n_components")
        ax.legend()
        ax.set_title("GMM (diag) information criteria - supporting evidence only")
    fig("06_gmm_bic_curve.png", f6)

    def f7():
        g = R[R.method == "leiden"]
        _, ax = plt.subplots(figsize=(7, 4))
        for k, gg in g.groupby("param_k_neighbors"):
            gg = gg.sort_values("param_resolution")
            ax.plot(gg.param_resolution, gg.n_clusters, "o-", label=f"k={int(k)}")
        ax.set_xlabel("resolution")
        ax.set_ylabel("communities")
        ax.legend()
        ax.set_title("Leiden resolution vs community count")
    fig("07_leiden_resolution.png", f7)

    def f8():
        g = R[R.method == "bertopic"]
        _, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(100 * g.noise_share, g.n_clusters, s=40)
        for r in g.itertuples():
            ax.annotate(f"{int(r.param_n_neighbors)}/{int(r.param_n_components)}/{int(r.param_min_cluster_size)}/{int(r.param_min_samples)}",
                        (100 * r.noise_share, r.n_clusters), fontsize=6)
        ax.set_xlabel("outlier %")
        ax.set_ylabel("topics")
        ax.set_title("BERTopic outliers vs topics (labels nn/nc/mcs/ms)")
    fig("08_bertopic_noise_topic_tradeoff.png", f8)

    def f9():
        metrics = ["semantic_cohesion_lift_incl_noise", "stability_ari", "paraphrase_kappa", "coverage", "granularity_score"]
        _, ax = plt.subplots(figsize=(10, 4.5))
        w = 0.2
        for i, (nm, (_, row)) in enumerate(zip(names, CMP.iterrows())):
            ax.bar(np.arange(len(metrics)) + i * w, [pd.to_numeric(row[m], errors="coerce") for m in metrics], w, label=nm)
        ax.set_xticks(np.arange(len(metrics)) + 1.5 * w, ["cohesion lift", "stability", "paraphrase kappa", "coverage", "granularity"])
        ax.legend()
        ax.set_title("All-method comparison (best configuration per method)")
    fig("09_final_all_method_comparison.png", f9)
    if errors:
        (fd / "figure_errors.txt").write_text("\n\n".join(errors))
    return {"created": done, "failed": [e.split(":")[0] for e in errors]}


# ====================================================================== report
def write_report(out, CMP, R, B, best, decisions, final, base_cid, base_why, repro, mem_info, T_common, timings, checks,
                 traces):
    C = CMP.set_index("method")
    nonpath = C[C.pathology_flags.fillna("") == ""]
    pool = nonpath if len(nonpath) else C

    def arg(col, higher=True, frame=pool):
        v = pd.to_numeric(frame[col], errors="coerce")
        if v.notna().sum() == 0:
            return None, None
        m = v.idxmax() if higher else v.idxmin()
        return m, v[m]

    def name(m):
        return METHOD_LABEL.get(m, str(m))

    def tbl(frame, cols):
        s = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for r in frame.to_dict("records"):
            s.append("| " + " | ".join(fnum(r.get(c)) if isinstance(r.get(c), (float, np.floating)) else str(r.get(c, ""))
                                       for c in cols) + " |")
        return "\n".join(s)

    def beat(m):
        d = decisions[m]
        if d["qualifies"]:
            return f"**Yes** - meaningful gains ({'; '.join(d['gains'])}) and no unacceptable cost."
        if d["gains"]:
            return f"**No** - gains ({'; '.join(d['gains'])}) come with unacceptable costs ({'; '.join(d['costs'])})."
        return "**No** - no gain beyond tolerance" +(f"; costs: {'; '.join(d['costs'])}" if d["costs"] else "") + "."

    coh_m, coh_v = arg("semantic_cohesion_lift_incl_noise")
    st_m, st_v = arg("stability_ari")
    pk_m, pk_v = arg("paraphrase_kappa")
    lc_m, lc_v = arg("low_confidence_rate_common", higher=False)
    gr = pool.assign(_g=pd.to_numeric(pool.granularity_score, errors="coerce"),
                     _d=(pd.to_numeric(pool.median_size, errors="coerce") - np.mean(s3.GRANULARITY_MEDIAN_BAND)).abs())
    gr_m = gr.sort_values(["_g", "_d"], ascending=[False, True]).index[0]
    no_value = [m for m in ("bertopic", "leiden", "gmm") if not decisions[m]["qualifies"]]
    no_value.sort(key=lambda m: -COMPLEXITY[m]["tuned_hyperparameters"])
    lines = [
        "# Sprint 3 Advanced Challenger Report - does complexity earn its place?", "",
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by `{Path(__file__).name}`. All numbers are "
        "computed by the script; none are hand-entered. Representation frozen: `sentence-transformers/sentence-t5-base` "
        "(embeddings reused, not regenerated).", "",
        "> **Known limitation.** " + s3.LIMITATION, "",
        "## Setup",
        f"- Baseline: `{base_cid}` ({base_why}), refit with identical parameters; reproduction check "
        f"(`baseline/baseline_reproduction_check.json`): " + ", ".join(
            f"{k} diff {fnum(v.get('abs_diff'), 6)}" for k, v in repro.items() if isinstance(v, dict)) +
        (f"; labels ARI vs recorded Sprint 3 membership {fnum(repro.get('labels_vs_recorded_final_membership_ARI'))}"
         if "labels_vs_recorded_final_membership_ARI" in repro else ""),
        f"- Validation: {json.dumps(checks)}",
        "- One evaluator for all methods (imported from the Sprint 3 script): semantic cohesion in the ORIGINAL T5 space "
        "with a size-preserving random-partition baseline (lift); silhouette in the shared Sprint 3 PCA space; the same "
        "5 x 80% subsamples; the same provider nearest-neighbour test.",
        "- Outlier-producing methods are compared on **coverage-adjusted** cohesion lift (outliers attached to their nearest "
        "centroid for the metric only) so that discarding hard statements is not rewarded; assigned-only values are also shown.",
        f"- Common low-confidence definition: primary-vs-secondary centroid margin below {fnum(T_common, 4)} (P10 of the "
        "baseline's margins); each method's native confidence is reported separately.",
        f"- Replacement rule: a challenger replaces K-Means only with >= 1 gain beyond tolerance {json.dumps(GAIN_TOL)} and "
        f"no cost beyond {json.dumps(COST_TOL)} or pathology flag.", "",
        "## Comparison (best configuration per method)",
        tbl(CMP, ["pipeline", "parameters", "n_clusters", "coverage", "semantic_cohesion_lift_incl_noise",
                  "kmeans_same_k_cohesion_lift", "stability_ari", "stability_ari_sd", "paraphrase_consistency",
                  "paraphrase_kappa", "provider_nn_gap_abs", "silhouette", "granularity_score",
                  "low_confidence_rate_common", "secondary_membership_rate", "pathology_flags"]), "",
        "`kmeans_same_k_cohesion_lift` = K-Means refit at the challenger's topic count: it shows whether a cohesion difference "
        "is method quality or just granularity.", "",
        "## Bootstrap 95% CIs (posting level, 100 resamples)",
        tbl(B, ["method", "metric", "point_estimate", "ci95_low", "ci95_high"]) if len(B) else "not computed", "",
        "## Answers",
        f"1. **Did BERTopic beat K-Means?** {beat('bertopic')}",
        f"2. **Did Leiden beat K-Means?** {beat('leiden')}",
        f"3. **Did GMM beat K-Means?** {beat('gmm')}",
        f"4. **Highest semantic cohesion:** {name(coh_m)} (coverage-adjusted lift {fnum(coh_v)}; assigned-only highest: "
        f"{name(arg('semantic_cohesion_lift')[0])}). Compare with `kmeans_same_k_cohesion_lift` before attributing it to the method.",
        f"5. **Most stable:** {name(st_m)} (subsample ARI {fnum(st_v)}).",
        f"6. **Best at preserving Gemini/Groq paraphrases:** {name(pk_m)} (chance-corrected kappa {fnum(pk_v)}; raw rates "
        "favour coarse partitions, so kappa is the fair comparison).",
        f"7. **Handles ambiguous tasks best:** {name(lc_m)} (lowest common low-confidence rate {pct(lc_v)}); secondary-membership "
        "rates: " + ", ".join(f"{name(m)} {pct(C.loc[m, 'secondary_membership_rate'])}" for m in C.index) + ".",
        f"8. **Most useful topic granularity:** {name(gr_m)} (granularity score {fnum(C.loc[gr_m, 'granularity_score'])}, "
        f"{fnum(C.loc[gr_m, 'n_clusters'])} topics, median size {fnum(C.loc[gr_m, 'median_size'])}).",
        "9. **Adds complexity without enough value:** " + (", ".join(
            f"{name(m)} ({COMPLEXITY[m]['tuned_hyperparameters']} tuned hyper-parameters, extra deps: "
            f"{COMPLEXITY[m]['extra_dependencies']})" for m in no_value) or "none - every challenger qualified") + ".",
        "", "## 10. FINAL RECOMMENDATION", "", f"**{RECOMMENDATION[final]}**", ""]
    if final == "kmeans":
        lines.append("Key reason: no challenger delivered a gain beyond the pre-declared tolerances without an unacceptable "
                     "cost, so the simpler baseline is kept (complexity did not earn its place).")
    else:
        d = decisions[final]
        lines.append(f"Key reason: {name(final)} shows {'; '.join(d['gains'])} with no unacceptable cost.")
    lines += ["", "## Multi-topic membership rules (derived from each method's observed distributions)"]
    lines += [f"- **{name(m)}**: {i['rule']} - thresholds {json.dumps(i['thresholds'], default=str)}" for m, i in mem_info.items()]
    lines += ["", "## Per-method selection traces"]
    for m in ("bertopic", "leiden", "gmm"):
        lines += [f"**{name(m)}** -> `{best[m]}`"] + [f"- {t}" for t in traces.get(m, [])] + [""]
    lines += [
              "## Limitations",
              "- " + s3.LIMITATION,
              "- No ground-truth topics: all metrics are internal or robustness measures, not accuracy.",
              "- BERTopic keywords (`bertopic/bertopic_keywords_all_configs.csv`) are for inspection only, not taxonomy names.",
              "- GMM BIC/AIC are supporting evidence only. Bootstrap CIs are conditional on the fitted partition.",
              "- Paraphrase pairs come from Sprint 2 cross-model agreement and skew towards easier paraphrases.", "",
              f"Runtime: {json.dumps(timings)}"]
    (out / "advanced_challenger_report.md").write_text("\n".join(lines), encoding="utf-8")


def zip_results(out, zpath):
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for d in ("bertopic", "leiden", "gmm", "baseline", "evaluation", "figures"):
            for p in sorted((out / d).rglob("*")):
                if p.is_file():
                    z.write(p, arcname=str(p.relative_to(out)))
        for f in ("run_manifest.json", "advanced_challenger_report.md"):
            if (out / f).exists():
                z.write(out / f, arcname=f)
    log(f"ZIP written: {zpath} ({zpath.stat().st_size / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-dir", required=True, help="Drive folder holding sprint3_task_input.csv and sprint3_results/")
    ap.add_argument("--output", required=True, help="results directory (created if missing)")
    ap.add_argument("--skip-install", action="store_true")
    args = ap.parse_args()
    try:
        run(args)
    except SystemExit:
        raise
    except Exception as e:
        traceback.print_exc()
        stop(f"{type(e).__name__}: {e}\nCompleted stages are checkpointed in {Path(args.output) / 'checkpoints'}; "
             "re-run the same command to resume.")


if __name__ == "__main__":
    main()
