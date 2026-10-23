#!/usr/bin/env python
"""Sprint 3 - Representation & Clustering. Workhuman Taxonomy of Work capstone. Google Colab experiment.

Controlled 2 x 2 experiment on ONE frozen input (data/experiment/sprint3_task_input.csv):

                    K-MEANS             HDBSCAN
    E5              EXP_1_E5_KMEANS     EXP_2_E5_HDBSCAN
    Sentence-T5     EXP_3_T5_KMEANS     EXP_4_T5_HDBSCAN

Run in Colab (after drive.mount('/content/drive')):

    !python "/content/drive/MyDrive/CAPESTONE PROJECT/sprint3_colab_experiment.py" \
      --input "/content/drive/MyDrive/CAPESTONE PROJECT/sprint3_task_input.csv" \
      --output "/content/drive/MyDrive/CAPESTONE PROJECT/sprint3_results"

No API keys, no LLM calls. Embeddings: intfloat/e5-base-v2 ("passage: " prefix, mean pooling, L2-normalised) and
sentence-transformers/sentence-t5-base (L2-normalised). Clustering runs on the SAME PCA (<= 50 components) procedure
for both models; cosine-based analyses (cohesion, neighbours, centroids, memberships) use the ORIGINAL normalised
embeddings. UMAP is used for one 2-D figure only.

Every stage is checkpointed under <output>/checkpoints (embeddings are cached in <output>/embeddings), so a restart
after a Colab disconnect resumes instead of recomputing. Figures are best-effort: a failing figure is logged and the
run continues. The final ZIP is written next to the output folder: <output>/../Sprint3_Experiment_Results.zip
"""
from __future__ import annotations

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONHASHSEED", "42")

import argparse
import hashlib
import importlib.metadata as md
import json
import math
import pickle
import platform
import random
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

# ====================================================================== configuration (pre-declared, not tuned)
EXPECTED_INPUT_SHA256 = "010795fed2864f9963f159204a857f2198a2246206009ab0dbbbb786799f4551"
SEED = 42
E5_MODEL = "intfloat/e5-base-v2"
T5_MODEL = "sentence-transformers/sentence-t5-base"
E5_PREFIX = "passage: "                      # every statement is a standalone passage; never mixed with "query: "
MODELS = {"e5": E5_MODEL, "sentence_t5": T5_MODEL}
MODEL_TAG = {"e5": "E5", "sentence_t5": "T5"}
PCA_MAX_COMPONENTS = 50
KMEANS_K = [25, 40, 60, 80, 100]
KMEANS_N_INIT = 10
HDB_MIN_CLUSTER_SIZE = [10, 20, 30, 40]
HDB_MIN_SAMPLES = [5, 10]
STABILITY_SEEDS = [101, 202, 303, 404, 505]   # K-Means seed-restart stability
N_SUBSAMPLES = 5                              # subsample stability (same protocol for K-Means and HDBSCAN)
SUBSAMPLE_FRAC = 0.80
N_BOOTSTRAP = 100                             # posting-level bootstrap for the 4 final candidates
N_RANDOM_BASELINE = 5                         # label permutations for the cohesion chance baseline
NN_SAMPLE = 25
NN_K = 5
VERY_SMALL_CLUSTER = 5                        # "very small" = fewer than 5 members
TINY_CLUSTER = 10                             # used by the fragmentation check
MEANINGFUL_CLUSTER = 5                        # error analysis: smallest *meaningful* cluster has >= 5 members
KMEANS_LOW_CONF = 0.10                        # K-Means ratio confidence 1 - d_own/d_second below this = low
HDB_LOW_CONF = 0.50                           # HDBSCAN membership probability below this = low
PATHOLOGY = {"min_stability_ari": 0.50, "max_dominant_share": 0.20, "max_tiny_cluster_share": 0.25,
             "max_noise_share": 0.50, "min_clusters": 10, "max_clusters": 300, "min_cohesion_lift": 0.05}
GRANULARITY_MEDIAN_BAND = (15, 60)            # sensible leaf-topic median size for ~3k task statements
# decision hierarchy (user-specified order). Lexicographic with pre-declared tolerances: a candidate survives a
# level if it is within `tol` of the best on that level; the next level only breaks ties among survivors.
HIERARCHY = [("semantic_cohesion_lift", 0.02), ("stability_ari", 0.05), ("granularity_score", 0.0),
             ("robustness", 0.05), ("silhouette", 0.01)]
J_ROBUST, J_SENSITIVE = 0.50, 0.30            # cross-representation cluster overlap categories
STRESS_SECTIONS = ["responsibilities"]        # higher-confidence subset for the contamination stress test
AUDIT_PURITY = {"audited_rows_after_filter": 85, "task_rows_after_filter": 79, "purity_after_filter": 0.929,
                "responsibilities": "53/59 TASK (89.8%)", "general_description": "26/26 TASK (100%)"}
LIMITATION = ("The Sprint 3 input was section-filtered following a manually labelled task-purity audit. "
              "Post-filter audited task purity was 92.9%. Residual non-task contamination may therefore remain "
              "and is treated as a known limitation of the clustering experiment.")
EXPERIMENTS = {("e5", "kmeans"): "EXP_1_E5_KMEANS", ("e5", "hdbscan"): "EXP_2_E5_HDBSCAN",
               ("sentence_t5", "kmeans"): "EXP_3_T5_KMEANS", ("sentence_t5", "hdbscan"): "EXP_4_T5_HDBSCAN"}
REQUIRED_COLUMNS = ["statement_id", "posting_id", "provider", "statement_text", "evidence_span", "source_section",
                    "evidence_valid"]
DEPENDENCIES = [("numpy", "numpy"), ("pandas", "pandas"), ("scipy", "scipy"), ("sklearn", "scikit-learn"),
                ("matplotlib", "matplotlib"), ("torch", "torch"), ("transformers", "transformers"),
                ("sentence_transformers", "sentence-transformers"), ("sentencepiece", "sentencepiece"),
                ("hdbscan", "hdbscan"), ("umap", "umap-learn")]
CONFIG_FINGERPRINT = hashlib.sha256(json.dumps(
    [EXPECTED_INPUT_SHA256, SEED, MODELS, E5_PREFIX, PCA_MAX_COMPONENTS, KMEANS_K, KMEANS_N_INIT, HDB_MIN_CLUSTER_SIZE,
     HDB_MIN_SAMPLES, STABILITY_SEEDS, N_SUBSAMPLES, SUBSAMPLE_FRAC, "v1"]).encode()).hexdigest()[:16]

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:6.0f}s] {msg}", flush=True)


def die(msg: str) -> None:
    print(f"\nERROR: {msg}\n", file=sys.stderr, flush=True)
    sys.exit(1)


# ====================================================================== dependencies
def ensure_dependencies(skip: bool) -> None:
    import importlib.util
    # find_spec does not import anything, so installing (which may upgrade numpy) happens before numpy is loaded
    missing = [pip_name for mod, pip_name in DEPENDENCIES if importlib.util.find_spec(mod) is None]
    if missing and not skip:
        log(f"installing missing packages: {missing}")
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing])
        if r.returncode != 0:
            log("WARNING: pip install returned a non-zero exit code; continuing with what is importable")
        importlib.invalidate_caches()
    for mod in ("numpy", "pandas", "scipy", "sklearn", "torch", "transformers", "sentence_transformers"):
        try:
            importlib.import_module(mod)
        except Exception as e:
            die(f"required package '{mod}' is not importable ({e}). In Colab run: !pip install {mod}")


def pkg_version(name: str):
    try:
        return md.version(name)
    except Exception:
        return None


# imported after ensure_dependencies() in main()
np = pd = None


def lazy_imports():
    global np, pd
    import numpy as _np
    import pandas as _pd
    np, pd = _np, _pd


def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def h(*parts) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()


def fnum(x, nd=3):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "n/a"
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}"
    return f"{float(x):.{nd}f}"


def pct(x, nd=1):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * float(x):.{nd}f}%"


# ====================================================================== checkpoints
class Checkpoints:
    def __init__(self, root: Path, input_sha: str):
        self.dir = root / "checkpoints"
        self.dir.mkdir(parents=True, exist_ok=True)
        meta = self.dir / "meta.json"
        want = {"input_sha256": input_sha, "config": CONFIG_FINGERPRINT}
        if meta.exists():
            try:
                have = json.loads(meta.read_text())
            except Exception:
                have = {}
            if have != want:
                log("checkpoint fingerprint changed (different input or configuration) -> clearing checkpoints")
                for f in self.dir.glob("*.pkl"):
                    f.unlink()
        meta.write_text(json.dumps(want))

    def path(self, name):
        return self.dir / f"{name}.pkl"

    def get(self, name):
        p = self.path(name)
        if p.exists():
            try:
                with open(p, "rb") as f:
                    return pickle.load(f)
            except Exception:
                log(f"checkpoint {name} unreadable - recomputing")
        return None

    def put(self, name, obj):
        tmp = self.path(name).with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, self.path(name))
        return obj


# ====================================================================== input
def norm_text(s: str) -> str:
    import re
    t = (s or "").translate(str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", " ": " "}))
    return re.sub(r"\s+", " ", t.lower().strip()).rstrip(" .;:,")


def load_input(path: Path, allow_mismatch: bool):
    if not path.exists():
        die(f"input file not found: {path}\nUpload sprint3_task_input.csv to My Drive/CAPESTONE PROJECT/ "
            f"and mount Drive first (from google.colab import drive; drive.mount('/content/drive')).")
    sha = sha256_file(path)
    if sha != EXPECTED_INPUT_SHA256:
        msg = f"input SHA-256 {sha} != frozen Sprint 3 input {EXPECTED_INPUT_SHA256}"
        if not allow_mismatch:
            die(msg + "\nUse the exact frozen file (re-upload it; do not open/save it in Excel), "
                      "or pass --allow-input-mismatch to run anyway (the run will be marked NON-FROZEN).")
        log("WARNING: " + msg + " (continuing because --allow-input-mismatch was given)")
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    miss = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if miss:
        die(f"input is missing required columns {miss}; found {list(df.columns)}")
    if not df.statement_id.is_unique:
        die("statement_id is not unique in the input")
    if (df.statement_text.str.strip() == "").any():
        die("input contains blank statement_text rows")
    if "normalized_text" not in df.columns:
        df["normalized_text"] = df.statement_text.map(norm_text)
    if df.normalized_text.duplicated().any():
        log(f"WARNING: {int(df.normalized_text.duplicated().sum())} exact normalized duplicates present in input")
    for c in ("cross_provider_paired_text", "duplicate_statement_ids", "duplicate_posting_ids", "duplicate_providers"):
        if c not in df.columns:
            df[c] = ""
    df = df.reset_index(drop=True)
    return df, sha


# ====================================================================== embeddings
def device_info():
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if dev == "cuda" else None
    return dev, gpu


def _encode_e5(texts, device, batch_size):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(E5_MODEL)
    model = AutoModel.from_pretrained(E5_MODEL).to(device).eval()
    out = []
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            batch = [E5_PREFIX + t for t in texts[i:i + batch_size]]
            enc = tok(batch, max_length=512, padding=True, truncation=True, return_tensors="pt").to(device)
            hs = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(hs.dtype)
            emb = (hs * mask).sum(1) / mask.sum(1).clamp(min=1e-9)          # official E5 average pooling
            out.append(F.normalize(emb, p=2, dim=1).float().cpu().numpy())
            if (i // batch_size) % 10 == 0:
                log(f"  e5: {min(i + batch_size, len(texts))}/{len(texts)}")
    del model
    return np.vstack(out).astype(np.float32)


def _encode_t5(texts, device, batch_size):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(T5_MODEL, device=device)
    emb = model.encode(list(texts), batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True,
                       show_progress_bar=True)
    del model
    return np.asarray(emb, dtype=np.float32)


def encode(model_key, texts, device, batch_size):
    X = _encode_e5(texts, device, batch_size) if model_key == "e5" else _encode_t5(texts, device, batch_size)
    X = X / np.linalg.norm(X, axis=1, keepdims=True).clip(min=1e-12)
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return X.astype(np.float32)


def get_embeddings(model_key, df, paired, out: Path, device, batch_size, input_sha):
    """Corpus embeddings (+ embeddings of the Groq paired wordings used by robustness checks), cached on disk."""
    ed = out / "embeddings"
    ed.mkdir(parents=True, exist_ok=True)
    npy, mp, meta = ed / f"{model_key}_embeddings.npy", ed / f"{model_key}_embedding_mapping.csv", ed / f"{model_key}_meta.json"
    pnpy = ed / f"{model_key}_paired_text_embeddings.npy"
    ids_hash = h(*df.statement_id)
    want = {"model": MODELS[model_key], "input_sha256": input_sha, "rows": len(df), "ids_hash": ids_hash,
            "prefix": E5_PREFIX if model_key == "e5" else "", "paired_rows": len(paired)}
    if npy.exists() and meta.exists() and pnpy.exists():
        try:
            if json.loads(meta.read_text()) == want:
                X, P = np.load(npy), np.load(pnpy)
                if X.shape[0] == len(df) and P.shape[0] == len(paired):
                    log(f"{model_key}: loaded cached embeddings {X.shape}")
                    return X, P
        except Exception:
            pass
    log(f"{model_key}: encoding {len(df)} statements with {MODELS[model_key]} on {device}")
    X = encode(model_key, df.statement_text.tolist(), device, batch_size)
    log(f"{model_key}: encoding {len(paired)} cross-provider paired wordings (robustness checks only)")
    P = encode(model_key, paired.paired_text.tolist(), device, batch_size) if len(paired) else np.zeros((0, X.shape[1]), np.float32)
    np.save(npy, X)
    np.save(pnpy, P)
    pd.DataFrame({"embedding_row": np.arange(len(df)), "statement_id": df.statement_id, "posting_id": df.posting_id,
                  "provider": df.provider, "statement_text": df.statement_text}).to_csv(mp, index=False)
    paired.assign(embedding_row=np.arange(len(paired)))[["embedding_row", "statement_id", "paired_text"]].to_csv(
        ed / f"{model_key}_paired_text_mapping.csv", index=False)
    meta.write_text(json.dumps({**want, "dims": int(X.shape[1]), "created_utc": datetime.now(timezone.utc).isoformat()}))
    log(f"{model_key}: saved {npy.name} {X.shape}")
    return X, P


# ====================================================================== PCA
def controlled_pca(X):
    from sklearn.decomposition import PCA
    n_comp = int(min(PCA_MAX_COMPONENTS, X.shape[0] - 1, X.shape[1]))
    pca = PCA(n_components=n_comp, svd_solver="full", random_state=SEED)
    Z = pca.fit_transform(X).astype(np.float64)
    evr = pca.explained_variance_ratio_
    return pca, Z, {"n_components": n_comp, "explained_variance_ratio": [float(v) for v in evr],
                    "cumulative_explained_variance": float(evr.sum()), "svd_solver": "full", "whiten": False}


# ====================================================================== clustering
_HDB_IMPL = None


def hdb_impl():
    global _HDB_IMPL
    if _HDB_IMPL is None:
        try:
            import hdbscan  # noqa: F401
            _HDB_IMPL = "hdbscan"
        except Exception:
            _HDB_IMPL = "sklearn"
            log("WARNING: 'hdbscan' package unavailable -> using sklearn.cluster.HDBSCAN (same algorithm; "
                "out-of-sample prediction falls back to nearest centroid)")
    return _HDB_IMPL


def fit_kmeans(Z, k, seed=SEED):
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, init="k-means++", n_init=KMEANS_N_INIT, random_state=seed)
    labels = km.fit_predict(Z)
    return km, labels.astype(int)


def fit_hdbscan(Z, mcs, ms):
    if hdb_impl() == "hdbscan":
        import hdbscan
        cl = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms, metric="euclidean",
                             cluster_selection_method="eom", prediction_data=True, core_dist_n_jobs=1)
    else:
        from sklearn.cluster import HDBSCAN
        cl = HDBSCAN(min_cluster_size=mcs, min_samples=ms, metric="euclidean", cluster_selection_method="eom")
    cl.fit(Z)
    return cl, cl.labels_.astype(int), np.asarray(cl.probabilities_, dtype=float)


def fit_config(method, params, Z, seed=SEED):
    if method == "kmeans":
        m, lab = fit_kmeans(Z, params["k"], seed)
        return m, lab, None
    return fit_hdbscan(Z, params["min_cluster_size"], params["min_samples"])


def config_grid():
    g = [("kmeans", {"k": k}) for k in KMEANS_K]
    g += [("hdbscan", {"min_cluster_size": a, "min_samples": b}) for a in HDB_MIN_CLUSTER_SIZE for b in HDB_MIN_SAMPLES]
    return g


def params_str(method, p):
    return f"k={p['k']}" if method == "kmeans" else f"min_cluster_size={p['min_cluster_size']},min_samples={p['min_samples']}"


def config_id(model_key, method, p):
    return f"{EXPERIMENTS[(model_key, method)]}__" + params_str(method, p).replace(",", "_")


# ====================================================================== metrics
def size_stats(labels):
    lab = labels[labels >= 0]
    n = len(labels)
    if len(lab) == 0:
        return {"n_clusters": 0, "assigned": 0, "noise_count": int(n), "noise_share": 1.0}
    sizes = np.bincount(lab)
    sizes = sizes[sizes > 0]
    return {"n_clusters": int(len(sizes)), "assigned": int(len(lab)), "noise_count": int(n - len(lab)),
            "noise_share": float((n - len(lab)) / n), "min_size": int(sizes.min()), "median_size": float(np.median(sizes)),
            "max_size": int(sizes.max()), "mean_size": float(sizes.mean()),
            "very_small_clusters": int((sizes < VERY_SMALL_CLUSTER).sum()),
            "tiny_cluster_share": float((sizes < TINY_CLUSTER).mean()),
            "dominant_share": float(sizes.max() / len(lab)),
            "size_p10": float(np.percentile(sizes, 10)), "size_p90": float(np.percentile(sizes, 90)),
            "size_distribution": ";".join(map(str, sorted(sizes.tolist(), reverse=True)))}


def centroids(X, labels):
    ids = np.array(sorted(set(labels[labels >= 0].tolist())), dtype=int)
    if len(ids) == 0:
        return ids, np.zeros((0, X.shape[1]))
    C = np.vstack([X[labels == c].mean(0) for c in ids])
    C = C / np.linalg.norm(C, axis=1, keepdims=True).clip(min=1e-12)
    return ids, C


def own_similarity(X, labels, ids, C):
    """cosine of every statement to its own cluster centroid (NaN for noise)."""
    pos = {c: i for i, c in enumerate(ids)}
    s = np.full(len(labels), np.nan)
    m = labels >= 0
    if m.any():
        idx = np.array([pos[c] for c in labels[m]])
        s[m] = np.einsum("ij,ij->i", X[m], C[idx])
    return s


def cohesion(X, labels):
    ids, C = centroids(X, labels)
    s = own_similarity(X, labels, ids, C)
    m = labels >= 0
    if not m.any():
        return {"semantic_cohesion": None, "cohesion_cluster_mean": None, "cohesion_median": None}, s, ids, C
    per = pd.DataFrame({"c": labels[m], "s": s[m]}).groupby("c").s
    return {"semantic_cohesion": float(np.nanmean(s[m])), "cohesion_cluster_mean": float(per.mean().mean()),
            "cohesion_median": float(np.nanmedian(s[m]))}, s, ids, C


def cohesion_baseline(X, labels):
    """chance level: same cluster sizes, members permuted at random (fixed seeds)."""
    m = np.where(labels >= 0)[0]
    if len(m) == 0:
        return None
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(N_RANDOM_BASELINE):
        perm = labels.copy()
        perm[m] = rng.permutation(labels[m])
        vals.append(cohesion(X, perm)[0]["semantic_cohesion"])
    return float(np.mean(vals))


def lift(v, base):
    if v is None or base is None or base >= 1:
        return None
    return float((v - base) / (1 - base))


def internal_metrics(Z, labels):
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_samples
    m = labels >= 0
    k = len(set(labels[m].tolist()))
    out = {"silhouette": None, "davies_bouldin": None, "calinski_harabasz": None}
    sil = np.full(len(labels), np.nan)
    if k >= 2 and m.sum() > k:
        sil[m] = silhouette_samples(Z[m], labels[m], metric="euclidean")
        out = {"silhouette": float(np.nanmean(sil[m])), "davies_bouldin": float(davies_bouldin_score(Z[m], labels[m])),
               "calinski_harabasz": float(calinski_harabasz_score(Z[m], labels[m]))}
    return out, sil


def sims_to_centroids(X, C):
    return X @ C.T if len(C) else np.zeros((len(X), 0))


def kmeans_ratio_confidence(km, Z):
    d = km.transform(Z)
    d.sort(axis=1)
    return 1.0 - d[:, 0] / np.clip(d[:, 1], 1e-12, None)


def ari_pair(a, b):
    from sklearn.metrics import adjusted_rand_score
    return float(adjusted_rand_score(a, b))


def subsample_indices(n):
    rng = np.random.default_rng(SEED)
    size = int(round(SUBSAMPLE_FRAC * n))
    return [np.sort(rng.choice(n, size=size, replace=False)) for _ in range(N_SUBSAMPLES)]


def subsample_stability(method, params, Z, subs):
    labs = []
    for idx in subs:
        _, lab, _ = fit_config(method, params, Z[idx])
        full = np.full(len(Z), -2, dtype=int)   # -2 = not in this subsample
        full[idx] = lab
        labs.append(full)
    a_all, a_nn, flips = [], [], []
    for i, j in combinations(range(len(labs)), 2):
        shared = (labs[i] != -2) & (labs[j] != -2)
        li, lj = labs[i][shared], labs[j][shared]
        a_all.append(ari_pair(li, lj))                         # noise (-1) treated as its own label
        both = (li >= 0) & (lj >= 0)
        if both.sum() > 1:
            a_nn.append(ari_pair(li[both], lj[both]))
        flips.append(float(((li < 0) != (lj < 0)).mean()))
    return {"stability_ari": float(np.mean(a_all)), "stability_ari_sd": float(np.std(a_all, ddof=1)),
            "stability_ari_nonnoise": float(np.mean(a_nn)) if a_nn else None,
            "stability_ari_nonnoise_sd": float(np.std(a_nn, ddof=1)) if len(a_nn) > 1 else None,
            "stability_noise_flip_rate": float(np.mean(flips)), "stability_pairs": len(a_all)}


def seed_stability(Z, k):
    labs = [fit_kmeans(Z, k, s)[1] for s in STABILITY_SEEDS]
    a = [ari_pair(labs[i], labs[j]) for i, j in combinations(range(len(labs)), 2)]
    return {"seed_ari_mean": float(np.mean(a)), "seed_ari_sd": float(np.std(a, ddof=1)), "seed_pairs": len(a)}


def granularity_score(n_clusters, median_size):
    if not n_clusters or median_size is None or n_clusters < PATHOLOGY["min_clusters"]:
        return 0.0
    lo, hi = GRANULARITY_MEDIAN_BAND
    if lo <= median_size <= hi:
        return 1.0
    bound = lo if median_size < lo else hi
    return float(max(0.0, 1.0 - abs(math.log(median_size / bound)) / math.log(2)))


def partition_agreement(la, lb):
    """agreement between two labelings of the same rows: ARI with noise as a label, ARI on rows assigned in both, NMI."""
    from sklearn.metrics import normalized_mutual_info_score
    both = (la >= 0) & (lb >= 0)
    return {"ari_noise_as_label": ari_pair(la, lb),
            "ari_both_assigned": ari_pair(la[both], lb[both]) if both.sum() > 1 else None,
            "nmi_both_assigned": float(normalized_mutual_info_score(la[both], lb[both])) if both.sum() > 1 else None,
            "rows_both_assigned": int(both.sum())}


def best_overlap(la, lb):
    """for every cluster of A: best Jaccard with any cluster of B (sets of shared statement ids)."""
    a_ids = sorted(set(la[la >= 0].tolist()))
    b_ids = sorted(set(lb[lb >= 0].tolist()))
    if not a_ids or not b_ids:
        return pd.DataFrame(columns=["cluster_a", "size_a", "best_cluster_b", "size_b", "intersection", "jaccard"])
    ct = pd.crosstab(pd.Series(la, name="a"), pd.Series(lb, name="b"))
    ct = ct.loc[[i for i in ct.index if i >= 0], [j for j in ct.columns if j >= 0]] if len(ct) else ct
    sa = pd.Series(la[la >= 0]).value_counts()
    sb = pd.Series(lb[lb >= 0]).value_counts()
    rows = []
    for a in a_ids:
        if a in ct.index and len(ct.columns):
            inter = ct.loc[a]
            jac = inter / (sa[a] + sb.reindex(inter.index).values - inter)
            b = int(jac.idxmax())
            rows.append({"cluster_a": a, "size_a": int(sa[a]), "best_cluster_b": b, "size_b": int(sb[b]),
                         "intersection": int(inter[b]), "jaccard": float(jac[b])})
        else:
            rows.append({"cluster_a": a, "size_a": int(sa[a]), "best_cluster_b": None, "size_b": 0,
                         "intersection": 0, "jaccard": 0.0})
    return pd.DataFrame(rows)


def overlap_category(j):
    return "robust" if j >= J_ROBUST else ("partial" if j >= J_SENSITIVE else "representation_sensitive")


# ====================================================================== provider robustness helpers
def provider_nn_pairs(X, providers):
    """for every statement: its nearest neighbour from the SAME provider and from the OTHER provider (cosine)."""
    S = X @ X.T
    np.fill_diagonal(S, -np.inf)
    prov = np.asarray(providers)
    rows = []
    for p in sorted(set(prov)):
        same = prov == p
        other = ~same
        if other.sum() == 0 or same.sum() < 2:
            continue
        Si = S[same]
        j_same = np.where(same)[0][np.argmax(Si[:, same], axis=1)]
        j_other = np.where(other)[0][np.argmax(Si[:, other], axis=1)]
        ii = np.where(same)[0]
        rows.append(pd.DataFrame({"i": ii, "j": j_same, "pair_type": "same_provider", "cosine": S[ii, j_same]}))
        rows.append(pd.DataFrame({"i": ii, "j": j_other, "pair_type": "cross_provider", "cosine": S[ii, j_other]}))
    d = pd.concat(rows, ignore_index=True)
    d["bin"] = pd.qcut(d.cosine, 5, labels=False, duplicates="drop")
    return d


def provider_robustness(labels, method, model_obj, pca, X, P, paired, nn_pairs, ids, C, providers):
    out = {}
    # (1) same task, two providers' wording: Gemini statement vs the Groq wording Sprint 2 matched to it
    if len(paired):
        gi = paired.row.values
        if method == "kmeans":
            pred = model_obj.predict(pca.transform(P).astype(np.float64))
        elif hdb_impl() == "hdbscan":
            import hdbscan
            pred, _ = hdbscan.approximate_predict(model_obj, pca.transform(P).astype(np.float64))
        else:
            pred = ids[np.argmax(sims_to_centroids(P, C), axis=1)] if len(ids) else np.full(len(P), -1)
        pred = np.asarray(pred, dtype=int)
        own = labels[gi]
        valid = own >= 0
        both = valid & (pred >= 0)
        lab = labels[labels >= 0]
        shares = np.bincount(lab) / len(lab) if len(lab) else np.array([1.0])
        chance = float((shares ** 2).sum())
        rate_all = float((pred[valid] == own[valid]).mean()) if valid.any() else None   # paraphrase noise counts as miss
        rate_both = float((pred[both] == own[both]).mean()) if both.any() else None
        out.update({"paired_n": int(len(gi)), "paired_gemini_assigned": int(valid.sum()),
                    "paired_groq_wording_noise": int((valid & (pred < 0)).sum()),
                    "paired_same_topic_rate": rate_all, "paired_same_topic_rate_both_assigned": rate_both,
                    "paired_chance_rate": chance,
                    "provider_pair_kappa": (rate_all - chance) / (1 - chance) if rate_all is not None and chance < 1 else None})
    # (2) nearest same-provider vs nearest cross-provider neighbour, compared within cosine bins
    d = nn_pairs.copy()
    li, lj = labels[d.i.values], labels[d.j.values]
    d["both"] = (li >= 0) & (lj >= 0)
    d["same_topic"] = li == lj
    d = d[d.both]
    if len(d):
        g = d.groupby(["bin", "pair_type"]).same_topic.agg(["mean", "size"]).unstack("pair_type")
        if ("mean", "cross_provider") in g.columns and ("mean", "same_provider") in g.columns:
            diff = (g[("mean", "cross_provider")] - g[("mean", "same_provider")])
            w = g[("size", "cross_provider")].fillna(0)
            out["nn_cross_minus_same_provider_same_topic"] = float(np.nansum(diff * w) / max(w.sum(), 1))
            out["nn_same_topic_rate_cross_provider"] = float(d[d.pair_type == "cross_provider"].same_topic.mean())
            out["nn_same_topic_rate_same_provider"] = float(d[d.pair_type == "same_provider"].same_topic.mean())
    # (3) provider composition of topics (descriptive; content differs between providers by construction)
    try:
        from scipy.stats import chi2_contingency
        m = labels >= 0
        ct = pd.crosstab(labels[m], np.asarray(providers)[m])
        if ct.shape[1] > 1 and ct.shape[0] > 1:
            chi2, p, _, _ = chi2_contingency(ct)
            out["provider_cramers_v"] = float(math.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1))))
            out["provider_chi2_p"] = float(p)
    except Exception:
        pass
    return out


# ====================================================================== membership (primary + optional secondary)
def membership(X, labels, ids, C, probs, method, km_conf):
    """Data-derived multi-topic rule. Returns per-statement frame + the derived thresholds."""
    S = sims_to_centroids(X, C)
    pos = {c: i for i, c in enumerate(ids)}
    n = len(labels)
    own = np.full(n, np.nan)
    sec_sim = np.full(n, np.nan)
    sec_id = np.full(n, -1)
    top1_id = ids[np.argmax(S, axis=1)] if len(ids) else np.full(n, -1)
    top1 = S.max(axis=1) if len(ids) else np.full(n, np.nan)
    for i in range(n):
        if labels[i] >= 0:
            p = pos[labels[i]]
            own[i] = S[i, p]
            if len(ids) > 1:
                row = S[i].copy()
                row[p] = -np.inf
                q = int(np.argmax(row))
                sec_sim[i], sec_id[i] = row[q], ids[q]
        elif len(ids) > 1:
            order = np.argsort(-S[i])
            sec_sim[i], sec_id[i] = S[i, order[1]], ids[order[1]]
    m = labels >= 0
    gap = own - sec_sim
    thr = {"secondary_min_similarity": float(np.nanpercentile(own[m], 25)) if m.any() else None,
           "secondary_max_gap": float(np.nanpercentile(gap[m & (gap >= 0)], 10)) if (m & (gap >= 0)).any() else None,
           "noise_post_hoc_min_similarity": float(np.nanpercentile(own[m], 50)) if m.any() else None}
    thr["rule"] = ("secondary topic iff similarity to the best other centroid >= P25 of members' own-centroid "
                   "similarity AND (primary - secondary) <= P10 of non-negative primary/secondary gaps; HDBSCAN noise "
                   "is assigned post hoc only if its best centroid similarity >= P50 of members' own-centroid similarity "
                   "AND its top1-top2 gap >= the same P10 gap; otherwise it stays UNASSIGNED; max 2 memberships")
    if method == "kmeans":
        conf = km_conf
    else:
        conf = probs if probs is not None else np.full(n, np.nan)
    d = pd.DataFrame({"primary": labels, "own_sim": own, "second_id": sec_id, "second_sim": sec_sim,
                      "gap": gap, "top1_id": top1_id, "top1_sim": top1, "confidence": conf})
    d["has_secondary"] = False
    d["post_hoc"] = False
    if thr["secondary_min_similarity"] is not None and thr["secondary_max_gap"] is not None:
        d.loc[m, "has_secondary"] = (d.second_sim[m] >= thr["secondary_min_similarity"]) & (d.gap[m] <= thr["secondary_max_gap"])
        noise = ~m
        top_gap = d.top1_sim - d.second_sim
        d.loc[noise, "post_hoc"] = (d.top1_sim[noise] >= thr["noise_post_hoc_min_similarity"]) & (top_gap[noise] >= thr["secondary_max_gap"])
    d["centroid_inconsistent"] = m & (d.top1_id != d.primary)
    return d, thr


def low_confidence_mask(d, method):
    cut = KMEANS_LOW_CONF if method == "kmeans" else HDB_LOW_CONF
    return (d.primary >= 0) & (d.confidence < cut)


# ====================================================================== selection
def pathology_flags(r):
    f = []
    if r.get("n_clusters", 0) < PATHOLOGY["min_clusters"]:
        f.append("too_few_clusters")
    if r.get("n_clusters", 0) > PATHOLOGY["max_clusters"]:
        f.append("over_fragmented_count")
    if (r.get("stability_ari") or 0) < PATHOLOGY["min_stability_ari"]:
        f.append("unstable")
    if (r.get("dominant_share") or 1) > PATHOLOGY["max_dominant_share"]:
        f.append("dominant_cluster")
    if (r.get("tiny_cluster_share") or 0) > PATHOLOGY["max_tiny_cluster_share"]:
        f.append("over_fragmented_tiny_clusters")
    if (r.get("noise_share") or 0) > PATHOLOGY["max_noise_share"]:
        f.append("excessive_noise")
    if (r.get("semantic_cohesion_lift") or 0) < PATHOLOGY["min_cohesion_lift"]:
        f.append("poor_cohesion")
    return f


def lexicographic_select(rows: pd.DataFrame):
    """Apply HIERARCHY with tolerances to non-pathological rows. Returns (chosen index, trace, deciding level)."""
    ok = rows[rows.pathology_flags == ""]
    pool = ok if len(ok) else rows.assign(_nf=rows.pathology_flags.str.count(";") + 1).nsmallest(
        1, "_nf", keep="all").drop(columns="_nf")
    trace, decided = [], None
    if not len(ok):
        trace.append("no non-pathological configuration - choosing among the least-flagged")
    for col, tol in HIERARCHY:
        v = pool[col].astype(float).fillna(-np.inf)
        best = v.max()
        keep = v >= best - tol
        trace.append(f"{col}: best={fnum(best)} tol={tol} -> kept {', '.join(pool.index[keep.values].astype(str))}")
        pool = pool[keep]
        if len(pool) == 1:
            decided = col
            break
    if len(pool) > 1:
        pool = pool.sort_values(["semantic_cohesion_lift", "stability_ari"], ascending=False)
        decided = "tie_break(semantic_cohesion_lift, stability_ari)"
        trace.append(f"still tied -> tie-break on raw cohesion lift then stability -> {pool.index[0]}")
    return pool.index[0], trace, decided


# ====================================================================== main pipeline
def run(args):
    out = Path(args.output)
    for sub in ("embeddings", "evaluation", "final", "figures", "input"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    ensure_dependencies(args.skip_install)
    lazy_imports()
    set_seeds()
    import sklearn
    import torch
    device, gpu = device_info()
    log(f"device={device} gpu={gpu or 'none'} torch={torch.__version__} sklearn={sklearn.__version__} "
        f"hdbscan_impl={hdb_impl()}")
    if device == "cpu":
        log("NOTE: no GPU detected - embeddings will run on CPU (slower but identical procedure). "
            "In Colab: Runtime > Change runtime type > GPU.")

    df, input_sha = load_input(Path(args.input), args.allow_input_mismatch)
    frozen = input_sha == EXPECTED_INPUT_SHA256
    N = len(df)
    log(f"input rows={N} postings={df.posting_id.nunique()} providers={df.provider.value_counts().to_dict()} "
        f"sha256={input_sha[:16]}... frozen={frozen}")
    try:
        import shutil
        shutil.copyfile(args.input, out / "input" / "sprint3_task_input.csv")
    except Exception as e:
        log(f"could not copy input into results ({e})")
    ck = Checkpoints(out, input_sha)
    timings = {}

    # paired Groq wordings (exclude identical-after-normalisation pairs: they would be trivially identical vectors)
    pt = df.cross_provider_paired_text.fillna("").str.strip()
    pm = (pt != "") & (pt.map(norm_text) != df.normalized_text)
    paired = pd.DataFrame({"row": np.where(pm)[0], "statement_id": df.statement_id[pm].values, "paired_text": pt[pm].values})
    log(f"cross-provider paired wordings available (non-identical): {len(paired)}")

    # ------------------------------------------------------------------ embeddings + PCA
    emb, pcas, Zs, pca_info, paired_emb = {}, {}, {}, {}, {}
    for mk in MODELS:
        t = time.time()
        log(f"=== representation {mk}: {MODELS[mk]} ===")
        emb[mk], paired_emb[mk] = get_embeddings(mk, df, paired, out, device, args.batch_size, input_sha)
        norms = np.linalg.norm(emb[mk], axis=1)
        assert np.allclose(norms, 1, atol=1e-3), f"{mk} embeddings not normalised"
        pcas[mk], Zs[mk], pca_info[mk] = controlled_pca(emb[mk])
        log(f"{mk}: dims={emb[mk].shape[1]} PCA components={pca_info[mk]['n_components']} "
            f"cumulative explained variance={pca_info[mk]['cumulative_explained_variance']:.3f}")
        timings[f"embed_pca_{mk}"] = round(time.time() - t, 1)
    pd.DataFrame([{"embedding_model": MODELS[mk], "component": i + 1, "explained_variance_ratio": v,
                   "cumulative": float(np.cumsum(pca_info[mk]["explained_variance_ratio"])[i])}
                  for mk in MODELS for i, v in enumerate(pca_info[mk]["explained_variance_ratio"])]).to_csv(
        out / "evaluation" / "pca_explained_variance.csv", index=False)

    providers = df.provider.values
    subs = subsample_indices(N)
    nn_pairs = {mk: provider_nn_pairs(emb[mk], providers) for mk in MODELS}

    # ------------------------------------------------------------------ representation-level semantic tests
    rep = ck.get("representation_tests")
    if rep is None:
        rep = {}
        rr = {}
        for mk in MODELS:
            X, P = emb[mk], paired_emb[mk]
            r = {}
            if len(paired):
                S = P @ X.T
                tgt = S[np.arange(len(paired)), paired.row.values]
                rank = 1 + (S > tgt[:, None]).sum(1)
                rr[mk] = 1.0 / rank
                r.update({"paraphrase_queries": int(len(paired)), "paraphrase_recall_at_1": float((rank <= 1).mean()),
                          "paraphrase_recall_at_5": float((rank <= 5).mean()),
                          "paraphrase_recall_at_10": float((rank <= 10).mean()), "paraphrase_mrr": float(rr[mk].mean()),
                          "paraphrase_median_rank": float(np.median(rank))})
            S = X @ X.T
            np.fill_diagonal(S, np.nan)
            rng = np.random.default_rng(SEED)
            a, b = rng.integers(0, N, 20000), rng.integers(0, N, 20000)
            keep = a != b
            rand = S[a[keep], b[keep]]
            top1 = np.nanmax(S, axis=1)
            r.update({"random_pair_cosine_mean": float(np.nanmean(rand)), "random_pair_cosine_sd": float(np.nanstd(rand)),
                      "top1_neighbor_cosine_mean": float(np.nanmean(top1)),
                      "neighbor_contrast_z": float((np.nanmean(top1) - np.nanmean(rand)) / np.nanstd(rand))})
            rep[mk] = r
        if len(paired) and len(rr) == 2:
            diff = rr["e5"] - rr["sentence_t5"]
            rng = np.random.default_rng(SEED)
            boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(1000)]
            rep["mrr_difference_e5_minus_t5"] = {"mean": float(diff.mean()), "ci95_low": float(np.percentile(boots, 2.5)),
                                                 "ci95_high": float(np.percentile(boots, 97.5))}
        # neighbour-set overlap between the two representations
        nn = {}
        for mk in MODELS:
            S = emb[mk] @ emb[mk].T
            np.fill_diagonal(S, -np.inf)
            nn[mk] = np.argsort(-S, axis=1)[:, :NN_K]
        jac = [len(set(nn["e5"][i]) & set(nn["sentence_t5"][i])) / len(set(nn["e5"][i]) | set(nn["sentence_t5"][i])) for i in range(N)]
        rep["top5_neighbor_jaccard_e5_vs_t5"] = float(np.mean(jac))
        ck.put("representation_tests", rep)
    # nearest-neighbour examples for human inspection
    q = df.assign(_h=[h(SEED, "nn", s) for s in df.statement_id]).sort_values("_h").head(NN_SAMPLE).index.values
    rows = []
    for mk in MODELS:
        X = emb[mk]
        S = X[q] @ X.T
        for qi, i in enumerate(q):
            S[qi, i] = -np.inf
            for rank, j in enumerate(np.argsort(-S[qi])[:NN_K], 1):
                rows.append({"embedding_model": MODELS[mk], "query_statement_id": df.statement_id[i],
                             "query_statement": df.statement_text[i], "neighbor_rank": rank,
                             "neighbor_statement_id": df.statement_id[j], "neighbor_statement": df.statement_text[j],
                             "cosine_similarity": round(float(S[qi, j]), 4)})
    pd.DataFrame(rows).to_csv(out / "evaluation" / "nearest_neighbor_examples.csv", index=False)
    pd.DataFrame([{"embedding_model": MODELS[mk], **rep[mk]} for mk in MODELS]).to_csv(
        out / "evaluation" / "representation_tests.csv", index=False)
    log(f"representation tests: { {mk: {k: round(v, 3) for k, v in rep[mk].items()} for mk in MODELS} }")

    # ------------------------------------------------------------------ the 26 clustering configurations
    results, labels_store, extra_store = {}, {}, {}
    grid = config_grid()
    total = len(grid) * len(MODELS)
    n_done = 0
    for mk in MODELS:
        X, Z, P = emb[mk], Zs[mk], paired_emb[mk]
        for method, params in grid:
            n_done += 1
            cid = config_id(mk, method, params)
            cached = ck.get(f"cfg_{cid}")
            if cached is not None:
                results[cid], labels_store[cid], extra_store[cid] = cached
                log(f"[{n_done}/{total}] {cid}: loaded from checkpoint")
                continue
            t = time.time()
            log(f"[{n_done}/{total}] {cid}: fitting")
            model_obj, labels, probs = fit_config(method, params, Z)
            r = {"config_id": cid, "experiment_id": EXPERIMENTS[(mk, method)], "embedding_model": MODELS[mk],
                 "embedding_key": mk, "clustering_method": method, "parameters": params_str(method, params),
                 **{f"param_{k}": v for k, v in params.items()}, "pca_components": pca_info[mk]["n_components"],
                 "pca_cumulative_variance": pca_info[mk]["cumulative_explained_variance"]}
            r.update(size_stats(labels))
            im, sil = internal_metrics(Z, labels)
            r.update(im)
            coh, own_s, ids, C = cohesion(X, labels)
            r.update(coh)
            base = cohesion_baseline(X, labels)
            r["cohesion_random_baseline"] = base
            r["semantic_cohesion_lift"] = lift(coh["semantic_cohesion"], base)
            if method == "hdbscan" and len(ids):
                lab_all = labels.copy()
                noise = lab_all < 0
                lab_all[noise] = ids[np.argmax(sims_to_centroids(X[noise], C), axis=1)] if noise.any() else lab_all[noise]
                coh_all = cohesion(X, lab_all)[0]["semantic_cohesion"]
                r["semantic_cohesion_incl_noise_nearest"] = coh_all
                r["semantic_cohesion_lift_incl_noise"] = lift(coh_all, cohesion_baseline(X, lab_all))
                r["hdbscan_prob_mean"] = float(np.mean(probs[labels >= 0])) if (labels >= 0).any() else None
                r["hdbscan_prob_median"] = float(np.median(probs[labels >= 0])) if (labels >= 0).any() else None
                r["hdbscan_prob_low_share"] = float((probs[labels >= 0] < HDB_LOW_CONF).mean()) if (labels >= 0).any() else None
            else:
                r["semantic_cohesion_incl_noise_nearest"] = coh["semantic_cohesion"]
                r["semantic_cohesion_lift_incl_noise"] = r["semantic_cohesion_lift"]
            km_conf = kmeans_ratio_confidence(model_obj, Z) if method == "kmeans" else None
            if method == "kmeans":
                r.update(seed_stability(Z, params["k"]))
            r.update(subsample_stability(method, params, Z, subs))
            r.update(provider_robustness(labels, method, model_obj, pcas[mk], X, P, paired, nn_pairs[mk], ids, C, providers))
            mem, thr = membership(X, labels, ids, C, probs, method, km_conf)
            lc = low_confidence_mask(mem, method)
            r["low_confidence_rate"] = float(lc.sum() / max((labels >= 0).sum(), 1))
            r["centroid_inconsistency_rate"] = float(mem.centroid_inconsistent.sum() / max((labels >= 0).sum(), 1))
            r["secondary_membership_rate"] = float(mem.has_secondary.mean())
            r["noise_post_hoc_assignable"] = int(mem.post_hoc.sum())
            r["granularity_score"] = granularity_score(r.get("n_clusters"), r.get("median_size"))
            r["fit_seconds"] = round(time.time() - t, 1)
            results[cid] = r
            labels_store[cid] = labels
            extra_store[cid] = {"probs": probs, "km_conf": km_conf, "sil": sil, "own_sim": own_s}
            ck.put(f"cfg_{cid}", (r, labels, extra_store[cid]))
            log(f"    clusters={r['n_clusters']} noise={pct(r['noise_share'])} cohesion_lift={fnum(r['semantic_cohesion_lift'])} "
                f"stability_ari={fnum(r['stability_ari'])} silhouette={fnum(r['silhouette'])} ({r['fit_seconds']}s)")

    # ------------------------------------------------------------------ cross-representation agreement (same method+params)
    cross_rows, cross_summary = [], []

    def cross_compare(cid_a, cid_b, comparison):
        la, lb = labels_store[cid_a], labels_store[cid_b]
        agr = partition_agreement(la, lb)
        ab, ba = best_overlap(la, lb), best_overlap(lb, la)
        for direction, frame, src, dst in (("e5->t5", ab, cid_a, cid_b), ("t5->e5", ba, cid_b, cid_a)):
            for rr in frame.to_dict("records"):
                cross_rows.append({"comparison": comparison, "direction": direction, "config_a": src, "config_b": dst,
                                   **rr, "category": overlap_category(rr["jaccard"])})
        def wmean(f):
            return float((f.jaccard * f.size_a).sum() / f.size_a.sum()) if len(f) else None
        s = {"comparison": comparison, "config_e5": cid_a, "config_t5": cid_b, **agr,
             "mean_best_jaccard_e5_to_t5": float(ab.jaccard.mean()) if len(ab) else None,
             "mean_best_jaccard_t5_to_e5": float(ba.jaccard.mean()) if len(ba) else None,
             "size_weighted_best_jaccard_e5_to_t5": wmean(ab), "size_weighted_best_jaccard_t5_to_e5": wmean(ba),
             "e5_robust_groups": int((ab.jaccard >= J_ROBUST).sum()) if len(ab) else 0,
             "e5_partial_groups": int(((ab.jaccard >= J_SENSITIVE) & (ab.jaccard < J_ROBUST)).sum()) if len(ab) else 0,
             "e5_representation_sensitive_groups": int((ab.jaccard < J_SENSITIVE).sum()) if len(ab) else 0,
             "t5_robust_groups": int((ba.jaccard >= J_ROBUST).sum()) if len(ba) else 0,
             "t5_representation_sensitive_groups": int((ba.jaccard < J_SENSITIVE).sum()) if len(ba) else 0}
        both_dirs = [v for v in (s["mean_best_jaccard_e5_to_t5"], s["mean_best_jaccard_t5_to_e5"]) if v is not None]
        s["mean_best_jaccard"] = float(np.mean(both_dirs)) if both_dirs else None
        cross_summary.append(s)
        return s

    for method, params in grid:
        a, b = config_id("e5", method, params), config_id("sentence_t5", method, params)
        s = cross_compare(a, b, f"same_parameters:{method}:{params_str(method, params)}")
        for cid in (a, b):
            results[cid]["cross_rep_ari"] = s["ari_both_assigned"]
            results[cid]["cross_rep_mean_best_jaccard"] = s["mean_best_jaccard"]
    for cid, r in results.items():
        comps = [v for v in (r.get("provider_pair_kappa"), r.get("cross_rep_ari")) if v is not None]
        r["robustness"] = float(np.mean(comps)) if comps else None
        r["pathology_flags"] = ";".join(pathology_flags(r))

    # ------------------------------------------------------------------ per-experiment candidate selection
    R = pd.DataFrame(results.values()).set_index("config_id", drop=False)
    candidates, selection_trace = {}, {}
    for (mk, method), exp in EXPERIMENTS.items():
        rows = R[R.experiment_id == exp]
        cid, trace, decided = lexicographic_select(rows)
        candidates[exp] = cid
        selection_trace[exp] = {"chosen": cid, "decided_by": decided, "trace": trace}
        log(f"{exp}: candidate {cid} (decided by {decided})")
    R["experiment_candidate"] = R.config_id.isin(candidates.values())

    # cross-representation agreement of the final candidates (same method, possibly different parameters)
    for method in ("kmeans", "hdbscan"):
        a = candidates[EXPERIMENTS[("e5", method)]]
        b = candidates[EXPERIMENTS[("sentence_t5", method)]]
        cross_compare(a, b, f"final_candidates:{method}")
    pd.DataFrame(cross_rows).to_csv(out / "evaluation" / "cross_representation_agreement.csv", index=False)
    pd.DataFrame(cross_summary).to_csv(out / "evaluation" / "cross_representation_summary.csv", index=False)

    # ------------------------------------------------------------------ bootstrap CIs (posting-level) for the 4 candidates
    boot = ck.get("bootstrap")
    if boot is None:
        boot = []
        posts = df.posting_id.values
        up = np.array(sorted(set(posts)))
        by_post = {p: np.where(posts == p)[0] for p in up}
        rng = np.random.default_rng(SEED)
        draws = [rng.choice(up, size=len(up), replace=True) for _ in range(N_BOOTSTRAP)]
        for exp, cid in candidates.items():
            lab, ex = labels_store[cid], extra_store[cid]
            vals = {"semantic_cohesion": [], "silhouette": [], "noise_share": []}
            for dr in draws:
                idx = np.concatenate([by_post[p] for p in dr])
                m = lab[idx] >= 0
                vals["semantic_cohesion"].append(np.nanmean(ex["own_sim"][idx][m]) if m.any() else np.nan)
                vals["silhouette"].append(np.nanmean(ex["sil"][idx][m]) if m.any() and not np.all(np.isnan(ex["sil"])) else np.nan)
                vals["noise_share"].append(float((lab[idx] < 0).mean()))
            for metric, v in vals.items():
                v = np.array(v, dtype=float)
                if metric == "noise_share" and results[cid]["clustering_method"] == "kmeans":
                    continue
                if np.all(np.isnan(v)):
                    continue
                boot.append({"experiment_id": exp, "config_id": cid, "metric": metric,
                             "point_estimate": results[cid].get(metric), "bootstrap_mean": float(np.nanmean(v)),
                             "ci95_low": float(np.nanpercentile(v, 2.5)), "ci95_high": float(np.nanpercentile(v, 97.5)),
                             "n_resamples": N_BOOTSTRAP, "resampling_unit": "posting",
                             "note": "conditional on the fitted partition (refit uncertainty is measured by stability ARI)"})
        ck.put("bootstrap", boot)
    B = pd.DataFrame(boot)
    B.to_csv(out / "evaluation" / "bootstrap_confidence_intervals.csv", index=False)

    # ------------------------------------------------------------------ final selection across the 4 candidates
    C4 = R.loc[list(candidates.values())].copy()
    winner_cid, final_trace, final_decided = lexicographic_select(C4)
    rest = C4.drop(index=winner_cid)
    second_cid, second_trace, second_decided = lexicographic_select(rest)
    # head-to-head: at which hierarchy level do winner and second-best first differ by more than the tolerance
    h2h = []
    for col, tol in HIERARCHY:
        a, b = C4.loc[winner_cid, col], C4.loc[second_cid, col]
        h2h.append({"criterion": col, "tolerance": tol, "winner": a, "second": b,
                    "difference": (a - b) if a is not None and b is not None and not pd.isna(a) and not pd.isna(b) else None})
    win = results[winner_cid]
    log(f"WINNER {winner_cid} | second {second_cid}")

    # Q1 representation evidence (matched configurations + paraphrase test)
    q1 = []
    for col in ("semantic_cohesion_lift", "stability_ari", "provider_pair_kappa", "silhouette", "low_confidence_rate"):
        e, t = [], []
        for method, params in grid:
            ve, vt = results[config_id("e5", method, params)].get(col), results[config_id("sentence_t5", method, params)].get(col)
            if ve is not None and vt is not None:
                e.append(ve)
                t.append(vt)
        if e:
            e, t = np.array(e), np.array(t)
            better = (e < t) if col == "low_confidence_rate" else (e > t)
            q1.append({"criterion": col, "matched_configs": len(e), "e5_mean": float(e.mean()), "t5_mean": float(t.mean()),
                       "e5_better_in": int(better.sum()), "lower_is_better": col == "low_confidence_rate"})
    if "mrr_difference_e5_minus_t5" in rep:
        d = rep["mrr_difference_e5_minus_t5"]
        q1.append({"criterion": "paraphrase_mrr", "matched_configs": None, "e5_mean": rep["e5"].get("paraphrase_mrr"),
                   "t5_mean": rep["sentence_t5"].get("paraphrase_mrr"), "e5_better_in": None,
                   "ci95_diff_low": d["ci95_low"], "ci95_diff_high": d["ci95_high"]})
    Q1 = pd.DataFrame(q1)

    def q1_verdict():
        votes = {"e5": 0, "sentence_t5": 0, "tie": 0}
        notes = []
        for r in q1:
            if r["criterion"] == "paraphrase_mrr":
                lo, hi = r["ci95_diff_low"], r["ci95_diff_high"]
                w = "e5" if lo > 0 else ("sentence_t5" if hi < 0 else "tie")
                notes.append(f"paraphrase MRR E5 {fnum(r['e5_mean'])} vs T5 {fnum(r['t5_mean'])} "
                             f"(diff 95% CI [{fnum(lo)}, {fnum(hi)}]) -> {w}")
            elif r["criterion"] in ("semantic_cohesion_lift", "stability_ari", "provider_pair_kappa"):
                n, k = r["matched_configs"], r["e5_better_in"]
                w = "e5" if k >= 0.75 * n else ("sentence_t5" if k <= 0.25 * n else "tie")
                notes.append(f"{r['criterion']}: E5 mean {fnum(r['e5_mean'])} vs T5 {fnum(r['t5_mean'])}; "
                             f"E5 higher in {k}/{n} matched configurations -> {w}")
            else:
                continue
            votes[w] += 1
        if votes["e5"] > votes["sentence_t5"] and votes["e5"] >= 2:
            v = "e5"
        elif votes["sentence_t5"] > votes["e5"] and votes["sentence_t5"] >= 2:
            v = "sentence_t5"
        else:
            v = "inconclusive"
        return v, votes, notes

    q1_winner, q1_votes, q1_notes = q1_verdict()

    # ------------------------------------------------------------------ final topic tables for the winner
    mk, method = win["embedding_key"], win["clustering_method"]
    X, Z = emb[mk], Zs[mk]
    labels = labels_store[winner_cid]
    ex = extra_store[winner_cid]
    ids, Cm = centroids(X, labels)
    mem, thr = membership(X, labels, ids, Cm, ex["probs"], method, ex["km_conf"])
    sizes = pd.Series(labels[labels >= 0]).value_counts()
    order = sorted(sizes.index, key=lambda c: (-sizes[c], c))
    tid = {c: f"TASK_TOPIC_{i + 1:03d}" for i, c in enumerate(order)}
    topic_rows = []
    for c in order:
        members = np.where(labels == c)[0]
        s = mem.own_sim.values[members]
        o = members[np.argsort(-s)]
        reps, seen = [], set()
        for i in o:
            t = df.normalized_text[i]
            if t not in seen:
                reps.append(df.statement_text[i])
                seen.add(t)
            if len(reps) == 3:
                break
        reps += [""] * (3 - len(reps))
        topic_rows.append({"topic_id": tid[c], "size": int(len(members)), "representative_statement_1": reps[0],
                           "representative_statement_2": reps[1], "representative_statement_3": reps[2],
                           "centroid_statement_id": df.statement_id[o[0]], "semantic_cohesion": float(np.mean(s)),
                           "embedding_model": MODELS[mk], "clustering_method": method,
                           "clustering_parameters": win["parameters"],
                           "secondary_members": int(((mem.second_id == c) & mem.has_secondary).sum()),
                           "post_hoc_noise_members": int(((mem.top1_id == c) & mem.post_hoc).sum())})
    topics = pd.DataFrame(topic_rows)
    topics.to_csv(out / "final" / "task_topics.csv", index=False)
    mrows = []
    for i in range(N):
        base = {"statement_id": df.statement_id[i], "posting_id": df.posting_id[i], "provider": df.provider[i],
                "embedding_model": MODELS[mk], "clustering_method": method}
        r = mem.iloc[i]
        conf = None if pd.isna(r.confidence) else float(r.confidence)
        if r.primary >= 0:
            mrows.append({**base, "topic_id": tid[r.primary], "membership_rank": 1, "is_primary": True,
                          "membership_score": float(r.own_sim), "assignment_confidence": conf, "assignment_source": method})
            if r.has_secondary:
                mrows.append({**base, "topic_id": tid[int(r.second_id)], "membership_rank": 2, "is_primary": False,
                              "membership_score": float(r.second_sim), "assignment_confidence": None,
                              "assignment_source": "secondary_centroid_rule"})
        elif r.post_hoc:
            mrows.append({**base, "topic_id": tid[int(r.top1_id)], "membership_rank": 1, "is_primary": True,
                          "membership_score": float(r.top1_sim), "assignment_confidence": None,
                          "assignment_source": "post_hoc_centroid_rule (HDBSCAN noise)"})
        else:
            mrows.append({**base, "topic_id": "UNASSIGNED", "membership_rank": 0, "is_primary": False,
                          "membership_score": None if pd.isna(r.top1_sim) else float(r.top1_sim),
                          "assignment_confidence": conf, "assignment_source": "noise_unassigned"})
    M = pd.DataFrame(mrows, columns=["statement_id", "posting_id", "provider", "topic_id", "membership_rank", "is_primary",
                                     "membership_score", "assignment_confidence", "embedding_model", "clustering_method",
                                     "assignment_source"])
    M.to_csv(out / "final" / "task_topic_membership.csv", index=False)
    # expanded membership: exact duplicates removed at export inherit the membership of the row they were folded into
    try:
        extra = []
        for r in df.itertuples():
            ids_, posts_, provs_ = (str(x).split(";") for x in (r.duplicate_statement_ids, r.duplicate_posting_ids,
                                                                  r.duplicate_providers))
            if len(ids_) < 2 or not (len(ids_) == len(posts_) == len(provs_)):
                continue
            mine = M[M.statement_id == r.statement_id]
            for sid, pid, prv in zip(ids_, posts_, provs_):
                if sid != r.statement_id:
                    extra.append(mine.assign(statement_id=sid, posting_id=pid, provider=prv))
        Mx = pd.concat([M.assign(kept_statement_id=M.statement_id, via_exact_duplicate=False)] +
                       [e.assign(kept_statement_id=e.index.map(M.statement_id), via_exact_duplicate=True) for e in extra],
                       ignore_index=True)
        Mx.to_csv(out / "final" / "task_topic_membership_expanded_with_duplicates.csv", index=False)
    except Exception as e:
        log(f"expanded membership skipped ({e})")
    thr_rows = {"config_id": winner_cid, **{k: v for k, v in thr.items()},
                "primary_similarity_quantiles": {q: float(np.nanpercentile(mem.own_sim[mem.primary >= 0], q * 100)) for q in (0.05, 0.1, 0.25, 0.5, 0.75, 0.9)},
                "gap_quantiles": {q: float(np.nanpercentile(mem.gap[mem.primary >= 0], q * 100)) for q in (0.05, 0.1, 0.25, 0.5, 0.75, 0.9)},
                "statements_with_secondary": int(mem.has_secondary.sum()),
                "noise_post_hoc_assigned": int(mem.post_hoc.sum()),
                "noise_unassigned": int(((mem.primary < 0) & ~mem.post_hoc).sum())}
    (out / "evaluation" / "membership_threshold_derivation.json").write_text(json.dumps(thr_rows, indent=2))

    # ------------------------------------------------------------------ error analysis (4 candidates)
    ea = []
    for exp, cid in candidates.items():
        r = results[cid]
        mkc, mth = r["embedding_key"], r["clustering_method"]
        Xc, lab = emb[mkc], labels_store[cid]
        exc = extra_store[cid]
        idc, Cc = centroids(Xc, lab)
        memc, _ = membership(Xc, lab, idc, Cc, exc["probs"], mth, exc["km_conf"])
        sz = pd.Series(lab[lab >= 0]).value_counts()
        def add(i, item):
            p = int(memc.primary.iloc[i])
            ea.append({"experiment_id": exp, "config_id": cid, "review_item": item, "statement_id": df.statement_id[i],
                       "posting_id": df.posting_id[i], "provider": df.provider[i], "statement_text": df.statement_text[i],
                       "topic_id": f"C{p}" if p >= 0 else "NOISE", "cluster_size": int(sz.get(p, 0)) if p >= 0 else None,
                       "assignment_confidence": None if pd.isna(memc.confidence.iloc[i]) else float(memc.confidence.iloc[i]),
                       "centroid_similarity": None if pd.isna(memc.own_sim.iloc[i]) else float(memc.own_sim.iloc[i]),
                       "second_topic_id": f"C{int(memc.second_id.iloc[i])}" if memc.second_id.iloc[i] >= 0 else None,
                       "second_similarity": None if pd.isna(memc.second_sim.iloc[i]) else float(memc.second_sim.iloc[i]),
                       "source_section": df.source_section[i], "error_review_label": ""})
        def sample_cluster(c, item):
            members = np.where(lab == c)[0]
            o = members[np.argsort(-memc.own_sim.values[members])]
            for i in list(o[:5]) + [x for x in o[-5:] if x not in o[:5]]:
                add(i, f"{item} (C{c}, n={len(members)})")
        ordered = list(sz.sort_values(ascending=False).index)
        for c in ordered[:5]:
            sample_cluster(c, "largest_cluster")
        mid = len(ordered) // 2
        for c in ordered[max(0, mid - 2): mid + 3]:
            sample_cluster(c, "medium_cluster")
        small = [c for c in sz.sort_values().index if sz[c] >= MEANINGFUL_CLUSTER][:5]
        for c in small:
            sample_cluster(c, "smallest_meaningful_cluster")
        assigned = np.where(lab >= 0)[0]
        for i in assigned[np.argsort(memc.confidence.values[assigned])][:20]:
            add(i, "lowest_confidence_assignment")
        noise = np.where(lab < 0)[0]
        if len(noise):
            for i in noise[np.argsort(-memc.top1_sim.values[noise])][:10]:
                add(i, "noise_example (closest to a centroid)")
            for i in noise[np.argsort(memc.top1_sim.values[noise])][:10]:
                add(i, "noise_example (farthest from all centroids)")
        gapv = memc.gap.values.copy()
        gapv[lab < 0] = np.inf
        for i in np.argsort(np.abs(gapv))[:20]:
            if np.isfinite(gapv[i]):
                add(i, "ambiguous_top1_vs_top2")
    EA = pd.DataFrame(ea)
    EA.to_csv(out / "evaluation" / "cluster_error_analysis.csv", index=False)
    ea_summary = EA.groupby("experiment_id").agg(rows=("statement_id", "size"),
                                                 residual_section_general_description=("source_section", lambda s: float((s == "general_description").mean()))).to_dict("index")

    # ------------------------------------------------------------------ provider consistency table
    pcols = ["config_id", "experiment_id", "embedding_model", "clustering_method", "parameters", "experiment_candidate",
             "paired_n", "paired_gemini_assigned", "paired_groq_wording_noise", "paired_same_topic_rate",
             "paired_same_topic_rate_both_assigned", "paired_chance_rate", "provider_pair_kappa",
             "nn_same_topic_rate_same_provider", "nn_same_topic_rate_cross_provider",
             "nn_cross_minus_same_provider_same_topic", "provider_cramers_v", "provider_chi2_p"]
    R.reindex(columns=pcols).to_csv(out / "evaluation" / "provider_cluster_consistency.csv", index=False)

    # ------------------------------------------------------------------ contamination stress test (winner only)
    stress = ck.get("stress")
    if stress is None:
        mask = df.source_section.isin(STRESS_SECTIONS).values
        sub_idx = np.where(mask)[0]
        Xs = X[sub_idx]
        _, Zs_sub, pinfo_sub = controlled_pca(Xs)                    # same controlled procedure on the subset
        params = {k.replace("param_", ""): win[k] for k in win if k.startswith("param_")}
        params = {k: int(v) for k, v in params.items()}
        _, lab_b, _ = fit_config(method, params, Zs_sub)
        lab_a = labels[sub_idx]
        agr = partition_agreement(lab_a, lab_b)
        bo = best_overlap(lab_a, lab_b)
        full_sizes = pd.Series(labels[labels >= 0]).value_counts()
        major_min = max(20, int(0.01 * N))
        major = [c for c in full_sizes.index if full_sizes[c] >= major_min]
        bo_major = bo[bo.cluster_a.isin(major)]
        stress = {"winner_config": winner_cid, "subset_sections": ";".join(STRESS_SECTIONS),
                  "full_rows": N, "subset_rows": int(len(sub_idx)), "subset_share": float(len(sub_idx) / N),
                  "subset_pca_components": pinfo_sub["n_components"],
                  "subset_pca_cumulative_variance": pinfo_sub["cumulative_explained_variance"],
                  **{f"full_{k}": v for k, v in size_stats(labels).items() if k != "size_distribution"},
                  **{f"subset_{k}": v for k, v in size_stats(lab_b).items() if k != "size_distribution"},
                  **agr, "mean_best_jaccard_full_to_subset": float(bo.jaccard.mean()) if len(bo) else None,
                  "major_topic_min_size": major_min, "major_topics": len(major),
                  "major_topics_preserved_jaccard_ge_0_5": int((bo_major.jaccard >= J_ROBUST).sum()),
                  "major_topic_preservation_rate": float((bo_major.jaccard >= J_ROBUST).mean()) if len(bo_major) else None,
                  "audit_note": (f"audit purity by section: responsibilities {AUDIT_PURITY['responsibilities']}, "
                                 f"general_description {AUDIT_PURITY['general_description']} - the responsibilities-only "
                                 f"subset is the requested 'duty-type' subset, not one the audit proves cleaner")}
        stress_rows = bo.assign(major=bo.cluster_a.isin(major), category=bo.jaccard.map(overlap_category))
        ck.put("stress", (stress, stress_rows))
    else:
        stress, stress_rows = stress
    pd.concat([pd.DataFrame([stress]).assign(row_type="summary"),
               stress_rows.assign(row_type="full_topic_vs_best_subset_topic")], ignore_index=True).to_csv(
        out / "evaluation" / "contamination_stress_test.csv", index=False)

    # ------------------------------------------------------------------ experiment tables
    R["selected_final"] = R.config_id == winner_cid
    R["selected_second"] = R.config_id == second_cid
    R.drop(columns=[], errors="ignore").to_csv(out / "evaluation" / "clustering_experiments.csv", index=False)
    comp_cols = ["experiment_id", "config_id", "embedding_model", "clustering_method", "parameters", "n_clusters",
                 "noise_share", "semantic_cohesion", "cohesion_random_baseline", "semantic_cohesion_lift",
                 "semantic_cohesion_lift_incl_noise", "stability_ari", "stability_ari_sd", "stability_ari_nonnoise",
                 "seed_ari_mean", "silhouette", "davies_bouldin", "calinski_harabasz", "min_size", "median_size",
                 "max_size", "dominant_share", "tiny_cluster_share", "very_small_clusters", "granularity_score",
                 "low_confidence_rate", "centroid_inconsistency_rate", "secondary_membership_rate",
                 "provider_pair_kappa", "nn_cross_minus_same_provider_same_topic", "cross_rep_ari",
                 "cross_rep_mean_best_jaccard", "robustness", "pathology_flags"]
    CMP = R.loc[list(candidates.values())].reindex(columns=comp_cols)
    CMP["nearest_neighbor_paraphrase_mrr"] = [rep[results[c]["embedding_key"]].get("paraphrase_mrr") for c in CMP.index]
    CMP["error_analysis_rows"] = [ea_summary.get(results[c]["experiment_id"], {}).get("rows") for c in CMP.index]
    for metric in ("semantic_cohesion", "silhouette", "noise_share"):
        sub = B[B.metric == metric].set_index("config_id") if len(B) else pd.DataFrame()
        CMP[f"{metric}_ci95"] = [f"[{fnum(sub.loc[c, 'ci95_low'])}, {fnum(sub.loc[c, 'ci95_high'])}]" if c in sub.index else ""
                                 for c in CMP.index]
    CMP["final_rank"] = ["1 (winner)" if c == winner_cid else ("2 (second)" if c == second_cid else "") for c in CMP.index]
    CMP.to_csv(out / "evaluation" / "experiment_comparison.csv", index=False)

    # ------------------------------------------------------------------ figures (best effort)
    figs = make_figures(out, R, candidates, rep, pca_info, B, labels, emb[mk], tid, winner_cid, mem)

    # ------------------------------------------------------------------ manifest + reports
    timings["total_seconds"] = round(time.time() - T0, 1)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "script": Path(__file__).name,
        "config_fingerprint": CONFIG_FINGERPRINT, "input_filename": Path(args.input).name, "input_path": str(args.input),
        "input_sha256": input_sha, "expected_input_sha256": EXPECTED_INPUT_SHA256, "input_is_frozen_version": frozen,
        "row_count": N, "unique_postings": int(df.posting_id.nunique()),
        "provider_counts": df.provider.value_counts().to_dict(), "section_counts": df.source_section.value_counts().to_dict(),
        "e5_model": E5_MODEL, "e5_input_format": f"'{E5_PREFIX}<statement_text>', mean pooling, L2-normalised",
        "sentence_t5_model": T5_MODEL, "sentence_t5_input_format": "<statement_text>, L2-normalised",
        "library_versions": {p: pkg_version(p) for p in ("numpy", "pandas", "scipy", "scikit-learn", "torch", "transformers",
                                                         "sentence-transformers", "hdbscan", "umap-learn", "matplotlib")},
        "python": sys.version.split()[0], "platform": platform.platform(), "device": device, "gpu": gpu,
        "hdbscan_implementation": hdb_impl(),
        "random_seeds": {"global": SEED, "kmeans_stability_seeds": STABILITY_SEEDS, "subsample_rng": SEED,
                         "bootstrap_rng": SEED, "umap_random_state": SEED},
        "pca": {mkk: {k: v for k, v in pca_info[mkk].items() if k != "explained_variance_ratio"} for mkk in MODELS},
        "pca_max_components": PCA_MAX_COMPONENTS,
        "grids": {"kmeans_k": KMEANS_K, "kmeans_n_init": KMEANS_N_INIT, "kmeans_init": "k-means++",
                  "hdbscan_min_cluster_size": HDB_MIN_CLUSTER_SIZE, "hdbscan_min_samples": HDB_MIN_SAMPLES,
                  "hdbscan_metric": "euclidean (on PCA space)", "hdbscan_cluster_selection_method": "eom"},
        "stability_protocol": {"subsamples": N_SUBSAMPLES, "fraction": SUBSAMPLE_FRAC, "kmeans_seed_restarts": len(STABILITY_SEEDS)},
        "bootstrap": {"resamples": N_BOOTSTRAP, "unit": "posting"},
        "pathology_thresholds": PATHOLOGY, "decision_hierarchy": HIERARCHY, "granularity_median_band": GRANULARITY_MEDIAN_BAND,
        "experiment_candidates": candidates, "selection_trace": selection_trace,
        "chosen_configuration": {"config_id": winner_cid, "embedding_model": win["embedding_model"],
                                 "clustering_method": win["clustering_method"], "parameters": win["parameters"],
                                 "decided_by": final_decided, "trace": final_trace},
        "second_best_configuration": {"config_id": second_cid, "decided_by": second_decided},
        "membership_rule": thr, "runtime_seconds": timings, "figures": figs,
        "known_input_quality_limitation": LIMITATION,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    write_model_selection(out, R, CMP, candidates, selection_trace, winner_cid, second_cid, final_trace, final_decided,
                          h2h, Q1, q1_winner, q1_notes, rep, stress, B)
    write_report(out, df, input_sha, frozen, pca_info, R, CMP, candidates, winner_cid, second_cid, final_decided, h2h,
                 rep, Q1, q1_winner, q1_notes, pd.DataFrame(cross_summary), stress, B, thr_rows, topics, M, EA, timings,
                 len(paired))
    zip_results(out, Path(args.zip) if args.zip else out.parent / "Sprint3_Experiment_Results.zip")
    log("DONE")


# ====================================================================== figures
def make_figures(out, R, candidates, rep, pca_info, B, labels, X, tid, winner_cid, mem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fd = out / "figures"
    done, errors = [], []

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

    tag = {E5_MODEL: "E5", T5_MODEL: "Sentence-T5"}

    def f1():
        fig_, ax = plt.subplots(1, 2, figsize=(11, 4))
        ms = [k for k in MODELS]
        keys = ["paraphrase_recall_at_1", "paraphrase_recall_at_5", "paraphrase_recall_at_10", "paraphrase_mrr"]
        w = 0.38
        for i, mkk in enumerate(ms):
            ax[0].bar(np.arange(len(keys)) + i * w, [rep[mkk].get(k, np.nan) for k in keys], w, label=MODEL_TAG[mkk])
        ax[0].set_xticks(np.arange(len(keys)) + w / 2, ["R@1", "R@5", "R@10", "MRR"])
        ax[0].set_title("Cross-provider paraphrase retrieval")
        ax[0].legend()
        g = R.groupby(["embedding_model", "clustering_method"]).semantic_cohesion_lift.mean().unstack()
        g.index = [tag.get(i, i) for i in g.index]
        g.plot.bar(ax=ax[1], rot=0)
        ax[1].set_title("Mean cohesion lift over the sweep (chance-adjusted)")
    fig("01_representation_comparison.png", f1)

    def f2():
        K = R[R.clustering_method == "kmeans"]
        fig_, ax = plt.subplots(1, 3, figsize=(14, 4))
        for m, g in K.groupby("embedding_model"):
            g = g.sort_values("param_k")
            ax[0].plot(g.param_k, g.semantic_cohesion_lift, "o-", label=tag.get(m, m))
            ax[1].plot(g.param_k, g.silhouette, "o-", label=tag.get(m, m))
            ax[2].errorbar(g.param_k, g.stability_ari, yerr=g.stability_ari_sd, fmt="o-", label=tag.get(m, m), capsize=3)
        for a, t in zip(ax, ["cohesion lift", "silhouette (PCA space)", "subsample stability ARI"]):
            a.set_xlabel("k")
            a.set_title(t)
            a.legend()
    fig("02_kmeans_sweep.png", f2)

    def f3():
        Hh = R[R.clustering_method == "hdbscan"]
        fig_, ax = plt.subplots(figsize=(7, 5))
        for m, g in Hh.groupby("embedding_model"):
            ax.scatter(100 * g.noise_share, g.n_clusters, label=tag.get(m, m), s=50)
            for r in g.itertuples():
                ax.annotate(f"{int(r.param_min_cluster_size)}/{int(r.param_min_samples)}", (100 * r.noise_share, r.n_clusters), fontsize=7)
        ax.set_xlabel("noise %")
        ax.set_ylabel("clusters (excl. noise)")
        ax.set_title("HDBSCAN cluster count vs noise (labels: min_cluster_size/min_samples)")
        ax.legend()
    fig("03_hdbscan_cluster_noise_tradeoff.png", f3)

    def f4():
        fig_, ax = plt.subplots(figsize=(12, 4.5))
        d = R.sort_values(["clustering_method", "embedding_model", "config_id"])
        cols = ["tab:blue" if m == E5_MODEL else "tab:orange" for m in d.embedding_model]
        ax.bar(range(len(d)), d.stability_ari, yerr=d.stability_ari_sd, color=cols, capsize=2)
        ax.set_xticks(range(len(d)), [c.split("__")[0].replace("EXP_", "")[:8] + "\n" + c.split("__")[1] for c in d.config_id], rotation=90, fontsize=6)
        ax.axhline(PATHOLOGY["min_stability_ari"], ls="--", c="gray")
        ax.set_title("Subsample stability (mean pairwise ARI, 5 x 80%) - blue E5, orange Sentence-T5")
    fig("04_stability_comparison.png", f4)

    def f5():
        d = B[B.metric == "semantic_cohesion"] if len(B) else pd.DataFrame()
        fig_, ax = plt.subplots(figsize=(7, 4))
        ax.errorbar(range(len(d)), d.point_estimate, yerr=[d.point_estimate - d.ci95_low, d.ci95_high - d.point_estimate], fmt="o", capsize=4)
        ax.set_xticks(range(len(d)), d.experiment_id, rotation=20, fontsize=8)
        ax.set_title("Semantic cohesion of the 4 candidates (95% posting bootstrap CI)\n"
                     "raw cosine: not comparable across embedding models - see cohesion lift")
    fig("05_semantic_cohesion_comparison.png", f5)

    def f6():
        s = pd.Series(labels[labels >= 0]).value_counts()
        fig_, ax = plt.subplots(figsize=(7, 4))
        ax.hist(s.values, bins=30)
        ax.set_xlabel("topic size")
        ax.set_ylabel("topics")
        ax.set_title(f"Final topic size distribution ({winner_cid})")
    fig("06_final_cluster_size_distribution.png", f6)

    def f7():
        fig_, ax = plt.subplots(figsize=(6, 4))
        for mkk in MODELS:
            ax.plot(np.arange(1, pca_info[mkk]["n_components"] + 1), np.cumsum(pca_info[mkk]["explained_variance_ratio"]), label=MODEL_TAG[mkk])
        ax.set_xlabel("components")
        ax.set_ylabel("cumulative explained variance")
        ax.legend()
        ax.set_title("Controlled PCA (same procedure for both models)")
    fig("07_pca_explained_variance.png", f7)

    def f8():
        import umap
        Y = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=SEED).fit_transform(X)
        fig_, ax = plt.subplots(figsize=(9, 8))
        top = pd.Series(labels[labels >= 0]).value_counts().index[:20]
        ax.scatter(Y[:, 0], Y[:, 1], s=3, c="lightgray")
        cmap = plt.get_cmap("tab20")
        for i, c in enumerate(top):
            m = labels == c
            ax.scatter(Y[m, 0], Y[m, 1], s=4, color=cmap(i % 20), label=tid[c])
        ax.legend(markerscale=3, fontsize=6, ncol=2)
        ax.set_title(f"UMAP (visualisation only - not used for clustering)\n{winner_cid}; 20 largest topics coloured")
    fig("08_final_umap.png", f8)
    if errors:
        (fd / "figure_errors.txt").write_text("\n\n".join(errors))
    return {"created": done, "failed": [e.split(":")[0] for e in errors]}


# ====================================================================== reports
def _row(r, cols):
    return "| " + " | ".join(str(r.get(c, "")) for c in cols) + " |"


def write_model_selection(out, R, CMP, candidates, selection_trace, winner, second, final_trace, final_decided, h2h,
                          Q1, q1_winner, q1_notes, rep, stress, B):
    W, S = R.loc[winner], R.loc[second]
    rep_label = {"e5": "E5 (intfloat/e5-base-v2)", "sentence_t5": "Sentence-T5 (sentence-transformers/sentence-t5-base)",
                 "inconclusive": "INCONCLUSIVE - the evidence does not clearly favour either representation"}
    # Q2 inside the winning representation
    mk = W.embedding_key
    km, hd = candidates[EXPERIMENTS[(mk, "kmeans")]], candidates[EXPERIMENTS[(mk, "hdbscan")]]
    q2_idx, q2_trace, q2_dec = lexicographic_select(R.loc[[km, hd]])
    q2m = R.loc[q2_idx].clustering_method
    close_levels = [x for x in h2h if x["difference"] is not None and abs(x["difference"]) <= x["tolerance"]]
    lines = ["# Sprint 3 Model Selection", "",
             "Decision procedure (pre-declared in the script): (1) reject pathological configurations "
             f"({json.dumps(PATHOLOGY)}); (2) lexicographic hierarchy with tolerances "
             f"{', '.join(f'{c} (tol {t})' for c, t in HIERARCHY)}. A level only breaks ties among configurations "
             "that are within tolerance of the best on all higher levels. No single metric decides.", "",
             "Cosine-based cohesion is compared across embedding models only as a chance-adjusted **lift** "
             "((cohesion - random-partition cohesion) / (1 - random-partition cohesion)) because raw cosine scales differ "
             "between E5 and Sentence-T5. Silhouette is measured in each model's own PCA space and is the lowest-priority "
             "criterion.", "",
             "## 1. E5 vs Sentence-T5", "", f"**Verdict: {rep_label[q1_winner]}**", ""]
    lines += [f"- {n}" for n in q1_notes]
    lines += ["", "| criterion | matched configs | E5 | T5 | E5 better in |", "|---|---:|---:|---:|---:|"]
    for r in Q1.to_dict("records"):
        lines.append(f"| {r['criterion']} | {r.get('matched_configs') if r.get('matched_configs') is not None else '-'} | "
                     f"{fnum(r['e5_mean'])} | {fnum(r['t5_mean'])} | {r.get('e5_better_in') if r.get('e5_better_in') is not None else '-'} |")
    lines += ["", f"Top-5 neighbour sets agree between E5 and T5 with mean Jaccard {fnum(rep.get('top5_neighbor_jaccard_e5_vs_t5'))}. "
              "Inspect `nearest_neighbor_examples.csv` before accepting the verdict.", "",
              f"## 2. K-Means vs HDBSCAN (within {rep_label.get(mk, mk).split(' (')[0]})", "",
              f"Candidates: K-Means `{km}` vs HDBSCAN `{hd}` -> **{q2m}** (decided by {q2_dec}).", ""]
    lines += [f"- {t}" for t in q2_trace]
    cols = ["config_id", "n_clusters", "noise_share", "semantic_cohesion_lift", "stability_ari", "granularity_score",
            "robustness", "silhouette", "low_confidence_rate", "pathology_flags"]
    lines += ["", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for c in (km, hd):
        rr = R.loc[c]
        lines.append("| " + " | ".join(fnum(rr[x]) if isinstance(rr[x], (float, np.floating)) else str(rr[x]) for x in cols) + " |")
    lines += ["", "## 3. Winning pipeline", "",
              f"- configuration: `{winner}`",
              f"- embedding model: {W.embedding_model}", f"- clustering method: {W.clustering_method}",
              f"- exact parameters: {W.parameters} (PCA {int(W.pca_components)} components, "
              f"cumulative variance {fnum(W.pca_cumulative_variance)}; seed {SEED})",
              f"- leaf topic count: {int(W.n_clusters)}",
              f"- noise / unassigned (before post-hoc rule): {int(W.noise_count)} ({pct(W.noise_share)})",
              f"- semantic cohesion: {fnum(W.semantic_cohesion)} (chance {fnum(W.cohesion_random_baseline)}, lift {fnum(W.semantic_cohesion_lift)})",
              f"- stability: subsample ARI {fnum(W.stability_ari)} ± {fnum(W.stability_ari_sd)}"
              + (f"; seed-restart ARI {fnum(W.seed_ari_mean)}" if not pd.isna(W.get("seed_ari_mean", np.nan)) else ""),
              f"- robustness: provider-pair kappa {fnum(W.provider_pair_kappa)}, cross-representation ARI {fnum(W.cross_rep_ari)}, "
              f"contamination stress-test major-topic preservation {pct(stress.get('major_topic_preservation_rate'))} "
              f"(ARI {fnum(stress.get('ari_both_assigned'))})",
              f"- decided by: {final_decided}", ""]
    lines += [f"  - {t}" for t in final_trace]
    lines += ["", "## 4. Second-best pipeline", "", f"`{second}` - {S.embedding_model}, {S.clustering_method}, {S.parameters}; "
              f"{int(S.n_clusters)} topics, noise {pct(S.noise_share)}, cohesion lift {fnum(S.semantic_cohesion_lift)}, "
              f"stability ARI {fnum(S.stability_ari)}.", "", "## 5. Why the winner beat the second-best", "",
              "| criterion | tolerance | winner | second | difference |", "|---|---:|---:|---:|---:|"]
    for x in h2h:
        lines.append(f"| {x['criterion']} | {x['tolerance']} | {fnum(x['winner'])} | {fnum(x['second'])} | {fnum(x['difference'])} |")
    decisive = next((x for x in h2h if x["difference"] is not None and abs(x["difference"]) > x["tolerance"]), None)
    if decisive is None:
        lines += ["", "**The result is CLOSE: the two pipelines are within tolerance on every criterion.** The winner was chosen "
                  "by the tie-break; treat both as viable and prefer the one whose error-analysis sample reads better."]
    else:
        lines += ["", f"First criterion (in priority order) where they differ beyond tolerance: **{decisive['criterion']}** "
                  f"(winner {fnum(decisive['winner'])} vs {fnum(decisive['second'])})."]
        if close_levels:
            lines.append(f"Within tolerance (effectively tied) on: {', '.join(x['criterion'] for x in close_levels)}.")
        if decisive["criterion"] in ("robustness", "silhouette"):
            lines.append("Because the decision falls to a low-priority criterion, the margin is **small**; do not over-interpret it.")
    lines += ["", "## Per-experiment candidate selection", ""]
    for exp, t in selection_trace.items():
        lines += [f"**{exp}** -> `{t['chosen']}` (decided by {t['decided_by']})"] + [f"- {x}" for x in t["trace"]] + [""]
    (out / "evaluation" / "model_selection.md").write_text("\n".join(lines), encoding="utf-8")


def write_report(out, df, sha, frozen, pca_info, R, CMP, candidates, winner, second, decided, h2h, rep, Q1, q1_winner,
                 q1_notes, XS, stress, B, thr, topics, M, EA, timings, n_paired):
    W = R.loc[winner]

    def tbl(frame, cols, fmt=None):
        s = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for r in frame.to_dict("records"):
            s.append("| " + " | ".join((fnum(r.get(c)) if isinstance(r.get(c), (float, np.floating)) else str(r.get(c, "")))
                                       for c in cols) + " |")
        return "\n".join(s)

    def ci(cid, metric):
        d = B[(B.config_id == cid) & (B.metric == metric)] if len(B) else B
        return f"[{fnum(d.ci95_low.iloc[0])}, {fnum(d.ci95_high.iloc[0])}]" if len(d) else "n/a"

    K = R[R.clustering_method == "kmeans"].sort_values(["embedding_model", "param_k"])
    Hh = R[R.clustering_method == "hdbscan"].sort_values(["embedding_model", "param_min_cluster_size", "param_min_samples"])
    sec = M[M.membership_rank == 2]
    un = M[M.topic_id == "UNASSIGNED"]
    xs_same = XS[XS.comparison.str.startswith("same_parameters")]
    xs_final = XS[XS.comparison.str.startswith("final_candidates")]
    rep_label = {"e5": "E5", "sentence_t5": "Sentence-T5", "inconclusive": "inconclusive"}
    lines = [
        "# Sprint 3 - Representation & Clustering: Stability Report", "",
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by `{Path(__file__).name}`. "
        "All numbers below are computed by the script from the frozen input; nothing is hand-entered.", "",
        "> **Known limitation.** " + LIMITATION, "",
        "## 1. Objective",
        "Choose a semantic representation for extracted work-task statements and a clustering method that yields stable, "
        "semantically cohesive leaf topics, via a controlled 2x2 experiment (E5 / Sentence-T5 x K-Means / HDBSCAN). "
        "No labels exist for clusters: every metric below is an internal or robustness measure, **not accuracy**.", "",
        "## 2. Frozen input",
        f"- file `{Path(df.attrs.get('path', 'sprint3_task_input.csv')).name}`, SHA-256 `{sha}` "
        f"({'matches the frozen version' if frozen else 'DOES NOT MATCH the frozen version - non-frozen run'})",
        f"- {len(df):,} statements (after removing exact normalised duplicates in export), {df.posting_id.nunique()} postings, "
        f"providers {df.provider.value_counts().to_dict()}, sections {df.source_section.value_counts().to_dict()}",
        f"- {n_paired} Gemini statements carry a non-identical Groq wording of the same task (Sprint 2 cross-model agreement); "
        "these are used only for robustness/representation tests, never clustered.", "",
        "## 3. Representation methodology",
        f"- **E5**: `{E5_MODEL}`, input `\"{E5_PREFIX}<statement>\"` for every statement (no query/passage mixing), "
        "attention-masked mean pooling, L2 normalisation.",
        f"- **Sentence-T5**: `{T5_MODEL}`, raw statement text, L2 normalisation.",
        "- Original normalised embeddings are kept for cosine similarity, nearest neighbours, centroid cohesion and memberships.",
        "- Clustering space: identical controlled PCA for both models (max 50 components, full SVD, no whitening): "
        + "; ".join(f"{MODEL_TAG[m]} {pca_info[m]['n_components']} components, cumulative explained variance "
                    f"{fnum(pca_info[m]['cumulative_explained_variance'])}" for m in MODELS) + ". UMAP is used only for one figure.", "",
        "## 4. Clustering methodology",
        f"- K-Means: k in {KMEANS_K}, k-means++, n_init={KMEANS_N_INIT}, random_state={SEED}.",
        f"- HDBSCAN ({'hdbscan package' if hdb_impl() == 'hdbscan' else 'sklearn implementation'}): min_cluster_size in "
        f"{HDB_MIN_CLUSTER_SIZE} x min_samples in {HDB_MIN_SAMPLES}, Euclidean on PCA space, EOM selection; noise is never forced.",
        f"- Stability: {N_SUBSAMPLES} deterministic {int(SUBSAMPLE_FRAC * 100)}% subsamples, identical for every configuration; pairwise "
        "ARI on shared statements (noise counted as its own label; ARI on rows assigned in both runs also reported). K-Means "
        f"additionally: {len(STABILITY_SEEDS)} seed restarts on the full data.",
        "- Semantic cohesion: mean cosine of members to their cluster centroid in the original embedding space; "
        f"chance baseline from {N_RANDOM_BASELINE} size-preserving random partitions; **cohesion lift** = (cohesion - chance)/(1 - chance).", "",
        "## 5. 2x2 experimental comparison (one candidate per cell)",
        tbl(CMP.reset_index(drop=True), ["experiment_id", "parameters", "n_clusters", "noise_share", "semantic_cohesion",
                                         "semantic_cohesion_lift", "stability_ari", "silhouette", "granularity_score",
                                         "robustness", "pathology_flags", "final_rank"]), "",
        "## 6. Parameter sweep", "### K-Means",
        tbl(K, ["embedding_model", "param_k", "silhouette", "davies_bouldin", "calinski_harabasz", "min_size", "median_size",
                "max_size", "very_small_clusters", "semantic_cohesion_lift", "stability_ari", "seed_ari_mean", "pathology_flags"]),
        "", "### HDBSCAN",
        tbl(Hh, ["embedding_model", "param_min_cluster_size", "param_min_samples", "n_clusters", "noise_count", "noise_share",
                 "min_size", "median_size", "max_size", "hdbscan_prob_mean", "silhouette", "semantic_cohesion_lift",
                 "semantic_cohesion_lift_incl_noise", "stability_ari", "stability_ari_nonnoise", "pathology_flags"]), "",
        "## 7. Stability",
        tbl(R.sort_values("stability_ari", ascending=False).head(10), ["config_id", "stability_ari", "stability_ari_sd",
                                                                        "stability_ari_nonnoise", "stability_noise_flip_rate", "seed_ari_mean"]),
        f"\nConfigurations flagged unstable (subsample ARI < {PATHOLOGY['min_stability_ari']}): "
        f"{int(R.pathology_flags.str.contains('unstable').sum())} of {len(R)}.", "",
        "## 8. Semantic cohesion (95% posting-bootstrap CIs, conditional on the fitted partition)",
        tbl(CMP.reset_index(drop=True).assign(ci=[ci(c, "semantic_cohesion") for c in CMP.config_id]),
            ["experiment_id", "semantic_cohesion", "ci", "cohesion_random_baseline", "semantic_cohesion_lift"]),
        "\nRaw cohesion is not comparable across embedding models (different cosine scales); compare the lift.", "",
        "## 9. Nearest-neighbour analysis",
        f"`evaluation/nearest_neighbor_examples.csv` lists the top-{NN_K} neighbours of {NN_SAMPLE} deterministic sample statements "
        "under each model for human inspection. Quantitative proxy - cross-provider paraphrase retrieval (rank of the Gemini "
        "statement when its Groq wording is the query, over the full corpus):",
        tbl(pd.DataFrame([{"model": MODEL_TAG[m], **rep[m]} for m in MODELS]),
            ["model", "paraphrase_queries", "paraphrase_recall_at_1", "paraphrase_recall_at_5", "paraphrase_mrr",
             "paraphrase_median_rank", "random_pair_cosine_mean", "neighbor_contrast_z"]),
        f"\nTop-5 neighbour-set Jaccard E5 vs T5: {fnum(rep.get('top5_neighbor_jaccard_e5_vs_t5'))}. Caveat: Sprint 2 accepted "
        "these pairs by lexical/semantic agreement, so they skew towards easy paraphrases.", "",
        f"## 10. E5 vs Sentence-T5: **{rep_label[q1_winner]}**", *[f"- {n}" for n in q1_notes], "",
        "## 11. K-Means vs HDBSCAN", "See `evaluation/model_selection.md` section 2 for the within-representation comparison; "
        f"the overall winner uses **{W.clustering_method}**.", "",
        "## 12. Cross-representation agreement (E5 vs Sentence-T5, same method and parameters)",
        tbl(pd.concat([xs_same, xs_final]), ["comparison", "ari_both_assigned", "nmi_both_assigned", "mean_best_jaccard",
                                             "e5_robust_groups", "e5_representation_sensitive_groups", "t5_robust_groups",
                                             "t5_representation_sensitive_groups"]),
        f"\nRobust group: best Jaccard >= {J_ROBUST}; representation-sensitive: < {J_SENSITIVE}. Cluster-level rows: "
        "`evaluation/cross_representation_agreement.csv`.", "",
        "## 13. Gemini/Groq provider robustness",
        "Primary statistic: for Gemini statements whose task Groq also extracted (different wording, same evidence), "
        "the Groq wording is assigned out-of-sample by the fitted model (K-Means predict / HDBSCAN approximate_predict) and we "
        "measure how often it lands in the Gemini statement's topic, chance-corrected (kappa-style: (rate - chance)/(1 - chance)). "
        "Secondary: same-topic rate of nearest cross-provider vs nearest same-provider neighbours within matched cosine bins "
        "(a value near 0 means topics follow meaning, not provider wording). Provider equality is not treated as ground truth.",
        tbl(CMP.reset_index(drop=True), ["experiment_id", "provider_pair_kappa", "nn_cross_minus_same_provider_same_topic"]), "",
        "## 14. Contamination stress test (winner only)",
        f"Winner refit with identical parameters on the {', '.join(STRESS_SECTIONS)}-only subset "
        f"({stress['subset_rows']:,} of {stress['full_rows']:,} statements) vs the full run restricted to the same statements: "
        f"ARI (both assigned) {fnum(stress.get('ari_both_assigned'))}, NMI {fnum(stress.get('nmi_both_assigned'))}, "
        f"mean best Jaccard {fnum(stress.get('mean_best_jaccard_full_to_subset'))}, major topics (size >= "
        f"{stress['major_topic_min_size']}) preserved at Jaccard >= {J_ROBUST}: {stress['major_topics_preserved_jaccard_ge_0_5']}"
        f"/{stress['major_topics']} ({pct(stress.get('major_topic_preservation_rate'))}).",
        f"Caveat: {stress['audit_note']}.", "",
        "## 15. Bootstrap uncertainty",
        tbl(B, ["experiment_id", "metric", "point_estimate", "ci95_low", "ci95_high", "n_resamples"]) if len(B) else "not computed", "",
        "## 16. Multi-topic membership",
        f"Rule (derived from the winner's own distributions; quantiles in `evaluation/membership_threshold_derivation.json`): "
        f"{thr['rule']}.",
        f"- secondary similarity floor {fnum(thr.get('secondary_min_similarity'))}, maximum primary-secondary gap "
        f"{fnum(thr.get('secondary_max_gap'))}, noise post-hoc floor {fnum(thr.get('noise_post_hoc_min_similarity'))}",
        f"- statements with a secondary topic: {len(sec):,} ({pct(len(sec) / len(df))}); HDBSCAN noise assigned post hoc: "
        f"{thr.get('noise_post_hoc_assigned', 0):,}; left UNASSIGNED: {len(un):,}.",
        "- Decision: one statement MAY belong to two topics (primary + at most one secondary) when the rule fires; "
        "unresolved statements stay unassigned rather than being forced.", "",
        "## 17. Error analysis",
        f"`evaluation/cluster_error_analysis.csv` holds {len(EA):,} rows (largest / medium / smallest meaningful clusters, "
        "lowest-confidence assignments, noise examples, ambiguous top-1 vs top-2) for the 4 candidates, with "
        "`error_review_label` left blank for human review (GOOD, MIXED_TOPIC, TOO_BROAD, OVER_SPLIT, RESIDUAL_NON_TASK, "
        "AMBIGUOUS_TASK, OUTLIER). No LLM was used to label them.",
        tbl(CMP.reset_index(drop=True), ["experiment_id", "low_confidence_rate", "centroid_inconsistency_rate",
                                         "secondary_membership_rate"]), "",
        "## 18. Final chosen configuration",
        f"**{winner}** - {W.embedding_model} + {W.clustering_method} ({W.parameters}); {int(W.n_clusters)} leaf topics "
        f"(`final/task_topics.csv`, IDs TASK_TOPIC_001..{len(topics):03d}, unnamed - naming is Sprint 4); memberships in "
        f"`final/task_topic_membership.csv`. Decided by: {decided}. Second-best: {second}. Full reasoning: "
        "`evaluation/model_selection.md`.", "",
        "## 19. Limitations",
        "- " + LIMITATION,
        "- No ground-truth topics exist; cohesion, silhouette, stability and agreement are internal/robustness measures, not accuracy.",
        "- Raw cosine scales differ between models; cross-model comparisons use chance-adjusted lift and retrieval ranks.",
        "- HDBSCAN cohesion/silhouette are computed on non-noise points (noise-inclusive cohesion lift reported alongside).",
        "- Bootstrap CIs are conditional on the fitted partition; refit variability is captured by the stability ARI.",
        "- Paraphrase pairs come from Sprint 2 cross-model agreement and are biased towards easier paraphrases.",
        "- Gemini contributes ~90% of statements; provider-composition statistics reflect genuine content differences too.", "",
        "## 20. Sprint 3 conclusion",
        f"Representation verdict: {rep_label[q1_winner]}. Selected pipeline: {W.embedding_model} + {W.clustering_method} "
        f"({W.parameters}) with {int(W.n_clusters)} leaf topics, semantic cohesion {fnum(W.semantic_cohesion)} "
        f"(lift {fnum(W.semantic_cohesion_lift)}), subsample stability ARI {fnum(W.stability_ari)}, "
        f"{pct(W.noise_share)} noise before the post-hoc rule. These topics are the input to Sprint 4 (naming, hierarchy, "
        "O*NET evaluation), which was not started here.", "",
        f"Runtime: {timings.get('total_seconds')} s.",
    ]
    (out / "sprint3_stability_report.md").write_text("\n".join(lines), encoding="utf-8")


def zip_results(out: Path, zpath: Path):
    include = ["embeddings", "evaluation", "final", "figures", "input"]
    files = ["run_manifest.json", "sprint3_stability_report.md"]
    zpath.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for d in include:
            for p in sorted((out / d).rglob("*")):
                if p.is_file():
                    z.write(p, arcname=str(p.relative_to(out)))
        for f in files:
            if (out / f).exists():
                z.write(out / f, arcname=f)
    log(f"ZIP written: {zpath} ({zpath.stat().st_size / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="frozen sprint3_task_input.csv")
    ap.add_argument("--output", required=True, help="results directory (created if missing)")
    ap.add_argument("--zip", default=None, help="zip path (default: <output>/../Sprint3_Experiment_Results.zip)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--skip-install", action="store_true", help="do not pip-install missing packages")
    ap.add_argument("--allow-input-mismatch", action="store_true", help="run even if the input SHA-256 is not the frozen one")
    args = ap.parse_args()
    try:
        run(args)
    except SystemExit:
        raise
    except Exception as e:
        traceback.print_exc()
        die(f"{type(e).__name__}: {e}\nCompleted stages are checkpointed in {Path(args.output) / 'checkpoints'}; "
            f"re-running the same command resumes from them.")


if __name__ == "__main__":
    main()
