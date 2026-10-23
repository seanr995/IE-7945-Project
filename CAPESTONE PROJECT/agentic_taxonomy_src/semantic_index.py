"""Validated loading of the frozen input + cached embeddings, and deterministic semantic-diversity sampling.

No embedding is ever regenerated. No K-Means / historical topic assignment is used for sampling.
"""
from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import E5_MODEL, FROZEN_INPUT_SHA256, T5_MODEL, Paths, StopRun
from .provenance import read_json, sha256_file

CHARS_PER_TOKEN = 3.5
CODE_TOKENS = 4


def norm_text(s: str) -> str:
    t = (s or "").translate(str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", " ": " "}))
    return re.sub(r"\s+", " ", t.lower().strip()).rstrip(" .;:,")


def ids_hash(ids) -> str:
    return hashlib.sha256("|".join(map(str, ids)).encode()).hexdigest()


def paired_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Groq wording of the same task (Sprint 2 cross-model agreement), derived exactly as in Sprint 3."""
    pt = df.get("cross_provider_paired_text", pd.Series([""] * len(df))).fillna("").str.strip()
    nt = df.normalized_text if "normalized_text" in df.columns else df.statement_text.map(norm_text)
    pm = (pt != "") & (pt.map(norm_text) != nt)
    return pd.DataFrame({"row": np.where(pm)[0], "statement_id": df.statement_id[pm].values, "paired_text": pt[pm].values})


def load_project(project_dir, need_e5: bool = False) -> dict:
    """Load + validate the frozen input and cached embeddings. Raises StopRun on any mismatch."""
    P = Paths(Path(project_dir))
    emb = P.embeddings
    req = {"input": P.input_csv, "manifest": P.sprint3_results / "run_manifest.json",
           "t5": emb / "sentence_t5_embeddings.npy", "t5_map": emb / "sentence_t5_embedding_mapping.csv",
           "t5_meta": emb / "sentence_t5_meta.json", "t5_paired": emb / "sentence_t5_paired_text_embeddings.npy",
           "t5_paired_map": emb / "sentence_t5_paired_text_mapping.csv"}
    opt = {"e5": emb / "e5_embeddings.npy", "e5_map": emb / "e5_embedding_mapping.csv", "e5_meta": emb / "e5_meta.json",
           "e5_paired": emb / "e5_paired_text_embeddings.npy", "e5_paired_map": emb / "e5_paired_text_mapping.csv"}
    missing = [f"{k}: {v}" for k, v in req.items() if not v.exists()]
    if missing:
        raise StopRun("required Sprint 3 artifacts missing (embeddings are never regenerated):\n  " + "\n  ".join(missing))
    sha = sha256_file(req["input"])
    manifest = read_json(req["manifest"])
    if sha != FROZEN_INPUT_SHA256 or sha != manifest.get("input_sha256"):
        raise StopRun(f"input SHA-256 {sha} does not match the frozen Sprint 3 input")
    df = pd.read_csv(req["input"], dtype=str, keep_default_na=False, encoding="utf-8")
    if not df.statement_id.is_unique or (df.statement_text.str.strip() == "").any() or len(df) != manifest.get("row_count"):
        raise StopRun("input failed uniqueness / blank-text / row-count validation")
    pr = paired_rows(df)
    out = {"df": df, "manifest": manifest, "input_sha256": sha, "paired": pr, "hashes": {}, "checks": {"rows": len(df)}}

    def load(prefix, model_id, fs):
        X = np.load(fs[prefix]).astype(np.float32)
        mp = pd.read_csv(fs[f"{prefix}_map"], dtype=str, keep_default_na=False)
        meta = read_json(fs[f"{prefix}_meta"])
        if X.shape[0] != len(df) or mp.statement_id.tolist() != df.statement_id.tolist() or \
                mp.embedding_row.astype(int).tolist() != list(range(len(df))):
            raise StopRun(f"{prefix} embeddings / mapping misaligned with the frozen input")
        if meta.get("model") != model_id or meta.get("input_sha256") != sha or meta.get("ids_hash") != ids_hash(df.statement_id):
            raise StopRun(f"{prefix}_meta.json does not describe these embeddings")
        if not np.allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-3):
            raise StopRun(f"{prefix} embeddings are not L2-normalised")
        Pm = np.load(fs[f"{prefix}_paired"]).astype(np.float32)
        pmp = pd.read_csv(fs[f"{prefix}_paired_map"], dtype=str, keep_default_na=False)
        if Pm.shape[0] != len(pr) or pmp.statement_id.tolist() != pr.statement_id.tolist():
            raise StopRun(f"{prefix} paired-wording embeddings do not match the paired rows")
        out["hashes"][prefix] = sha256_file(fs[prefix])
        out["checks"][f"{prefix}_aligned"] = True
        return X, Pm

    out["X_t5"], out["P_t5"] = load("t5", T5_MODEL, req)
    out["X_e5"] = out["P_e5"] = None
    if need_e5 and all(v.exists() for v in opt.values()):
        out["X_e5"], out["P_e5"] = load("e5", E5_MODEL, opt)
    return out


# ---------------------------------------------------------------- sampling
def est_tokens(text: str) -> int:
    return int(math.ceil(len(text) / CHARS_PER_TOKEN)) + CODE_TOKENS


def farthest_point_order(X, n):
    """First n farthest-point-sampling picks (cosine), starting at the most central statement, + coverage radius."""
    n = min(n, len(X))
    c = X.mean(0)
    c /= np.linalg.norm(c)
    first = int(np.argmax(X @ c))
    order, best = [first], X @ X[first]
    picked = np.zeros(len(X), dtype=bool)
    picked[first] = True
    radius = [1.0 - float(best[~picked].min()) if (~picked).any() else 0.0]
    for _ in range(1, n):
        i = int(np.argmin(np.where(picked, np.inf, best)))
        order.append(i)
        picked[i] = True
        best = np.maximum(best, X @ X[i])
        radius.append(1.0 - float(best[~picked].min()) if (~picked).any() else 0.0)
    return np.array(order), np.array(radius)


def knee(y):
    if len(y) < 3:
        return len(y) - 1
    x = np.linspace(0, 1, len(y))
    yn = (y - y.min()) / max(y.max() - y.min(), 1e-12)
    return int(np.argmax((1 - x) - yn))


def choose_sample(X, texts, n_calls, tokens_per_call, min_per_call=30):
    """FPS + knee of the coverage-radius curve, capped by the measured token budget of the discovery calls."""
    tok = np.array([est_tokens(t) for t in texts])
    mean_tok = float(tok.mean())
    n_cap = max(1, min(int(n_calls * tokens_per_call // mean_tok), len(X)))
    order, radius = farthest_point_order(X, n_cap)
    k = knee(radius) + 1
    n = int(min(max(k, n_calls * min_per_call), n_cap))
    return {"indices": order[:n], "radius_curve": radius, "knee_n": k, "token_cap_n": n_cap, "chosen_n": n,
            "mean_tokens_per_statement": mean_tok, "est_statement_tokens_per_call": float(tok[order[:n]].sum() / max(n_calls, 1)),
            "coverage_radius_at_chosen": float(radius[n - 1]),
            "rule": "n = max(knee of FPS coverage-radius curve, calls*min_per_call), capped by calls*tokens_per_call"}


def balanced_regions(X, idx, n_regions):
    """Balanced, semantically contiguous shards (FPS anchors + capacity-constrained greedy assignment). A work-
    distribution device so each discovery call sees a different part of the task space - not topics."""
    idx = np.asarray(idx)
    if n_regions <= 1:
        return [idx]
    Xs = X[idx]
    order, _ = farthest_point_order(Xs, n_regions)
    S = Xs @ Xs[order].T
    cap = int(math.ceil(len(idx) / n_regions))
    assign, load = np.full(len(idx), -1), np.zeros(n_regions, dtype=int)
    for flat in np.argsort(-S, axis=None, kind="stable"):
        i, r = divmod(int(flat), n_regions)
        if assign[i] == -1 and load[r] < cap:
            assign[i], load[r] = r, load[r] + 1
    return [idx[assign == r] for r in range(n_regions)]


def fps_within(X, rows, start_row=None):
    rows = np.asarray(rows)
    if len(rows) <= 1:
        return rows
    Xs = X[rows]
    first = int(np.where(rows == start_row)[0][0]) if start_row is not None and start_row in set(rows.tolist()) else 0
    order, best = [first], Xs @ Xs[first]
    picked = np.zeros(len(rows), dtype=bool)
    picked[first] = True
    for _ in range(1, len(rows)):
        i = int(np.argmin(np.where(picked, np.inf, best)))
        order.append(i)
        picked[i] = True
        best = np.maximum(best, Xs @ Xs[i])
    return rows[np.array(order)]


def normalize(v):
    return v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12, None)
