"""Stages 5 & 7: O*NET task embedding (cached), deterministic top-10 retrieval, quantitative evaluation."""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import pandas as pd

from .config import ONET_DIR_CANDIDATES, StopRun, T5_META, Settings, log, read_json, write_json
from .evidence import Frozen, l2

T5_MODEL = "sentence-transformers/sentence-t5-base"


def load_onet(s: Settings):
    d = next((s.find(c) for c in ONET_DIR_CANDIDATES if s.find(c + "/task_statements.csv").exists()), None)
    if d is None:
        raise StopRun("O*NET task_statements.csv not found (expected data/reference/onet/extracted/31.0/db_31_0_csv)")
    t = pd.read_csv(d / "task_statements.csv")
    t = t.rename(columns={"O*NET-SOC Code": "soc", "Title": "occupation", "Task ID": "task_id", "Task": "task", "Task Type": "task_type"})
    t = t.drop_duplicates("task_id").reset_index(drop=True)
    dwa = pd.read_csv(d / "tasks_to_dwas.csv")
    dwa_map = defaultdict(list)
    for tid, eid, name in zip(dwa["Task ID"], dwa["DWA Element ID"], dwa["DWA Element Name"]):
        dwa_map[int(tid)].append((str(eid), str(name)))
    t["dwa_ids"] = [json.dumps(sorted({e for e, _ in dwa_map.get(int(i), [])})) for i in t.task_id]
    t["dwa_names"] = [json.dumps(sorted({n for _, n in dwa_map.get(int(i), [])})) for i in t.task_id]
    return t[["task_id", "soc", "occupation", "task_type", "task", "dwa_ids", "dwa_names"]]


def hash_embedder(texts, dim=768):
    """MOCK/TEST ONLY: deterministic bag-of-words embedding. Never used in a real run."""
    from sklearn.feature_extraction.text import HashingVectorizer
    hv = HashingVectorizer(n_features=dim, ngram_range=(1, 2), stop_words="english", alternate_sign=False, norm="l2")
    return hv.transform(texts).toarray().astype("float32")


def t5_embedder():
    from sentence_transformers import SentenceTransformer
    import torch
    m = SentenceTransformer(T5_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")

    def enc(texts):
        return m.encode(list(texts), batch_size=128, show_progress_bar=False, normalize_embeddings=True, convert_to_numpy=True)
    return enc


def verify_same_model(fz: Frozen, embedder, n=24):
    """Re-embed a sample of frozen tasks with the live model; they must reproduce the stored Sprint 3 vectors."""
    rng = np.random.default_rng(0)
    pick = rng.choice(len(fz.members), n, replace=False)
    new = l2(embedder(fz.members.task_text.iloc[pick].tolist()).astype("float32"))
    cos = (new * fz.emb[pick]).sum(1)
    if cos.min() < 0.98:
        raise StopRun(f"Sentence-T5 sanity check failed: re-embedded tasks differ from the frozen Sprint 3 embeddings "
                      f"(min cosine {cos.min():.4f}). O*NET must be embedded with the identical model.")
    return {"n": n, "min_cosine": float(cos.min()), "mean_cosine": float(cos.mean())}


def onet_embeddings(s: Settings, onet: pd.DataFrame, embedder, tag: str):
    s.cache_root.mkdir(parents=True, exist_ok=True)
    p = s.cache_root / f"onet_task_embeddings_{tag}.npy"
    meta = s.cache_root / f"onet_task_embeddings_{tag}.json"
    ids = onet.task_id.tolist()
    m = read_json(meta)
    if p.exists() and m and m.get("n") == len(ids) and m.get("first_last") == [ids[0], ids[-1]]:
        log(f"  O*NET embeddings: cache hit ({len(ids)} tasks)")
        return np.load(p)
    log(f"  embedding {len(ids)} O*NET tasks (once; cached)")
    E = l2(embedder(onet.task.tolist()).astype("float32"))
    np.save(p, E)
    write_json(meta, {"n": len(ids), "first_last": [ids[0], ids[-1]], "model": T5_MODEL if tag == "t5" else "mock-hash",
                      "dims": int(E.shape[1])})
    return E


def retrieve_top10(centroids, E, k=10):
    """Exact cosine ranking (both sides L2-normalised). Deterministic tie-break on index."""
    S = centroids @ E.T
    order = np.argsort(-S, axis=1, kind="stable")[:, :k]
    return order, np.take_along_axis(S, order, 1), S


def build_top10_table(fz: Frozen, onet: pd.DataFrame, order, sims, names) -> pd.DataFrame:
    rows = []
    for i, t in enumerate(fz.topic_ids):
        for r in range(order.shape[1]):
            o = onet.iloc[order[i, r]]
            rows.append({"topic_id": t, "topic_name": names.get(t, ""), "rank": r + 1, "onet_task_id": int(o.task_id),
                         "onet_soc_code": o.soc, "occupation_title": o.occupation, "task_type": o.task_type,
                         "onet_task": o.task, "cosine_similarity": round(float(sims[i, r]), 6),
                         "dwa_names": o.dwa_names})
    return pd.DataFrame(rows)


def otsu(values, bins=64):
    v = np.sort(np.asarray(values, dtype=float))
    if len(v) < 4 or v[0] == v[-1]:
        return float(np.median(v))
    best, thr = -1, float(np.median(v))
    for c in np.unique(v)[1:]:
        lo, hi = v[v < c], v[v >= c]
        if len(lo) < 2 or len(hi) < 2:
            continue
        sb = len(lo) * len(hi) * (lo.mean() - hi.mean()) ** 2
        if sb > best:
            best, thr = sb, float((lo.max() + hi.min()) / 2)
    return thr


def evaluate_onet(fz: Frozen, onet: pd.DataFrame, E, order, sims, S_all, claude_eval: dict, reps_n=5) -> tuple:
    """Quantitative O*NET metrics. Thresholds are derived from the observed score distributions:
      tau_noise = 99th percentile of ALL centroid x O*NET cosines (background); a top-1 below it is noise-level.
      tau_split = Otsu threshold over the 80 observed top-1 scores; below it a topic is flagged weak for the critic."""
    top1 = sims[:, 0]
    tau_noise = float(np.percentile(S_all, 99))
    tau_split = otsu(top1)
    dwa_sets = [set(json.loads(x)) for x in onet.dwa_ids]
    ids = onet.task_id.values
    # nearest O*NET task for EVERY member task (member-task retrieval)
    nn_member = np.zeros(len(fz.members), dtype=int)
    for a in range(0, len(fz.members), 512):
        nn_member[a:a + 512] = np.argmax(fz.emb[a:a + 512] @ E.T, axis=1)
    rows = []
    for i, t in enumerate(fz.topic_ids):
        ev = claude_eval.get(t, {})
        best = [int(x) for x in ev.get("best_onet_task_ids", [])]
        top_ids = [int(ids[j]) for j in order[i]]
        dom_ids = best or [top_ids[0]]
        dom_idx = [int(np.where(ids == b)[0][0]) for b in dom_ids]
        dom_dwa = set().union(*[dwa_sets[j] for j in dom_idx])
        # purity: share of member tasks whose nearest O*NET task expresses the dominant concept
        def same_concept(j):
            return (ids[j] in dom_ids) or bool(dom_dwa & dwa_sets[j]) if dom_dwa else ids[j] in dom_ids
        r = fz.tid_to_rows[t]
        purity = float(np.mean([same_concept(nn_member[k]) for k in r]))
        # nearest-neighbour agreement: representative tasks' own top-1 O*NET task vs centroid top-10
        core, _ = fz.core_boundary(t, n_core=reps_n, n_boundary=0 or 1)
        rep_top1 = [int(ids[nn_member[k]]) for k in core]
        agree = float(np.mean([x in set(top_ids) for x in rep_top1]))
        exact = float(np.mean([x == top_ids[0] for x in rep_top1]))
        align = ev.get("alignment", "UNEVALUATED")
        meaningful = bool(align in ("STRONG_MATCH", "PARTIAL_MATCH") and top1[i] >= tau_noise)
        rows.append({"topic_id": t, "alignment": align, "best_onet_task_ids": json.dumps(best),
                     "dominant_onet_concept": ev.get("dominant_onet_concept", ""),
                     "semantic_alignment_confidence": ev.get("semantic_alignment_confidence"),
                     "reason": ev.get("reason", ""), "top1_similarity": round(float(top1[i]), 6),
                     "top5_mean_similarity": round(float(sims[i, :5].mean()), 6),
                     "centroid_top1_onet_task_id": top_ids[0], "purity": round(purity, 4),
                     "nn_agreement_top10": round(agree, 4), "nn_agreement_top1_exact": round(exact, 4),
                     "meaningful_alignment": meaningful, "below_split_threshold": bool(top1[i] < tau_split)})
    df = pd.DataFrame(rows)
    top5 = sims[:, :5]
    summ = {"threshold_noise_p99_background": round(tau_noise, 6), "threshold_otsu_top1": round(tau_split, 6),
            "threshold_derivation": "tau_noise = 99th percentile of all 80 x %d centroid-O*NET cosines; tau_split = Otsu "
                                    "threshold of the observed top-1 scores" % len(onet),
            "coverage_meaningful_alignment": round(float(df.meaningful_alignment.mean()), 4),
            "purity_mean": round(float(df.purity.mean()), 4), "purity_median": round(float(df.purity.median()), 4),
            "nn_agreement_top10_mean": round(float(df.nn_agreement_top10.mean()), 4),
            "nn_agreement_top1_exact_mean": round(float(df.nn_agreement_top1_exact.mean()), 4),
            "top1_mean": round(float(top1.mean()), 4), "top1_median": round(float(np.median(top1)), 4),
            "alignment_counts": {k: int((df.alignment == k).sum()) for k in ["STRONG_MATCH", "PARTIAL_MATCH", "NO_MATCH", "UNEVALUATED"]},
            "top5_similarity_distribution": {f"rank{r + 1}": {"mean": round(float(top5[:, r].mean()), 4),
                                                              "median": round(float(np.median(top5[:, r])), 4),
                                                              "min": round(float(top5[:, r].min()), 4),
                                                              "max": round(float(top5[:, r].max()), 4)} for r in range(5)}}
    return df, summ
