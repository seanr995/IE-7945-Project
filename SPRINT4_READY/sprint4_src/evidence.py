"""Stage 1: frozen inputs, leaf centroids, core/boundary tasks, c-TF-IDF keywords."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import (FROZEN_MEMBERS, FROZEN_TOPICS, N_LEAVES, StopRun, T5_EMB, T5_MAP, Settings, log, read_json)


def l2(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(n, 1e-12, None)


class Frozen:
    """Frozen Sprint 3 leaves + aligned Sentence-T5 task embeddings. Never mutated."""

    def __init__(self, s: Settings, task_embedder=None):
        self.topics = pd.read_csv(s.find(FROZEN_TOPICS))
        self.members = pd.read_csv(s.find(FROZEN_MEMBERS))
        if len(self.topics) != N_LEAVES or self.topics.topic_id.nunique() != N_LEAVES:
            raise StopRun("frozen leaf file must contain exactly 80 unique topics")
        if len(self.members) != 3015 or self.members.statement_id.nunique() != 3015:
            raise StopRun("frozen membership file must contain 3015 unique tasks")
        mp = pd.read_csv(s.find(T5_MAP))
        emb = np.load(s.find(T5_EMB)).astype("float32")
        if len(mp) != len(emb):
            raise StopRun("embedding / mapping length mismatch")
        idx = pd.Series(np.arange(len(mp)), index=mp.statement_id)
        missing = set(self.members.statement_id) - set(idx.index)
        if missing:
            raise StopRun(f"{len(missing)} frozen tasks have no Sentence-T5 embedding")
        self.members = self.members.reset_index(drop=True)
        if task_embedder is not None:             # mock/test mode only: hashed embeddings replace T5
            self.emb = l2(task_embedder(self.members.task_text.tolist()).astype("float32"))
            self.emb_source = "mock-hash"
        else:
            self.emb = l2(emb[idx.loc[self.members.statement_id].values])
            self.emb_source = "sentence-transformers/sentence-t5-base (frozen Sprint 3 embeddings)"
        self.topic_ids = self.topics.topic_id.tolist()
        self.tid_to_rows = {t: np.where(self.members.topic_id.values == t)[0] for t in self.topic_ids}
        sizes = {t: len(r) for t, r in self.tid_to_rows.items()}
        if any(sizes[t] != n for t, n in zip(self.topics.topic_id, self.topics.topic_size)):
            raise StopRun("frozen topic sizes do not match membership")
        self.centroids = l2(np.stack([self.emb[self.tid_to_rows[t]].mean(0) for t in self.topic_ids]))
        self.sizes = np.array([sizes[t] for t in self.topic_ids])
        self.sim_to_centroid = np.zeros(len(self.members), dtype="float32")
        for i, t in enumerate(self.topic_ids):
            r = self.tid_to_rows[t]
            self.sim_to_centroid[r] = self.emb[r] @ self.centroids[i]
        self.topic_index = {t: i for i, t in enumerate(self.topic_ids)}

    def core_boundary(self, t, n_core=10, n_boundary=5):
        r = self.tid_to_rows[t]
        sims = self.sim_to_centroid[r]
        order = sorted(range(len(r)), key=lambda k: (-sims[k], self.members.statement_id.iat[r[k]]))
        n = len(r)
        nb = n_boundary if n >= n_core + n_boundary else max(1, n // 3)     # small topics: keep core/boundary disjoint
        nc = min(n_core, n - nb)
        core, bound = order[:nc], order[::-1][:nb]
        return [r[k] for k in core], [r[k] for k in bound]


def ctfidf_keywords(task_texts_by_topic: dict, topn=10) -> dict:
    """Class-based TF-IDF (c-TF-IDF, Grootendorst 2022): one document per topic;
    W[t,w] = tf[t,w] * log(1 + A / f_w) with tf = word count / words in topic, A = mean words per topic,
    f_w = word frequency across all topics. Unigrams+bigrams, English stop words, min_df=2 topics-docs not required."""
    from sklearn.feature_extraction.text import CountVectorizer
    ids = list(task_texts_by_topic)
    docs = [" ".join(task_texts_by_topic[t]) for t in ids]
    cv = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1, token_pattern=r"(?u)\b[A-Za-z][A-Za-z\-]+\b")
    X = cv.fit_transform(docs).toarray().astype("float64")
    words = np.array(cv.get_feature_names_out())
    tf = X / np.clip(X.sum(1, keepdims=True), 1, None)
    A = X.sum() / X.shape[0]
    idf = np.log(1 + A / np.clip(X.sum(0), 1, None))
    W = tf * idf
    out = {}
    for i, t in enumerate(ids):
        top = np.argsort(-W[i], kind="stable")[:topn]
        out[t] = [words[j] for j in top]
    return out


def build_leaf_evidence(fz: Frozen) -> pd.DataFrame:
    kw = ctfidf_keywords({t: fz.members.task_text.iloc[fz.tid_to_rows[t]].tolist() for t in fz.topic_ids})
    rows = []
    for t, raw in zip(fz.topics.topic_id, fz.topics.raw_leiden_label):
        core, bound = fz.core_boundary(t)
        m = fz.members
        r = fz.tid_to_rows[t]
        rows.append({
            "topic_id": t, "raw_leiden_label": int(raw), "topic_size": len(r),
            "mean_sim_to_centroid": round(float(fz.sim_to_centroid[r].mean()), 6),
            "core_statement_ids": json.dumps([m.statement_id.iat[k] for k in core]),
            "core_tasks": json.dumps([m.task_text.iat[k] for k in core], ensure_ascii=False),
            "core_similarities": json.dumps([round(float(fz.sim_to_centroid[k]), 5) for k in core]),
            "boundary_statement_ids": json.dumps([m.statement_id.iat[k] for k in bound]),
            "boundary_tasks": json.dumps([m.task_text.iat[k] for k in bound], ensure_ascii=False),
            "boundary_similarities": json.dumps([round(float(fz.sim_to_centroid[k]), 5) for k in bound]),
            "boundary_mean_sim": round(float(np.mean([fz.sim_to_centroid[k] for k in bound])), 6),
            "ctfidf_keywords": json.dumps(kw[t], ensure_ascii=False)})
    return pd.DataFrame(rows)
