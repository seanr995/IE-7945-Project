"""Reference alignment: TASK -> O*NET, SKILL -> ESCO. PROTOTYPE / EXPERIMENTAL / FOR VALIDATION.

    python -m src.prototype.alignment

1. Embed reference concepts once (cached): O*NET task statements, O*NET Generalized Work
   Activities (content model 4.A.x.x.x) and ESCO skills (label + description).
2. For each surviving prototype statement retrieve top-k by cosine similarity.
3. AMBIGUOUS mappings only (top-1 similarity below AMBIGUOUS_BELOW or top-1/top-2 margin
   below MARGIN) are reranked independently by Gemini and Groq; each model may only return ids
   from the candidate list (enforced by `constrain_rerank`), or reject all.
4. final_candidate_score:
     not reranked : embedding_similarity
     reranked     : 0.6 * embedding_similarity + 0.4 * rerank_support,
                    rerank_support = mean over models of (6 - rank)/5 if ranked else 0
5. mapping_status is always 'candidate' (no official posting->O*NET/ESCO mapping exists for this data).
6. candidate_gap = both models rejected every candidate, OR top-1 similarity < GAP_BELOW
   ("candidate reference gap for human review" - never a confirmed missing concept).
"""
from __future__ import annotations

import hashlib
import json

import duckdb
import numpy as np
import pandas as pd

from src.llm.base import load_config
from src.llm.embeddings import Embedder
from src.llm.router import Router, constrain_rerank
from src.utils.common import DB_PATH, get_logger, load_env

log = get_logger("prototype.alignment")

# Thresholds (cosine, gemini-embedding-001 768-d SEMANTIC_SIMILARITY). Chosen from the observed
# distribution of top-1 similarities in this experiment; documented in the experiment report.
AMBIGUOUS_BELOW = 0.80
MARGIN = 0.01
GAP_BELOW = 0.70


def reference_frames(con) -> dict[str, pd.DataFrame]:
    tasks = con.execute("""SELECT CAST(task_id AS VARCHAR) ref_id, onetsoc_code, task AS text, title AS occupation
                           FROM onet_tasks""").df()
    gwa = con.execute("""SELECT element_id AS ref_id, NULL AS onetsoc_code, element_name || ': ' || description AS text,
                                element_name AS occupation FROM onet_content_model_reference
                         WHERE regexp_matches(element_id, '^4\\.A\\.[0-9]+\\.[a-z]\\.[0-9]+$')""").df()
    esco = con.execute("""SELECT concept_uri AS ref_id, preferred_label AS label, skill_type, reuse_level,
                                 preferred_label || coalesce('. ' || substr(description, 1, 240), '') AS text
                          FROM esco_skills WHERE status = 'released'""").df()
    return {"onet_task": tasks, "onet_gwa": gwa, "esco_skill": esco}


def topk(q: np.ndarray, m: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    s = q @ m.T
    idx = np.argpartition(-s, kth=min(k, s.shape[1] - 1), axis=1)[:, :k]
    rows = np.arange(s.shape[0])[:, None]
    order = np.argsort(-s[rows, idx], axis=1)
    idx = idx[rows, order]
    return idx, s[rows, idx]


def _h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def main() -> dict:
    load_env()
    cfg = load_config()
    ex = cfg["experiment"]
    k = int(ex["top_k_candidates"])
    run_id = "align_" + _h(json.dumps(ex, sort_keys=True))[:8]
    emb = Embedder(run_id=run_id)
    con = duckdb.connect(str(DB_PATH), read_only=True)
    refs = reference_frames(con)
    st = con.execute("""SELECT statement_id, posting_id, kind, statement_text, evidence_text, phase, consensus_score
                        FROM prototype.prototype_statements
                        WHERE coalesce(adjudication_decision, 'keep') <> 'reject'""").df()
    con.close()
    log.info("Statements to align: %d (%s)", len(st), st.kind.value_counts().to_dict())
    mats = {name: emb.embed(df.text.tolist()) for name, df in refs.items()}
    log.info("Reference embeddings ready: %s", {n: m.shape for n, m in mats.items()})
    sv = emb.embed(st.statement_text.tolist())

    task_rows, skill_rows = [], []
    tasks = st[st.kind == "task"].reset_index(drop=True)
    skills = st[st.kind == "skill"].reset_index(drop=True)
    tv, kv = sv[(st.kind == "task").values], sv[(st.kind == "skill").values]
    if len(tasks):
        for ref_type, kk in (("onet_task", k), ("onet_gwa", 3)):
            idx, sims = topk(tv, mats[ref_type], kk)
            R = refs[ref_type]
            for i, r in tasks.iterrows():
                for rank, (j, s) in enumerate(zip(idx[i], sims[i]), 1):
                    task_rows.append({"statement_id": r.statement_id, "onet_soc_code": R.onetsoc_code.iat[j],
                                      "onet_reference_id": R.ref_id.iat[j], "onet_text": R.text.iat[j],
                                      "onet_occupation_or_element": R.occupation.iat[j], "reference_type": ref_type,
                                      "embedding_similarity": round(float(s), 4), "rank": rank})
    if len(skills):
        idx, sims = topk(kv, mats["esco_skill"], k)
        R = refs["esco_skill"]
        for i, r in skills.iterrows():
            for rank, (j, s) in enumerate(zip(idx[i], sims[i]), 1):
                skill_rows.append({"statement_id": r.statement_id, "esco_uri": R.ref_id.iat[j], "esco_label": R.label.iat[j],
                                   "esco_skill_type": R.skill_type.iat[j], "embedding_similarity": round(float(s), 4),
                                   "rank": rank})
    tdf, kdf = pd.DataFrame(task_rows), pd.DataFrame(skill_rows)

    # ---------------- ambiguity + two-model reranking (constrained to candidate list)
    def ambiguous(df, ref_filter=None):
        d = df if ref_filter is None else df[df.reference_type == ref_filter]
        top = d[d["rank"] <= 2].pivot_table(index="statement_id", columns="rank", values="embedding_similarity")
        amb = top[(top[1] < AMBIGUOUS_BELOW) | ((top[1] - top[2]) < MARGIN)]
        return set(amb.index)

    amb_t = ambiguous(tdf, "onet_task") if len(tdf) else set()
    amb_s = ambiguous(kdf) if len(kdf) else set()
    meta = st.set_index("statement_id")
    order = sorted(amb_t | amb_s, key=lambda s: (meta.loc[s, "phase"] != "dual_model", _h(s)))
    chosen = order[: int(ex["rerank_max_statements"])]
    log.info("Ambiguous: %d tasks, %d skills; reranking %d (cap %s)", len(amb_t), len(amb_s), len(chosen), ex["rerank_max_statements"])
    router = Router(run_id=run_id, cfg=cfg)
    rr = {}  # (statement_id, provider) -> decision
    bs = int(ex["rerank_batch_size"])
    for kind, reference, df, id_col, text_col in (("task", "onet", tdf[tdf.reference_type == "onet_task"] if len(tdf) else tdf,
                                                    "onet_reference_id", "onet_text"),
                                                   ("skill", "esco", kdf, "esco_uri", "esco_label")):
        ids = [s for s in chosen if meta.loc[s, "kind"] == kind]
        for b0 in range(0, len(ids), bs):
            bids = ids[b0:b0 + bs]
            batch = []
            for sidx in bids:
                c = df[df.statement_id == sidx].sort_values("rank")
                batch.append({"statement_text": meta.loc[sidx, "statement_text"], "evidence_text": meta.loc[sidx, "evidence_text"],
                              "kind": kind, "candidates": [{"id": f"c{n}", "text": t, "ref": ref}
                                                           for n, (t, ref) in enumerate(zip(c[text_col], c[id_col]), 1)]})
            for prov in ("gemini", "groq"):
                try:
                    res = router.rerank_candidates(prov, batch, reference)
                    for d in constrain_rerank(res.data, batch):
                        rr[(bids[d["index"]], prov)] = {**d, "model": res.model,
                                                        "id_to_ref": {c["id"]: c["ref"] for c in batch[d["index"]]["candidates"]}}
                except Exception as e:
                    log.error("rerank %s failed: %s", prov, e)
                    for sidx in bids:
                        rr[(sidx, prov)] = {"status": "error", "ranked": [], "reject_all": None, "invalid_ids": [], "id_to_ref": {}}

    def apply(df, id_col):
        if not len(df):
            return df
        g_rank, q_rank, status, final, agree, rej_g, rej_q, invalid = [], [], [], [], [], [], [], []
        for r in df.itertuples():
            dg, dq = rr.get((r.statement_id, "gemini")), rr.get((r.statement_id, "groq"))
            if dg is None and dq is None:
                g_rank.append(None); q_rank.append(None); status.append("not_required" if r.statement_id not in chosen else "not_run")
                final.append(r.embedding_similarity); agree.append(None); rej_g.append(None); rej_q.append(None); invalid.append(0)
                continue
            ranks = []
            for d, store in ((dg, g_rank), (dq, q_rank)):
                rank = None
                if d and d.get("status") == "ok":
                    refs_ranked = [d["id_to_ref"][cid] for cid in d["ranked"]]
                    rank = refs_ranked.index(getattr(r, id_col)) + 1 if getattr(r, id_col) in refs_ranked else None
                store.append(rank)
                ranks.append(rank)
            support = np.mean([(6 - x) / 5 if x else 0.0 for x in ranks])
            final.append(round(0.6 * r.embedding_similarity + 0.4 * support, 4))
            ok = [d for d in (dg, dq) if d and d.get("status") == "ok"]
            status.append("reranked_both" if len(ok) == 2 else ("reranked_one" if ok else "rerank_failed"))
            tg = dg["ranked"][0] if dg and dg.get("ranked") else None
            tq = dq["ranked"][0] if dq and dq.get("ranked") else None
            agree.append(None if len(ok) < 2 else (tg == tq and tg is not None) or (dg["reject_all"] and dq["reject_all"]))
            rej_g.append(dg.get("reject_all") if dg else None)
            rej_q.append(dq.get("reject_all") if dq else None)
            invalid.append(len((dg or {}).get("invalid_ids", [])) + len((dq or {}).get("invalid_ids", [])))
        df = df.copy()
        df["gemini_rank"], df["groq_rank"], df["reranker_status"] = g_rank, q_rank, status
        df["rerank_top_choice_agreement"], df["gemini_rejected_all"], df["groq_rejected_all"] = agree, rej_g, rej_q
        df["invalid_ids_discarded"] = invalid
        df["final_candidate_score"] = final
        top1 = df.groupby("statement_id").embedding_similarity.transform("max")
        both_rej = df.gemini_rejected_all.eq(True) & df.groq_rejected_all.eq(True)
        df["weak_similarity"] = top1 < GAP_BELOW
        df["candidate_gap"] = both_rej | df.weak_similarity
        df["mapping_status"] = "candidate"
        return df

    tdf = apply(tdf, "onet_reference_id")
    kdf = apply(kdf, "esco_uri")
    con = duckdb.connect(str(DB_PATH))
    for name, df in (("prototype_task_onet_candidates", tdf), ("prototype_skill_esco_candidates", kdf)):
        con.register("df", df)
        con.execute(f"CREATE OR REPLACE TABLE prototype.{name} AS SELECT * FROM df")
        con.unregister("df")
    con.close()
    emb.close()
    out = {"tasks": len(tasks), "skills": len(skills), "reranked": len(chosen), "ambiguous_tasks": len(amb_t),
           "ambiguous_skills": len(amb_s), "embedding_api_calls": emb.api_calls, "embedding_cache_hits": emb.cache_hits}
    log.info("Alignment done: %s", out)
    return out


if __name__ == "__main__":
    print(main())
