"""Evidence-Backed Workforce Intelligence Prototype: initiative -> work. PROTOTYPE / EXPERIMENTAL.

    python -m src.demo.initiative_to_work "Modernize our data reporting"

No generative model is asked "what tasks are required?". The pipeline is retrieval-only:
  1. embed the initiative (Gemini embedding API, cached)
  2. retrieve extracted TASK statements from real postings by cosine similarity
     (only statements that passed the evidence gates and were not rejected by adjudication)
  3. merge near-duplicate tasks (cosine >= MERGE_AT) so one task can cite several postings
  4. rank: rank_score = 0.7 * relevance + 0.3 * consensus_score
  5. skills: ONLY skill statements extracted from the same posting(s) as the task, ranked by
     similarity to (task + initiative)
  6. attach candidate O*NET task / ESCO skill alignments (mapping_status = candidate)
  7. every result carries an explanation object and a runtime provenance check
     (posting exists and evidence_text is found in job_postings.posting_text).
If nothing clears MIN_RELEVANCE the run returns status 'insufficient_evidence' instead of guessing.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone

import duckdb
import numpy as np
import pandas as pd

from src.llm.embeddings import Embedder
from src.utils.common import DB_PATH, load_env

MIN_RELEVANCE = 0.62
MERGE_AT = 0.90
TOP_TASKS = 6
SKILLS_PER_TASK = 4
PARAMS = {"min_relevance": MIN_RELEVANCE, "merge_at": MERGE_AT, "top_tasks": TOP_TASKS,
          "skills_per_task": SKILLS_PER_TASK, "rank_score": "0.7*relevance + 0.3*consensus_score"}

EXAMPLE_INITIATIVES = [
    "Introduce an AI-assisted customer support capability.",
    "Modernize the organization's data analytics and reporting infrastructure.",
    "Strengthen construction project oversight and building-safety inspections.",
]

LIMITATIONS = [
    "Evidence comes from a 200-posting experimental sample (NYC public sector + Arbeitnow, English only).",
    "Statements are machine-extracted; human gold labels are not yet available.",
    "O*NET/ESCO links are nearest-neighbour candidates, not validated mappings.",
    "Relevance is embedding similarity to the initiative text, not a causal requirement.",
]


def _load(con):
    st = con.execute("""
        SELECT s.statement_id, s.posting_id, s.kind, s.statement_text, s.evidence_text, s.evidence_section,
               s.evidence_start, s.evidence_end, s.agreement_status, s.consensus_score, s.confidence_tier,
               s.extractor_provider, s.extractor_model, s.validator_provider, s.paired_statement_text,
               p.source, p.job_title, p.company_name, p.source_url, p.retrieval_id, p.source_record_ref, p.date_posted
        FROM prototype.prototype_statements s JOIN job_postings p USING (posting_id)
        WHERE coalesce(s.adjudication_decision, 'keep') <> 'reject'""").df()
    onet = con.execute("""SELECT statement_id, onet_soc_code, onet_reference_id, onet_text, onet_occupation_or_element,
                                 embedding_similarity, final_candidate_score, reranker_status, gemini_rank, groq_rank,
                                 candidate_gap
                          FROM prototype.prototype_task_onet_candidates WHERE reference_type='onet_task' AND rank=1""").df()
    esco = con.execute("""SELECT statement_id, esco_uri, esco_label, embedding_similarity, final_candidate_score,
                                 reranker_status, gemini_rank, groq_rank, candidate_gap
                          FROM prototype.prototype_skill_esco_candidates WHERE rank=1""").df()
    return st, onet.set_index("statement_id"), esco.set_index("statement_id")


def _verify(con, posting_id: str, evidence: str) -> bool:
    r = con.execute("SELECT posting_text FROM job_postings WHERE posting_id = ?", [posting_id]).fetchone()
    return bool(r and evidence and evidence in r[0])


def _ref(df, sid):
    if sid not in df.index:
        return None
    r = df.loc[sid]
    d = {k: (None if (isinstance(v, float) and np.isnan(v)) else (v.item() if hasattr(v, "item") else v))
         for k, v in r.to_dict().items()}
    d["mapping_status"] = "candidate"
    return d


def analyze(initiative: str, con=None, embedder: Embedder | None = None, persist: bool = True) -> dict:
    load_env()
    own = con is None
    con = con or duckdb.connect(str(DB_PATH), read_only=not persist)
    emb = embedder or Embedder(run_id="initiative")
    st, onet, esco = _load(con)
    tasks, skills = st[st.kind == "task"].reset_index(drop=True), st[st.kind == "skill"].reset_index(drop=True)
    q = emb.embed([initiative])[0]
    tv = emb.embed(tasks.statement_text.tolist())
    kv = emb.embed(skills.statement_text.tolist()) if len(skills) else np.zeros((0, len(q)))
    rel = tv @ q
    order = np.argsort(-rel)
    cand = [i for i in order[:60] if rel[i] >= MIN_RELEVANCE]
    groups = []  # merge near-duplicate tasks
    for i in cand:
        for g in groups:
            if float(tv[i] @ tv[g[0]]) >= MERGE_AT:
                g.append(i)
                break
        else:
            groups.append([i])
    scored = []
    for g in groups:
        rep = max(g, key=lambda i: 0.7 * rel[i] + 0.3 * tasks.consensus_score.iat[i])
        scored.append((0.7 * rel[rep] + 0.3 * tasks.consensus_score.iat[rep], rep, g))
    scored.sort(key=lambda x: -x[0])
    results = []
    for rank, (score, rep, g) in enumerate(scored[:TOP_TASKS], 1):
        t = tasks.iloc[rep]
        posts = list(dict.fromkeys(tasks.posting_id.iat[i] for i in g))
        evidence = [{"posting_id": tasks.posting_id.iat[i], "job_title": tasks.job_title.iat[i],
                     "source": tasks.source.iat[i], "company_name": tasks.company_name.iat[i],
                     "statement_text": tasks.statement_text.iat[i], "evidence_text": tasks.evidence_text.iat[i],
                     "evidence_section": tasks.evidence_section.iat[i],
                     "evidence_offsets": [None if pd.isna(tasks.evidence_start.iat[i]) else int(tasks.evidence_start.iat[i]),
                                          None if pd.isna(tasks.evidence_end.iat[i]) else int(tasks.evidence_end.iat[i])],
                     "source_record_ref": tasks.source_record_ref.iat[i], "retrieval_id": tasks.retrieval_id.iat[i],
                     "source_url": tasks.source_url.iat[i] if isinstance(tasks.source_url.iat[i], str) else None,
                     "provenance_verified": _verify(con, tasks.posting_id.iat[i], tasks.evidence_text.iat[i])}
                    for i in g]
        mask = skills.posting_id.isin(posts).values
        sk = []
        if mask.any():
            ctx = tv[rep] + q
            ctx = ctx / np.linalg.norm(ctx)
            idx = np.where(mask)[0]
            sims = kv[idx] @ ctx
            seen = set()
            for j in idx[np.argsort(-sims)]:
                s = skills.iloc[j]
                key = s.statement_text.lower()
                if key in seen:
                    continue
                seen.add(key)
                sk.append({"statement_id": s.statement_id, "skill": s.statement_text, "relevance_to_task": round(float(kv[j] @ ctx), 4),
                           "posting_id": s.posting_id, "evidence_text": s.evidence_text, "evidence_section": s.evidence_section,
                           "agreement_status": s.agreement_status, "consensus_score": float(s.consensus_score),
                           "confidence_tier": s.confidence_tier, "esco_candidate": _ref(esco, s.statement_id),
                           "provenance_verified": _verify(con, s.posting_id, s.evidence_text),
                           "explanation": {
                               "result": s.statement_text,
                               "why": (f"Extracted from posting {s.posting_id} ('{s.job_title}'), the same posting that evidences "
                                       f"task '{t.statement_text}'. Similarity to task+initiative = {float(kv[j] @ ctx):.3f}."),
                               "evidence_postings": [s.posting_id],
                               "reference_alignment": {"esco": _ref(esco, s.statement_id)},
                               "model_agreement": {"agreement_status": s.agreement_status, "consensus_score": float(s.consensus_score),
                                                   "confidence_tier": s.confidence_tier},
                               "limitations": LIMITATIONS}})
                if len(sk) >= SKILLS_PER_TASK:
                    break
        onet_c = _ref(onet, t.statement_id)
        results.append({
            "rank": rank, "task_statement": t.statement_text, "statement_id": t.statement_id,
            "rank_score": round(float(score), 4), "relevance_to_initiative": round(float(rel[rep]), 4),
            "n_supporting_postings": len(posts), "evidence": evidence,
            "model_agreement": {"agreement_status": t.agreement_status, "consensus_score": float(t.consensus_score),
                                "confidence_tier": t.confidence_tier, "extractor": f"{t.extractor_provider}:{t.extractor_model}",
                                "other_model_statement": t.paired_statement_text if isinstance(t.paired_statement_text, str) else None},
            "onet_candidate": onet_c, "skills": sk,
            "explanation": {
                "result": t.statement_text,
                "why": (f"Semantic similarity {float(rel[rep]):.3f} between the initiative and this task statement, which was "
                        f"extracted with verbatim evidence from {len(posts)} real posting(s); rank_score = 0.7*relevance + "
                        f"0.3*consensus ({float(t.consensus_score):.3f})."),
                "evidence_postings": posts,
                "reference_alignment": {"onet_task": onet_c},
                "model_agreement": {"agreement_status": t.agreement_status, "consensus_score": float(t.consensus_score)},
                "limitations": LIMITATIONS}})
    status = "ok" if results else "insufficient_evidence"
    run_id = hashlib.sha256(json.dumps({"i": initiative, "p": PARAMS, "n": len(st)}, sort_keys=True).encode()).hexdigest()[:16]
    out = {"run_id": run_id, "initiative": initiative, "status": status, "params": PARAMS,
           "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "results": results,
           "all_results_have_evidence": all(r["evidence"] and all(e["provenance_verified"] for e in r["evidence"]) and
                                            all(s["provenance_verified"] for s in r["skills"]) for r in results),
           "label": "EXPERIMENTAL PROTOTYPE - candidate results for validation"}
    if persist:
        _persist(con, out)
    if own:
        con.close()
    if embedder is None:
        emb.close()
    return out


def _persist(con, out):
    con.execute("CREATE SCHEMA IF NOT EXISTS prototype")
    con.execute("""CREATE TABLE IF NOT EXISTS prototype.prototype_initiative_runs (run_id VARCHAR PRIMARY KEY, initiative VARCHAR,
                   status VARCHAR, n_tasks INTEGER, n_skills INTEGER, all_results_have_evidence BOOLEAN, params_json VARCHAR,
                   created_at VARCHAR)""")
    con.execute("""CREATE TABLE IF NOT EXISTS prototype.prototype_initiative_results (run_id VARCHAR, result_rank INTEGER,
                   result_type VARCHAR, statement_id VARCHAR, parent_statement_id VARCHAR, text VARCHAR, score DOUBLE,
                   evidence_postings VARCHAR, provenance_verified BOOLEAN, explanation_json VARCHAR)""")
    con.execute("DELETE FROM prototype.prototype_initiative_runs WHERE run_id = ?", [out["run_id"]])
    con.execute("DELETE FROM prototype.prototype_initiative_results WHERE run_id = ?", [out["run_id"]])
    con.execute("INSERT INTO prototype.prototype_initiative_runs VALUES (?,?,?,?,?,?,?,?)",
                [out["run_id"], out["initiative"], out["status"], len(out["results"]),
                 sum(len(r["skills"]) for r in out["results"]), out["all_results_have_evidence"],
                 json.dumps(out["params"]), out["created_at"]])
    rows = []
    for r in out["results"]:
        rows.append([out["run_id"], r["rank"], "task", r["statement_id"], None, r["task_statement"], r["rank_score"],
                     json.dumps(r["explanation"]["evidence_postings"]), all(e["provenance_verified"] for e in r["evidence"]),
                     json.dumps(r["explanation"], default=str)])
        for s in r["skills"]:
            rows.append([out["run_id"], r["rank"], "skill", s["statement_id"], r["statement_id"], s["skill"],
                         s["relevance_to_task"], json.dumps([s["posting_id"]]), s["provenance_verified"],
                         json.dumps(s["explanation"], default=str)])
    if rows:
        con.executemany("INSERT INTO prototype.prototype_initiative_results VALUES (?,?,?,?,?,?,?,?,?,?)", rows)


def render_text(out: dict) -> str:
    L = [f"INITIATIVE: {out['initiative']}", f"[{out['label']}]  status={out['status']}", ""]
    for r in out["results"]:
        o = r["onet_candidate"] or {}
        L += [f"TASK {r['rank']}: {r['task_statement']}",
              f"  rank_score={r['rank_score']}  relevance={r['relevance_to_initiative']}  supporting postings={r['n_supporting_postings']}",
              f"  agreement: {r['model_agreement']['agreement_status']} (consensus {r['model_agreement']['consensus_score']:.2f}, "
              f"{r['model_agreement']['confidence_tier']})",
              f"  evidence [{r['evidence'][0]['posting_id']} | {r['evidence'][0]['job_title']}]: \"{r['evidence'][0]['evidence_text']}\"",
              f"  O*NET candidate: {o.get('onet_text')} ({o.get('onet_soc_code')}, sim {o.get('embedding_similarity')})",
              "  SKILLS ASSOCIATED WITH TASK:"]
        for s in r["skills"]:
            e = s["esco_candidate"] or {}
            L.append(f"    - {s['skill']}  | evidence: \"{s['evidence_text'][:90]}\" | ESCO: {e.get('esco_label')} "
                     f"(sim {e.get('embedding_similarity')})")
        L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) or EXAMPLE_INITIATIVES[0]
    print(render_text(analyze(text)))
