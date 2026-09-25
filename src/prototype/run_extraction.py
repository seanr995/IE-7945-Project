"""Dual-model evidence-preserving extraction experiment. PROTOTYPE / EXPERIMENTAL / FOR VALIDATION.

    python -m src.prototype.run_extraction

Idempotent: every LLM response is cached, tables are CREATE OR REPLACE'd in schema `prototype`.
"""
from __future__ import annotations

import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import duckdb
import numpy as np
import pandas as pd

from src.llm.base import load_config
from src.llm.embeddings import Embedder
from src.llm.router import Router
from src.prototype.evidence import (consensus, jaccard, locate_evidence, match_items, split_events,
                                    support_overlap, _norm)
from src.prototype.sample import build_sample
from src.prototype.sections import detect_sections, model_input
from src.utils.common import DB_PATH, OUTPUTS, get_logger, load_env

log = get_logger("prototype.extraction")
ADV = OUTPUTS / "advanced"


def sid(*parts) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:20]


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------- gating one provider output

def gate_items(posting: dict, provider: str, res, sections_method: str) -> list[dict]:
    rows, seen = [], set()
    src = posting["posting_text"]
    for k, it in enumerate(res.data["items"] if res else []):
        loc = locate_evidence(src, it["evidence_text"])
        sup = support_overlap(it["statement_text"], loc["exact_text"] or it["evidence_text"])
        key = (it["kind"], _norm(it["statement_text"]))
        if loc["match_type"] == "none":
            status = "rejected_evidence_not_found"
        elif sup < 0.34:
            status = "rejected_unsupported_statement"
        elif key in seen:
            status = "rejected_duplicate_in_output"
        else:
            status = "accepted"
        seen.add(key)
        rows.append({
            "output_id": sid(posting["posting_id"], provider, k, it["statement_text"]),
            "posting_id": posting["posting_id"], "provider": provider, "model": res.model,
            "prompt_version": res_prompt_version(), "request_id": res.request_id, "cache_hit": res.cache_hit,
            "item_index": k, "kind": it["kind"], "statement_text": it["statement_text"].strip(),
            "evidence_text_model": it["evidence_text"], "evidence_section": it["evidence_section"],
            "evidence_match_type": loc["match_type"], "evidence_start": loc["start"], "evidence_end": loc["end"],
            "evidence_text": loc["exact_text"], "evidence_fuzzy_ratio": loc["ratio"], "support_overlap": sup,
            "gate_status": status, "section_detection_method": sections_method,
            "start": loc["start"], "end": loc["end"],
        })
    return rows


_PV = {}


def res_prompt_version():
    return _PV.get("extract", "extract-v1")


def triggers(rows: list[dict], ok: bool) -> list[str]:
    """Quality rules that make the Gemini-primary phase call Groq."""
    if not ok:
        return ["gemini_failed_or_schema_invalid"]
    t = []
    acc = [r for r in rows if r["gate_status"] == "accepted"]
    bad = [r for r in rows if r["gate_status"].startswith("rejected_evidence") or r["gate_status"].startswith("rejected_unsupported")]
    if rows and len(bad) / len(rows) >= 0.2:
        t.append("evidence_failure_rate>=20%")
    if len(acc) < 3 or not any(r["kind"] == "task" for r in acc):
        t.append("unusually_empty_or_no_tasks")
    if any(r["evidence_match_type"] == "fuzzy" for r in acc):
        t.append("evidence_only_fuzzy_traceable")
    kinds = {}
    for r in acc:
        kinds.setdefault(_norm(r["statement_text"]), set()).add(r["kind"])
    if any(len(v) > 1 for v in kinds.values()):
        t.append("structurally_ambiguous_task_vs_skill")
    return t


# ---------------------------------------------------------------- main

def main(limit: int | None = None) -> dict:
    load_env()
    cfg = load_config()
    _PV.update(cfg["prompt_versions"])
    ex = cfg["experiment"]
    run_id = "run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    router = Router(run_id=run_id, cfg=cfg)
    con = duckdb.connect(str(DB_PATH))
    con.execute("CREATE SCHEMA IF NOT EXISTS prototype")

    sample = build_sample(con, ex["sample_seed"], ex["sample_size"], ex["dual_model_postings"], ex["min_description_words"])
    if limit:
        sample = sample.head(limit)
    con.register("s", sample)
    con.execute("CREATE OR REPLACE TABLE prototype.prototype_extraction_sample AS SELECT * FROM s")
    con.unregister("s")
    log.info("Sample: %d postings (%s)", len(sample), sample.groupby(["phase", "source"]).size().to_dict())

    ids = sample.posting_id.tolist()
    posts = con.execute("SELECT posting_id, job_title, description, requirements, preferred_skills, posting_text, source "
                        "FROM job_postings WHERE posting_id IN (SELECT unnest(?))", [ids]).df()
    posts = {r.posting_id: {k: (v if isinstance(v, str) else None) for k, v in r._asdict().items()}
             for r in posts.itertuples(index=False)}
    phase = dict(zip(sample.posting_id, sample.phase))

    prep = {}
    for pid in ids:
        p = posts[pid]
        det = detect_sections(p["description"], p["requirements"], p["preferred_skills"])
        prep[pid] = {"sections": det["sections"], "method": det["method"],
                     "input": model_input(det["sections"], ex["max_input_chars"])}
    sec_rows = [{"posting_id": pid, "section": s["section"], "origin": s["origin"], "start_in_origin": s["start"],
                 "end_in_origin": s["end"], "section_detection_method": s["method"], "chars": len(s["text"])}
                for pid in ids for s in prep[pid]["sections"]]
    con.register("sr", pd.DataFrame(sec_rows))
    con.execute("CREATE OR REPLACE TABLE prototype.prototype_sections AS SELECT * FROM sr")
    con.unregister("sr")
    con.close()  # release the DuckDB write lock during the long API phase

    calls = {}  # (pid, provider) -> LLMResult | Exception

    def run(pid, provider):
        p = posts[pid]
        try:
            return pid, provider, router.extract_statements(provider, pid, p["job_title"], prep[pid]["input"])
        except Exception as e:
            log.error("%s extraction failed for %s: %s", provider, pid, e)
            return pid, provider, e

    dual = [p for p in ids if phase[p] == "dual_model"]
    prim = [p for p in ids if phase[p] == "gemini_primary"]
    log.info("Extracting: %d dual-model postings, %d gemini-primary postings", len(dual), len(prim))
    with ThreadPoolExecutor(cfg["gemini"]["max_concurrency"]) as gx, ThreadPoolExecutor(1) as qx:
        fg = [gx.submit(run, p, "gemini") for p in dual + prim]
        fq = [qx.submit(run, p, "groq") for p in dual]
        for f in fg + fq:
            pid, prov, r = f.result()
            calls[(pid, prov)] = r
        # quality-rule triggered Groq calls for the gemini-primary phase
        out_rows = {}
        trig = {}
        for pid in prim:
            r = calls[(pid, "gemini")]
            ok = not isinstance(r, Exception)
            rows = gate_items(posts[pid], "gemini", r, prep[pid]["method"]) if ok else []
            out_rows[(pid, "gemini")] = rows
            t = triggers(rows, ok)
            if t:
                trig[pid] = t
        log.info("Gemini-primary phase: Groq triggered for %d/%d postings", len(trig), len(prim))
        for f in [qx.submit(run, p, "groq") for p in trig]:
            pid, prov, r = f.result()
            calls[(pid, prov)] = r

    # gate everything
    for (pid, prov), r in calls.items():
        if (pid, prov) not in out_rows:
            out_rows[(pid, prov)] = gate_items(posts[pid], prov, r, prep[pid]["method"]) if not isinstance(r, Exception) else []
    all_out = [row for rows in out_rows.values() for row in rows]
    runs = [{"posting_id": pid, "provider": prov, "phase": phase[pid], "status": "error" if isinstance(r, Exception) else "ok",
             "error": str(r)[:200] if isinstance(r, Exception) else None,
             "model": None if isinstance(r, Exception) else r.model,
             "cache_hit": None if isinstance(r, Exception) else r.cache_hit,
             "n_items": len(out_rows.get((pid, prov), [])),
             "schema_invalid_items": None if isinstance(r, Exception) else int(r.data.get("schema_invalid_items", 0)),
             "n_accepted": sum(x["gate_status"] == "accepted" for x in out_rows.get((pid, prov), [])),
             "trigger_reasons": "; ".join(trig.get(pid, [])) if prov == "groq" and phase[pid] == "gemini_primary" else None}
            for (pid, prov), r in calls.items()]

    # semantic similarity between providers' accepted statements (embeddings, not chat models)
    emb = Embedder(run_id=run_id)
    compared = [pid for pid in ids if (pid, "groq") in calls and not isinstance(calls[(pid, "groq")], Exception)
                and not isinstance(calls.get((pid, "gemini")), Exception)]
    texts = sorted({r["statement_text"] for pid in compared for prov in ("gemini", "groq")
                    for r in out_rows[(pid, prov)] if r["gate_status"] == "accepted"})
    vec = dict(zip(texts, emb.embed(texts))) if texts else {}

    statements, match_log, adjud_items = [], [], []
    for pid in ids:
        g = [r for r in out_rows.get((pid, "gemini"), []) if r["gate_status"] == "accepted"]
        if pid in compared:
            q = [r for r in out_rows[(pid, "groq")] if r["gate_status"] == "accepted"]
            sim = {(i, j): float(np.dot(vec[a["statement_text"]], vec[b["statement_text"]]))
                   for i, a in enumerate(g) for j, b in enumerate(q) if a["kind"] == b["kind"]}
            m = match_items(g, q, sim)
            match_log.append({"posting_id": pid, "phase": phase[pid], "n_gemini": len(g), "n_groq": len(q),
                              "n_pairs": len(m["pairs"]),
                              "gemini_task": sum(x["kind"] == "task" for x in g), "groq_task": sum(x["kind"] == "task" for x in q),
                              "gemini_skill": sum(x["kind"] == "skill" for x in g), "groq_skill": sum(x["kind"] == "skill" for x in q),
                              "pairs_task": sum(g[i]["kind"] == "task" for i, _, _, _ in m["pairs"]),
                              "pairs_skill": sum(g[i]["kind"] == "skill" for i, _, _, _ in m["pairs"]),
                              "pairs_lexical": sum(t == "lexical" for _, _, t, _ in m["pairs"]),
                              "pairs_semantic": sum(t == "semantic" for _, _, t, _ in m["pairs"]),
                              "gemini_oversplit_events": split_events(g, q), "groq_oversplit_events": split_events(q, g),
                              "gemini_unmatched_kinds": json.dumps({k: sum(v[0] == k for v in m["gemini_unmatched"].values())
                                                                    for k in ("boundary", "kind_conflict", "missing_in_other")}),
                              "groq_unmatched_kinds": json.dumps({k: sum(v[0] == k for v in m["groq_unmatched"].values())
                                                                  for k in ("boundary", "kind_conflict", "missing_in_other")})})
            for i, j, mt, sc in m["pairs"]:
                a, b = g[i], q[j]
                statements.append(_stmt(a, "gemini", "groq", b, f"{mt}_agree", mt, sc, phase[pid], cfg))
            for side, items, other, un in (("gemini", g, q, m["gemini_unmatched"]), ("groq", q, g, m["groq_unmatched"])):
                for i, (cls, js) in un.items():
                    o = other[js[0]] if js else None
                    st = _stmt(items[i], side, None, o, f"{side}_only", "boundary" if cls == "boundary" else "single_model",
                               None, phase[pid], cfg, disagreement=cls)
                    statements.append(st)
                    if cls in ("kind_conflict", "boundary"):
                        adjud_items.append(st)
        else:
            for a in g:
                statements.append(_stmt(a, "gemini", None, None, "not_compared", "single_model", None, phase[pid], cfg))

    # AI adjudication (Groq) for kind conflicts & boundary disagreements - blind to which model proposed
    by_post = {}
    for st in adjud_items:
        by_post.setdefault(st["posting_id"], []).append(st)
    adj_posts = sorted(by_post)[:int(ex.get('adjudication_max_postings', 30))]
    for pid in adj_posts:
        items = by_post[pid][:20]
        try:
            r = router.adjudicate_disagreement("groq", pid, "\n\n".join(s["text"] for s in prep[pid]["input"]),
                                               [{"proposed_by": "extractor", "kind": s["kind"], "statement_text": s["statement_text"],
                                                 "evidence_text": s["evidence_text"]} for s in items])
            dec = {d["item_index"]: d for d in r.data["decisions"]}
            for k, s in enumerate(items):
                d = dec.get(k)
                if d:
                    s["adjudication_decision"] = d["decision"]
                    s["adjudicated_kind"] = d["kind"]
                    s["adjudicator"] = f"groq:{r.model}"
        except Exception as e:
            log.error("adjudication failed for %s: %s", pid, e)
    for s in statements:
        s["adjudication_required"] = s["disagreement_type"] in ("kind_conflict", "boundary")

    con = duckdb.connect(str(DB_PATH))
    st_df = pd.DataFrame(statements).drop_duplicates("statement_id")
    mo_df = pd.DataFrame(all_out).drop(columns=["start", "end"])
    for name, df in (("prototype_statements", st_df), ("prototype_model_outputs", mo_df),
                     ("prototype_extraction_calls", pd.DataFrame(runs)), ("prototype_model_matching", pd.DataFrame(match_log))):
        con.register("df", df)
        con.execute(f"CREATE OR REPLACE TABLE prototype.{name} AS SELECT * FROM df")
        con.unregister("df")
    build_review_queue(con, out_rows, posts)
    con.close()
    emb.close()
    log.info("Extraction experiment done: %d statements", len(st_df))
    return {"run_id": run_id, "statements": len(st_df)}


def _stmt(a, provider, validator, other, agreement_status, agreement_key, score, phase, cfg, disagreement=None):
    kind_conf = disagreement == "kind_conflict"
    cons, tier = consensus(a["evidence_match_type"], agreement_key, a["support_overlap"], kind_conf)
    return {
        "statement_id": sid(a["posting_id"], a["kind"], _norm(a["statement_text"])),
        "posting_id": a["posting_id"], "kind": a["kind"], "statement_text": a["statement_text"],
        "evidence_text": a["evidence_text"], "evidence_text_model": a["evidence_text_model"],
        "evidence_section": a["evidence_section"], "evidence_start": a["evidence_start"], "evidence_end": a["evidence_end"],
        "evidence_offset_basis": "job_postings.posting_text" if a["evidence_start"] is not None else "offset unavailable",
        "evidence_match_type": a["evidence_match_type"], "source_language": "en",
        "extractor_provider": provider, "extractor_model": a["model"], "prompt_version": a["prompt_version"],
        "schema_version": cfg["schema_version"],
        "validation_status": ("confirmed_by_independent_extraction" if validator else
                              ("not_confirmed_by_other_model" if agreement_status.endswith("_only") else "not_validated")),
        "validator_provider": validator or (("groq" if provider == "gemini" else "gemini") if agreement_status.endswith("_only") else None),
        "validator_model": other["model"] if (validator and other) else None,
        "agreement_status": agreement_status, "match_score": score,
        "paired_statement_text": other["statement_text"] if other else None,
        "disagreement_type": disagreement, "phase": phase,
        "schema_valid": True, "evidence_valid": a["evidence_match_type"] in ("exact", "normalized"),
        "model_agreement": agreement_key in ("lexical", "semantic"), "semantic_agreement": agreement_key == "semantic",
        "support_overlap": a["support_overlap"], "kind_conflict": kind_conf,
        "adjudication_required": False, "adjudication_decision": None, "adjudicated_kind": None, "adjudicator": None,
        "consensus_score": cons, "confidence_tier": tier, "section_detection_method": a["section_detection_method"],
        "created_at": now(),
    }


def build_review_queue(con, out_rows, posts):
    st = con.execute("SELECT * FROM prototype.prototype_statements").df()
    items = []
    for r in st.itertuples():
        reasons, pr = [], 3
        if r.kind_conflict:
            reasons.append("ambiguous task-vs-skill (models disagree on kind)")
            pr = 1
        if r.adjudication_decision == "reject":
            reasons.append("AI adjudicator rejected (needs human confirmation)")
            pr = 1
        if r.evidence_match_type == "fuzzy":
            reasons.append("evidence only fuzzy-traceable")
            pr = min(pr, 1)
        if r.disagreement_type == "boundary":
            reasons.append("statement-boundary disagreement between models")
            pr = min(pr, 2)
        if r.confidence_tier == "low":
            reasons.append("low consensus score")
            pr = min(pr, 2)
        if reasons:
            items.append((r, reasons, pr))
    rows = []
    for r, reasons, pr in items:
        g = [x for x in out_rows.get((r.posting_id, "gemini"), []) if x["evidence_start"] is not None and r.evidence_start is not None
             and max(x["evidence_start"], r.evidence_start) < min(x["evidence_end"], r.evidence_end)]
        q = [x for x in out_rows.get((r.posting_id, "groq"), []) if x["evidence_start"] is not None and r.evidence_start is not None
             and max(x["evidence_start"], r.evidence_start) < min(x["evidence_end"], r.evidence_end)]
        rows.append({"review_id": sid("review", r.statement_id), "posting_id": r.posting_id, "statement_id": r.statement_id,
                     "source_text": r.evidence_text,
                     "gemini_output": json.dumps([{"kind": x["kind"], "statement": x["statement_text"]} for x in g], ensure_ascii=False),
                     "groq_output": json.dumps([{"kind": x["kind"], "statement": x["statement_text"]} for x in q], ensure_ascii=False),
                     "reason_for_review": "; ".join(reasons), "priority": pr,
                     "ai_adjudication": r.adjudication_decision, "human_decision": None, "reviewer": None, "reviewed_at": None})
    # rejected-by-gate outputs from the dual phase are also worth a human look (possible hallucination)
    for (pid, prov), outs in out_rows.items():
        for x in outs:
            if x["gate_status"] in ("rejected_evidence_not_found", "rejected_unsupported_statement"):
                rows.append({"review_id": sid("review", x["output_id"]), "posting_id": pid, "statement_id": None,
                             "source_text": x["evidence_text_model"],
                             "gemini_output": json.dumps([{"kind": x["kind"], "statement": x["statement_text"]}]) if prov == "gemini" else "[]",
                             "groq_output": json.dumps([{"kind": x["kind"], "statement": x["statement_text"]}]) if prov == "groq" else "[]",
                             "reason_for_review": f"invalid evidence: {x['gate_status']} ({prov})", "priority": 1,
                             "ai_adjudication": None, "human_decision": None, "reviewer": None, "reviewed_at": None})
    df = pd.DataFrame(rows).drop_duplicates("review_id").sort_values(["priority", "posting_id"])
    df = df.astype({"human_decision": "object", "reviewer": "object", "reviewed_at": "object"})
    con.register("rq", df)
    con.execute("""CREATE OR REPLACE TABLE prototype.prototype_review_queue AS
                   SELECT review_id, posting_id, statement_id, source_text, gemini_output, groq_output, reason_for_review,
                          priority, ai_adjudication, CAST(human_decision AS VARCHAR) human_decision,
                          CAST(reviewer AS VARCHAR) reviewer, CAST(reviewed_at AS TIMESTAMP) reviewed_at FROM rq""")
    con.unregister("rq")


if __name__ == "__main__":
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else None
    print(main(limit=lim))
