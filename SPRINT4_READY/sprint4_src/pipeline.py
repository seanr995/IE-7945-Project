"""Sprint 4 orchestration (Gemini provider): evidence -> leaf labels -> hierarchy search -> hierarchy builder -> O*NET retrieval ->
O*NET evaluator -> metrics -> critic + convergence -> stability / gap analysis / cards / human template -> validation/zips."""
from __future__ import annotations

import json
import os
import re
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import agents as A
from . import extras as X
from . import hierarchy as H
from . import onet as O
from .budget import Ledger, historical_claude_spend
from .config import (CODE_DIR, FROZEN_MEMBERS, FROZEN_REPORT, FROZEN_TOPICS, LEAF_BATCH, ONET_BATCH, ONET_DIR_CANDIDATES, PLANNED_NORMAL_CALLS,
                     T5_EMB, T5_MAP, T5_META, CallBudgetError, Settings, StopRun, append_jsonl, env_search_paths, load_env, log, read_json,
                     utc, write_json)
from .evidence import Frozen, build_leaf_evidence, l2
from .llm import LLM, GeminiTransport, MockTransport
from .schemas import CriticBatch, HierarchyResult, LeafBatch, OnetBatch

OUT = ["leaf_evidence.csv", "leaf_topics_labeled.csv", "task_hierarchy.csv", "onet_top10_matches.csv", "onet_evaluation.csv",
       "agent_decisions.jsonl", "hierarchy_diagnostics.csv", "sprint4_evaluation_report.md"]
ZIP_NAME = "Sprint4_Agentic_Results.zip"


def chunks(xs, n):
    return [xs[i:i + n] for i in range(0, len(xs), n)]


def jl(x):
    return json.loads(x) if isinstance(x, str) else x


def make_settings(base, mock=None) -> Settings:
    load_env(Path(base))
    return Settings.from_env(base, mock)


def make_embedder(s: Settings):
    return O.hash_embedder if s.mock else O.t5_embedder()


def safe_config(base, mock=None) -> dict:
    """Configuration summary that is safe to print (never contains key values)."""
    s = make_settings(base, mock)
    return {"LLM provider": s.provider, "Gemini key present": bool(os.environ.get("GEMINI_API_KEY")), "Gemini model": s.model,
            "Mock mode": s.mock, "env files searched": [str(p) + (" (found)" if p.exists() else " (missing)") for p in env_search_paths(base)]}


# ------------------------------------------------------------------ preflight (no generation request is spent)
REQUIRED_FILES = [FROZEN_TOPICS, FROZEN_MEMBERS, FROZEN_REPORT, T5_EMB, T5_MAP, T5_META]


def preflight(base, mock=None) -> dict:
    s = make_settings(base, mock)
    checks = {}
    checks["llm_provider"] = s.provider
    checks["gemini_model_configured"] = bool(s.model)
    if not s.mock:
        if not os.environ.get("GEMINI_API_KEY"):
            raise StopRun("GEMINI_API_KEY missing: add it to SPRINT4_READY/.env (or the project-folder .env)")
        try:
            from google import genai      # noqa: F401
        except ImportError:
            raise StopRun("google-genai is not installed (pip install google-genai)")
        checks["google_genai_import"] = "ok"
    else:
        checks["google_genai_import"] = "skipped (mock)"
    missing = [f for f in REQUIRED_FILES if not s.find(f).exists()]
    onet_ok = any(s.find(c + "/task_statements.csv").exists() and s.find(c + "/tasks_to_dwas.csv").exists() for c in ONET_DIR_CANDIDATES)
    if missing or not onet_ok:
        raise StopRun(f"missing required input files: {missing + ([] if onet_ok else ['O*NET task_statements.csv / tasks_to_dwas.csv'])}")
    checks["frozen_and_sentence_t5_files"] = "present"
    checks["onet_source"] = "present"
    for d in (s.results, s.prov, s.cache):
        d.mkdir(parents=True, exist_ok=True)
        probe = d / ".write_test"
        probe.write_text("ok")
        probe.unlink()
    checks["output_and_checkpoint_folders_writable"] = True
    led = Ledger(s)                                                       # raises if the ledger is unreadable
    n_cache = len(list(s.cache.rglob("*.json")))
    checks["ledger_and_gemini_cache_readable"] = {"ledger_rows": len(led.d["rows"]), "cached_responses": n_cache}
    if led.real_used >= s.max_real_calls:
        raise CallBudgetError(f"GEMINI_CALL_CAP_REACHED: {led.real_used}/{s.max_real_calls} real calls already used")
    fz = Frozen(s, O.hash_embedder if s.mock else None)
    r = {"utc": utc(), "mock": s.mock, "status": "OK", "provider": s.provider, "model": s.model, "checks": checks,
         "normal_call_plan": PLANNED_NORMAL_CALLS, "hard_call_cap": s.max_real_calls, "real_calls_used_so_far": led.real_used,
         "real_calls_remaining": s.max_real_calls - led.real_used, "min_seconds_between_calls": s.min_seconds,
         "frozen_leaves": len(fz.topics), "frozen_tasks": len(fz.members), "historical_sprint3_claude": historical_claude_spend(s)}
    write_json(s.prov / ("preflight_mock.json" if s.mock else "preflight_status.json"), r)
    log(f"PREFLIGHT OK | provider {s.provider} | model {s.model} | normal plan {PLANNED_NORMAL_CALLS} calls, hard cap {s.max_real_calls}, "
        f"used so far {led.real_used} | cached responses {n_cache} | frozen {len(fz.topics)} leaves / {len(fz.members)} tasks")
    return r


# ------------------------------------------------------------------ state helpers
class State:
    def __init__(self):
        self.labels, self.l2, self.l2_info, self.l1_of_l2, self.l1_info = {}, {}, {}, {}, {}
        self.unresolved, self.last_action, self.history, self.provisional = {}, {}, [], set()


def ids_by_size(fz, assign):
    t = pd.Series(np.asarray(fz.sizes)).groupby(np.asarray(assign)).sum()
    return list(t.sort_values(ascending=False, kind="stable").index)


def short(text, n=20):
    return " ".join(str(text).split()[:n])


def log_decision(dec, role, target, d, **kw):
    append_jsonl(dec, {"utc": utc(), "kind": "decision", "role": role, "target": target, "decision": d, **kw})


# ------------------------------------------------------------------ the run
def run(base, mock=None, responder=None, sleep=None, transport=None) -> dict:
    s = make_settings(base, mock)
    s.results.mkdir(parents=True, exist_ok=True)
    s.prov.mkdir(parents=True, exist_ok=True)
    dec = s.results / "agent_decisions.jsonl"
    if dec.exists():
        shutil.move(dec, s.prov / f"agent_decisions_{utc().replace(':', '-')}.jsonl")
    pre = preflight(base, mock)
    led = Ledger(s)
    try:
        return _run(s, led, pre, dec, responder, sleep, transport)
    finally:                                         # ledger/usage files are always written, also after quota/auth stops
        X.export_api_usage(s, led)


def _run(s, led, pre, dec, responder, sleep, transport):
    R = s.results
    embed = make_embedder(s)
    fz = Frozen(s, O.hash_embedder if s.mock else None)
    if transport is None:
        if s.mock:
            from .mock import mock_responder
            transport = MockTransport(responder or mock_responder)
        else:
            transport = GeminiTransport(os.environ["GEMINI_API_KEY"], s.model)
    llm = LLM(s, led, transport, sleep=sleep or time.sleep)

    # ---- 1. leaf evidence (deterministic)
    log("[1/9] leaf evidence")
    ev = build_leaf_evidence(fz)
    ev.to_csv(R / "leaf_evidence.csv", index=False)
    fz.ev = ev
    evi = ev.set_index("topic_id")

    # ---- 2. Gemini leaf labeler (20 topics per call)
    log("[2/9] Gemini leaf labeler")
    st = State()

    def leaf_payload(ids, errs):
        d = {"topics": [{"id": t, "n": int(fz.sizes[fz.topic_index[t]]), "core": jl(evi.loc[t].core_tasks)[:7],
                         "edge": jl(evi.loc[t].boundary_tasks)[:3], "kw": jl(evi.loc[t].ctfidf_keywords)[:8]} for t in ids]}
        if errs:
            d["repair"] = errs
        return d
    for i, ids in enumerate(chunks(fz.topic_ids, LEAF_BATCH), 1):
        res, failed = llm.batch("leaf_labeler", A.SYS_LEAF, LeafBatch, ids, leaf_payload, lambda o: A.by_id(o["topics"], "topic_id"),
                                A.leaf_validate, f"leaf-{i}")
        for t, r in res.items():
            st.labels[t] = r
            log_decision(dec, "leaf_labeler", t, r)
        for t, msg in failed.items():                # FLAG_UNRESOLVED: keyword placeholder, never silently accepted
            kw = jl(evi.loc[t].ctfidf_keywords)[:2]
            st.labels[t] = {"topic_id": t, "topic_name": "Unresolved Label " + " ".join(k.title() for k in kw), "description": "Gemini label was invalid after repair; see evidence.",
                            "core_action": "review", "object_or_outcome": "needs human label", "include_if": "see core tasks", "exclude_if": "see boundary tasks",
                            "coherent": False, "mixed_topic": True, "confidence": 0.0, "reason": "unresolved"}
            st.unresolved[t] = f"leaf label invalid after batched repair: {msg}"
            log_decision(dec, "leaf_labeler", t, {"status": "FLAG_UNRESOLVED", "error": msg})
    assert len(st.labels) == 80

    # ---- 3. hierarchy candidate search (deterministic)
    log("[3/9] hierarchy candidate search")
    diag, Z, k2, k1 = H.search(fz)
    l2lab = H.order_labels(H.cut(Z, k2), fz.sizes)
    l1raw = H.cut(Z, k1)
    l2_ids = [f"F2_{i + 1:02d}" for i in range(k2)]
    st.l2 = {t: l2_ids[int(l2lab[i])] for i, t in enumerate(fz.topic_ids)}
    l1_for_l2 = {}
    for i, t in enumerate(fz.topic_ids):
        l1_for_l2.setdefault(st.l2[t], set()).add(int(l1raw[i]))
    assert all(len(v) == 1 for v in l1_for_l2.values()), "L2/L1 cuts are not nested"
    raw_l1_of_l2 = {k: next(iter(v)) for k, v in l1_for_l2.items()}
    l1_order = ids_by_size(fz, [raw_l1_of_l2[st.l2[t]] for t in fz.topic_ids])
    l1_ids = {raw: f"D1_{i + 1:02d}" for i, raw in enumerate(l1_order)}
    st.l1_of_l2 = {k: l1_ids[v] for k, v in raw_l1_of_l2.items()}
    diag_rows = [diag]
    sel_link = str(diag[diag.selected].iloc[0].linkage)
    log_decision(dec, "hierarchy_search", "candidates", {"selected_level2": k2, "selected_level1": k1, "linkage": sel_link, "weights": H.SCORE_WEIGHTS,
                                                         "candidates": diag.drop(columns=["stage"]).to_dict("records")})

    # ---- 4. Gemini hierarchy builder (ONE call: families + domains)
    log("[4/9] Gemini hierarchy builder")
    dom_ids = sorted(set(st.l1_of_l2.values()))
    kids = {p: [t for t in fz.topic_ids if st.l2[t] == p] for p in l2_ids}
    kids1 = {d: [p for p in l2_ids if st.l1_of_l2[p] == d] for d in dom_ids}

    def hier_payload(ids, errs):
        fam = [p for p in l2_ids if p in ids]
        dom = [d for d in dom_ids if d in ids]
        d = {"families": [{"id": p, "leaves": [{"id": t, "name": st.labels[t]["topic_name"], "act": st.labels[t]["core_action"],
                                                "d": short(st.labels[t]["description"], 20)} for t in kids[p]]} for p in fam],
             "domains": [{"id": x, "families": kids1[x], "leaf_names": [st.labels[t]["topic_name"] for p in kids1[x] for t in kids[p]]} for x in dom]}
        if errs:
            d["repair"] = errs
        return d
    all_h = l2_ids + dom_ids
    children = {**kids, **kids1}
    res, failed = llm.batch("hierarchy_builder", A.SYS_HIER, HierarchyResult, all_h, hier_payload,
                            lambda o: {**A.by_id(o["families"], "parent_id"), **A.by_id(o["domains"], "parent_id")},
                            lambda r, i: A.parent_validate(r, i, children[i]), "hier-1", splittable=False)
    for pid in all_h:
        info = st.l2_info if pid in kids else st.l1_info
        if pid in res:
            r = res[pid]
            info[pid] = {"name": r["parent_name"], "description": r["parent_description"], "confidence": r["confidence"],
                         "misplaced": r["misplaced_child_ids"], "reason": r["reason"], "level": 2 if pid in kids else 1}
            log_decision(dec, "hierarchy_builder", pid, r)
        else:
            info[pid] = {"name": f"Unresolved {'Family' if pid in kids else 'Domain'} {pid}", "description": "Gemini naming invalid after repair.",
                         "confidence": 0.0, "misplaced": [], "reason": "unresolved", "level": 2 if pid in kids else 1}
            st.unresolved[pid] = f"hierarchy naming invalid after batched repair: {failed.get(pid)}"
            log_decision(dec, "hierarchy_builder", pid, {"status": "FLAG_UNRESOLVED", "error": failed.get(pid)})
    diag_rows.append(pd.DataFrame([H.diagnostics_row(fz, st.l2, st.l1_of_l2, "after_hierarchy_builder")]))

    # ---- 5. O*NET retrieval (deterministic; Gemini never searches)
    log("[5/9] O*NET retrieval (Sentence-T5, exact top-10 by cosine)")
    onet = O.load_onet(s)
    if not s.mock:
        chk = O.verify_same_model(fz, embed)
        log(f"  Sentence-T5 re-embedding check: min cosine {chk['min_cosine']:.4f}")
    E = O.onet_embeddings(s, onet, embed, "mock" if s.mock else "t5")
    order, sims, S_all = O.retrieve_top10(fz.centroids, E)
    top10 = O.build_top10_table(fz, onet, order, sims, {t: st.labels[t]["topic_name"] for t in fz.topic_ids})
    top10.to_csv(R / "onet_top10_matches.csv", index=False)
    top_by = {t: g.sort_values("rank") for t, g in top10.groupby("topic_id")}

    # ---- 6. Gemini O*NET evaluator (20 topics per call; sees top-5 of the retrieved top-10)
    log("[6/9] Gemini O*NET evaluator")
    cev = {}

    def onet_payload(ids, errs):
        d = {"topics": [{"id": t, "name": st.labels[t]["topic_name"], "tasks": jl(evi.loc[t].core_tasks)[:3],
                         "cands": [{"oid": int(r.onet_task_id), "t": r.onet_task, "occ": r.occupation_title, "cos": round(float(r.cosine_similarity), 3)}
                                   for r in top_by[t].head(5).itertuples()]} for t in ids]}
        if errs:
            d["repair"] = errs
        return d
    for i, ids in enumerate(chunks(fz.topic_ids, ONET_BATCH), 1):
        res, failed = llm.batch("onet_evaluator", A.SYS_ONET, OnetBatch, ids, onet_payload, lambda o: A.by_id(o["evaluations"], "topic_id"),
                                lambda r, t: A.onet_validate(r, t, [int(x) for x in top_by[t].head(5).onet_task_id]), f"onet-{i}")
        for t, r in res.items():
            cev[t] = r
            log_decision(dec, "onet_evaluator", t, r)
        for t, msg in failed.items():
            st.unresolved[t] = (st.unresolved.get(t, "") + " | " if t in st.unresolved else "") + f"O*NET judgment invalid after batched repair: {msg}"
            log_decision(dec, "onet_evaluator", t, {"status": "FLAG_UNRESOLVED", "error": msg})

    # ---- 7. quantitative O*NET metrics
    log("[7/9] O*NET quantitative evaluation")
    onet_df, onet_sum = O.evaluate_onet(fz, onet, E, order, sims, S_all, cev)

    # ---- 8. critic (one batched call) + convergence controller
    log("[8/9] Gemini critic and convergence")
    pre_stab, _ = X.hierarchy_stability(fz, st, sel_link, k2, k1)        # deterministic; used only as a flag for the critic
    conv = critic_loop(s, llm, fz, st, onet_df, diag_rows, dec, pre_stab)

    # ---- 9. outputs + advancements
    log("[9/9] outputs, stability, gap analysis, cards, human-review template")
    onet_df, onet_sum = O.evaluate_onet(fz, onet, E, order, sims, S_all, cev)      # recomputed on the converged state
    top10["topic_name"] = top10.topic_id.map({t: st.labels[t]["topic_name"] for t in fz.topic_ids})
    top10.to_csv(R / "onet_top10_matches.csv", index=False)
    onet_df.insert(1, "topic_name", onet_df.topic_id.map({t: st.labels[t]["topic_name"] for t in fz.topic_ids}))
    onet_df.to_csv(R / "onet_evaluation.csv", index=False)
    write_json(R / "onet_summary.json", onet_sum)
    dg = pd.concat([d if isinstance(d, pd.DataFrame) else pd.DataFrame([d]) for d in diag_rows], ignore_index=True)
    dg.to_csv(R / "hierarchy_diagnostics.csv", index=False)
    write_outputs(s, fz, st, onet_df, cev)
    stab_df, stab_sum = X.hierarchy_stability(fz, st, sel_link, k2, k1)           # validation only; hierarchy not altered
    stab_df.to_csv(R / "hierarchy_stability.csv", index=False)
    write_json(R / "hierarchy_stability_summary.json", stab_sum)
    gap_df, gap_sum = X.gap_analysis(fz, st, onet_df, top10)
    gap_df.to_csv(R / "onet_gap_analysis.csv", index=False)
    write_json(R / "onet_gap_summary.json", gap_sum)
    X.explainability_cards(s, fz, st, onet_df, top10, gap_df, stab_df)
    X.human_review_template(s, fz, st)
    X.export_api_usage(s, led)
    summary = {"level1": len(set(st.l1_of_l2.values())), "level2": len(set(st.l2.values())), "leaves": 80,
               "onet_tasks": len(onet), "ledger": led.summary(), "historical_claude": historical_claude_spend(s), "convergence": conv,
               "onet": onet_sum, "selected": {"level2": k2, "level1": k1, "linkage": sel_link}, "embedding_source": fz.emb_source,
               "mock": s.mock, "stability": stab_sum, "gap": gap_sum, "unresolved": dict(st.unresolved)}
    write_json(R / "run_summary.json", summary)
    from .report import write_report
    write_report(s, fz, st, diag, onet_sum, onet_df, summary, preflight=pre, diag_all=dg)
    make_archives(s)
    return summary


# ------------------------------------------------------------------ critic
def derived_flags(fz, st, onet_df, ev):
    """All thresholds are quantiles of the observed distributions (no hand-set cutoffs)."""
    conf = np.array([st.labels[t]["confidence"] for t in fz.topic_ids])
    q_conf = float(np.percentile(conf, 25))
    q_pur = float(np.percentile(onet_df.purity, 25))
    q_bound = float(np.percentile(fz.ev.boundary_mean_sim, 25))
    return q_conf, q_pur, q_bound


def select_cases(fz, st, onet_df, exclude=(), pre_stab=None):
    q_conf, q_pur, q_bound = derived_flags(fz, st, onet_df, fz.ev)
    o = onet_df.set_index("topic_id")
    bm = fz.ev.set_index("topic_id").boundary_mean_sim
    mis = {c for info in st.l2_info.values() for c in info["misplaced"]}
    pers = pre_stab.set_index("topic_id").level2_parent_persistence if pre_stab is not None else None
    q_pers = float(np.percentile(pers, 25)) if pers is not None else None
    leaves = []
    for t in fz.topic_ids:
        if t in exclude:
            continue
        r = []
        L = st.labels[t]
        if L["mixed_topic"] or not L["coherent"]:
            r.append("mixed_topic")
        if L["confidence"] <= q_conf:
            r.append("low_label_confidence")
        if o.loc[t, "alignment"] in ("NO_MATCH", "UNEVALUATED") or not o.loc[t, "meaningful_alignment"] or \
                (o.loc[t, "alignment"] == "PARTIAL_MATCH" and o.loc[t, "below_split_threshold"]):
            r.append("weak_onet_alignment")
        if o.loc[t, "purity"] <= q_pur:
            r.append("low_onet_purity")
        if bm.loc[t] <= q_bound:
            r.append("inconsistent_boundary_tasks")
        if t in mis:
            r.append("misplaced_child_recommended")
        if pers is not None and pers[t] <= q_pers and pers[t] < 1.0:
            r.append("unstable_hierarchy_parent")
        if r:
            leaves.append((len(r), t, r))
    leaves.sort(key=lambda x: (-x[0], x[1]))
    leaves = leaves[:24]
    pt = H.parent_table(fz, st.l2).set_index("parent_id")
    q_coh = float(np.percentile(pt.cohesion, 25))
    q_ov = float(np.percentile(pt.nearest_parent_sim, 75))
    q_pc = float(np.percentile([i["confidence"] for i in st.l2_info.values()], 25))
    pmean = ({p: float(np.mean([pers[t] for t in fz.topic_ids if st.l2[t] == p])) for p in pt.index} if pers is not None else None)
    q_pm = float(np.percentile(list(pmean.values()), 25)) if pmean else None
    parents = []
    for p in pt.index:
        if p in exclude:
            continue
        r = []
        if pt.loc[p, "cohesion"] <= q_coh:
            r.append("low_cohesion_parent")
        if pt.loc[p, "nearest_parent_sim"] >= q_ov:
            r.append("overlapping_parent_definition")
        if st.l2_info[p]["misplaced"]:
            r.append("misplaced_children")
        if st.l2_info[p]["confidence"] <= q_pc:
            r.append("low_parent_confidence")
        if pmean and pmean[p] <= q_pm:
            r.append("unstable_hierarchy_parent")
        if len(r) >= 2 or "misplaced_children" in r:
            parents.append((len(r), p, r))
    parents.sort(key=lambda x: (-x[0], x[1]))
    return leaves, parents[:10], {"q_label_conf": q_conf, "q_purity": q_pur, "q_boundary_sim": q_bound, "q_parent_cohesion": q_coh,
                                  "q_parent_overlap": q_ov, "q_parent_conf": q_pc, "q_leaf_l2_persistence": q_pers}


def critic_loop(s, llm, fz, st, onet_df, diag_rows, dec, pre_stab):
    """ONE batched critic call (plus at most the shared batched repair). Then recompute diagnostics and evaluate convergence."""
    conv = {"rounds": [], "stop_reason": None}
    prev_l2 = dict(st.l2)
    prev_names = {t: st.labels[t]["topic_name"] for t in fz.topic_ids}
    prev_coh = H.diagnostics_row(fz, st.l2, st.l1_of_l2, "x")["L2_cohesion"]
    leaves, parents, thr = select_cases(fz, st, onet_df, pre_stab=pre_stab)
    if not leaves and not parents:
        conv["stop_reason"] = "no problematic cases flagged; critic not called"
        conv["unresolved"] = dict(st.unresolved)
        return conv
    o = onet_df.set_index("topic_id")
    evi = fz.ev.set_index("topic_id")
    simp = H.parent_table(fz, st.l2).set_index("parent_id")
    l2ids = sorted(set(st.l2.values()))
    cases = []
    for _, t, reasons in leaves:
        alt = H.best_alternative_parent(fz, st.l2, t)
        top_alt = sorted(((p, v) for p, v in alt.items() if p != st.l2[t]), key=lambda x: -x[1])[:2]
        cases.append({"type": "leaf", "id": t, "why": reasons, "name": st.labels[t]["topic_name"], "d": short(st.labels[t]["description"], 30),
                      "core": jl(evi.loc[t].core_tasks)[:3], "edge": jl(evi.loc[t].boundary_tasks)[:2], "conf": st.labels[t]["confidence"],
                      "parent": [st.l2[t], st.l2_info[st.l2[t]]["name"], round(alt.get(st.l2[t], 0.0), 3)],
                      "alt": [[p, st.l2_info[p]["name"], round(v, 3)] for p, v in top_alt],
                      "onet": [o.loc[t, "alignment"], o.loc[t, "dominant_onet_concept"], o.loc[t, "top1_similarity"], o.loc[t, "purity"]]})
    for _, p, reasons in parents:
        cases.append({"type": "parent", "id": p, "why": reasons, "name": st.l2_info[p]["name"], "d": short(st.l2_info[p]["description"], 30),
                      "coh": round(float(simp.loc[p, "cohesion"]), 3),
                      "near": [simp.loc[p, "nearest_parent_id"], st.l2_info[simp.loc[p, "nearest_parent_id"]]["name"], round(float(simp.loc[p, "nearest_parent_sim"]), 3)],
                      "mis": st.l2_info[p]["misplaced"], "kids": [[t, st.labels[t]["topic_name"]] for t in fz.topic_ids if st.l2[t] == p]})
    case_types = {c["id"]: c["type"] for c in cases}
    l2set, l1set = set(l2ids), set(st.l1_info)

    def payload(ids, errs):
        d = {"cases": [c for c in cases if c["id"] in ids], "l2": [{"id": p, "name": st.l2_info[p]["name"]} for p in l2ids], "l1": sorted(l1set)}
        if errs:
            d["repair"] = errs
        return d
    res, failed = llm.batch("critic", A.SYS_CRITIC, CriticBatch, list(case_types), payload, lambda o_: A.by_id(o_["decisions"], "target_id"),
                            lambda r, i: A.critic_validate(r, i, case_types[i], l2set, l1set), "critic-1", splittable=False)
    decisions = [res[i] for i in case_types if i in res]
    for i, msg in failed.items():
        st.unresolved[i] = f"critic decision invalid after batched repair: {msg}"
        log_decision(dec, "critic", i, {"status": "FLAG_UNRESOLVED", "error": msg})
    stats = apply_decisions(fz, st, decisions, dec, 1)
    stats["flagged"] += len(failed)
    row = H.diagnostics_row(fz, st.l2, st.l1_of_l2, "after_critic_round_1")
    diag_rows.append(pd.DataFrame([row]))
    changed = sum(prev_l2[t] != st.l2[t] for t in fz.topic_ids) / len(fz.topic_ids)
    renamed = sum(prev_names[t] != st.labels[t]["topic_name"] for t in fz.topic_ids)
    d_coh = row["L2_cohesion"] - prev_coh
    stats.update({"round": 1, "cases": len(case_types), "leaf_assignment_change_share": round(changed, 4), "leaf_labels_changed": renamed,
                  "delta_L2_cohesion": round(d_coh, 5)})
    conv["rounds"].append(stats)
    stable_labels = renamed == 0 and stats["parent_renames"] == 0 and stats["refinements"] == 0
    no_struct = stats["merges"] == 0 and stats["splits"] == 0
    if changed < 0.05 and no_struct and (stable_labels or d_coh < 0.005):
        conv["stop_reason"] = ("converged after the critic round: <5% leaf-to-parent assignments changed, no merge/split requested, "
                               "labels stable or metrics not materially improving")
    else:
        conv["stop_reason"] = ("single critic round completed (the call plan allows one critic call); leaf-to-parent change "
                               f"{changed:.1%}, merges {stats['merges']}, splits {stats['splits']} - a further round would need extra calls, "
                               "so remaining differences are listed for human review")
    conv["unresolved"] = dict(st.unresolved)
    return conv


def apply_decisions(fz, st, decisions, dec, rnd):
    stats = {"accepts": 0, "leaf_renames": 0, "parent_renames": 0, "refinements": 0, "moves": 0, "merges": 0, "splits": 0,
             "flagged": 0, "rejected": 0}
    for d in decisions:
        tid, act, applied = d["target_id"], d["action"], True
        st.last_action[tid] = act
        is_leaf = d["target_type"] == "leaf"
        info = st.labels.get(tid) if is_leaf else (st.l2_info.get(tid) or st.l1_info.get(tid))
        key_n, key_d = ("topic_name", "description") if is_leaf else ("name", "description")
        if act == "ACCEPT":
            stats["accepts"] += 1
        elif act == "RENAME":
            info[key_n] = d["new_name"].strip()
            stats["leaf_renames" if is_leaf else "parent_renames"] += 1
            st.provisional.discard(tid)
        elif act == "REFINE_DESCRIPTION":
            info[key_d] = d["new_description"].strip()
            stats["refinements"] += 1
        elif act == "MOVE_PARENT":
            src, dst = st.l2[tid], d["move_to_parent_id"]
            if dst == src or dst not in st.l2_info or sum(1 for v in st.l2.values() if v == src) <= 1:
                applied = False
                stats["rejected"] += 1
            else:
                st.l2[tid] = dst
                stats["moves"] += 1
        elif act == "MERGE_PARENT":
            src, dst = tid, d["merge_with_parent_id"]
            if src in st.l2_info and dst in st.l2_info:
                for t in fz.topic_ids:
                    if st.l2[t] == src:
                        st.l2[t] = dst
                st.l2_info.pop(src)
                st.l1_of_l2.pop(src)
                stats["merges"] += 1
            elif src in st.l1_info and dst in st.l1_info:
                for p, dd in st.l1_of_l2.items():
                    if dd == src:
                        st.l1_of_l2[p] = dst
                st.l1_info.pop(src)
                stats["merges"] += 1
            else:
                applied = False
                stats["rejected"] += 1
        elif act == "SPLIT_PARENT":
            leaves = [t for t in fz.topic_ids if st.l2[t] == tid]
            if len(leaves) >= 4:
                idx = [fz.topic_index[t] for t in leaves]
                D = H.cosine_distance(fz.centroids[idx])
                g = H.cut(H.build_linkage(D), 2)
                nid = f"F2_{max(int(p.split('_')[1]) for p in st.l2_info) + 1:02d}"
                for t, gi in zip(leaves, g):
                    if gi == 1:
                        st.l2[t] = nid
                base = st.l2_info[tid]
                st.l2_info[tid] = dict(base, name=base["name"] + " (Part 1)")
                st.l2_info[nid] = dict(base, name=base["name"] + " (Part 2)", reason="provisional name after critic SPLIT_PARENT")
                st.l1_of_l2[nid] = st.l1_of_l2[tid]
                st.provisional |= {tid, nid}
                stats["splits"] += 1
            else:
                applied = False
                stats["rejected"] += 1
        elif act == "FLAG_UNRESOLVED":
            st.unresolved[tid] = d["reason"]
            stats["flagged"] += 1
        append_jsonl(dec, {"utc": utc(), "kind": "decision", "role": "critic", "round": rnd, "target": tid, "applied": applied, "decision": d})
    return stats


# ------------------------------------------------------------------ outputs
def write_outputs(s, fz, st, onet_df, cev):
    R = s.results
    ev = fz.ev.set_index("topic_id")
    o = onet_df.set_index("topic_id")
    rows = []
    for t, raw in zip(fz.topics.topic_id, fz.topics.raw_leiden_label):
        L, p = st.labels[t], st.l2[t]
        d = st.l1_of_l2[p]
        rows.append({"topic_id": t, "raw_leiden_label": int(raw), "topic_size": int(fz.sizes[fz.topic_index[t]]),
                     "topic_name": L["topic_name"], "description": L["description"], "core_action": L["core_action"],
                     "object_or_outcome": L["object_or_outcome"], "include_if": L["include_if"], "exclude_if": L["exclude_if"],
                     "coherent": L["coherent"], "mixed_topic": L["mixed_topic"], "label_confidence": L["confidence"],
                     "label_reason": L["reason"], "ctfidf_keywords": ev.loc[t, "ctfidf_keywords"],
                     "level2_id": p, "level2_name": st.l2_info[p]["name"], "level1_id": d, "level1_name": st.l1_info[d]["name"],
                     "onet_alignment": o.loc[t, "alignment"], "onet_dominant_concept": o.loc[t, "dominant_onet_concept"],
                     "critic_action": st.last_action.get(t, "NOT_REVIEWED"),
                     "final_status": "UNRESOLVED" if t in st.unresolved else "LABELED"})
    pd.DataFrame(rows).to_csv(R / "leaf_topics_labeled.csv", index=False)
    m = fz.members
    th = pd.DataFrame({"statement_id": m.statement_id, "posting_id": m.posting_id, "task_text": m.task_text,
                       "leaf_topic_id": m.topic_id, "leaf_topic_name": m.topic_id.map(lambda t: st.labels[t]["topic_name"]),
                       "level2_id": m.topic_id.map(st.l2), "level2_name": m.topic_id.map(lambda t: st.l2_info[st.l2[t]]["name"]),
                       "level1_id": m.topic_id.map(lambda t: st.l1_of_l2[st.l2[t]]),
                       "level1_name": m.topic_id.map(lambda t: st.l1_info[st.l1_of_l2[st.l2[t]]]["name"]),
                       "membership_strength": m.membership_strength})
    th.to_csv(R / "task_hierarchy.csv", index=False)
    write_json(R / "hierarchy_structure.json", {
        "level1": {d: {**st.l1_info[d], "children": sorted(p for p, dd in st.l1_of_l2.items() if dd == d)} for d in sorted(st.l1_info)},
        "level2": {p: {**st.l2_info[p], "children": [t for t in fz.topic_ids if st.l2[t] == p]} for p in sorted(st.l2_info)},
        "provisional_names": sorted(st.provisional)})


RETURN_INSTRUCTIONS = r"""1. Download Sprint4_RETURN_TO_REPO.zip from Google Drive.
2. Extract it into:
   D:\Capstone IE7945\Taxonomy Project\CAPESTONE PROJECT\
3. Allow the new Sprint 4 files to merge into the project.
4. Do not delete or overwrite Sprint 3 result folders.
5. Run `git status`.
6. Commit the new Sprint 4 files.
"""
RETURN_ZIP = "Sprint4_RETURN_TO_REPO.zip"
NOTEBOOK = "Sprint4_Agentic_Labeling_Hierarchy_Colab.ipynb"


def _result_files(s: Settings):
    """sprint4_results minus large/duplicate/temporary material: no embedding .npy, no mock cache, no .tmp / pycache.
    Claude response caches (small JSON) are kept: they let a re-run replay for $0."""
    for p in sorted(s.results.rglob("*")):
        rel = p.relative_to(s.results)
        if p.is_dir() or p.suffix in (".tmp", ".npy", ".pyc") or "__pycache__" in rel.parts or "gemini_mock" in rel.parts:
            continue
        if p.name == ".env" or ("mock" in p.name and not s.mock):
            continue
        yield p, Path("sprint4_results") / rel


def make_archives(s: Settings):
    for name in (ZIP_NAME, RETURN_ZIP):
        zp = s.base / name
        if zp.exists():
            zp.unlink()
    with zipfile.ZipFile(s.base / ZIP_NAME, "w", zipfile.ZIP_DEFLATED) as z:
        for p, arc in _result_files(s):
            z.write(p, arc.as_posix())
    with zipfile.ZipFile(s.base / RETURN_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for p, arc in _result_files(s):
            z.write(p, arc.as_posix())
        for f in (NOTEBOOK, "README_RUN_SPRINT4.md"):
            src = s.find(f)
            if src.exists():
                z.write(src, f)
        z.writestr("RETURN_TO_REPO_INSTRUCTIONS.txt", RETURN_INSTRUCTIONS)
    return s.base / ZIP_NAME, s.base / RETURN_ZIP


# ------------------------------------------------------------------ validation
def validate(base, mock=None) -> dict:
    s = make_settings(base, mock)
    R = s.results
    checks = {}

    def ck(name, ok, detail=""):
        checks[name] = {"ok": bool(ok), "detail": str(detail)}
    for f in OUT:
        ck(f"exists:{f}", (R / f).exists())
    fm = pd.read_csv(s.find(FROZEN_MEMBERS))
    ft = pd.read_csv(s.find(FROZEN_TOPICS))
    th = pd.read_csv(R / "task_hierarchy.csv")
    lab = pd.read_csv(R / "leaf_topics_labeled.csv")
    ev = pd.read_csv(R / "leaf_evidence.csv")
    t10 = pd.read_csv(R / "onet_top10_matches.csv")
    oe = pd.read_csv(R / "onet_evaluation.csv")
    ck("80 leaves labeled", len(lab) == 80 and lab.topic_id.nunique() == 80 and set(lab.topic_id) == set(ft.topic_id))
    ck("80 leaves evidence", len(ev) == 80)
    ck("3015 tasks in hierarchy, unique", len(th) == 3015 and th.statement_id.nunique() == 3015)
    mm = th.merge(fm[["statement_id", "topic_id"]], on="statement_id")
    ck("leaf membership unchanged vs frozen", len(mm) == 3015 and (mm.leaf_topic_id == mm.topic_id).all())
    ck("every task has level1/level2", th[["level1_id", "level2_id", "level1_name", "level2_name"]].notna().all().all())
    nest = th.groupby("level2_id").level1_id.nunique()
    ck("level2 nested in level1", (nest == 1).all())
    ck("every leaf has a single level2", th.groupby("leaf_topic_id").level2_id.nunique().eq(1).all())
    ck("no empty parent names", (th.level2_name.str.len() > 0).all() and (th.level1_name.str.len() > 0).all())
    ck("top-10 exactly 10 per topic", len(t10) == 800 and (t10.groupby("topic_id").size() == 10).all())
    ck("top-10 sorted by cosine desc", all(g.cosine_similarity.is_monotonic_decreasing for _, g in t10.groupby("topic_id")))
    ok_best = True
    for r in oe.itertuples():
        best = set(json.loads(r.best_onet_task_ids))
        ok_best &= best <= set(t10[t10.topic_id == r.topic_id].onet_task_id)
    ck("O*NET best ids only from the deterministic retrieved top-10 (and only the top-5 shown to Gemini)", ok_best)
    ck("80 O*NET evaluations", len(oe) == 80 and oe.alignment.isin(["STRONG_MATCH", "PARTIAL_MATCH", "NO_MATCH"]).all())
    led = Ledger(s)
    sm = led.summary()
    ck("Gemini real calls <= hard cap 12", sm["gemini_real_calls"] <= s.max_real_calls, sm)
    ck("successful normal calls <= 10 (plan) and repair calls <= 2", sm["normal_successful_calls"] <= 10 and sm["repair_calls"] <= 2, sm)
    ck("ledger provider is gemini only; Claude Sprint 4 calls = 0", all(r["provider"] == "gemini" for r in led.d["rows"]) and sm["claude_sprint4_calls"] == 0)
    src = chr(10).join(p.read_text(encoding="utf-8") for p in list((CODE_DIR / "sprint4_src").glob("*.py")) + [CODE_DIR / "sprint4_agentic_pipeline.py"] if p.exists())
    banned = "|".join([r"import\s+anthrop" + "ic", r"from\s+anthrop" + "ic", r"Anthrop" + r"ic\s*\(", "ANTHROP" + "IC_API_KEY", "claude" + "-sonnet"])
    ck("no active Anthropic dependency in runtime code", re.search(banned, src) is None)
    # --- advancement outputs
    sb = pd.read_csv(R / "hierarchy_stability.csv")
    ck("hierarchy_stability covers all 80 leaves", len(sb) == 80 and set(sb.topic_id) == set(ft.topic_id))
    ck("stability persistence in [0,1] and multiples of 1/5", sb[["level1_parent_persistence", "level2_parent_persistence"]].apply(
        lambda c: ((c >= 0) & (c <= 1) & ((c * 5 - (c * 5).round()).abs() < 1e-6)).all()).all())
    ss = read_json(R / "hierarchy_stability_summary.json") or {}
    ck("stability used the 5 required seeds", ss.get("seeds") == [101, 202, 303, 404, 505] and ss.get("leaves_kept") == 80, ss.get("seeds"))
    gp = pd.read_csv(R / "onet_gap_analysis.csv")
    ck("O*NET gap analysis covers all 80 leaves", len(gp) == 80 and set(gp.topic_id) == set(ft.topic_id)
       and gp.coverage_class.isin(["WELL_COVERED", "PARTIALLY_COVERED", "POSSIBLE_EMERGING_WORK"]).all())
    cards = pd.read_csv(R / "topic_explainability_cards.csv")
    md_files = sorted(p.name for p in (R / "topic_explainability_cards").glob("*"))
    ck("80 explainability CSV rows", len(cards) == 80 and set(cards.topic_id) == set(ft.topic_id))
    ck("exactly 80 explainability Markdown cards LT_001..LT_080", md_files == [f"LT_{i:03d}.md" for i in range(1, 81)])
    hr = pd.read_csv(R / "human_review_template.csv", keep_default_na=False)
    human = ["label_correct", "coherent", "boundary_ok", "over_merged", "under_merged", "suggested_name", "reviewer_notes"]
    ck("human_review_template: 80 rows, human columns blank", len(hr) == 80 and (hr[human] == "").all().all())
    au = pd.read_csv(R / "api_usage.csv")
    ck("api_usage.csv logs every ledger row", len(au) == len(led.d["rows"]) and (R / "api_ledger.jsonl").exists())
    def no_secret(blob: bytes) -> bool:
        key = os.environ.get("GEMINI_API_KEY", "")
        return b"AIza" not in blob and b"sk-ant-" not in blob and not (key and key.encode() in blob)
    # --- archives (two passes so the zips also contain validation.json)
    def archive_checks():
        for nm in (ZIP_NAME, RETURN_ZIP):
            zp = s.base / nm
            ck(f"exists:{nm}", zp.exists())
            if not zp.exists():
                continue
            with zipfile.ZipFile(zp) as z:
                names = z.namelist()
                blob = b"".join(z.read(n) for n in names if n.endswith((".json", ".jsonl", ".md", ".csv", ".txt", ".ipynb", ".py")))
            ck(f"{nm}: all 8 required outputs + extras", all(f"sprint4_results/{f}" in names for f in OUT + [
                "hierarchy_stability.csv", "hierarchy_stability_summary.json", "onet_gap_analysis.csv", "topic_explainability_cards.csv",
                "human_review_template.csv", "api_usage.csv", "api_ledger.jsonl"]))
            ck(f"{nm}: 80 cards", sum(n.startswith("sprint4_results/topic_explainability_cards/LT_") for n in names) == 80)
            ck(f"{nm}: no .env / caches / npy / secrets", not any(n.endswith(".env") or "__pycache__" in n or n.endswith((".npy", ".pyc"))
                                                                   or "gemini_mock" in n for n in names) and no_secret(blob))
            if nm == RETURN_ZIP:
                ck("return zip: relative paths + notebook + README + instructions",
                   all(not n.startswith(("/", "..")) and ":" not in n for n in names) and NOTEBOOK in names and "README_RUN_SPRINT4.md" in names
                   and "RETURN_TO_REPO_INSTRUCTIONS.txt" in names and all(n.startswith("sprint4_results/") or n in
                                                                          (NOTEBOOK, "README_RUN_SPRINT4.md", "RETURN_TO_REPO_INSTRUCTIONS.txt") for n in names))
                with zipfile.ZipFile(zp) as z:
                    ck("return zip: instructions text exact", z.read("RETURN_TO_REPO_INSTRUCTIONS.txt").decode("utf-8") == RETURN_INSTRUCTIONS)
    dj = (R / "agent_decisions.jsonl").read_text(encoding="utf-8") if (R / "agent_decisions.jsonl").exists() else ""
    ck("no API key in decisions log", no_secret(dj.encode("utf-8")))
    for _ in range(2):
        make_archives(s)
        archive_checks()
        out = {"status": "PASS" if all(c["ok"] for c in checks.values()) else "FAIL", "checks": checks}
        write_json(R / "validation.json", out)
    make_archives(s)
    return out
