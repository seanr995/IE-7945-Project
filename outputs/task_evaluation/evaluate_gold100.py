"""Gold-100 evaluation of the existing Sprint 2 task extraction (no LLM/API calls, read-only DB).

    .venv\\Scripts\\python.exe outputs/task_evaluation/evaluate_gold100.py

Reuses src.prototype.task_eval (validate / evaluate / match) on the completed human gold file.
Steps
  1. clean gold  : drop Excel padding rows (blank posting_id), map status 'labeled' -> 'labelled'
                   -> outputs/task_evaluation/gold100_clean.csv   (data/annotations/task_gold_100.csv is NOT modified)
  2. validate    : row checks -> gold100_row_validation.csv, gold_validation_report.md
  3. evaluate    : task_eval.evaluate under several matching modes -> gold100/<mode>/...
                   strict     = normalized exact only
                   sem_0.75 / sem_0.80 / sem_0.85 = normalized exact OR cosine >= t (lexical stage disabled)
                   cascade    = project default (exact -> Jaccard 0.6 -> cosine 0.85), Sprint 2 continuity
  4. calibrate   : cross-posting null distribution of cosine (false-match rate per threshold)
  5. triage      : deterministic FP / FN categories (heuristic, for manual confirmation)
  6. purity      : existing kind flag counts + 100-statement blind audit sample
  -> extraction_metrics.csv, gold100/summary.json
"""
import hashlib
import json
import re
import sys
from argparse import Namespace
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.prototype.evidence import _norm, containment, jaccard, locate_evidence  # noqa: E402
from src.prototype.task_eval import SYSTEMS, Sim, evaluate, prf  # noqa: E402
from src.prototype.task_gold import read_gold  # noqa: E402
from src.prototype.task_statements import QUALIFICATION_LIKE  # noqa: E402

SRC = ROOT / "data/annotations/task_gold_100.csv"
OUT = ROOT / "outputs/task_evaluation"
RUNS = OUT / "gold100"
DB = ROOT / "database/taxonomy.duckdb"
SEED = "gold100-eval-2026-10-22"
OFF = 1.01  # threshold that can never be reached -> disables a matching stage
MODES = {"strict": dict(lexical_threshold=OFF, semantic_threshold=OFF, review_threshold=OFF),
         "sem_0.75": dict(lexical_threshold=OFF, semantic_threshold=0.75, review_threshold=0.75),
         "sem_0.80": dict(lexical_threshold=OFF, semantic_threshold=0.80, review_threshold=0.80),
         "sem_0.85": dict(lexical_threshold=OFF, semantic_threshold=0.85, review_threshold=0.85),
         "cascade": dict(lexical_threshold=0.6, semantic_threshold=0.85, review_threshold=0.75)}
SKILL_TAIL = re.compile(r"\b(skills?|knowledge|proficiency|experience|abilit(y|ies)|degree|certification)\s*$", re.I)
QUAL_SECTIONS = {"requirements", "preferred_qualifications", "skills"}


def h(*p):
    return hashlib.sha256("|".join(p).encode()).hexdigest()


# ---------------------------------------------------------------- 1. clean
raw = read_gold(SRC)
pad = raw.posting_id == ""
g = raw[~pad].copy()
status_raw = g.adjudication_status.value_counts().to_dict()
g["adjudication_status"] = g.adjudication_status.replace({"labeled": "labelled"})
g.to_csv(OUT / "gold100_clean.csv", index=False, encoding="utf-8-sig")
GOLD = OUT / "gold100_clean.csv"

# ---------------------------------------------------------------- 2. validate
con = duckdb.connect(str(DB), read_only=True)
pids = sorted(g.posting_id.unique())
txt = dict(con.execute("SELECT posting_id, posting_text FROM job_postings WHERE posting_id IN (SELECT unnest(?))",
                       [pids]).fetchall())
sample = con.execute("SELECT posting_id, phase, in_gold_100 FROM task_pipeline.experiment_sample").df()
calls = con.execute("SELECT posting_id, provider FROM prototype.prototype_extraction_calls WHERE status='ok'").df()
first_words = g.gold_task_statement.str.strip().str.split().str[0].str.lower().value_counts()
verbs = set(first_words[first_words >= 2].index)  # lexicon of action verbs the annotator used to open a task


def compound(t):
    """Candidate two-action row: '<verb> X and <verb> Y' (shared-object 'Plan and execute X' is flagged separately)."""
    w = t.lower().replace("&", " and ").replace(",", " , ").split()
    for i, x in enumerate(w[:-1]):
        if x in ("and", ",", ";", "/") and w[i + 1] in verbs and i > 1:
            return "two_actions"
    return "shared_object_verb_pair" if len(w) > 2 and w[1] == "and" and w[2] in verbs else ""


rows = []
for r in g.itertuples():
    t, ev, src = r.gold_task_statement.strip(), r.evidence_span.strip(), txt.get(r.posting_id)
    m = locate_evidence(src, ev)["match_type"] if (src and ev) else "n/a"
    rows.append({"posting_id": r.posting_id, "task_index": r.task_index, "gold_task_statement": t,
                 "task_non_empty": bool(t), "evidence_non_empty": bool(ev), "posting_in_job_postings": src is not None,
                 "evidence_match": m, "evidence_verbatim": m in ("exact", "normalized"), "evidence_chars": len(ev),
                 "task_tokens_in_evidence": round(containment(t, ev), 3) if t and ev else None,
                 "words": len(t.split()), "compound_flag": compound(t),
                 "skill_like": bool(QUALIFICATION_LIKE.match(t) or SKILL_TAIL.search(t)),
                 "first_word_not_verb": t.split()[0].lower() not in verbs if t else True})
v = pd.DataFrame(rows)
v["dup_exact_in_posting"] = v.assign(n=v.gold_task_statement.map(_norm)).duplicated(["posting_id", "n"], keep=False)
near = []
for pid, grp in v.groupby("posting_id"):
    ts = grp.gold_task_statement.tolist()
    for i in range(len(ts)):
        for j in range(i + 1, len(ts)):
            if _norm(ts[i]) != _norm(ts[j]) and jaccard(ts[i], ts[j]) >= 0.75:
                near.append({"posting_id": pid, "task_a": ts[i], "task_b": ts[j], "jaccard": round(jaccard(ts[i], ts[j]), 3)})
near = pd.DataFrame(near, columns=["posting_id", "task_a", "task_b", "jaccard"])
v.to_csv(OUT / "gold100_row_validation.csv", index=False)
near.to_csv(OUT / "gold100_near_duplicates.csv", index=False)
per_post = v.groupby("posting_id").size()
spans_per_post = g.groupby("posting_id").evidence_span.nunique()
in_sample = set(sample[sample.in_gold_100].posting_id)
cover = {p: set(calls[calls.provider == p].posting_id) for p in ("gemini", "groq")}
gp = set(pids)
val = {
    "file": str(SRC.relative_to(ROOT)), "physical_rows": len(raw), "padding_rows_blank_posting_id": int(pad.sum()),
    "padding_rows_nonblank_columns": sorted(c for c in raw.columns if (raw.loc[pad, c].str.strip() != "").any()),
    "task_rows": len(g), "unique_posting_ids": len(gp), "status_values_raw": status_raw,
    "tasks_per_posting": per_post.describe().round(2).to_dict(),
    "tasks_per_posting_min_max": [int(per_post.min()), int(per_post.max())],
    "empty_gold_task": int((~v.task_non_empty).sum()), "empty_evidence": int((~v.evidence_non_empty).sum()),
    "posting_not_in_job_postings": int((~v.posting_in_job_postings).sum()),
    "evidence_match_types": v.evidence_match.value_counts().to_dict(),
    "distinct_evidence_spans_per_posting": spans_per_post.value_counts().sort_index().to_dict(),
    "median_evidence_chars": float(v.evidence_chars.median()),
    "dup_exact_rows": int(v.dup_exact_in_posting.sum()), "near_dup_pairs_jaccard_ge_0.75": len(near),
    "compound_two_actions": int((v.compound_flag == "two_actions").sum()),
    "compound_shared_object": int((v.compound_flag == "shared_object_verb_pair").sum()),
    "skill_like_rows": int(v.skill_like.sum()), "first_word_not_common_verb": int(v.first_word_not_verb.sum()),
    "words_per_task": v.words.describe().round(2).to_dict(),
    "gold_postings_not_in_frozen_gold100": sorted(gp - in_sample), "frozen_gold100_not_labelled": sorted(in_sample - gp),
    "gold_postings_with_gemini_call": len(gp & cover["gemini"]), "gold_postings_with_groq_call": len(gp & cover["groq"]),
    "gold_postings_by_phase": sample[sample.posting_id.isin(gp)].phase.value_counts().to_dict(),
    "groq_covered_by_phase": sample[sample.posting_id.isin(gp & cover["groq"])].phase.value_counts().to_dict(),
}

# ---------------------------------------------------------------- model-output inventory
inv = {
    "task_statements": con.execute("""SELECT provider, count(*) n, sum(evidence_valid::int) evidence_valid,
        sum(is_valid_task::int) is_valid_task, count(DISTINCT posting_id) postings FROM task_pipeline.task_statements
        GROUP BY ALL ORDER BY 1""").df(),
    "task_statements_total": con.execute("""SELECT count(*) n, sum(evidence_valid::int) ev, sum(is_valid_task::int) valid_task,
        count(DISTINCT posting_id) postings FROM task_pipeline.task_statements""").df(),
    "task_statements_gold": con.execute("""SELECT provider, count(*) n, sum(evidence_valid::int) evidence_valid,
        count(DISTINCT posting_id) postings FROM task_pipeline.task_statements
        WHERE posting_id IN (SELECT unnest(?)) GROUP BY ALL ORDER BY 1""", [pids]).df(),
    "raw_outputs": con.execute("""SELECT provider, kind, count(*) n, count(DISTINCT posting_id) postings
        FROM prototype.prototype_model_outputs WHERE gate_status='accepted' GROUP BY ALL ORDER BY 1,2""").df(),
    "raw_outputs_gold": con.execute("""SELECT provider, kind, count(*) n, count(DISTINCT posting_id) postings
        FROM prototype.prototype_model_outputs WHERE gate_status='accepted' AND posting_id IN (SELECT unnest(?))
        GROUP BY ALL ORDER BY 1,2""", [pids]).df(),
    "review_status": con.execute("""SELECT review_status, count(*) n FROM task_pipeline.task_statements
        GROUP BY ALL ORDER BY 2 DESC""").df(),
}
ts_all = con.execute("""SELECT task_statement_id, posting_id, provider, atomic_task, evidence_span, source_section,
    evidence_valid, looks_like_qualification, review_status FROM task_pipeline.task_statements""").df()
preds = {k: con.execute(q).df() for k, (q, _) in SYSTEMS.items() if k != "claude"}
con.close()

# ---------------------------------------------------------------- 3. evaluate
metrics, summ = [], {"validation": val, "modes": {}}
for mode, th in MODES.items():
    print(f"\n=== {mode} ===")
    s = evaluate(Namespace(gold=str(GOLD), out=str(RUNS / mode), **th))
    summ["modes"][mode] = s
    for sysname, r in s["systems"].items():
        metrics.append({"system": sysname, "match_mode": mode, "postings": r["postings_evaluated"],
                        **{k: r["micro"][k] for k in ("predicted", "gold", "tp", "fp", "fn", "precision", "recall", "f1")},
                        **{f"macro_{k}": r["macro_over_postings"][k] for k in ("precision", "recall", "f1")}})
    # head-to-head: every system restricted to the postings Groq actually ran on
    groq_posts = set(pd.read_csv(RUNS / mode / "groq" / "per_posting.csv").posting_id)
    for sysname in s["systems"]:
        pp = pd.read_csv(RUNS / mode / sysname / "per_posting.csv")
        pp = pp[pp.posting_id.isin(groq_posts)]
        r = prf(int(pp.tp.sum()), int(pp.predicted.sum()), int(pp.gold.sum()))
        metrics.append({"system": f"{sysname}@groq_postings", "match_mode": mode, "postings": len(pp),
                        **{k: r[k] for k in ("predicted", "gold", "tp", "fp", "fn", "precision", "recall", "f1")}})
M = pd.DataFrame(metrics)
for c in ("precision", "recall", "f1", "macro_precision", "macro_recall", "macro_f1"):
    M[c] = M[c].astype(float).round(4)
M.to_csv(OUT / "extraction_metrics.csv", index=False)

# ---------------------------------------------------------------- 4. calibrate thresholds
gt = g[["posting_id", "gold_task_statement"]].rename(columns={"gold_task_statement": "t"}).assign(t=lambda x: x.t.str.strip())
sim = Sim()
pr = {k: d[d.posting_id.isin(gp)].drop_duplicates(["posting_id", "task"]) for k, d in preds.items()}
sim.prime(list(gt.t) + [t for d in pr.values() for t in d.task])
calib = []
for sysname in ("gemini", "groq"):
    d = pr[sysname]
    posts = sorted(set(d.posting_id))
    by = {p: d[d.posting_id == p].task.tolist() for p in posts}
    same, null = [], []
    for i, p in enumerate(posts):
        other = posts[int(h(SEED, "null", p), 16) % len(posts)]
        if other == p:
            other = posts[(i + 1) % len(posts)]
        for t in gt[gt.posting_id == p].t:
            same.append(max(sim(t, x) for x in by[p]))
            null.append(max(sim(t, x) for x in by[other]))
    same, null = np.array(same), np.array(null)
    for t in (0.70, 0.75, 0.80, 0.85, 0.90):
        calib.append({"system": sysname, "threshold": t, "gold_tasks": len(same),
                      "same_posting_best_cos_ge_t": round(float((same >= t).mean()), 4),
                      "other_posting_best_cos_ge_t (false-match proxy)": round(float((null >= t).mean()), 4)})
calib = pd.DataFrame(calib)
calib.to_csv(OUT / "gold100_threshold_calibration.csv", index=False)
band = []
for mode in ("sem_0.75", "sem_0.80", "sem_0.85"):
    for sysname in ("gemini", "groq"):
        tp = pd.read_csv(RUNS / mode / sysname / "true_positives.csv")
        tp = tp[tp.match_method == "semantic"]
        lo = float(mode.split("_")[1])
        b = tp[(tp.score >= lo) & (tp.score < lo + 0.05)].assign(k=lambda x: [h(SEED, "band", a, b) for a, b in zip(x.gold_task, x.predicted_task)])
        band.append(b.sort_values("k").head(25).drop(columns="k").assign(system=sysname, band=f"[{lo:.2f},{lo + 0.05:.2f})"))
pd.concat(band).drop_duplicates(["posting_id", "gold_task", "predicted_task"]).to_csv(
    OUT / "gold100_threshold_band_samples.csv", index=False)

# ---------------------------------------------------------------- 5. triage (heuristic)
def tri(mode):
    out = []
    for sysname in ("gemini", "groq", "baseline_agreed"):
        d = RUNS / mode / sysname
        fp, fn = (pd.read_csv(d / f, keep_default_na=False) for f in ("false_positives.csv", "false_negatives.csv"))
        P = pr[sysname]
        for r in fp.itertuples():
            golds = gt[gt.posting_id == r.posting_id].t.tolist()
            best = max(sim(r.predicted_task, x) for x in golds)
            same_src = max(containment(x, str(r.evidence_span)) for x in golds)
            if "hallucinated" in r.error_category or QUALIFICATION_LIKE.match(r.predicted_task) or SKILL_TAIL.search(r.predicted_task):
                cat = "unsupported_or_non_task (qualification/skill wording or weak evidence)"
            elif r.section in QUAL_SECTIONS:
                cat = "qualification_section_item (likely non-task; verify)"
            elif best >= 0.70 or same_src >= 0.6:
                cat = "wording_or_splitting_mismatch (same source sentence / near paraphrase)"
            else:
                cat = "possible_legit_task_missing_from_gold"
            out.append({"system": sysname, "side": "model_only (FP)", "posting_id": r.posting_id, "text": r.predicted_task,
                        "section": r.section, "best_cos_to_gold": round(best, 3), "gold_tokens_in_pred_evidence": round(same_src, 3),
                        "category": cat})
        for r in fn.itertuples():
            pp = P[P.posting_id == r.posting_id]
            best = max(sim(r.gold_task, x) for x in pp.task) if len(pp) else 0.0
            same_src = max((containment(r.gold_task, str(e)) for e in pp.evidence_span), default=0.0)
            if QUALIFICATION_LIKE.match(r.gold_task) or SKILL_TAIL.search(r.gold_task):
                cat = "gold_row_is_skill_or_qualification"
            elif best >= 0.70 or same_src >= 0.6:
                cat = "wording_or_splitting_mismatch (model used the same sentence / near paraphrase)"
            else:
                cat = "true_missed_human_task"
            out.append({"system": sysname, "side": "gold_only (FN)", "posting_id": r.posting_id, "text": r.gold_task,
                        "section": "", "best_cos_to_gold": round(best, 3), "gold_tokens_in_pred_evidence": round(same_src, 3),
                        "category": cat})
    return pd.DataFrame(out)


triage = {}
for mode in ("sem_0.80", "sem_0.85"):
    t = tri(mode)
    t.to_csv(OUT / f"gold100_error_triage_{mode}.csv", index=False)
    triage[mode] = t.groupby(["system", "side", "category"]).size().rename("n").reset_index()
    triage[mode].to_csv(OUT / f"gold100_error_triage_summary_{mode}.csv", index=False)

# gold exhaustiveness proxy: responsibilities-section predictions whose source sentence contains no gold task
exh = []
for sysname in ("gemini", "groq"):
    d = pr[sysname]
    d = d[d.section == "responsibilities"]
    hit = [max(containment(x, str(e)) for x in gt[gt.posting_id == p].t) >= 0.6 for p, e in zip(d.posting_id, d.evidence_span)]
    exh.append({"system": sysname, "responsibility_predictions": len(d), "source_sentence_has_a_gold_task": int(sum(hit)),
                "share": round(float(np.mean(hit)), 4)})
sim.close()
summ["exhaustiveness_proxy"] = exh
summ["calibration"] = calib.to_dict("records")

# ---------------------------------------------------------------- 6. purity
pur = {"raw_kind_accepted": inv["raw_outputs"].to_dict("records"),
       "task_statements_looks_like_qualification": int(ts_all.looks_like_qualification.sum())}
ev = ts_all[ts_all.evidence_valid.fillna(False).astype(bool)].copy()
ev["heur_qual_regex"] = ev.atomic_task.str.match(QUALIFICATION_LIKE) | ev.atomic_task.str.contains(SKILL_TAIL)
ev["heur_short_no_verb"] = ev.atomic_task.str.split().str.len().le(2)
pur["evidence_valid_by_section"] = ev.groupby(["provider", "source_section"]).size().rename("n").reset_index().to_dict("records")
pur["evidence_valid_heur_qual_regex"] = int(ev.heur_qual_regex.sum())
pur["evidence_valid_le_2_words"] = int(ev.heur_short_no_verb.sum())
pur["evidence_valid_from_qual_sections"] = int(ev.source_section.isin(QUAL_SECTIONS).sum())
pur["evidence_valid_total"] = len(ev)
ev["k"] = [h(SEED, "purity", s) for s in ev.task_statement_id]
aud = pd.concat([ev[ev.provider == p].sort_values("k").head(50) for p in ("gemini", "groq")])
aud = aud.rename(columns={"task_statement_id": "statement_id", "atomic_task": "statement_text"})
aud = aud[["statement_id", "posting_id", "provider", "statement_text", "evidence_span"]].assign(human_label="")
aud.to_csv(OUT / "task_purity_audit_100.csv", index=False, encoding="utf-8-sig")
pur["audit_sample"] = {"rows": len(aud), "by_provider": aud.provider.value_counts().to_dict(),
                       "rule": "50 per provider, evidence_valid task_statements, ordered by sha256(seed|purity|statement_id)"}
summ["purity"] = pur
summ["inventory"] = {k: v.to_dict("records") for k, v in inv.items()}
(RUNS / "summary.json").write_text(json.dumps(summ, indent=2, default=str), encoding="utf-8")

# ---------------------------------------------------------------- console digest
pd.set_option("display.width", 220)
print("\nVALIDATION", json.dumps(val, indent=1, default=str))
for k, d in inv.items():
    print(f"\n{k}\n{d.to_string(index=False)}")
print("\nMETRICS\n", M[["system", "match_mode", "postings", "predicted", "gold", "tp", "fp", "fn", "precision", "recall", "f1"]].to_string(index=False))
print("\nCALIBRATION\n", calib.to_string(index=False))
for m, t in triage.items():
    print(f"\nTRIAGE {m}\n", t.to_string(index=False))
print("\nEXHAUSTIVENESS", exh)
print("\nPURITY", json.dumps({k: v for k, v in pur.items() if k != "evidence_valid_by_section"}, indent=1, default=str))
