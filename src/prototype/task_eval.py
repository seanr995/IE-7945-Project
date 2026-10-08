"""Task-extraction evaluation: HUMAN GOLD vs MODEL-EXTRACTED tasks. SECTION A - TASK EXTRACTION.

Run ONLY after data/annotations/task_gold_100.csv has been completed by humans:

    python -m src.prototype.task_eval validate     # format + verbatim evidence-span check of the gold file(s)
    python -m src.prototype.task_eval evaluate     # P / R / F1 per system  -> outputs/task_evaluation/
    python -m src.prototype.task_eval agreement    # Annotator A vs B on the 20-posting audit (before adjudication)

Options: --semantic-threshold 0.85  --review-threshold 0.75  --lexical-threshold 0.6  --gold PATH  --out DIR

Matching (within the SAME posting, never across postings):
  1. normalized exact   lower-case, quotes/dashes/whitespace unified, trailing punctuation dropped
  2. lexical            stemmed content-word Jaccard >= lexical threshold
  3. semantic           cosine (local bge-small-en-v1.5, cached) >= semantic threshold
  Candidate pairs are ranked (stage, score) and assigned greedily ONE-TO-ONE: a gold task can match at most one
  prediction and vice versa. Pairs with review <= cosine < semantic (and no stronger match) are written to
  ambiguous_review.csv for a human decision; they are NOT counted as true positives (strict metrics), and an
  upper bound counting them is reported separately as `if_ambiguous_accepted`.
No LLM is used as a judge. Error categories are deterministic HEURISTICS for triage, to be confirmed manually.

Systems (each evaluated only on gold postings where it produced output; coverage reported):
  gemini, groq             raw evidence-gated model outputs (prototype.prototype_model_outputs, gate accepted)
  baseline_valid           task_pipeline.task_statements, baseline arm, is_valid_task
  baseline_agreed          baseline arm, review_status = accepted_cross_model_agreement (dual-model postings)
  claude                   Claude third-provider arm, if it was run
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.llm.base import load_config
from src.prototype.evidence import _norm, containment, jaccard, locate_evidence
from src.prototype.task_gold import GOLD_A, GOLD_B, gold_tasks, labelled_postings, read_gold
from src.prototype.task_statements import QUALIFICATION_LIKE
from src.utils.common import DB_PATH, OUTPUTS

OUT = OUTPUTS / "task_evaluation"
STAGE_RANK = {"normalized_exact": 3, "lexical": 2, "semantic": 1}

SYSTEMS = {
    "gemini": ("""SELECT posting_id, statement_text AS task, evidence_text AS evidence_span, evidence_section AS section,
                         evidence_match_type IN ('exact','normalized') AS evidence_valid, support_overlap
                  FROM prototype.prototype_model_outputs WHERE provider='gemini' AND kind='task' AND gate_status='accepted'""",
               "SELECT posting_id FROM prototype.prototype_extraction_calls WHERE provider='gemini' AND status='ok'"),
    "groq": ("""SELECT posting_id, statement_text AS task, evidence_text AS evidence_span, evidence_section AS section,
                       evidence_match_type IN ('exact','normalized') AS evidence_valid, support_overlap
                FROM prototype.prototype_model_outputs WHERE provider='groq' AND kind='task' AND gate_status='accepted'""",
             "SELECT posting_id FROM prototype.prototype_extraction_calls WHERE provider='groq' AND status='ok'"),
    "baseline_valid": ("""SELECT posting_id, atomic_task AS task, evidence_span, source_section AS section, evidence_valid,
                                 support_overlap FROM task_pipeline.task_statements
                          WHERE experiment_arm='baseline_gemini_groq' AND is_valid_task""",
                       "SELECT posting_id FROM prototype.prototype_extraction_calls WHERE provider='gemini' AND status='ok'"),
    "baseline_agreed": ("""SELECT posting_id, atomic_task AS task, evidence_span, source_section AS section, evidence_valid,
                                  support_overlap FROM task_pipeline.task_statements
                           WHERE experiment_arm='baseline_gemini_groq' AND review_status='accepted_cross_model_agreement'""",
                        """SELECT g.posting_id FROM prototype.prototype_extraction_calls g
                           JOIN prototype.prototype_extraction_calls q USING (posting_id)
                           WHERE g.provider='gemini' AND q.provider='groq' AND g.status='ok' AND q.status='ok'"""),
    "claude": ("""SELECT posting_id, statement_text AS task, evidence_text AS evidence_span, evidence_section AS section,
                         evidence_match_type IN ('exact','normalized') AS evidence_valid, support_overlap
                  FROM prototype.prototype_model_outputs_anthropic WHERE kind='task' AND gate_status='accepted'""",
               "SELECT posting_id FROM prototype.prototype_extraction_calls_anthropic WHERE status='ok'"),
}


# ---------------------------------------------------------------- matching

class Sim:
    """Cached cosine similarity with the configured local embedding model (lazy: only if needed)."""

    def __init__(self):
        self._emb = None
        self.vec = {}

    def prime(self, texts):
        texts = sorted({t for t in texts if t and t not in self.vec})
        if not texts:
            return
        if self._emb is None:
            from src.llm.embeddings import Embedder
            self._emb = Embedder(run_id="task_eval")
        self.vec.update(zip(texts, self._emb.embed(texts)))

    def __call__(self, a, b) -> float:
        return float(np.dot(self.vec[a], self.vec[b]))

    def close(self):
        if self._emb is not None:
            self._emb.close()


def match(gold: list[str], pred: list[str], sim: Sim, th: dict) -> dict:
    """One-to-one matching of gold vs predicted task strings for ONE posting."""
    cands, review = [], []
    for i, g in enumerate(gold):
        for j, p in enumerate(pred):
            if _norm(g) == _norm(p):
                cands.append((STAGE_RANK["normalized_exact"], 1.0, i, j, "normalized_exact"))
                continue
            jac = jaccard(g, p)
            if jac >= th["lexical"]:
                cands.append((STAGE_RANK["lexical"], jac, i, j, "lexical"))
                continue
            s = sim(g, p)
            if s >= th["semantic"]:
                cands.append((STAGE_RANK["semantic"], s, i, j, "semantic"))
            elif s >= th["review"]:
                review.append((s, i, j))
    cands.sort(key=lambda c: (-c[0], -c[1], c[2], c[3]))
    ug, up, pairs = set(), set(), []
    for _, sc, i, j, how in cands:
        if i in ug or j in up:
            continue
        ug.add(i)
        up.add(j)
        pairs.append((i, j, how, round(sc, 4)))
    amb, ag, ap = [], set(), set()
    for s, i, j in sorted(review, key=lambda r: (-r[0], r[1], r[2])):
        if i in ug or j in up or i in ag or j in ap:
            continue
        ag.add(i)
        ap.add(j)
        amb.append((i, j, round(s, 4)))
    return {"pairs": pairs, "ambiguous": amb, "fn": [i for i in range(len(gold)) if i not in ug],
            "fp": [j for j in range(len(pred)) if j not in up]}


def prf(tp, n_pred, n_gold) -> dict:
    p = tp / n_pred if n_pred else None
    r = tp / n_gold if n_gold else None
    f1 = (2 * p * r / (p + r) if (p + r) else 0.0) if p is not None and r is not None else None
    return {"tp": tp, "fp": n_pred - tp, "fn": n_gold - tp, "predicted": n_pred, "gold": n_gold,
            "precision": p, "recall": r, "f1": f1}


def fp_category(p: dict, pred_texts: list[str], gold_texts: list[str]) -> str:
    """Triage heuristic for a false positive (order matters)."""
    if not p.get("evidence_valid", True) or (p.get("support_overlap") is not None and p["support_overlap"] < 0.6):
        return "hallucinated_or_implied (weak/invalid evidence)"
    if QUALIFICATION_LIKE.match(p["task"]) or (p.get("section") in ("requirements", "skills", "preferred_qualifications")
                                               and len(p["task"].split()) <= 3):
        return "skill_or_qualification_as_task"
    for g in gold_texts:
        if containment(p["task"], g) >= 0.7 and sum(containment(x, g) >= 0.7 for x in pred_texts) >= 2:
            return "compound_over_split (gold kept it as one task)"
    return "extra_task_not_in_gold (possible gold omission - check)"


def fn_category(g: str, pred_texts: list[str], gold_texts: list[str], ambiguous: bool) -> str:
    if ambiguous:
        return "ambiguous_match_pending_review"
    for p in pred_texts:
        if containment(g, p) >= 0.7 and sum(containment(x, p) >= 0.7 for x in gold_texts) >= 2:
            return "compound_under_split (model merged several gold tasks)"
    return "missed_task"


# ---------------------------------------------------------------- commands

def _thresholds(a) -> dict:
    ev = load_config().get("evaluation", {})
    return {"lexical": a.lexical_threshold or ev.get("lexical_jaccard_min", 0.6),
            "semantic": a.semantic_threshold or ev.get("semantic_match_min", 0.85),
            "review": a.review_threshold or ev.get("semantic_review_min", 0.75)}


def validate(path: Path) -> pd.DataFrame:
    d = read_gold(path)
    con = duckdb.connect(str(DB_PATH), read_only=True)
    txt = dict(con.execute("SELECT posting_id, posting_text FROM job_postings WHERE posting_id IN (SELECT unnest(?))",
                           [d.posting_id.unique().tolist()]).fetchall())
    con.close()
    rows = []
    bad_status = sorted(set(d.adjudication_status) - {"not_started", "in_progress", "labelled", "no_tasks", "adjudicated"})
    for r in gold_tasks(d).itertuples():
        src = txt.get(r.posting_id)
        if src is None:
            rows.append({"posting_id": r.posting_id, "gold_task_statement": r.gold_task_statement, "problem": "unknown posting_id"})
            continue
        if not r.evidence_span:
            rows.append({"posting_id": r.posting_id, "gold_task_statement": r.gold_task_statement, "problem": "missing evidence_span"})
            continue
        m = locate_evidence(src, r.evidence_span)["match_type"]
        if m not in ("exact", "normalized"):
            rows.append({"posting_id": r.posting_id, "gold_task_statement": r.gold_task_statement,
                         "problem": f"evidence_span not verbatim in posting_text (match={m})"})
    rep = pd.DataFrame(rows, columns=["posting_id", "gold_task_statement", "problem"])
    print(f"{path.name}: {len(labelled_postings(d))} labelled postings, {len(gold_tasks(d))} gold tasks, "
          f"{len(rep)} problems" + (f", unknown statuses {bad_status}" if bad_status else ""))
    return rep


def evaluate(a) -> dict:
    th = _thresholds(a)
    gpath = Path(a.gold)
    gd = read_gold(gpath)
    lab = labelled_postings(gd)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not lab:
        res = {"status": "no_human_labels_yet", "gold_file": str(gpath), "labelled_postings": 0}
        (out / "metrics.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(json.dumps(res, indent=2))
        return res
    problems = validate(gpath)
    problems.to_csv(out / "gold_validation_problems.csv", index=False)
    gt = gold_tasks(gd)
    con = duckdb.connect(str(DB_PATH), read_only=True)
    preds, cover = {}, {}
    for name, (q, cq) in SYSTEMS.items():
        try:
            preds[name] = con.execute(q).df()
            cover[name] = set(con.execute(cq).df().posting_id)
        except duckdb.CatalogException:
            continue  # system not run (e.g. Claude arm)
    con.close()
    sim = Sim()
    sim.prime(list(gt.gold_task_statement) + [t for p in preds.values() for t in p[p.posting_id.isin(lab)].task])
    summary = {"status": "ok", "gold_file": str(gpath), "labelled_postings": len(lab), "gold_tasks": len(gt),
               "gold_span_problems": len(problems), "thresholds": th, "matching": "one-to-one greedy by (stage, score)",
               "systems": {}}
    for name, pdf in preds.items():
        posts = sorted(lab & cover[name])
        tp_rows, fp_rows, fn_rows, amb_rows, per = [], [], [], [], []
        for pid in posts:
            g = gt[gt.posting_id == pid].gold_task_statement.tolist()
            pp = pdf[pdf.posting_id == pid].drop_duplicates("task").to_dict("records")
            pt = [x["task"] for x in pp]
            m = match(g, pt, sim, th)
            amb_g = {i for i, _, _ in m["ambiguous"]}
            for i, j, how, sc in m["pairs"]:
                tp_rows.append({"posting_id": pid, "gold_task": g[i], "predicted_task": pt[j], "match_method": how, "score": sc})
            for j in m["fp"]:
                fp_rows.append({"posting_id": pid, "predicted_task": pt[j], "evidence_span": pp[j]["evidence_span"],
                                "section": pp[j]["section"], "error_category": fp_category(pp[j], pt, g)})
            for i in m["fn"]:
                fn_rows.append({"posting_id": pid, "gold_task": g[i],
                                "error_category": fn_category(g[i], pt, g, i in amb_g)})
            for i, j, s in m["ambiguous"]:
                amb_rows.append({"posting_id": pid, "gold_task": g[i], "predicted_task": pt[j], "cosine": s,
                                 "human_decision": ""})
            per.append({"posting_id": pid, **prf(len(m["pairs"]), len(pt), len(g)), "ambiguous": len(m["ambiguous"])})
        per_df = pd.DataFrame(per)
        tp, n_pred, n_gold = int(per_df.tp.sum()), int(per_df.predicted.sum()), int(per_df.gold.sum())
        n_amb = int(per_df.ambiguous.sum())
        macro = {k: (float(per_df[k].dropna().mean()) if per_df[k].notna().any() else None)
                 for k in ("precision", "recall", "f1")}
        d = out / name
        d.mkdir(exist_ok=True)
        per_df.to_csv(d / "per_posting.csv", index=False)
        pd.DataFrame(tp_rows).to_csv(d / "true_positives.csv", index=False)
        fpd, fnd = pd.DataFrame(fp_rows), pd.DataFrame(fn_rows)
        fpd.to_csv(d / "false_positives.csv", index=False)
        fnd.to_csv(d / "false_negatives.csv", index=False)
        pd.DataFrame(amb_rows, columns=["posting_id", "gold_task", "predicted_task", "cosine", "human_decision"]).to_csv(
            d / "ambiguous_review.csv", index=False)
        errs = pd.concat([fpd.assign(side="false_positive") if len(fpd) else fpd,
                          fnd.assign(side="false_negative") if len(fnd) else fnd])
        if len(errs):
            errs.groupby(["side", "error_category"]).size().rename("n").to_csv(d / "error_summary.csv")
        summary["systems"][name] = {
            "postings_evaluated": len(posts), "labelled_postings_without_system_output": len(lab) - len(posts),
            "micro": prf(tp, n_pred, n_gold), "macro_over_postings": macro,
            "if_ambiguous_accepted": prf(tp + n_amb, n_pred, n_gold),
            "match_methods": pd.DataFrame(tp_rows).match_method.value_counts().to_dict() if tp_rows else {},
            "error_categories": errs.groupby("error_category").size().to_dict() if len(errs) else {}}
    sim.close()
    (out / "metrics.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    rows = [{"system": k, **{f"micro_{m}": v["micro"][m] for m in ("precision", "recall", "f1", "tp", "fp", "fn")},
             "postings": v["postings_evaluated"]} for k, v in summary["systems"].items()]
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    return summary


def agreement(a) -> dict:
    """Annotator A vs Annotator B on the audit postings both have labelled (A treated as reference)."""
    th = _thresholds(a)
    A, B = read_gold(Path(a.gold)), read_gold(Path(a.gold_b))
    both = sorted(labelled_postings(A) & labelled_postings(B) & set(B.posting_id))
    out = Path(a.out) / "inter_annotator_audit20"
    out.mkdir(parents=True, exist_ok=True)
    if not both:
        res = {"status": "no_postings_labelled_by_both_annotators_yet"}
        print(json.dumps(res, indent=2))
        return res
    ga, gb = gold_tasks(A), gold_tasks(B)
    sim = Sim()
    sim.prime(list(ga[ga.posting_id.isin(both)].gold_task_statement) + list(gb.gold_task_statement))
    per, dis = [], []
    for pid in both:
        x = ga[ga.posting_id == pid].gold_task_statement.tolist()
        y = gb[gb.posting_id == pid].gold_task_statement.tolist()
        m = match(x, y, sim, th)
        per.append({"posting_id": pid, **prf(len(m["pairs"]), len(y), len(x)), "ambiguous": len(m["ambiguous"])})
        dis += [{"posting_id": pid, "side": "only_annotator_a", "task": x[i]} for i in m["fn"]]
        dis += [{"posting_id": pid, "side": "only_annotator_b", "task": y[j]} for j in m["fp"]]
        dis += [{"posting_id": pid, "side": "ambiguous_pair", "task": f"A: {x[i]} || B: {y[j]} (cos {s})"}
                for i, j, s in m["ambiguous"]]
    sim.close()
    per = pd.DataFrame(per)
    per.to_csv(out / "per_posting.csv", index=False)
    pd.DataFrame(dis).to_csv(out / "disagreements.csv", index=False)
    res = {"status": "ok", "postings_compared": len(both), "thresholds": th,
           "b_vs_a": prf(int(per.tp.sum()), int(per.predicted.sum()), int(per.gold.sum())),
           "note": "F1 here is symmetric pairwise agreement; precision/recall treat A as reference."}
    (out / "agreement.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["validate", "evaluate", "agreement"])
    ap.add_argument("--gold", default=str(GOLD_A))
    ap.add_argument("--gold-b", default=str(GOLD_B))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--semantic-threshold", type=float)
    ap.add_argument("--review-threshold", type=float)
    ap.add_argument("--lexical-threshold", type=float)
    a = ap.parse_args()
    if a.command == "validate":
        for p in (a.gold, a.gold_b):
            rep = validate(Path(p))
            if len(rep):
                print(rep.to_string(index=False))
    elif a.command == "evaluate":
        evaluate(a)
    else:
        agreement(a)


if __name__ == "__main__":
    main()
