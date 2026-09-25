"""Gold annotation set (EMPTY labels) + evaluation. PROTOTYPE.

    python -m src.prototype.gold build      # writes data/annotation/sprint2_gold_100.csv (never overwrites labels)
    python -m src.prototype.gold evaluate   # precision / recall / F1 once humans fill gold_tasks / gold_skills

Gold label format: one statement per line inside the cell (or separated by ' | ').
Matching rule (same as the model-agreement matcher): same kind and stemmed content-word
Jaccard >= 0.5, greedy one-to-one.
"""
from __future__ import annotations

import sys

import duckdb
import pandas as pd

from src.llm.base import load_config
from src.prototype.evidence import match_items
from src.prototype.sections import detect_sections, model_input
from src.utils.common import DB_PATH, ROOT

GOLD = ROOT / "data" / "annotation" / "sprint2_gold_100.csv"
COLS = ["posting_id", "source", "job_title", "relevant_text", "gold_tasks", "gold_skills", "reviewer", "review_status", "notes"]


def build() -> pd.DataFrame:
    if GOLD.exists():
        old = pd.read_csv(GOLD, dtype=str, keep_default_na=False)
        if (old[["gold_tasks", "gold_skills"]] != "").any().any():
            print(f"{GOLD} already contains human labels - not overwritten.")
            return old
    cfg = load_config()
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute("""SELECT s.posting_id, s.source, p.job_title, p.description, p.requirements, p.preferred_skills
                        FROM prototype.prototype_extraction_sample s JOIN job_postings p USING (posting_id)
                        WHERE s.in_gold_100 ORDER BY s.sample_rank""").df()
    con.close()
    rows = []
    for r in df.itertuples():
        f = lambda x: x if isinstance(x, str) else None
        secs = model_input(detect_sections(f(r.description), f(r.requirements), f(r.preferred_skills))["sections"],
                           cfg["experiment"]["max_input_chars"])
        rows.append({"posting_id": r.posting_id, "source": r.source, "job_title": r.job_title,
                     "relevant_text": "\n\n".join(f"[{s['section']}]\n{s['text']}" for s in secs),
                     "gold_tasks": "", "gold_skills": "", "reviewer": "", "review_status": "not_started", "notes": ""})
    out = pd.DataFrame(rows, columns=COLS)
    assert len(out) == 100, f"expected 100 gold postings, got {len(out)}"
    GOLD.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(GOLD, index=False, encoding="utf-8")
    return out


def _labels(cell: str) -> list[str]:
    if not isinstance(cell, str) or not cell.strip():
        return []
    parts = cell.replace(" | ", "\n").split("\n")
    return [p.strip(" -•\t") for p in parts if p.strip(" -•\t")]


def prf(tp: int, n_pred: int, n_gold: int) -> dict:
    p = tp / n_pred if n_pred else None
    r = tp / n_gold if n_gold else None
    f1 = 2 * p * r / (p + r) if p and r else (0.0 if p is not None and r is not None else None)
    return {"tp": tp, "predicted": n_pred, "gold": n_gold, "precision": p, "recall": r, "f1": f1}


def evaluate(gold: pd.DataFrame, predictions: pd.DataFrame) -> dict:
    """gold: COLS; predictions: posting_id, kind, statement_text, system. Only labelled postings count."""
    lab = gold[gold.review_status.isin(["done", "reviewed", "complete"]) |
               (gold.gold_tasks.fillna("") != "") | (gold.gold_skills.fillna("") != "")]
    if lab.empty:
        return {"status": "no_human_labels_yet", "labelled_postings": 0}
    res = {"status": "ok", "labelled_postings": len(lab), "systems": {}}
    for system, pred in predictions.groupby("system"):
        acc = {"task": [0, 0, 0], "skill": [0, 0, 0]}
        for r in lab.itertuples():
            g = [{"kind": "task", "statement_text": t} for t in _labels(r.gold_tasks)] + \
                [{"kind": "skill", "statement_text": t} for t in _labels(r.gold_skills)]
            p = [{"kind": x.kind, "statement_text": x.statement_text} for x in pred[pred.posting_id == r.posting_id].itertuples()]
            m = match_items(p, g)
            for kind in ("task", "skill"):
                acc[kind][0] += sum(p[i]["kind"] == kind for i, _, _, _ in m["pairs"])
                acc[kind][1] += sum(x["kind"] == kind for x in p)
                acc[kind][2] += sum(x["kind"] == kind for x in g)
        res["systems"][system] = {k: prf(*v) for k, v in acc.items()}
    return res


def predictions_from_db() -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    mo = con.execute("""SELECT posting_id, kind, statement_text, provider AS system FROM prototype.prototype_model_outputs
                        WHERE gate_status = 'accepted'""").df()
    cs = con.execute("""SELECT posting_id, kind, statement_text, 'consensus_high_medium' AS system
                        FROM prototype.prototype_statements WHERE confidence_tier IN ('high','medium')
                        AND coalesce(adjudication_decision,'keep') <> 'reject'""").df()
    con.close()
    return pd.concat([mo, cs])


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        print(f"gold file: {GOLD} ({len(build())} postings, gold labels empty)")
    else:
        g = pd.read_csv(GOLD, dtype=str, keep_default_na=False)
        import json
        print(json.dumps(evaluate(g, predictions_from_db()), indent=2))
