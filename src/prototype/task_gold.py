"""Blind human TASK gold annotation files. SECTION A - TASK EXTRACTION.

    python -m src.prototype.task_gold build

Writes (never overwrites a file that already contains human labels):
  data/annotations/task_gold_100.csv                  Annotator A - all 100 gold postings
  data/annotations/task_gold_audit_20_annotator_b.csv Annotator B - blind audit subset (20 of the same 100)
  data/annotations/task_gold_100_assignment.csv       who labels what

Selection (deterministic, no model output involved):
  gold 100  = the existing `in_gold_100` flag of the frozen sample (src/prototype/sample.py: 50 postings per
              experiment phase in sha256(seed|gold|posting_id) order -> 50 NYC + 50 Arbeitnow).
  audit 20  = 10 per source from those 100, ordered by sha256(seed|audit|posting_id), first taking one posting
              per not-yet-covered title group (analysis grouping), then filling in hash order.

Content shown to annotators: posting text and a rule-based (regex heading) responsibilities excerpt only.
NO Gemini / Groq / Claude output, NO O*NET tasks, NO machine suggestions. gold columns start EMPTY.

Long format: one row per gold task. Each posting starts with one empty row (task_index 1); add rows for
further tasks with the same posting_id (the text columns may be left blank on the added rows).
"""
from __future__ import annotations

import hashlib
import sys

import duckdb
import pandas as pd

from src.llm.base import load_config
from src.prototype.freeze import load_frozen
from src.prototype.sections import detect_sections
from src.utils.common import DATA, DB_PATH

ANN = DATA / "annotations"
GOLD_A = ANN / "task_gold_100.csv"
GOLD_B = ANN / "task_gold_audit_20_annotator_b.csv"
ASSIGN = ANN / "task_gold_100_assignment.csv"
LABEL_COLS = ["gold_task_statement", "evidence_span"]
COLS = ["posting_id", "source", "job_title", "task_index", "responsibilities_text", "full_posting_text",
        "gold_task_statement", "evidence_span", "annotator", "adjudication_status", "notes"]
STATUSES = ["not_started", "in_progress", "labelled", "no_tasks", "adjudicated"]
LABELLED = {"labelled", "no_tasks", "adjudicated"}


def _h(*parts) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def responsibilities_excerpt(description, requirements, preferred) -> str:
    """Rule-based responsibilities sections; if none detected, say so (annotators then use the full text)."""
    secs = detect_sections(description, requirements, preferred)["sections"]
    resp = [s["text"] for s in secs if s["section"] == "responsibilities"]
    if resp:
        return "\n\n".join(resp)
    return "(no responsibilities heading detected by rules - read full_posting_text)"


def audit_subset(gold: pd.DataFrame, seed: str, per_source: int = 10) -> list[str]:
    out = []
    for src, g in gold.groupby("source", sort=True):
        g = g.assign(h=[_h(seed, "audit", p) for p in g.posting_id]).sort_values("h")
        chosen, seen = [], set()
        for r in g.itertuples():
            if r.title_group_analysis_only not in seen:
                chosen.append(r.posting_id)
                seen.add(r.title_group_analysis_only)
            if len(chosen) == per_source:
                break
        for p in g.posting_id:
            if len(chosen) == per_source:
                break
            if p not in chosen:
                chosen.append(p)
        out += chosen
    return out


def has_labels(path) -> bool:
    if not path.exists():
        return False
    d = pd.read_csv(path, dtype=str, keep_default_na=False)
    return bool((d[LABEL_COLS] != "").any().any())


def build() -> dict:
    for p in (GOLD_A, GOLD_B):
        if has_labels(p):
            print(f"{p} already contains human labels - nothing overwritten.")
            return {"status": "kept_existing_labels"}
    seed = load_config()["experiment"]["sample_seed"]
    fz = load_frozen()
    gold = fz[fz.in_gold_100.astype(str).str.lower() == "true"].sort_values("sample_rank")
    assert len(gold) == 100 and gold.posting_id.is_unique, f"expected 100 unique gold postings, got {len(gold)}"
    con = duckdb.connect(str(DB_PATH), read_only=True)
    txt = con.execute("""SELECT posting_id, source, job_title, language, description, requirements, preferred_skills,
                                posting_text FROM job_postings WHERE posting_id IN (SELECT unnest(?))""",
                      [gold.posting_id.tolist()]).df()
    uniq = set(con.execute("SELECT posting_id FROM job_postings_unique WHERE posting_id IN (SELECT unnest(?))",
                           [gold.posting_id.tolist()]).df().posting_id)
    con.close()
    assert len(txt) == 100 and set(txt.language) == {"en"} and len(uniq) == 100, "gold postings must be unique English"
    txt = txt.set_index("posting_id")
    f = lambda x: x if isinstance(x, str) else None
    rows = []
    for r in gold.itertuples():
        t = txt.loc[r.posting_id]
        rows.append({"posting_id": r.posting_id, "source": t.source, "job_title": t.job_title, "task_index": 1,
                     "responsibilities_text": responsibilities_excerpt(f(t.description), f(t.requirements),
                                                                       f(t.preferred_skills)),
                     "full_posting_text": t.posting_text, "gold_task_statement": "", "evidence_span": "",
                     "annotator": "", "adjudication_status": "not_started", "notes": ""})
    a = pd.DataFrame(rows, columns=COLS)
    audit = audit_subset(gold, seed)
    assert len(audit) == 20 and set(audit) <= set(a.posting_id)
    b = a[a.posting_id.isin(audit)].copy()
    ANN.mkdir(parents=True, exist_ok=True)
    a.to_csv(GOLD_A, index=False, encoding="utf-8-sig")  # BOM so Excel opens UTF-8 correctly
    b.to_csv(GOLD_B, index=False, encoding="utf-8-sig")
    asg = gold[["posting_id", "source", "job_title", "title_group_analysis_only", "sample_rank", "phase"]].copy()
    asg["annotator_a"] = "required"
    asg["in_audit_20"] = asg.posting_id.isin(audit)
    asg["annotator_b"] = asg.in_audit_20.map({True: "required (blind to A)", False: ""})
    asg["label_file_a"] = GOLD_A.relative_to(DATA.parent).as_posix()
    asg["label_file_b"] = asg.in_audit_20.map({True: GOLD_B.relative_to(DATA.parent).as_posix(), False: ""})
    asg["selection_rule"] = asg.in_audit_20.map(
        {True: "audit: 10/source, sha256(seed|audit|posting_id), title-group spread first",
         False: "gold: in_gold_100 flag of frozen sample"})
    asg.to_csv(ASSIGN, index=False, encoding="utf-8-sig")
    return {"status": "written", "gold_postings": len(a), "audit_postings": len(b),
            "gold_by_source": a.source.value_counts().to_dict(), "audit_by_source": b.source.value_counts().to_dict()}


def read_gold(path) -> pd.DataFrame:
    """Read a (partly) labelled long-format file; text columns of added rows may be blank."""
    d = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    d["posting_id"] = d.posting_id.str.strip()
    d["adjudication_status"] = d.adjudication_status.str.strip().str.lower()
    return d


def labelled_postings(d: pd.DataFrame) -> set[str]:
    return set(d[d.adjudication_status.isin(LABELLED)].posting_id)


def gold_tasks(d: pd.DataFrame) -> pd.DataFrame:
    g = d[(d.gold_task_statement.str.strip() != "") & d.posting_id.isin(labelled_postings(d))].copy()
    g["gold_task_statement"] = g.gold_task_statement.str.strip()
    g["evidence_span"] = g.evidence_span.str.strip()
    return g[["posting_id", "gold_task_statement", "evidence_span", "annotator", "notes"]].reset_index(drop=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd != "build":
        raise SystemExit("usage: python -m src.prototype.task_gold build")
    print(build())
