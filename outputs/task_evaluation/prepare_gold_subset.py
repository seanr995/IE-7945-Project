"""Build the human-labeled evaluation subset from the annotator's working file (no API calls).

Source: data/annotations/task_gold_100_3rows 1.csv (annotator file; adjudication_status was left 'not_started').
A row counts as COMPLETED when gold_task_statement is non-empty and its posting_id exists in job_postings.
Output: outputs/task_evaluation/gold_subset_labelled.csv (status set to 'labelled' so task_eval picks it up)
        outputs/task_evaluation/gold_row_validation.csv (per-row checks)
"""
import sys
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.prototype.evidence import _norm, locate_evidence  # noqa: E402
from src.prototype.task_gold import read_gold  # noqa: E402

SRC = ROOT / "data/annotations/task_gold_100_3rows 1.csv"
OUT = ROOT / "outputs/task_evaluation"

d = read_gold(SRC)
con = duckdb.connect(str(ROOT / "database/taxonomy.duckdb"), read_only=True)
txt = dict(con.execute("SELECT posting_id, posting_text FROM job_postings WHERE posting_id IN (SELECT unnest(?))",
                       [d.posting_id.unique().tolist()]).fetchall())
con.close()
rows = []
for r in d.itertuples():
    g, ev, src = r.gold_task_statement.strip(), r.evidence_span.strip(), txt.get(r.posting_id)
    span_m = locate_evidence(src, ev)["match_type"] if (src and ev) else "n/a"
    task_m = locate_evidence(src, g)["match_type"] if (src and g) else "n/a"
    rows.append({"posting_id": r.posting_id, "task_index": r.task_index, "gold_task_statement": g,
                 "completed": bool(g), "posting_id_exists": src is not None, "evidence_span_present": bool(ev),
                 "evidence_span_match": span_m, "evidence_verbatim": span_m in ("exact", "normalized"),
                 "evidence_span_chars": len(ev), "gold_task_in_source_match": task_m,
                 "gold_task_in_evidence_span": bool(g) and _norm(g.rstrip(" .;,")) in _norm(ev)})
v = pd.DataFrame(rows)
v.to_csv(OUT / "gold_row_validation.csv", index=False)
keep = v.completed & v.posting_id_exists
lab = d[keep.values].copy()
lab["adjudication_status"] = "labelled"
lab.to_csv(OUT / "gold_subset_labelled.csv", index=False, encoding="utf-8-sig")
print(f"rows={len(d)} completed={int(v.completed.sum())} kept={int(keep.sum())} postings={lab.posting_id.nunique()}")
print(f"unknown posting_id={int((~v.posting_id_exists).sum())} empty_span={int((~v.evidence_span_present).sum())}")
print("evidence_span match:", v.evidence_span_match.value_counts().to_dict())
print("gold task in source:", v.gold_task_in_source_match.value_counts().to_dict())
print("gold task inside its evidence span:", int(v.gold_task_in_evidence_span.sum()))
print("distinct spans per posting (median):", d.groupby("posting_id").evidence_span.nunique().median())
print("by source:", lab.drop_duplicates("posting_id").source.value_counts().to_dict())
