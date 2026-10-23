"""Evaluate the human TASK/SKILL/OTHER purity audit and build the Sprint 3 TASK-only input (no LLM/API calls).

    .venv\\Scripts\\python.exe outputs/task_evaluation/build_sprint3_task_input.py

Inputs : outputs/task_evaluation/task_purity_audit_100.csv (human_label filled), task_pipeline.task_statements (read only)
Outputs: outputs/task_evaluation/task_purity_results.csv
         task_pipeline.sprint3_task_input  (TABLE, CREATE OR REPLACE; a table rather than a view because
                                            build_db.preserve_prototype_schema only carries tables across rebuilds)
Filter : evidence-valid task statements, excluding source sections whose audited TASK rate is < 50%
         (requirements, preferred_qualifications) plus 'skills' (2 statements, unaudited, 0/2 gold matches).
"""
import sys
from math import sqrt
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "outputs/task_evaluation/task_purity_audit_100.csv"
OUT = ROOT / "outputs/task_evaluation/task_purity_results.csv"
DB = ROOT / "database/taxonomy.duckdb"
LABELS = {"TASK", "SKILL", "OTHER"}
UNSAFE = ("requirements", "preferred_qualifications", "skills")
FILTER_VERSION = "sprint3_task_input_v1_2026-10-22"


def wilson(k, n, z=1.96):
    if not n:
        return (None, None)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    w = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return round(c - w, 4), round(c + w, 4)


a = pd.read_csv(AUDIT, dtype=str, keep_default_na=False, encoding="utf-8-sig")
a["human_label"] = a.human_label.str.strip().str.upper()
bad = a[~a.human_label.isin(LABELS)]
assert len(a) == 100 and a.statement_id.is_unique, f"expected 100 unique rows, got {len(a)}"
if len(bad):
    sys.exit(f"{len(bad)} blank/invalid labels:\n{bad[['statement_id', 'human_label']].to_string(index=False)}")

con = duckdb.connect(str(DB))
ts = con.execute("""SELECT task_statement_id AS statement_id, posting_id, provider, atomic_task, source_section,
                           evidence_valid, is_valid_task FROM task_pipeline.task_statements""").df()
m = a.merge(ts, on="statement_id", how="left", suffixes=("", "_db"))
assert m.source_section.notna().all(), "audit statement_id not found in task_statements"
assert (m.statement_text == m.atomic_task).all() and (m.provider == m.provider_db).all(), "audit rows drifted from DB"
pop = ts[ts.evidence_valid.fillna(False).astype(bool) & ts.is_valid_task.fillna(False).astype(bool)]
pop_n = pop.groupby(["provider", "source_section"]).size()


def row(group, value, d, population=None):
    k, n = int((d.human_label == "TASK").sum()), len(d)
    lo, hi = wilson(k, n)
    return {"group": group, "value": value, "n_audited": n, "TASK": k, "SKILL": int((d.human_label == "SKILL").sum()),
            "OTHER": int((d.human_label == "OTHER").sum()), "task_purity": round(k / n, 4) if n else None,
            "ci95_low": lo, "ci95_high": hi, "population_evidence_valid": population}


def weighted(d, p):
    """Purity re-weighted to the population by provider x section strata (unaudited strata dropped)."""
    s = d.groupby(["provider", "source_section"]).human_label.apply(lambda x: (x == "TASK").mean())
    w = p.reindex(s.index)
    return round(float((s * w).sum() / w.sum()), 4), int(p.sum() - w.sum())


res = [row("overall", "all", m, len(pop))]
res += [row("provider", k, g, int((pop.provider == k).sum())) for k, g in m.groupby("provider")]
res += [row("source_section", k, g, int((pop.source_section == k).sum())) for k, g in m.groupby("source_section")]
res += [row("provider_x_section", f"{p}|{s}", g, int(pop_n.get((p, s), 0))) for (p, s), g in m.groupby(["provider", "source_section"])]
for s in sorted(set(pop.source_section) - set(m.source_section)):
    res.append({"group": "source_section", "value": s, "n_audited": 0, "population_evidence_valid": int((pop.source_section == s).sum())})
kept_m, kept_p = m[~m.source_section.isin(UNSAFE)], pop_n[~pop_n.index.get_level_values(1).isin(UNSAFE)]
res.append(row("filter", "unsafe_sections_removed", m[m.source_section.isin(UNSAFE)], int(pop.source_section.isin(UNSAFE).sum())))
res.append(row("filter", "kept_sections (sprint3 input)", kept_m, int(kept_p.sum())))
w_all, w_all_gap = weighted(m, pop_n)
w_kept, w_kept_gap = weighted(kept_m, kept_p)
res.append({"group": "weighted_estimate", "value": "all evidence-valid (provider x section)", "task_purity": w_all,
            "population_evidence_valid": len(pop), "note": f"{w_all_gap} statements in unaudited strata excluded from weights"})
res.append({"group": "weighted_estimate", "value": "sprint3 input (provider x section)", "task_purity": w_kept,
            "population_evidence_valid": int(kept_p.sum()), "note": f"{w_kept_gap} statements in unaudited strata excluded from weights"})
R = pd.DataFrame(res)
R.to_csv(OUT, index=False)

con.execute(f"""CREATE OR REPLACE TABLE task_pipeline.sprint3_task_input AS
    SELECT task_statement_id AS statement_id, posting_id, provider, atomic_task AS statement_text, evidence_span,
           source_section, evidence_valid, confirming_provider, phase, '{FILTER_VERSION}' AS filter_version
    FROM task_pipeline.task_statements
    WHERE evidence_valid AND is_valid_task AND source_section NOT IN {UNSAFE}
    ORDER BY posting_id, task_statement_id""")
n_final = con.execute("SELECT count(*) FROM task_pipeline.sprint3_task_input").fetchone()[0]
by = con.execute("SELECT provider, source_section, count(*) n FROM task_pipeline.sprint3_task_input GROUP BY ALL ORDER BY 1, 2").df()
con.close()

pd.set_option("display.width", 200)
print(R.to_string(index=False))
print(f"\noriginal evidence-valid = {len(pop)}  removed = {len(pop) - n_final}  sprint3_task_input = {n_final}")
print(by.to_string(index=False))
