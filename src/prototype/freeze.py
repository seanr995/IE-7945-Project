"""Freeze + profile the 200-posting task-extraction experiment corpus. SECTION A - TASK EXTRACTION.

    python -m src.prototype.freeze            # freeze (first run) or verify (later runs) + profile

The sample itself is produced by the EXISTING deterministic sampler `src.prototype.sample.build_sample`
(seed `experiment.sample_seed` in config/llm_config.json). This module does not re-sample: it

1. writes the exact posting IDs (plus a SHA-256 of each posting_text) to
   data/experiment/task_extraction_sample_200.csv - the frozen experiment definition;
2. on later runs verifies the database still yields the identical sample/texts and reports any drift
   (the frozen file is never silently overwritten);
3. mirrors it to task_pipeline.experiment_sample in taxonomy.duckdb;
4. profiles the sample (source, titles, a transparent keyword *title grouping*, seniority, length).

The title grouping is a SAMPLING/ANALYSIS grouping derived from job-title keywords only. It is NOT an
occupation taxonomy and NOT an O*NET/ESCO mapping.
"""
from __future__ import annotations

import hashlib
import re
import sys

import duckdb
import pandas as pd

from src.llm.base import load_config
from src.prototype.sample import build_sample
from src.utils.common import DATA, DB_PATH, OUTPUTS, get_logger

log = get_logger("prototype.freeze")

FROZEN = DATA / "experiment" / "task_extraction_sample_200.csv"
PROFILE_DIR = OUTPUTS / "task_extraction" / "sample_profile"
SCHEMA = "task_pipeline"

# Ordered keyword rules on the lower-cased job title; first match wins. Transparent and reproducible.
TITLE_GROUPS = [
    ("software / data / IT", r"software|developer|engineer.*(data|backend|front|full|platform|cloud|devops|ml|machine|"
                             r"site reliab|qa|test|security|it\b)|\bdata\b|devops|\bit\b|cyber|cloud|programm|"
                             r"machine learning|\bai\b|database|network|systems? (admin|analyst)|web|tech lead|sre"),
    ("product / design / UX", r"product|design|ux|ui\b|user research"),
    ("engineering / architecture (non-software)", r"engineer|architect|surveyor|construction|mechanic|electric|plumb|"
                                                  r"inspector|project manager.*(construction|capital)"),
    ("legal / compliance", r"attorney|counsel|legal|lawyer|paralegal|compliance|hearing officer|judge"),
    ("finance / accounting / audit", r"account|finance|financial|audit|budget|tax|payroll|treasur|controller|fiscal|"
                                     r"procurement|purchas|contract"),
    ("sales / marketing / customer", r"sales|account executive|marketing|growth|customer|client|business development|"
                                     r"partnership|community manager|content|social media|communications?|press|"
                                     r"public relations|outreach"),
    ("HR / people / admin", r"\bhr\b|human resources|recruit|talent|people|administrative|office|assistant|clerk|"
                            r"secretary|receptionist|executive assistant"),
    ("health / social services", r"nurse|health|medical|clinic|psycholog|social work|case ?work|counsel+or|therap|"
                                 r"physician|epidemiolog|benefits"),
    ("policy / research / planning", r"policy|research|planner|planning|analyst|program|evaluat|statistic|economist"),
    ("operations / management", r"operations|manager|director|chief|head of|lead|coordinator|supervisor|commissioner"),
    ("trades / facilities / public safety", r"maintenance|custodian|worker|driver|officer|police|fire|safety|"
                                            r"technician|operator|laborer|repair"),
    ("intern / student", r"intern|student|werkstudent|trainee|fellow|apprentice|praktik"),
]
_TG = [(g, re.compile(p)) for g, p in TITLE_GROUPS]


def title_group(title: str | None) -> str:
    t = (title or "").lower()
    for g, rx in _TG:
        if rx.search(t):
            return g
    return "unclassified"


def md_table(df: pd.DataFrame) -> str:
    """Minimal markdown table (avoids an optional `tabulate` dependency)."""
    d = df.reset_index()
    cell = lambda v: str(v).replace("|", "/").replace("\n", " ")
    lines = ["| " + " | ".join(cell(c) for c in d.columns) + " |", "|" + "---|" * len(d.columns)]
    lines += ["| " + " | ".join(cell(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join(lines)


def _sha(s) -> str:
    return hashlib.sha256((s if isinstance(s, str) else "").encode("utf-8")).hexdigest()


def current_sample(con) -> pd.DataFrame:
    ex = load_config()["experiment"]
    s = build_sample(con, ex["sample_seed"], ex["sample_size"], ex["dual_model_postings"], ex["min_description_words"])
    txt = con.execute("SELECT posting_id, posting_text FROM job_postings WHERE posting_id IN (SELECT unnest(?))",
                      [s.posting_id.tolist()]).df()
    s = s.merge(txt, on="posting_id", how="left")
    s["posting_text_sha256"] = s.posting_text.map(_sha)
    s["title_group_analysis_only"] = s.job_title.map(title_group)
    return s.drop(columns=["posting_text"])


def load_frozen() -> pd.DataFrame:
    return pd.read_csv(FROZEN, dtype={"posting_id": str}, keep_default_na=False).sort_values("sample_rank")


def freeze_or_verify(con) -> tuple[pd.DataFrame, dict]:
    cur = current_sample(con)
    if not FROZEN.exists():
        FROZEN.parent.mkdir(parents=True, exist_ok=True)
        cur.to_csv(FROZEN, index=False, encoding="utf-8")
        log.info("Frozen %d posting IDs -> %s", len(cur), FROZEN)
        return cur, {"status": "frozen_now", "n": len(cur)}
    fz = load_frozen()
    live = con.execute("SELECT posting_id, posting_text FROM job_postings WHERE posting_id IN (SELECT unnest(?))",
                       [fz.posting_id.tolist()]).df()
    live_sha = dict(zip(live.posting_id, live.posting_text.map(_sha)))
    rep = {"status": "verified", "n": len(fz),
           "missing_in_db": int(sum(p not in live_sha for p in fz.posting_id)),
           "text_changed": int(sum(live_sha.get(p) not in (None, h) for p, h in zip(fz.posting_id, fz.posting_text_sha256))),
           "sampler_reproduces_frozen_ids": list(cur.posting_id) == list(fz.posting_id)}
    if rep["missing_in_db"] or rep["text_changed"]:
        rep["status"] = "DRIFT"
        log.error("Frozen sample drift: %s", rep)
    return fz, rep


def mirror(con, fz: pd.DataFrame) -> None:
    con.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
    con.register("fz", fz)
    con.execute(f"CREATE OR REPLACE TABLE {SCHEMA}.experiment_sample AS SELECT * FROM fz")
    con.unregister("fz")


def profile(con, fz: pd.DataFrame) -> dict:
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    extra = con.execute("SELECT posting_id, job_category_source, posting_text_word_count FROM job_postings "
                        "WHERE posting_id IN (SELECT unnest(?))", [fz.posting_id.tolist()]).df()
    d = fz.merge(extra, on="posting_id", how="left")
    gold = d[d.in_gold_100.astype(str).str.lower() == "true"]
    out = {}

    def tab(name, col, frame=d):
        t = frame.groupby(["source", col]).size().unstack(0, fill_value=0)
        t["total"] = t.sum(axis=1)
        t = t.sort_values("total", ascending=False)
        t.to_csv(PROFILE_DIR / f"{name}.csv")
        out[name] = t
        return t

    tab("source_by_phase", "phase")
    tab("seniority", "seniority")
    tab("title_group_analysis_only", "title_group_analysis_only")
    tab("gold100_title_group_analysis_only", "title_group_analysis_only", gold)
    d["nyc_job_category_first"] = d.job_category_source.fillna("").str.split(",").str[0].replace("", "(none)")
    tab("source_job_category_first", "nyc_job_category_first")
    ln = d.groupby("source").description_word_count.describe().round(1)
    ln.loc["all"] = d.description_word_count.describe().round(1)
    ln.to_csv(PROFILE_DIR / "description_word_count.csv")
    out["length"] = ln
    d[["sample_rank", "posting_id", "source", "job_title", "title_group_analysis_only", "seniority",
       "description_word_count", "phase", "in_gold_100"]].to_csv(PROFILE_DIR / "titles.csv", index=False)
    stats = {"postings": len(d), "unique_titles": int(d.job_title.nunique()),
             "unique_normalized_titles": int(d.normalized_job_title.nunique()),
             "title_groups_non_empty": int(d.title_group_analysis_only.nunique()),
             "largest_title_group_share": round(d.title_group_analysis_only.value_counts(normalize=True).iloc[0], 3),
             "unclassified_share": round((d.title_group_analysis_only == "unclassified").mean(), 3)}
    md = ["# Task-extraction experiment sample - profile", "",
          f"Frozen file: `{FROZEN.relative_to(DATA.parent).as_posix()}` (sampler: `src/prototype/sample.py`, seed "
          f"`{d.sampling_seed.iat[0]}`).", "",
          "Title groups are a keyword grouping of job titles for sampling/analysis only - NOT an occupation taxonomy.", "",
          "## Summary", "", *[f"- {k}: {v}" for k, v in stats.items()], ""]
    for name, t in out.items():
        md += [f"## {name}", "", md_table(t), ""]
    (PROFILE_DIR / "sample_profile.md").write_text("\n".join(md), encoding="utf-8")
    return stats


def main() -> dict:
    con = duckdb.connect(str(DB_PATH))
    try:
        fz, rep = freeze_or_verify(con)
        mirror(con, fz)
        stats = profile(con, fz)
    finally:
        con.close()
    return {**rep, **stats}


if __name__ == "__main__":
    r = main()
    print(r)
    sys.exit(1 if r["status"] == "DRIFT" else 0)
