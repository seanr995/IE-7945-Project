"""Load O*NET and ESCO reference files into DuckDB.

Files are discovered by inspecting the extracted directories: a file is mapped to a
canonical table only if its stem matches AND its header contains the required
columns. Original values, codes, element IDs and scale IDs are preserved; column
names are converted to snake_case and a version column is added.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import duckdb

from src.utils.common import ESCO_EXT, ONET_EXT, get_logger, read_manifest, rel

log = get_logger("ingestion.reference")


def snake(c: str) -> str:
    c = c.replace("O*NET-SOC", "onetsoc").replace("*", "")
    c = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", c)  # camelCase (ESCO) -> snake
    return re.sub(r"[^0-9a-zA-Z]+", "_", c).strip("_").lower()


def header(p: Path) -> list[str]:
    with open(p, encoding="utf-8-sig", newline="") as f:
        return next(csv.reader(f))


# (table, candidate stems in priority order, required columns, extra label)
ONET_MAP = [
    ("onet_occupations", ["occupation_data"], {"O*NET-SOC Code", "Title", "Description"}),
    ("onet_tasks", ["task_statements"], {"O*NET-SOC Code", "Task ID", "Task"}),
    # O*NET <=30: skills.csv. O*NET 31.0 splits Skills into Essential (2.A) + Transferable (2.B)
    ("onet_skills", ["skills", "essential_skills", "transferable_skills"], {"O*NET-SOC Code", "Element ID", "Scale ID", "Data Value"}),
    ("onet_knowledge", ["knowledge"], {"O*NET-SOC Code", "Element ID", "Scale ID", "Data Value"}),
    ("onet_abilities", ["abilities"], {"O*NET-SOC Code", "Element ID", "Scale ID", "Data Value"}),
    ("onet_work_activities", ["work_activities"], {"O*NET-SOC Code", "Element ID", "Scale ID", "Data Value"}),
    # O*NET <=30: technology_skills.csv. O*NET 31.0: software_skills.csv (content model 2.E)
    ("onet_technology_skills", ["technology_skills", "software_skills"], {"O*NET-SOC Code", "Element ID"}),
    ("onet_content_model_reference", ["content_model_reference"], {"Element ID", "Element Name"}),
    ("onet_scales_reference", ["scales_reference"], {"Scale ID"}),
    ("onet_job_zones", ["job_zones"], {"O*NET-SOC Code", "Job Zone"}),
    ("onet_alternate_titles", ["job_titles", "alternate_titles"], {"O*NET-SOC Code"}),
]


def onet_version() -> str | None:
    vers = [r["version"] for r in read_manifest() if r["dataset_name"] == "onet_database" and r["status"] == "OK"]
    return max(vers, key=lambda v: tuple(int(x) for x in v.split("."))) if vers else None


def load_onet(con: duckdb.DuckDBPyConnection) -> list[dict]:
    version = onet_version()
    if not version:
        log.warning("No O*NET version in manifest; skipping O*NET")
        return []
    files = {p.stem.lower(): p for p in (ONET_EXT / version).rglob("*.csv")}
    report = []
    for table, stems, required in ONET_MAP:
        matched = [files[s] for s in stems if s in files and required <= set(header(files[s]))]
        if not matched:
            log.warning("O*NET %s: no file with required schema found (candidates %s)", table, stems)
            report.append({"table_name": table, "files": "", "row_count": 0, "status": "NOT FOUND"})
            continue
        selects = []
        for p in matched:
            cols = header(p)
            col_sql = ", ".join(f'"{c}" AS {snake(c)}' for c in cols)
            selects.append(f"SELECT {col_sql}, '{p.stem}' AS source_file, '{version}' AS onet_version "
                           f"FROM read_csv('{p.as_posix()}', header=true, auto_detect=true, "
                           f"sample_size=-1, types={{'O*NET-SOC Code':'VARCHAR'}})"
                           if "O*NET-SOC Code" in cols else
                           f"SELECT {col_sql}, '{p.stem}' AS source_file, '{version}' AS onet_version "
                           f"FROM read_csv('{p.as_posix()}', header=true, auto_detect=true, sample_size=-1)")
        sql = " UNION ALL BY NAME ".join(selects)
        con.execute(f"CREATE OR REPLACE TABLE {table} AS {sql}")
        n = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        report.append({"table_name": table, "files": "; ".join(rel(p) for p in matched), "row_count": n, "status": "LOADED"})
        log.info("O*NET %-30s %8d rows from %s", table, n, [p.name for p in matched])
    return report


# ------------------------------------------------------------------ ESCO

ESCO_OFFICIAL = [  # table, filename prefixes (case-insensitive), required columns
    ("esco_occupations", ["occupations_"], {"conceptUri", "preferredLabel"}),
    ("esco_skills", ["skills_"], {"conceptUri", "preferredLabel"}),
    ("esco_occupation_skill_relations", ["occupationskillrelations_"], {"occupationUri", "skillUri"}),
    ("esco_occupation_hierarchy", ["broaderrelationsoccpillar_"], {"conceptUri", "broaderUri"}),
    ("esco_skill_hierarchy", ["broaderrelationsskillpillar_"], {"conceptUri", "broaderUri"}),
    ("esco_isco_groups", ["iscogroups_"], {"conceptUri", "code"}),
    ("esco_skill_groups", ["skillgroups_"], {"conceptUri", "preferredLabel"}),
]
ESCO_API = {
    "esco_occupations": "occupations.csv", "esco_skills": "skills.csv",
    "esco_occupation_skill_relations": "occupation_skill_relations.csv",
    "esco_occupation_hierarchy": "occupation_hierarchy.csv", "esco_skill_hierarchy": "skill_hierarchy.csv",
}


def esco_state() -> dict:
    rows = read_manifest()
    off = [r for r in rows if r["dataset_name"] == "esco_classification_en_csv" and r["status"] == "OK"]
    if off:
        r = sorted(off, key=lambda r: r["version"])[-1]
        return {"mode": "official_zip", "version": r["version"], "dir": Path(r["extracted_path"])}
    api = [r for r in rows if r["dataset_name"] == "esco_api_fallback" and r["status"].startswith("OK")]
    if api:
        r = api[-1]
        return {"mode": "api_fallback", "version": r["version"], "dir": Path(r["extracted_path"])}
    return {"mode": None, "version": None, "dir": None}


def load_esco(con: duckdb.DuckDBPyConnection) -> tuple[dict, list[dict]]:
    from src.utils.common import ROOT
    st = esco_state()
    report = []
    if not st["mode"]:
        log.warning("ESCO_STATUS = MANUAL_DOWNLOAD_REQUIRED; no ESCO tables loaded")
        st["status"] = "MANUAL_DOWNLOAD_REQUIRED"
        return st, report
    d = ROOT / st["dir"]
    if st["mode"] == "official_zip":
        csvs = list(d.rglob("*.csv"))
        plan = []
        for table, prefixes, req in ESCO_OFFICIAL:
            hit = [p for p in csvs if any(p.name.lower().startswith(pf) for pf in prefixes) and req <= set(header(p))]
            plan.append((table, hit[0] if hit else None))
        st["status"] = "OFFICIAL_ZIP_LOADED"
    else:
        plan = [(t, d / f if (d / f).exists() else None) for t, f in ESCO_API.items()]
        st["status"] = "MANUAL_DOWNLOAD_REQUIRED (official ZIP); TEMPORARY_API_FALLBACK_LOADED"
    for table, p in plan:
        if p is None:
            report.append({"table_name": table, "files": "", "row_count": 0, "status": "NOT FOUND"})
            continue
        cols = header(p)
        col_sql = ", ".join(f'"{c}" AS {snake(c)}' for c in cols)
        con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT {col_sql}, '{st['version']}' AS esco_version, "
                    f"'{st['mode']}' AS esco_source_mode FROM read_csv('{p.as_posix()}', header=true, "
                    f"all_varchar=true, quote='\"', escape='\"', strict_mode=false)")
        n = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        report.append({"table_name": table, "files": rel(p), "row_count": n, "status": "LOADED"})
        log.info("ESCO %-34s %8d rows (%s %s)", table, n, st["mode"], st["version"])
    return st, report
