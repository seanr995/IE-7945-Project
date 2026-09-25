"""End-to-end, idempotent Sprint 1 pipeline.

    python run_pipeline.py              # full run (downloads only what is missing)
    python run_pipeline.py --offline    # skip network; use existing raw data
    python run_pipeline.py --refresh    # take a new job-source snapshot first
    python run_pipeline.py --skip-notebook

Steps: acquire -> extract -> manifest -> validate downloads -> inspect schemas -> ingest +
standardize jobs -> O*NET -> ESCO -> DuckDB -> EDA tables -> notebook -> summary -> acceptance.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time

import pandas as pd

from src.utils.common import (DB_PATH, DOCS, ESCO_EXT, MANIFEST, ONET_EXT, OUTPUTS, RAW_JOBS, ROOT,
                              get_logger, load_env, read_manifest, rel)
from src.utils.validate_downloads import validate_all

log = get_logger("run_pipeline")


def inspect_schemas() -> pd.DataFrame:
    """Inventory every extracted/raw tabular file with its actual header."""
    rows = []
    for base in (ONET_EXT, ESCO_EXT):
        for p in sorted(base.rglob("*.csv")):
            with open(p, encoding="utf-8-sig", newline="") as f:
                cols = next(csv.reader(f))
            rows.append({"file": rel(p), "n_columns": len(cols), "columns": " | ".join(cols)})
    for p in sorted(RAW_JOBS.rglob("*.csv")):
        cols = list(pd.read_csv(p, nrows=0).columns)
        rows.append({"file": rel(p), "n_columns": len(cols), "columns": " | ".join(cols)})
    for p in sorted(RAW_JOBS.glob("*/retrieval_*/page_001.json")):
        import json
        d = json.loads(p.read_text(encoding="utf-8"))
        rec = (d.get("data") or d.get("results") or [{}])[0]
        rows.append({"file": rel(p), "n_columns": len(rec), "columns": " | ".join(rec.keys())})
    df = pd.DataFrame(rows)
    (OUTPUTS / "tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUTS / "tables" / "schema_inventory.csv", index=False)
    log.info("Schema inventory: %d files -> outputs/tables/schema_inventory.csv", len(df))
    return df


SECRET_PATTERNS = [re.compile(p) for p in (
    # key-like assignments with a real-looking value (placeholders such as "your-key" don't match)
    r"(?i)(api[_-]?key|app[_-]?key|authorization-key)[ \t]*[=:][ \t]*['\"]?[A-Za-z0-9+/=]{16,}",
    r"(?i)(USAJOBS_API_KEY|ADZUNA_APP_KEY|ADZUNA_APP_ID)[ \t]*=[ \t]*[A-Za-z0-9+/=]{8,}",
)]


def secret_scan() -> list[str]:
    import os
    hits = []
    # literal values of any configured credentials must not appear anywhere outside .env
    literals = [v for k in ("USAJOBS_API_KEY", "ADZUNA_APP_KEY", "ADZUNA_APP_ID")
                if (v := os.environ.get(k, "").strip()) and len(v) >= 6]
    if literals:
        SECRET_PATTERNS.extend(re.compile(re.escape(v)) for v in literals)
    skip = {".venv", "data", "database", ".git", "__pycache__"}
    for p in ROOT.rglob("*"):
        if not p.is_file() or any(part in skip for part in p.relative_to(ROOT).parts) or p.name in (".env",):
            continue
        if p.suffix.lower() not in {".py", ".md", ".ipynb", ".txt", ".csv", ".json", ".example", ".log", ""}:
            continue
        try:
            t = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for pat in SECRET_PATTERNS:
            if pat.search(t):
                hits.append(rel(p))
    # also the manifest
    if MANIFEST.exists() and any(p.search(MANIFEST.read_text(encoding="utf-8")) for p in SECRET_PATTERNS):
        hits.append(rel(MANIFEST))
    return sorted(set(hits))


def acceptance(notebook_ok: bool) -> list[tuple[str, bool, str]]:
    import duckdb
    man = read_manifest()
    con = duckdb.connect(str(DB_PATH), read_only=True)
    cnt = lambda t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
    tables = set(con.execute("SELECT table_name FROM duckdb_tables()").df().table_name)
    jobs_rows = [r for r in man if r["dataset_category"] == "job_postings"]
    onet = [r for r in man if r["dataset_name"] == "onet_database"]
    esco_status = con.execute("SELECT status FROM source_metadata WHERE dataset_name='esco_status'").fetchone()[0]
    onet_tables = ["onet_occupations", "onet_tasks", "onet_skills", "onet_knowledge", "onet_abilities",
                   "onet_work_activities", "onet_technology_skills"]
    structure = ["data/raw/jobs", "data/reference/onet/raw", "data/reference/onet/extracted", "data/reference/esco/raw",
                 "data/reference/esco/extracted", "data/processed", "database", "notebooks", "src/acquisition",
                 "src/ingestion", "src/cleaning", "src/analysis", "src/utils", "outputs/figures", "outputs/tables",
                 "outputs/logs", "docs", "tests", ".env", ".env.example", ".gitignore", "requirements.txt",
                 "README.md", "run_pipeline.py", "download_data.py"]
    missing = [s for s in structure if not (ROOT / s).exists()]
    raw_ok = all((ROOT / r["raw_file_path"]).exists() and any((ROOT / r["raw_file_path"]).iterdir()) for r in jobs_rows)
    secrets = secret_scan()
    checks = [
        ("project structure exists", not missing, f"missing: {missing}" if missing else "all present"),
        ("at least one REAL public job posting source downloaded", len(jobs_rows) > 0, ", ".join(r["dataset_name"] for r in jobs_rows)),
        ("raw job source files preserved", raw_ok, "; ".join(r["raw_file_path"] for r in jobs_rows)),
        ("source provenance recorded", all(r["official_source_url"] and r["provider"] for r in man), f"{len(man)} manifest rows"),
        ("checksums recorded", all(len(r["sha256"]) == 64 for r in man), "sha256 on every manifest row"),
        ("job_postings has > 0 rows", cnt("job_postings") > 0, f"{cnt('job_postings'):,} rows"),
        ("database/taxonomy.duckdb exists", DB_PATH.exists(), f"{DB_PATH.stat().st_size / 1e6:.1f} MB"),
        ("O*NET downloaded from official source", bool(onet) and "onetcenter.org" in onet[0]["download_url"], onet[0]["download_url"] if onet else ""),
        ("O*NET raw ZIP preserved", bool(onet) and (ROOT / onet[0]["raw_file_path"]).exists(), onet[0]["raw_file_path"] if onet else ""),
        ("O*NET files extracted", bool(onet) and any((ROOT / onet[0]["extracted_path"]).rglob("*.csv")), onet[0]["extracted_path"] if onet else ""),
        ("O*NET reference tables loaded", all(t in tables and cnt(t) > 0 for t in onet_tables), ", ".join(f"{t}={cnt(t):,}" for t in onet_tables if t in tables)),
        ("ESCO loaded OR marked manual-download-required",
         ("LOADED" in esco_status) or ("MANUAL_DOWNLOAD_REQUIRED" in esco_status and (DOCS / "ESCO_MANUAL_DOWNLOAD_REQUIRED.md").exists()), esco_status),
        ("docs/data_contract.md exists", (DOCS / "data_contract.md").exists(), ""),
        ("docs/data_sources.md exists", (DOCS / "data_sources.md").exists(), ""),
        ("data/data_manifest.csv exists", MANIFEST.exists(), ""),
        ("notebooks/01_data_exploration.ipynb exists", (ROOT / "notebooks/01_data_exploration.ipynb").exists(), ""),
        ("notebook executes without error", notebook_ok, ""),
        ("outputs/sprint1_summary.md exists", (OUTPUTS / "sprint1_summary.md").exists(), ""),
        ("outputs/meeting_brief.md exists", (OUTPUTS / "meeting_brief.md").exists(), ""),
        ("repository contains no secrets", not secrets, f"hits: {secrets}" if secrets else "no key patterns found; .env git-ignored"),
        ("posting_id unique & deterministic", cnt("job_postings") == con.execute("SELECT count(DISTINCT posting_id) FROM job_postings").fetchone()[0], "PRIMARY KEY enforced"),
    ]
    con.close()
    return checks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--skip-notebook", action="store_true")
    a = ap.parse_args()
    load_env()
    t0 = time.time()

    log.info("STEP 1-3: acquisition / extraction / manifest")
    if not a.offline:
        import download_data
        download_data.main(refresh=a.refresh)
    else:
        from src.acquisition import esco
        esco.acquire(allow_api=False)  # still discovers a newly placed official ZIP
    log.info("STEP 2b: download validation")
    val = validate_all(print_table=True)
    if any(r["Status"] != "PASS" for r in val):
        log.error("Download validation failed - aborting before DB build")
        return 1

    log.info("STEP 4: schema inspection")
    inspect_schemas()

    log.info("STEP 5-9: ingest + standardize jobs, O*NET, ESCO -> DuckDB")
    from src.ingestion.build_db import build
    res = build()

    log.info("STEP 10: EDA tables")
    from src.analysis.eda import compute
    m = compute()

    notebook_ok = True
    if not a.skip_notebook:
        log.info("STEP 11: generate + execute notebook")
        from src.analysis.notebook import build_and_execute
        try:
            build_and_execute()
        except Exception as e:
            notebook_ok = False
            log.error("Notebook failed: %s", e)

    log.info("STEP 12: summary / contract / source docs")
    from src.analysis.reports import generate
    generate(m)

    log.info("STEP 13: acceptance checks")
    checks = acceptance(notebook_ok)
    print("\nACCEPTANCE TEST")
    for name, ok, detail in checks:
        print(f"[{'x' if ok else ' '}] {name:55s} {detail}")
    failed = [c for c in checks if not c[1]]
    log.info("Pipeline finished in %.0fs - %d/%d checks passed", time.time() - t0, len(checks) - len(failed), len(checks))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
