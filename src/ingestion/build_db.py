"""Build database/taxonomy.duckdb from raw + reference data.

The database is rebuilt into a temporary file and atomically swapped in, so
re-running never duplicates rows and a failed run never leaves a half-built DB.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import duckdb
import pandas as pd

from src.acquisition.jobs import BOOTSTRAP_LABEL
from src.cleaning.canonical import CONTRACT_VERSION, build as build_canonical
from src.ingestion.reference import load_esco, load_onet
from src.utils.common import DB_PATH, MANIFEST, PROCESSED, get_logger, read_manifest, utc_now

log = get_logger("ingestion.build_db")


def source_metadata(esco_state: dict, job_stats: dict) -> pd.DataFrame:
    rows = []
    for r in read_manifest():
        role = {"job_postings": "empirical corpus", "reference_taxonomy": "reference taxonomy"}[r["dataset_category"]]
        if r["dataset_name"].startswith("jobs_"):
            final = "TEMPORARY bootstrap - " + BOOTSTRAP_LABEL if "bootstrap" in r["status"].lower() else "candidate final"
        elif r["dataset_name"] == "esco_api_fallback":
            final = "TEMPORARY - replace with official ESCO CSV package (manual download)"
        else:
            final = "final (official reference)"
        rows.append({"dataset_name": r["dataset_name"], "role": role, "provider": r["provider"],
                     "official_source_url": r["official_source_url"], "version": r["version"],
                     "retrieved_utc": r["retrieval_timestamp_utc"], "raw_path": r["raw_file_path"],
                     "record_count": r.get("record_count"), "license_or_terms": r["license_or_terms"],
                     "status": r["status"], "final_or_temporary": final})
    rows.append({"dataset_name": "esco_status", "role": "reference taxonomy", "provider": "European Commission (ESCO)",
                 "official_source_url": "https://esco.ec.europa.eu/en/use-esco/download",
                 "version": esco_state.get("version"), "retrieved_utc": None, "raw_path": None,
                 "record_count": None, "license_or_terms": None,
                 "status": "ESCO_STATUS = " + esco_state.get("status", "MANUAL_DOWNLOAD_REQUIRED"),
                 "final_or_temporary": esco_state.get("mode")})
    return pd.DataFrame(rows)


def preserve_prototype_schema(con: duckdb.DuckDBPyConnection) -> None:
    """Sprint 1 rebuilds `main` from scratch; carry the separate `prototype` schema (Sprint 2
    experimental tables) over from the previous database file so it is never lost or mixed in."""
    if not DB_PATH.exists():
        return
    con.execute(f"ATTACH '{DB_PATH.as_posix()}' AS old_db (READ_ONLY)")
    try:
        tabs = con.execute("SELECT table_name FROM duckdb_tables() WHERE database_name='old_db' "
                           "AND schema_name='prototype'").fetchall()
        if tabs:
            con.execute("CREATE SCHEMA IF NOT EXISTS prototype")
            for (t,) in tabs:
                con.execute(f'CREATE TABLE prototype."{t}" AS SELECT * FROM old_db.prototype."{t}"')
            log.info("Preserved %d prototype.* tables from previous database", len(tabs))
    finally:
        con.execute("DETACH old_db")


def build() -> dict:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = DB_PATH.with_suffix(".building.duckdb")
    if tmp.exists():
        tmp.unlink()
    con = duckdb.connect(str(tmp))
    try:
        raw_df, jobs, job_stats = build_canonical()
        PROCESSED.mkdir(parents=True, exist_ok=True)
        jobs.to_parquet(PROCESSED / "job_postings_canonical.parquet", index=False)

        con.register("raw_df", raw_df)
        con.execute("CREATE TABLE job_postings_raw AS SELECT * FROM raw_df")
        con.execute("""CREATE TABLE job_postings (
            posting_id VARCHAR PRIMARY KEY, source VARCHAR NOT NULL, source_job_id VARCHAR,
            source_record_ref VARCHAR, retrieval_id VARCHAR, job_title VARCHAR, normalized_job_title VARCHAR,
            company_name VARCHAR, industry VARCHAR, industry_derived VARCHAR, job_category_source VARCHAR,
            location_raw VARCHAR, country VARCHAR, state VARCHAR, city VARCHAR, geo_basis VARCHAR,
            remote_flag BOOLEAN, employment_type VARCHAR, seniority_source VARCHAR, seniority_derived VARCHAR,
            seniority VARCHAR, seniority_basis VARCHAR, description VARCHAR, responsibilities VARCHAR,
            requirements VARCHAR, preferred_skills VARCHAR, salary_min DOUBLE, salary_max DOUBLE,
            salary_currency VARCHAR, salary_period VARCHAR, date_posted DATE, date_collected DATE,
            language VARCHAR, language_confidence DOUBLE, source_url VARCHAR, posting_text VARCHAR,
            description_char_count INTEGER, description_word_count INTEGER, posting_text_word_count INTEGER,
            exact_dup_key VARCHAR, normalized_dup_key VARCHAR, duplicate_group_id VARCHAR,
            is_exact_duplicate BOOLEAN, is_normalized_duplicate BOOLEAN, is_source_id_repeat BOOLEAN,
            contract_version VARCHAR)""")
        con.register("jobs_df", jobs)
        con.execute("INSERT INTO job_postings SELECT * FROM jobs_df")
        con.execute("""CREATE VIEW job_postings_unique AS
                       SELECT * FROM job_postings WHERE NOT is_normalized_duplicate""")

        onet_report = load_onet(con)
        esco_state, esco_report = load_esco(con)

        man = pd.read_csv(MANIFEST, dtype=str)
        con.register("man", man)
        con.execute("CREATE TABLE data_manifest AS SELECT * FROM man")
        sm = source_metadata(esco_state, job_stats)
        con.register("sm", sm)
        con.execute("CREATE TABLE source_metadata AS SELECT * FROM sm")
        load = pd.DataFrame(onet_report + esco_report)
        con.register("load", load)
        con.execute("CREATE TABLE reference_load_report AS SELECT * FROM load")
        con.execute("CREATE TABLE pipeline_runs AS SELECT ? AS run_utc, ? AS contract_version, ? AS job_stats_json",
                    [utc_now(), CONTRACT_VERSION, json.dumps(job_stats)])
        preserve_prototype_schema(con)
        con.execute("CHECKPOINT")
    finally:
        con.close()
    os.replace(tmp, DB_PATH)
    log.info("DuckDB written: %s", DB_PATH)
    return {"job_stats": job_stats, "onet": onet_report, "esco_state": {k: str(v) for k, v in esco_state.items()},
            "esco": esco_report}
