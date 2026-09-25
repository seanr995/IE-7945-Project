"""Reproducible, idempotent data acquisition.

    python download_data.py            # download anything missing, reuse existing raw data
    python download_data.py --refresh  # take a NEW snapshot of the job-posting sources
    python download_data.py --no-esco-api

Raw artifacts are never modified or deleted; provenance goes to data/data_manifest.csv.
"""
from __future__ import annotations

import argparse
import sys

from src.acquisition import esco, jobs, onet
from src.utils.common import get_logger, load_env
from src.utils.validate_downloads import validate_all

log = get_logger("download_data")


def main(refresh: bool = False, esco_api: bool = True) -> dict:
    load_env()
    summary = {}
    log.info("=== O*NET ===")
    summary["onet"] = onet.acquire()
    log.info("=== ESCO ===")
    summary["esco"] = esco.acquire(allow_api=esco_api)
    log.info("ESCO status: %s", summary["esco"]["status"])
    log.info("=== Job postings ===")
    summary["jobs"] = jobs.acquire(refresh=refresh)
    log.info("=== Download validation ===")
    results = validate_all(print_table=True)
    summary["validation"] = results
    failed = [r for r in results if r["Status"] != "PASS"]
    if failed:
        log.error("%d artifact(s) failed validation", len(failed))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="take a new job-source snapshot")
    ap.add_argument("--no-esco-api", action="store_true", help="skip ESCO API fallback")
    a = ap.parse_args()
    s = main(refresh=a.refresh, esco_api=not a.no_esco_api)
    sys.exit(1 if any(r["Status"] != "PASS" for r in s["validation"]) else 0)
