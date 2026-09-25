"""Job-posting acquisition following the agreed source priority:

 1. existing local files in data/raw/jobs/local/ (or loose files in data/raw/jobs/)
 2. USAJOBS Search API          (needs USAJOBS_EMAIL + USAJOBS_API_KEY)
 3. Adzuna API                  (needs ADZUNA_APP_ID + ADZUNA_APP_KEY)
 4. public no-auth bootstrap:   NYC Open Data "Jobs NYC Postings" (primary)
                                Arbeitnow free public job-board API (secondary)

Every source writes raw responses, byte-for-byte, into
data/raw/jobs/<source>/retrieval_<UTC timestamp>/ plus retrieval_metadata.json.
A retrieval is only considered usable once its `.complete` marker exists; existing
complete retrievals are reused (idempotent) unless refresh=True.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from src.utils.common import (DOCS, RAW_JOBS, dir_size, env, get_logger, rel, session,
                              sha256_dir, upsert_manifest, utc_now, utc_stamp)

log = get_logger("acquisition.jobs")

BOOTSTRAP_LABEL = ("Temporary bootstrap source - instructor/team approval required before final "
                   "corpus selection.")

SOURCES = {
    "nyc_jobs": {
        "provider": "City of New York - Department of Citywide Administrative Services (DCAS), via NYC Open Data",
        "official_source_url": "https://data.cityofnewyork.us/City-Government/Jobs-NYC-Postings/kpav-sd4t",
        "download_url": "https://data.cityofnewyork.us/api/views/kpav-sd4t/rows.csv?accessType=DOWNLOAD",
        "metadata_url": "https://data.cityofnewyork.us/api/views/kpav-sd4t.json",
        "license_or_terms": ("NYC Open Data public dataset; governed by NYC Open Data Terms of Use "
                             "(https://opendata.cityofnewyork.us/overview/#termsofuse) - requires verification"),
    },
    "arbeitnow": {
        "provider": "Arbeitnow (arbeitnow.com) free public Job Board API",
        "official_source_url": "https://www.arbeitnow.com/blog/job-board-api",
        "download_url": "https://www.arbeitnow.com/api/job-board-api?page=<n>",
        "license_or_terms": ("API response states: 'This is a free public API for jobs, please do not abuse. "
                             "I would appreciate linking back to the site. By using the API, you agree to the "
                             "terms of service present on Arbeitnow.com' - full ToS requires verification"),
    },
    "usajobs": {
        "provider": "U.S. Office of Personnel Management - USAJOBS",
        "official_source_url": "https://developer.usajobs.gov/",
        "download_url": "https://data.usajobs.gov/api/search",
        "license_or_terms": "USAJOBS API terms of service (https://developer.usajobs.gov/) - requires verification",
    },
    "adzuna": {
        "provider": "Adzuna",
        "official_source_url": "https://developer.adzuna.com/",
        "download_url": "https://api.adzuna.com/v1/api/jobs/{country}/search/{page}",
        "license_or_terms": "Adzuna API terms (https://developer.adzuna.com/) - requires verification",
    },
}


def latest_complete(source: str) -> Path | None:
    d = RAW_JOBS / source
    done = sorted(d.glob("retrieval_*/.complete")) if d.exists() else []
    return done[-1].parent if done else None


def _finish(source: str, out: Path, meta: dict, n_records: int, status: str, notes: str,
            fmt: str) -> dict:
    meta.update({"source": source, "record_count": n_records, "completed_utc": utc_now()})
    (out / "retrieval_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    # also maintain a source-level pointer file, as agreed for USAJOBS-style layout
    (RAW_JOBS / source / "retrieval_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out / ".complete").write_text(utc_now())
    register(source, out, status, notes, fmt)
    return {"source": source, "raw": out, "records": n_records}


def register(source: str, out: Path, status: str, notes: str, fmt: str) -> None:
    meta = json.loads((out / "retrieval_metadata.json").read_text(encoding="utf-8"))
    s = SOURCES.get(source, {})
    upsert_manifest({
        "dataset_name": f"jobs_{source}", "dataset_category": "job_postings",
        "provider": s.get("provider", source), "official_source_url": s.get("official_source_url", ""),
        "download_url": s.get("download_url", ""), "version": meta.get("source_version", f"snapshot {out.name}"),
        "retrieval_timestamp_utc": meta.get("started_utc", ""), "raw_file_path": rel(out),
        "extracted_path": "", "file_size_bytes": dir_size(out), "sha256": sha256_dir(out), "format": fmt,
        "license_or_terms": s.get("license_or_terms", "requires verification"), "status": status,
        "record_count": meta.get("record_count", ""), "notes": notes,
    })


# ---------------------------------------------------------------- priority 1

def check_local() -> list[Path]:
    exts = {".csv", ".json", ".jsonl", ".parquet", ".zip"}
    local = []
    for p in RAW_JOBS.iterdir() if RAW_JOBS.exists() else []:
        if p.is_file() and p.suffix.lower() in exts:
            local.append(p)
    ld = RAW_JOBS / "local"
    if ld.exists():
        local += [p for p in ld.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    log.info("Local job files found: %d", len(local))
    return local


# ---------------------------------------------------------------- priority 2

def write_usajobs_doc() -> None:
    (DOCS / "USAJOBS_API_SETUP.md").write_text("""# USAJOBS API setup

USAJOBS is an approved public source but its Search API requires a free API key.

1. Request a key at https://developer.usajobs.gov/APIRequest/Index (you provide your
   email address; the key is emailed to you).
2. Add both values to the project `.env` file (never commit it):

```
USAJOBS_EMAIL=you@northeastern.edu
USAJOBS_API_KEY=your-authorization-key
```

   `USAJOBS_EMAIL` is sent as the `User-Agent` header and `USAJOBS_API_KEY` as the
   `Authorization-Key` header, as required by https://developer.usajobs.gov/.
3. Run `python download_data.py` (or `python run_pipeline.py`). The downloader pages
   through `https://data.usajobs.gov/api/search` (500 results/page) and stores every raw
   page under `data/raw/jobs/usajobs/retrieval_<timestamp>/page_NNN.json`.
""", encoding="utf-8")


def acquire_usajobs(max_pages: int = 20) -> dict | None:
    email, key = env("USAJOBS_EMAIL"), env("USAJOBS_API_KEY")
    if not (email and key):
        write_usajobs_doc()
        log.info("USAJOBS credentials not configured -> docs/USAJOBS_API_SETUP.md; skipping")
        return None
    out = RAW_JOBS / "usajobs" / f"retrieval_{utc_stamp()}"
    out.mkdir(parents=True, exist_ok=True)
    meta = {"started_utc": utc_now(), "endpoint": SOURCES["usajobs"]["download_url"],
            "params": {"ResultsPerPage": 500}, "pages": 0}
    s = session()
    s.headers.update({"Host": "data.usajobs.gov", "User-Agent": email, "Authorization-Key": key})
    n = 0
    for page in range(1, max_pages + 1):
        r = s.get(SOURCES["usajobs"]["download_url"], params={"ResultsPerPage": 500, "Page": page}, timeout=120)
        r.raise_for_status()
        (out / f"page_{page:03d}.json").write_bytes(r.content)
        items = r.json().get("SearchResult", {}).get("SearchResultItems", [])
        n += len(items)
        meta["pages"] = page
        if not items:
            break
        time.sleep(1)
    return _finish("usajobs", out, meta, n, "OK", "Official USAJOBS Search API.", "JSON API pages")


# ---------------------------------------------------------------- priority 3

def acquire_adzuna(country: str = "us", max_pages: int = 40) -> dict | None:
    app_id, app_key = env("ADZUNA_APP_ID"), env("ADZUNA_APP_KEY")
    if not (app_id and app_key):
        log.info("Adzuna credentials not configured; skipping")
        return None
    out = RAW_JOBS / "adzuna" / f"retrieval_{utc_stamp()}"
    out.mkdir(parents=True, exist_ok=True)
    meta = {"started_utc": utc_now(), "country": country, "results_per_page": 50, "pages": 0}
    s, n = session(), 0
    for page in range(1, max_pages + 1):
        r = s.get(f"https://api.adzuna.com/v1/api/jobs/{country}/search/{page}",
                  params={"app_id": app_id, "app_key": app_key, "results_per_page": 50,
                          "content-type": "application/json"}, timeout=120)
        r.raise_for_status()
        (out / f"page_{page:03d}.json").write_bytes(r.content)
        res = r.json().get("results", [])
        n += len(res)
        meta["pages"] = page
        if not res:
            break
        time.sleep(1)
    return _finish("adzuna", out, meta, n, "OK", "Official Adzuna API (credentials not stored).", "JSON API pages")


# ---------------------------------------------------------------- priority 4

def acquire_nyc(refresh: bool = False) -> dict:
    prev = latest_complete("nyc_jobs")
    if prev and not refresh:
        log.info("NYC Jobs retrieval exists: %s (reuse)", rel(prev))
        register("nyc_jobs", prev, "OK - " + BOOTSTRAP_LABEL, _nyc_notes(), "CSV export + Socrata metadata JSON")
        return {"source": "nyc_jobs", "raw": prev}
    src = SOURCES["nyc_jobs"]
    out = RAW_JOBS / "nyc_jobs" / f"retrieval_{utc_stamp()}"
    out.mkdir(parents=True, exist_ok=True)
    meta = {"started_utc": utc_now(), "download_url": src["download_url"], "metadata_url": src["metadata_url"]}
    s = session()
    m = s.get(src["metadata_url"], timeout=120)
    m.raise_for_status()
    (out / "dataset_metadata.json").write_bytes(m.content)
    md = m.json()
    meta["source_version"] = f"rowsUpdatedAt={md.get('rowsUpdatedAt')}"
    meta["dataset_name"] = md.get("name")
    meta["attribution"] = md.get("attribution")
    r = s.get(src["download_url"], timeout=600)
    r.raise_for_status()
    if r.content[:200].lstrip().lower().startswith(b"<"):
        raise RuntimeError("NYC export returned HTML, not CSV")
    (out / "jobs_nyc_postings.csv").write_bytes(r.content)
    import pandas as pd
    n = len(pd.read_csv(out / "jobs_nyc_postings.csv", dtype=str))
    return _finish("nyc_jobs", out, meta, n, "OK - " + BOOTSTRAP_LABEL, _nyc_notes(), "CSV export + Socrata metadata JSON")


def _nyc_notes() -> str:
    return ("Official municipal open-data export of current City of New York job postings (public "
            "sector only). " + BOOTSTRAP_LABEL)


def acquire_arbeitnow(max_pages: int = 25, refresh: bool = False) -> dict:
    prev = latest_complete("arbeitnow")
    if prev and not refresh:
        log.info("Arbeitnow retrieval exists: %s (reuse)", rel(prev))
        register("arbeitnow", prev, "OK - " + BOOTSTRAP_LABEL, _arb_notes(), "JSON API pages")
        return {"source": "arbeitnow", "raw": prev}
    out = RAW_JOBS / "arbeitnow" / f"retrieval_{utc_stamp()}"
    out.mkdir(parents=True, exist_ok=True)
    meta = {"started_utc": utc_now(), "endpoint": "https://www.arbeitnow.com/api/job-board-api",
            "max_pages": max_pages, "pages": 0}
    s, n, seen = session(), 0, set()
    for page in range(1, max_pages + 1):
        r = s.get(meta["endpoint"], params={"page": page}, timeout=120)
        if r.status_code == 429:
            log.warning("Arbeitnow rate-limited at page %d; stopping politely", page)
            break
        r.raise_for_status()
        (out / f"page_{page:03d}.json").write_bytes(r.content)
        d = r.json()
        data = d.get("data", [])
        new = [x for x in data if x.get("slug") not in seen]
        seen.update(x.get("slug") for x in data)
        n += len(data)
        meta["pages"] = page
        meta["api_terms_text"] = d.get("meta", {}).get("terms")
        if not data or not new or not d.get("links", {}).get("next"):
            break
        time.sleep(2)  # polite pacing ("please do not abuse")
    return _finish("arbeitnow", out, meta, n, "OK - " + BOOTSTRAP_LABEL, _arb_notes(), "JSON API pages")


def _arb_notes() -> str:
    return ("Genuinely open job-board API (no key); mostly Germany-based postings in German/English, "
            "useful for language-mix testing. Aggregator content: employers are third parties. " + BOOTSTRAP_LABEL)


def acquire(refresh: bool = False) -> list[dict]:
    RAW_JOBS.mkdir(parents=True, exist_ok=True)
    results = []
    local = check_local()
    if local:
        results.append({"source": "local", "files": [rel(p) for p in local]})
    for fn in (acquire_usajobs, acquire_adzuna):
        try:
            r = fn()
            if r:
                results.append(r)
        except Exception as e:
            log.error("%s failed: %s", fn.__name__, type(e).__name__)  # never log credentials
    for fn in (acquire_nyc, acquire_arbeitnow):
        try:
            results.append(fn(refresh=refresh))
        except Exception as e:
            log.error("%s failed: %s", fn.__name__, e)
    return results
