"""ESCO acquisition.

Priority:
 1. An official ESCO ZIP placed manually in data/reference/esco/raw/ (the official
    download page requires a human to accept a privacy statement and give an email
    address, so we never automate it).
 2. Fallback: the official ESCO Web Services API (https://ec.europa.eu/esco/api),
    queried with an explicit selectedVersion that the API validates. Raw API pages
    are preserved (gzip of the exact response bytes) and flattened to CSV.
"""
from __future__ import annotations

import csv
import gzip
import json
import re
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.utils.common import (DOCS, ESCO_EXT, ESCO_RAW, dir_size, get_logger, rel, session,
                              sha256_dir, sha256_file, upsert_manifest, utc_now, utc_stamp,
                              validate_zip)

log = get_logger("acquisition.esco")

DOWNLOAD_PAGE = "https://esco.ec.europa.eu/en/use-esco/download"
API = "https://ec.europa.eu/esco/api"
API_DOCS = "https://esco.ec.europa.eu/en/use-esco/use-esco-services-api/esco-web-service-api"
TERMS = ("requires verification (ESCO is published by the European Commission; confirm reuse "
         "terms at https://esco.ec.europa.eu before redistribution)")
PAGE = 100


def discover_latest_version() -> str | None:
    try:
        html = session().get(DOWNLOAD_PAGE, timeout=60).text
    except Exception as e:  # network failure must not crash the pipeline
        log.warning("Could not reach ESCO download page: %s", e)
        return None
    vers = set(re.findall(r"ESCO dataset - v(\d+\.\d+\.\d+)", html))
    if not vers:
        return None
    return "v" + max(vers, key=lambda v: tuple(int(x) for x in v.split(".")))


def write_manual_doc(version: str | None) -> None:
    v = version or "<latest listed on the page>"
    DOCS.mkdir(exist_ok=True)
    (DOCS / "ESCO_MANUAL_DOWNLOAD_REQUIRED.md").write_text(f"""# ESCO manual download required

The official ESCO download requires a human to accept the European Commission
privacy statement and enter an email address; the download link is emailed.
The pipeline does **not** automate or bypass this step.

1. Open the official page: {DOWNLOAD_PAGE}
2. Version: **ESCO dataset - {v}**
3. Content: **classification**
4. Language: **en**
5. File type: **csv**
6. Accept the privacy statement, enter your email, open the emailed link, and save the ZIP
   (unchanged, any filename ending in `.zip`) into:

   `data/reference/esco/raw/`

Then run `python run_pipeline.py`. The pipeline auto-discovers the ZIP, verifies it,
extracts it to `data/reference/esco/extracted/<version>/`, and loads the official CSVs in
place of the temporary API-derived tables.
""", encoding="utf-8")


# ---------------------------------------------------------------- official ZIP

def find_official_zip() -> Path | None:
    zips = sorted(ESCO_RAW.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    for z in zips:
        ok, msg = validate_zip(z)
        if ok:
            return z
        log.warning("Ignoring invalid ESCO zip %s: %s", rel(z), msg)
    return None


def zip_version(z: Path) -> str:
    m = re.search(r"v?(\d+\.\d+\.\d+)", z.name)
    if m:
        return "v" + m.group(1)
    with zipfile.ZipFile(z) as zf:
        for n in zf.namelist():
            m = re.search(r"v(\d+\.\d+\.\d+)", n)
            if m:
                return "v" + m.group(1)
    return "unknown_version"


def ingest_official_zip(z: Path) -> dict:
    version = zip_version(z)
    ext = ESCO_EXT / version
    marker = ext / ".extracted_ok"
    if not marker.exists() or marker.read_text() != sha256_file(z):
        ext.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(ext)
        marker.write_text(sha256_file(z))
    files = list(ext.rglob("*.csv"))
    upsert_manifest({
        "dataset_name": "esco_classification_en_csv", "dataset_category": "reference_taxonomy",
        "provider": "European Commission (ESCO)", "official_source_url": DOWNLOAD_PAGE,
        "download_url": "manual download via official page (emailed link)", "version": version,
        "retrieval_timestamp_utc": utc_now(), "raw_file_path": rel(z), "extracted_path": rel(ext),
        "file_size_bytes": z.stat().st_size, "sha256": sha256_file(z), "format": "ZIP of CSV files",
        "license_or_terms": TERMS, "status": "OK", "record_count": f"{len(files)} csv files",
        "notes": "Official ESCO package placed manually in data/reference/esco/raw/.",
    })
    return {"mode": "official_zip", "version": version, "raw": z, "extracted": ext, "files": len(files)}


# ---------------------------------------------------------------- API fallback

def _fetch_page(kind: str, page: int, version: str, out_dir: Path) -> Path:
    # NB: the ESCO search API's `offset` parameter is a PAGE INDEX (see its `next` links),
    # not a record offset.
    out = out_dir / f"{kind}_page_{page:04d}.json.gz"
    if out.exists():
        return out
    url = f"{API}/search"
    params = {"type": kind, "language": "en", "limit": PAGE, "offset": page,
              "full": "true", "selectedVersion": version}
    s = session()
    for attempt in range(5):
        try:
            r = s.get(url, params=params, timeout=180)
            r.raise_for_status()
            json.loads(r.content)  # must be valid JSON
            tmp = out.with_suffix(".part")
            with gzip.open(tmp, "wb") as f:
                f.write(r.content)
            tmp.replace(out)
            return out
        except Exception as e:
            log.warning("ESCO %s page %d attempt %d failed: %s", kind, page, attempt + 1, e)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"ESCO API page {kind}@{page} failed")


def _total(kind: str, version: str) -> int:
    r = session().get(f"{API}/search", params={"type": kind, "language": "en", "limit": 1,
                                                "offset": 0, "selectedVersion": version}, timeout=60)
    r.raise_for_status()
    return int(r.json()["total"])


def _en(d, key="en"):
    if not isinstance(d, dict):
        return None
    v = d.get(key)
    if isinstance(v, dict):
        return v.get("literal")
    return v


def _alt(d):
    if not isinstance(d, dict):
        return None
    v = d.get("en") or []
    return "\n".join(v) if isinstance(v, list) else v


def flatten_api(raw_dir: Path, ext: Path) -> dict:
    ext.mkdir(parents=True, exist_ok=True)
    occ, sk, rel_rows, occ_h, sk_h = [], [], [], [], []
    for f in sorted(raw_dir.glob("occupation_page_*.json.gz")):
        for x in json.loads(gzip.open(f).read())["_embedded"]["results"]:
            L = x.get("_links", {})
            occ.append({
                "conceptUri": x["uri"], "preferredLabel": _en(x.get("preferredLabel")) or x.get("title"),
                "altLabels": _alt(x.get("alternativeLabel")), "description": _en(x.get("description")),
                "scopeNote": _en(x.get("scopeNote")), "code": x.get("code"), "status": x.get("status"),
                "iscoGroup": next((b["uri"].rsplit("/", 1)[-1] for b in L.get("broaderIscoGroup", [])), None),
                "naceCodes": ";".join(n.get("code", "") for n in L.get("hasNACECode", [])) or None,
            })
            for relation, key in (("essential", "hasEssentialSkill"), ("optional", "hasOptionalSkill")):
                for s in L.get(key, []):
                    rel_rows.append({"occupationUri": x["uri"], "relationType": relation,
                                     "skillType": s.get("skillType"), "skillUri": s["uri"],
                                     "skillLabel": s.get("title")})
            for key, items in L.items():
                if key.startswith("broader"):
                    for b in items:
                        occ_h.append({"conceptUri": x["uri"], "broaderUri": b["uri"],
                                      "broaderLabel": b.get("title"), "broaderType": key})
    for f in sorted(raw_dir.glob("skill_page_*.json.gz")):
        for x in json.loads(gzip.open(f).read())["_embedded"]["results"]:
            L = x.get("_links", {})
            sk.append({
                "conceptUri": x["uri"], "preferredLabel": _en(x.get("preferredLabel")) or x.get("title"),
                "altLabels": _alt(x.get("alternativeLabel")), "description": _en(x.get("description")),
                "status": x.get("status"),
                "skillType": next((t.get("title") for t in L.get("hasSkillType", [])), None),
                "reuseLevel": next((t.get("title") for t in L.get("hasReuseLevel", [])), None),
            })
            for key, items in L.items():
                if key.startswith("broader"):
                    for b in items:
                        sk_h.append({"conceptUri": x["uri"], "broaderUri": b["uri"],
                                     "broaderLabel": b.get("title"), "broaderType": key})
    out = {"occupations.csv": occ, "skills.csv": sk, "occupation_skill_relations.csv": rel_rows,
           "occupation_hierarchy.csv": occ_h, "skill_hierarchy.csv": sk_h}
    for name, rows in out.items():
        with open(ext / name, "w", newline="", encoding="utf-8") as fh:
            if rows:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
    return {k: len(v) for k, v in out.items()}


def _verify_complete(raw_dir: Path, totals: dict) -> None:
    """Every concept reported by the API must be present exactly once."""
    for kind, total in totals.items():
        uris = [x["uri"] for f in sorted(raw_dir.glob(f"{kind}_page_*.json.gz"))
                for x in json.loads(gzip.open(f).read())["_embedded"]["results"]]
        if len(uris) != total or len(set(uris)) != total:
            raise RuntimeError(f"ESCO {kind}: got {len(uris)} records ({len(set(uris))} unique), API total {total}")
        log.info("ESCO %s complete: %d unique concepts", kind, total)


def acquire_api(version: str) -> dict:
    base = ESCO_RAW / f"api_{version}"
    done = sorted(base.glob("retrieval_*/.complete")) if base.exists() else []
    if done:
        raw_dir = done[-1].parent
        log.info("ESCO API retrieval already complete: %s", rel(raw_dir))
    else:
        partial = sorted(base.glob("retrieval_*")) if base.exists() else []
        raw_dir = partial[-1] if partial else base / f"retrieval_{utc_stamp()}"
        raw_dir.mkdir(parents=True, exist_ok=True)
        jobs = []
        totals = {}
        for kind in ("occupation", "skill"):
            totals[kind] = _total(kind, version)
            jobs += [(kind, pg) for pg in range(-(-totals[kind] // PAGE))]
        log.info("ESCO API %s totals %s -> %d pages", version, totals, len(jobs))
        with ThreadPoolExecutor(max_workers=4) as ex:
            list(ex.map(lambda j: _fetch_page(j[0], j[1], version, raw_dir), jobs))
        _verify_complete(raw_dir, totals)
        (raw_dir / "retrieval_metadata.json").write_text(json.dumps({
            "api": API, "endpoint": "/search", "params": {"language": "en", "limit": PAGE, "full": "true",
                                                          "selectedVersion": version},
            "totals": totals, "pages": len(jobs), "retrieved_utc": utc_now(),
            "note": "Each file is the gzip of the exact API response body.",
        }, indent=2), encoding="utf-8")
        (raw_dir / ".complete").write_text(utc_now())
    ext = ESCO_EXT / f"api_{version}"
    counts = flatten_api(raw_dir, ext)
    meta = json.loads((raw_dir / "retrieval_metadata.json").read_text())
    upsert_manifest({
        "dataset_name": "esco_api_fallback", "dataset_category": "reference_taxonomy",
        "provider": "European Commission (ESCO Web Services API)", "official_source_url": API_DOCS,
        "download_url": f"{API}/search?type=<occupation|skill>&language=en&full=true&selectedVersion={version}",
        "version": version, "retrieval_timestamp_utc": meta["retrieved_utc"],
        "raw_file_path": rel(raw_dir), "extracted_path": rel(ext), "file_size_bytes": dir_size(raw_dir),
        "sha256": sha256_dir(raw_dir), "format": "gzip-compressed JSON API pages (sha256 over directory)",
        "license_or_terms": TERMS, "status": "OK (temporary API fallback)",
        "record_count": f"{counts['occupations.csv']} occupations; {counts['skills.csv']} skills",
        "notes": ("Temporary Sprint 1 fallback: official ZIP needs manual privacy acceptance. "
                  "Version explicitly requested via selectedVersion and accepted by the API "
                  "(the API rejects invalid versions)."),
    })
    return {"mode": "api_fallback", "version": version, "raw": raw_dir, "extracted": ext, "counts": counts}


def acquire(allow_api: bool = True) -> dict:
    ESCO_RAW.mkdir(parents=True, exist_ok=True)
    z = find_official_zip()
    latest = discover_latest_version()
    if z:
        log.info("Official ESCO ZIP found: %s", rel(z))
        return {"status": "OFFICIAL_ZIP_LOADED", "latest_listed": latest, **ingest_official_zip(z)}
    write_manual_doc(latest)
    log.warning("ESCO_STATUS = MANUAL_DOWNLOAD_REQUIRED (see docs/ESCO_MANUAL_DOWNLOAD_REQUIRED.md)")
    if allow_api and latest:
        try:
            return {"status": "MANUAL_DOWNLOAD_REQUIRED; API_FALLBACK_LOADED", "latest_listed": latest,
                    **acquire_api(latest)}
        except Exception as e:
            log.error("ESCO API fallback failed: %s", e)
    return {"status": "MANUAL_DOWNLOAD_REQUIRED", "latest_listed": latest, "mode": None}
