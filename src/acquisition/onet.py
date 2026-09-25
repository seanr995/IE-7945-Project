"""O*NET acquisition: discover current production version from the official
O*NET Resource Center page, download the complete CSV archive, verify, extract."""
from __future__ import annotations

import re
import zipfile
from pathlib import Path

from src.utils.common import (ONET_EXT, ONET_RAW, download_file, get_logger, manifest_row,
                              rel, session, sha256_file, upsert_manifest, utc_now, validate_zip)

log = get_logger("acquisition.onet")

DATABASE_PAGE = "https://www.onetcenter.org/database.html"
DL_BASE = "https://www.onetcenter.org/dl_files/database/"
LICENSE = ("Creative Commons Attribution 4.0 International (CC BY 4.0), as stated on the official "
           "O*NET database page / https://www.onetcenter.org/license_db.html; attribution to "
           "O*NET (USDOL/ETA) required")


def discover_version() -> tuple[str, str]:
    """Return (version like '31.0', file token like '31_0') from the official page."""
    html = session().get(DATABASE_PAGE, timeout=60).text
    tokens = set(re.findall(r"/dl_files/database/db_(\d+_\d+)_", html))
    if not tokens:
        raise RuntimeError("Could not find any db_<ver>_ links on the O*NET database page")
    token = max(tokens, key=lambda t: tuple(int(x) for x in t.split("_")))
    version = token.replace("_", ".")
    if f"O*NET {version} Database" not in html:
        log.warning("Version %s derived from links but heading text not found", version)
    return version, token


def acquire() -> dict:
    version, token = discover_version()
    log.info("Current O*NET production version on official site: %s", version)
    url = f"{DL_BASE}db_{token}_csv.zip"
    raw = ONET_RAW / f"onet_database_{version}_csv.zip"
    ok, msg = validate_zip(raw)
    if not ok:
        download_file(url, raw, log)
        ok, msg = validate_zip(raw)
        if not ok:
            bad = raw.with_suffix(".invalid")
            raw.replace(bad)
            raise RuntimeError(f"O*NET download invalid ({msg}); moved to {bad}")
    else:
        log.info("O*NET raw archive already present and valid: %s", rel(raw))

    ext = ONET_EXT / version
    marker = ext / ".extracted_ok"
    if not marker.exists():
        ext.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(raw) as z:
            z.extractall(ext)
        marker.write_text(sha256_file(raw))
        log.info("Extracted O*NET to %s", rel(ext))
    n_csv = len(list(ext.rglob("*.csv")))

    prev = manifest_row("onet_database", rel(raw))
    upsert_manifest({
        "dataset_name": "onet_database",
        "dataset_category": "reference_taxonomy",
        "provider": "O*NET Resource Center (National Center for O*NET Development, sponsored by USDOL/ETA)",
        "official_source_url": DATABASE_PAGE,
        "download_url": url,
        "version": version,
        "retrieval_timestamp_utc": prev["retrieval_timestamp_utc"] if prev and prev.get("sha256") == sha256_file(raw) else utc_now(),
        "raw_file_path": rel(raw),
        "extracted_path": rel(ext),
        "file_size_bytes": raw.stat().st_size,
        "sha256": sha256_file(raw),
        "format": "ZIP of CSV files",
        "license_or_terms": LICENSE,
        "status": "OK",
        "record_count": f"{n_csv} csv files",
        "notes": ("Complete CSV archive db_" + token + "_csv.zip served from the official dl_files "
                  "directory (the page lists the individual CSVs and other ZIP formats)."),
    })
    return {"version": version, "raw": raw, "extracted": ext, "zip_check": msg, "files": n_csv}
