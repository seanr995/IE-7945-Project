"""Shared helpers: paths, logging, hashing, HTTP, manifest management."""
from __future__ import annotations

import csv
import hashlib
import logging
import os
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW_JOBS = DATA / "raw" / "jobs"
ONET_RAW = DATA / "reference" / "onet" / "raw"
ONET_EXT = DATA / "reference" / "onet" / "extracted"
ESCO_RAW = DATA / "reference" / "esco" / "raw"
ESCO_EXT = DATA / "reference" / "esco" / "extracted"
PROCESSED = DATA / "processed"
DB_PATH = ROOT / "database" / "taxonomy.duckdb"
MANIFEST = DATA / "data_manifest.csv"
OUTPUTS = ROOT / "outputs"
DOCS = ROOT / "docs"

USER_AGENT = "IE7945-capstone-workforce-analytics/1.0 (academic research; Northeastern University)"

MANIFEST_FIELDS = [
    "dataset_name", "dataset_category", "provider", "official_source_url", "download_url",
    "version", "retrieval_timestamp_utc", "raw_file_path", "extracted_path", "file_size_bytes",
    "sha256", "format", "license_or_terms", "status", "record_count", "notes",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def rel(p: Path | str) -> str:
    """Project-relative POSIX path (manifest never stores absolute user paths)."""
    p = Path(p)
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    (OUTPUTS / "logs").mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(OUTPUTS / "logs" / "pipeline.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_dir(path: Path) -> str:
    """Deterministic digest over a directory of raw files (name + content hash)."""
    h = hashlib.sha256()
    for f in sorted(p for p in path.rglob("*") if p.is_file()):
        h.update(f.relative_to(path).as_posix().encode())
        h.update(sha256_file(f).encode())
    return h.hexdigest()


def dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def looks_like_html(path: Path) -> bool:
    with open(path, "rb") as f:
        head = f.read(512).lstrip().lower()
    return head.startswith(b"<!doctype html") or head.startswith(b"<html") or b"<html" in head[:200]


def validate_zip(path: Path) -> tuple[bool, str]:
    """A real ZIP: exists, non-empty, not an HTML page, passes CRC test."""
    if not path.exists():
        return False, "missing"
    if path.stat().st_size == 0:
        return False, "zero bytes"
    if looks_like_html(path):
        return False, "HTML page saved as zip (error/consent page)"
    if not zipfile.is_zipfile(path):
        return False, "not a zip file"
    try:
        with zipfile.ZipFile(path) as z:
            bad = z.testzip()
            if bad:
                return False, f"CRC failure in {bad}"
            return True, f"{len(z.namelist())} members"
    except zipfile.BadZipFile as e:
        return False, f"bad zip: {e}"


def download_file(url: str, dest: Path, logger: logging.Logger, timeout: int = 300) -> Path:
    """Stream download to a .part file then atomically rename (never leaves partial raw)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with session().get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    os.replace(tmp, dest)
    logger.info("downloaded %s -> %s (%d bytes)", url, rel(dest), dest.stat().st_size)
    return dest


# ---------------------------------------------------------------- manifest

def read_manifest() -> list[dict]:
    if not MANIFEST.exists():
        return []
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def upsert_manifest(row: dict) -> None:
    """Insert or replace a manifest row keyed by (dataset_name, raw_file_path)."""
    rows = read_manifest()
    key = (row["dataset_name"], row["raw_file_path"])
    rows = [r for r in rows if (r["dataset_name"], r["raw_file_path"]) != key]
    # drop stale rows of the same dataset whose raw artifact no longer exists
    rows = [r for r in rows if r["dataset_name"] != row["dataset_name"] or (ROOT / r["raw_file_path"]).exists()]
    rows.append({k: ("" if row.get(k) is None else row.get(k)) for k in MANIFEST_FIELDS})
    rows.sort(key=lambda r: (r["dataset_category"], r["dataset_name"], r["raw_file_path"]))
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def manifest_row(dataset_name: str, raw_file_path: str) -> dict | None:
    for r in read_manifest():
        if r["dataset_name"] == dataset_name and r["raw_file_path"] == raw_file_path:
            return r
    return None


def load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass


def env(name: str) -> str | None:
    v = os.environ.get(name, "").strip()
    return v or None
