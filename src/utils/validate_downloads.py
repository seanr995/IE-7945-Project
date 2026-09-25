"""Phase 2: verify every artifact recorded in data/data_manifest.csv."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

from src.utils.common import ROOT, looks_like_html, read_manifest, sha256_dir, sha256_file, validate_zip


def _check(row: dict) -> dict:
    p = ROOT / row["raw_file_path"]
    res = {"Dataset": row["dataset_name"], "Version": row["version"], "Raw path": row["raw_file_path"],
           "Size": 0, "SHA256": (row["sha256"] or "")[:16] + "...", "Extracted?": "n/a",
           "Records/files": row.get("record_count", ""), "Status": "FAIL"}
    if not p.exists():
        res["Status"] = "FAIL: missing"
        return res
    if p.is_file():
        res["Size"] = p.stat().st_size
        if res["Size"] == 0:
            res["Status"] = "FAIL: zero bytes"
            return res
        if p.suffix.lower() == ".zip":
            ok, msg = validate_zip(p)
            if not ok:
                res["Status"] = f"FAIL: {msg}"
                return res
        actual = sha256_file(p)
    else:
        files = [f for f in p.rglob("*") if f.is_file()]
        res["Size"] = sum(f.stat().st_size for f in files)
        if not files or res["Size"] == 0:
            res["Status"] = "FAIL: empty directory"
            return res
        for f in files:  # readability + HTML-masquerade checks
            if f.suffix == ".gz":
                json.loads(gzip.open(f).read())
            elif f.suffix == ".json":
                json.loads(f.read_bytes())
            elif f.suffix == ".csv" and looks_like_html(f):
                res["Status"] = f"FAIL: HTML saved as {f.name}"
                return res
        actual = sha256_dir(p)
    if actual != row["sha256"]:
        res["Status"] = "FAIL: sha256 mismatch vs manifest"
        return res
    if row.get("extracted_path"):
        e = ROOT / row["extracted_path"]
        res["Extracted?"] = "yes" if e.exists() and any(e.iterdir()) else "NO"
    res["Status"] = "PASS" if res["Extracted?"] != "NO" else "FAIL: not extracted"
    return res


def validate_all(print_table: bool = True) -> list[dict]:
    results = [_check(r) for r in read_manifest()]
    if print_table:
        import pandas as pd
        df = pd.DataFrame(results)
        df["Size"] = df["Size"].map(lambda b: f"{b/1e6:,.2f} MB")
        with pd.option_context("display.max_colwidth", 60, "display.width", 250):
            print(df.to_string(index=False))
    return results
