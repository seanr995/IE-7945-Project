"""Step 1-2 of skill extraction: filter raw postings to the task-side posting_ids and build the text
that skills will be extracted from.

* The posting_ids come from task_hierarchy.csv (the final task taxonomy).
* A raw record is matched by recomputing posting_id exactly as src/cleaning/canonical.py does.
* Unlike task extraction, requirements and preferred skills are KEPT: skills mostly live there.
* Pure Python, no API calls.

    python -m src.skills.prepare_postings --hierarchy task_hierarchy.csv --raw-dir <unzipped jobs dir>
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.cleaning.canonical import clean_text, sha
from src.utils.common import get_logger

log = get_logger("skills.prepare_postings")
csv.field_size_limit(10**9)

OUT_COLUMNS = ["posting_id", "source", "source_job_id", "job_title", "company_name", "description",
               "requirements", "preferred_skills", "source_tags", "skill_input_text", "char_count"]


def task_posting_ids(hierarchy: Path) -> set[str]:
    with hierarchy.open(encoding="utf-8-sig", newline="") as f:
        return {r["posting_id"] for r in csv.DictReader(f)}


def _latest(root: Path, source: str) -> Path:
    runs = sorted(p for p in (root / source).glob("retrieval_*") if (p / ".complete").exists() or p.is_dir())
    return runs[-1]


def read_nyc(root: Path) -> list[dict]:
    f = _latest(root, "nyc_jobs") / "jobs_nyc_postings.csv"
    out = []
    with f.open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            g = lambda k: (r.get(k) or "").strip() or None
            jid = f"{g('Job ID')}-{(g('Posting Type') or 'NA').lower()}"
            out.append({"source": "nyc_jobs", "source_job_id": jid, "job_title": clean_text(g("Business Title")),
                        "company_name": clean_text(g("Agency")), "description": clean_text(g("Job Description")),
                        "requirements": clean_text(g("Minimum Qual Requirements")),
                        "preferred_skills": clean_text(g("Preferred Skills")), "source_tags": None})
    return out


def read_arbeitnow(root: Path) -> list[dict]:
    out = []
    for p in sorted(_latest(root, "arbeitnow").glob("page_*.json")):
        for x in json.loads(p.read_text(encoding="utf-8")).get("data", []):
            out.append({"source": "arbeitnow", "source_job_id": x.get("slug"),
                        "job_title": clean_text(x.get("title")), "company_name": clean_text(x.get("company_name")),
                        "description": clean_text(x.get("description")), "requirements": None,
                        "preferred_skills": None, "source_tags": "; ".join(x.get("tags") or []) or None})
    return out


def skill_input_text(r: dict) -> str:
    """Labelled blocks; evidence spans must be verbatim substrings of this text."""
    parts = [("TITLE", r["job_title"]), ("DESCRIPTION", r["description"]),
             ("REQUIREMENTS", r["requirements"]), ("PREFERRED SKILLS", r["preferred_skills"])]
    return "\n\n".join(f"[{k}]\n{v}" for k, v in parts if v)


def build(hierarchy: Path, raw_dir: Path) -> tuple[list[dict], dict]:
    ids = task_posting_ids(hierarchy)
    found: dict[str, dict] = {}
    seen = 0
    for rows in (read_nyc(raw_dir), read_arbeitnow(raw_dir)):
        for r in rows:
            seen += 1
            pid = sha(f"{r['source']}|{r['source_job_id']}")
            if pid in ids and pid not in found:
                r["posting_id"] = pid
                r["skill_input_text"] = skill_input_text(r)
                r["char_count"] = len(r["skill_input_text"])
                found[pid] = r
    missing = sorted(ids - set(found))
    stats = {"task_posting_ids": len(ids), "raw_records_scanned": seen, "matched": len(found),
             "missing": missing, "by_source": {s: sum(1 for r in found.values() if r["source"] == s)
                                               for s in ("nyc_jobs", "arbeitnow")}}
    return [found[i] for i in sorted(found)], stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hierarchy", type=Path, required=True)
    ap.add_argument("--raw-dir", type=Path, required=True, help="folder that contains arbeitnow/ and nyc_jobs/")
    ap.add_argument("--out", type=Path, default=Path("outputs/skill_extraction/postings_188.csv"))
    a = ap.parse_args()
    rows, stats = build(a.hierarchy, a.raw_dir)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    chars = sorted(r["char_count"] for r in rows)
    stats["char_count"] = {"min": chars[0], "median": chars[len(chars) // 2], "max": chars[-1]}
    (a.out.parent / "postings_188_report.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    log.info("%s", stats)
    if stats["missing"]:
        raise SystemExit(f"{len(stats['missing'])} posting_ids not found in raw data")


if __name__ == "__main__":
    main()
