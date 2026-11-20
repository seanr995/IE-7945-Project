"""Merge per-batch skill extractions, re-validate them against the posting text, write one clean CSV.

Python only: dedup, posting-ID matching, verbatim-evidence checks and section tagging happen here.

    python -m src.skills.merge_skills --batches out_1.json out_2.json ... --out outputs/skill_extraction/posting_skills.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

csv.field_size_limit(10**9)
POSTINGS = Path("outputs/skill_extraction/postings_188.csv")
COLUMNS = ["posting_id", "skill_text", "normalized_skill", "type", "section", "evidence_span"]
# Unambiguous spelling variants only. Judgement calls (organization vs organizational skills,
# leadership vs team leadership) are left alone for a later step.
MERGE = {"excel": "microsoft excel", "word": "microsoft word", "powerpoint": "microsoft powerpoint",
         "outlook": "microsoft outlook", "ai": "artificial intelligence", "llm": "large language model",
         "large language models": "large language model", "pims": "pim", "databases": "database",
         "investigations": "investigation", "data pipelines": "data pipeline",
         "reasonable accommodations": "reasonable accommodation",
         "data-driven decision-making": "data-driven decision making"}
SECTION = re.compile(r"^\[(TITLE|DESCRIPTION|REQUIREMENTS|PREFERRED SKILLS)\]$", re.M)


def section_of(text: str, span: str) -> str:
    pos = text.find(span)
    last = None
    for m in SECTION.finditer(text):
        if m.start() <= pos:
            last = m.group(1)
    return (last or "").lower().replace(" ", "_")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("outputs/skill_extraction/posting_skills.csv"))
    a = ap.parse_args()
    with POSTINGS.open(encoding="utf-8-sig", newline="") as f:
        text = {r["posting_id"]: r["skill_input_text"] for r in csv.DictReader(f)}
    rows, dropped, seen = [], Counter(), set()
    for b in a.batches:
        for s in json.loads(b.read_text(encoding="utf-8")):
            pid = s["posting_id"]
            norm = re.sub(r"\s+", " ", s["normalized_skill"].strip().lower())
            norm = MERGE.get(norm, norm)
            if pid not in text:
                dropped["unknown posting_id"] += 1
            elif s["evidence_span"] not in text[pid]:
                dropped["span not verbatim"] += 1
            elif s["skill_text"].lower() not in s["evidence_span"].lower():
                dropped["skill_text not in span"] += 1
            elif s["type"] not in ("tool", "concept"):
                dropped["bad type"] += 1
            elif not norm or (pid, norm) in seen:
                dropped["empty or duplicate"] += 1
            else:
                seen.add((pid, norm))
                rows.append({"posting_id": pid, "skill_text": s["skill_text"].strip(), "normalized_skill": norm,
                             "type": s["type"], "section": section_of(text[pid], s["evidence_span"]),
                             "evidence_span": s["evidence_span"]})
    rows.sort(key=lambda r: (r["posting_id"], r["normalized_skill"]))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    per = Counter(r["posting_id"] for r in rows)
    print(f"rows={len(rows)} postings_with_skills={len(per)}/{len(text)} dropped={dict(dropped)}")
    print("no skills:", sorted(set(text) - set(per)))


if __name__ == "__main__":
    main()
