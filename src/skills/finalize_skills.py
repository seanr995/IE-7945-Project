"""Apply skill_vocab decisions to posting_skills.csv and write the final skill file + vocabulary table.

    python -m src.skills.finalize_skills
"""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

from src.skills.skill_vocab import GENERIC, LANGUAGE, MERGE

csv.field_size_limit(10**9)
D = Path("outputs/skill_extraction")
COLUMNS = ["posting_id", "skill_text", "normalized_skill", "raw_normalized_skill", "type",
           "is_generic", "is_language", "section", "evidence_span"]


def main() -> None:
    with (D / "posting_skills.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    final = defaultdict(list)
    for r in rows:
        final[MERGE.get(r["normalized_skill"], r["normalized_skill"])].append(r)
    # one type per final skill: majority vote, ties go to "concept"
    ftype = {k: ("tool" if Counter(r["type"] for r in v)["tool"] > len(v) / 2 else "concept") for k, v in final.items()}
    out, seen, dup = [], set(), 0
    for r in sorted(rows, key=lambda r: (r["posting_id"], r["normalized_skill"])):
        n = MERGE.get(r["normalized_skill"], r["normalized_skill"])
        if (r["posting_id"], n) in seen:
            dup += 1
            continue
        seen.add((r["posting_id"], n))
        out.append({"posting_id": r["posting_id"], "skill_text": r["skill_text"], "normalized_skill": n,
                    "raw_normalized_skill": r["normalized_skill"], "type": ftype[n],
                    "is_generic": int(n in GENERIC), "is_language": int(n in LANGUAGE),
                    "section": r["section"], "evidence_span": r["evidence_span"]})
    with (D / "posting_skills_final.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(out)
    vocab = defaultdict(lambda: {"rows": 0, "postings": set(), "from": set()})
    for r in out:
        v = vocab[r["normalized_skill"]]
        v["rows"] += 1
        v["postings"].add(r["posting_id"])
        if r["raw_normalized_skill"] != r["normalized_skill"]:
            v["from"].add(r["raw_normalized_skill"])
    with (D / "skill_vocabulary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["normalized_skill", "type", "is_generic", "is_language", "n_postings", "merged_from"])
        for k, v in sorted(vocab.items(), key=lambda t: (-len(t[1]["postings"]), t[0])):
            w.writerow([k, ftype[k], int(k in GENERIC), int(k in LANGUAGE), len(v["postings"]), "; ".join(sorted(v["from"]))])
    gen = sum(r["is_generic"] for r in out)
    lang = sum(r["is_language"] for r in out)
    flagged = {r["normalized_skill"] for r in out if r["is_generic"] or r["is_language"]}
    print(f"rows {len(rows)} -> {len(out)} (duplicates after merge: {dup}); distinct skills "
          f"{len({r['raw_normalized_skill'] for r in out})} -> {len(vocab)}")
    print(f"generic rows {gen}, language rows {lang}, rows left for task co-occurrence {len(out) - gen - lang}")
    absent = sorted((GENERIC | LANGUAGE) - set(vocab))
    print("listed in vocab but absent from data:", absent)


if __name__ == "__main__":
    main()
