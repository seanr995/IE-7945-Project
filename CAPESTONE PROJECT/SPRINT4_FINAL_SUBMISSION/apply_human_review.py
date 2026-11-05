#!/usr/bin/env python
"""Ingest the completed Sprint 4 human review.

Usage (from CAPESTONE PROJECT/ or anywhere):
    python apply_human_review.py [path/to/sprint4_results] [--review other_template.csv]

Reads  sprint4_results/human_review_template.csv  (human columns filled by teammates)
Writes sprint4_results/human_review_summary.md  and  sprint4_results/human_review_issues.csv

READ-ONLY with respect to the frozen taxonomy: leaf memberships (task_hierarchy.csv and the frozen Sprint 3 membership file)
are only read and compared; nothing is moved, renamed or overwritten. Suggested renames are REPORTED, never applied.
If no human judgment is present the summary says HUMAN REVIEW PENDING; nothing is fabricated. No API calls, no network.
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

HUMAN = ["label_correct", "coherent", "boundary_ok", "over_merged", "under_merged", "suggested_name", "reviewer_notes"]
YES = {"y", "yes", "true", "1", "correct", "ok", "pass"}
NO = {"n", "no", "false", "0", "incorrect", "fail"}


def tri(v):
    """True / False / None (blank or unrecognised)."""
    if pd.isna(v):
        return None
    s = str(v).strip().lower()
    if s in YES:
        return True
    if s in NO:
        return False
    return None


def filled(v):
    return not pd.isna(v) and str(v).strip() != ""


def main(argv):
    args = [a for a in argv if not a.startswith("--")]
    here = Path(__file__).resolve().parent
    res = Path(args[0]) if args else here / "sprint4_results"
    if not args and not res.exists():
        res = here          # flat submission folder
    review = res / "human_review_template.csv"
    if "--review" in argv:
        review = Path(argv[argv.index("--review") + 1])
    df = pd.read_csv(review, dtype=str)
    missing = [c for c in ["topic_id", "topic_name"] + HUMAN if c not in df.columns]
    if missing:
        sys.exit(f"review file lacks columns: {missing}")

    # ---- frozen-membership guard (read-only comparison)
    frozen_ok, frozen_note = None, "frozen file not found; guard skipped"
    th = res / "task_hierarchy.csv"
    fr = here / "final_taxonomy_results" / "final_deliverable" / "final_task_topic_membership_leiden80.csv"
    if th.exists() and fr.exists():
        a = pd.read_csv(fr)[["statement_id", "topic_id"]].set_index("statement_id").topic_id
        b = pd.read_csv(th)[["statement_id", "leaf_topic_id"]].set_index("statement_id").leaf_topic_id
        frozen_ok = len(a) == len(b) == 3015 and a.nunique() == 80 and (a.sort_index() == b.reindex(a.index).sort_index()).all()
        frozen_note = f"{len(a)} tasks / {a.nunique()} leaves; identical to task_hierarchy.csv: {bool(frozen_ok)}"

    n = len(df)
    any_judgment = df[HUMAN[:5]].apply(lambda col: col.map(filled)).any(axis=1)
    reviewed = df[df[HUMAN[:5]].apply(lambda col: col.map(filled)).all(axis=1)]
    partial = df[any_judgment & ~df.index.isin(reviewed.index)]
    pending = not bool(any_judgment.any())

    T = {c: df[c].map(tri) for c in HUMAN[:5]}
    cnt = lambda c, val: int((T[c] == val).sum())
    issues = []
    for i, r in df.iterrows():
        flags = []
        if T["label_correct"][i] is False: flags.append("LABEL_NEEDS_RENAME")
        if T["coherent"][i] is False: flags.append("MIXED_OR_INCOHERENT")
        if T["boundary_ok"][i] is False: flags.append("BOUNDARY_ISSUE")
        if T["over_merged"][i] is True: flags.append("OVER_MERGED")
        if T["under_merged"][i] is True: flags.append("UNDER_MERGED")
        if filled(r["suggested_name"]): flags.append("SUGGESTED_RENAME")
        if flags:
            issues.append({"topic_id": r["topic_id"], "topic_name": r["topic_name"], "issue_types": ";".join(flags),
                           "suggested_name": r["suggested_name"] if filled(r["suggested_name"]) else "",
                           "reviewer_notes": r["reviewer_notes"] if filled(r["reviewer_notes"]) else ""})
    iss_cols = ["topic_id", "topic_name", "issue_types", "suggested_name", "reviewer_notes"]
    pd.DataFrame(issues, columns=iss_cols).to_csv(res / "human_review_issues.csv", index=False)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    L = [f"# Sprint 4 Human Review Summary", "", f"Generated {ts} from `{review.name}`.", ""]
    if pending:
        L += ["## HUMAN REVIEW PENDING", "",
              "No human judgment has been entered in the review template (all five judgment columns are blank). "
              "No review result is reported and nothing has been inferred. Fill `human_review_template.csv` and re-run this script.", ""]
    L += ["## Status", "",
          f"- Topics in file: **{n}**",
          f"- Fully reviewed (all 5 judgments entered): **{len(reviewed)}**",
          f"- Partially reviewed: **{len(partial)}**",
          f"- Not started: **{n - len(reviewed) - len(partial)}**",
          f"- Review complete: **{'YES' if len(reviewed) == n else 'NO'}**",
          f"- Frozen-membership guard: {frozen_note}", ""]
    if not pending:
        L += ["## Judgments (counts among topics where the cell was filled)", "",
              "| Question | Yes | No | Blank/unrecognised |", "|---|---:|---:|---:|"]
        q = [("Label correct (accepted)", "label_correct"), ("Representative tasks coherent", "coherent"),
             ("Boundary tasks reasonably belong", "boundary_ok"), ("Over-merged", "over_merged"), ("Under-merged", "under_merged")]
        for lab, c in q:
            L.append(f"| {lab} | {cnt(c, True)} | {cnt(c, False)} | {n - cnt(c, True) - cnt(c, False)} |")
        L += ["", f"- Labels accepted: **{cnt('label_correct', True)}**; needing rename: **{cnt('label_correct', False)}**",
              f"- Coherent topics: **{cnt('coherent', True)}**; mixed/incoherent: **{cnt('coherent', False)}**",
              f"- Boundary issues: **{cnt('boundary_ok', False)}**",
              f"- Over-merged topics: **{cnt('over_merged', True)}**; under-merged topics: **{cnt('under_merged', True)}**", ""]
        for title, key in [("Over-merged topics", ("over_merged", True)), ("Under-merged topics", ("under_merged", True)),
                           ("Boundary issues", ("boundary_ok", False)), ("Mixed / incoherent topics", ("coherent", False))]:
            idx = [i for i in df.index if T[key[0]][i] is key[1]]
            L.append(f"### {title} ({len(idx)})")
            L += [f"- {df.topic_id[i]} {df.topic_name[i]}" + (f" - {df.reviewer_notes[i]}" if filled(df.reviewer_notes[i]) else "") for i in idx] or ["- none"]
            L.append("")
        ren = df[df.suggested_name.map(filled)]
        L.append(f"### Suggested renames ({len(ren)}) - reported only, NOT applied")
        L += [f"- {r.topic_id}: **{r.topic_name}** -> **{r.suggested_name}**" for r in ren.itertuples()] or ["- none"]
        L.append("")
        notes = df[df.reviewer_notes.map(filled)]
        L.append(f"### Reviewer notes ({len(notes)})")
        L += [f"- {r.topic_id} {r.topic_name}: {r.reviewer_notes}" for r in notes.itertuples()] or ["- none"]
        L.append("")
    L += ["---", "This script never edits leaf memberships, names or hierarchy files; applying renames or merges is a separate, deliberate step."]
    (res / "human_review_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("HUMAN REVIEW PENDING" if pending else f"reviewed {len(reviewed)}/{n}; issues rows: {len(issues)}")
    if frozen_ok is False:
        sys.exit("WARNING: frozen membership mismatch detected")


if __name__ == "__main__":
    main(sys.argv[1:])
