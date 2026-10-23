"""Freeze the Sprint 3 clustering input. SPRINT 3 - REPRESENTATION & CLUSTERING (input only; no embeddings here).

    python -m src.prototype.sprint3_freeze

Reads task_pipeline.sprint3_task_input (+ the Sprint 2 Groq wording of cross-model-agreed Gemini rows) READ-ONLY and
writes the frozen experiment file used by all four Colab experiments:
  data/experiment/sprint3_task_input.csv            frozen input (rows after exact-duplicate removal)
  data/experiment/sprint3_duplicate_provenance.csv  every removed duplicate -> the statement_id it was folded into
  data/experiment/sprint3_task_input.sha256         SHA-256 of the frozen CSV
  colab/sprint3_task_input.csv                      byte-identical copy for upload to Google Drive

Duplicates: exact duplicates of a minimally normalised statement_text (lower-case, trimmed, whitespace collapsed,
curly quotes/dashes unified, trailing . ; : , dropped). The lowest statement_id of each group is kept; the others are
removed from the experiment input only and stay traceable through duplicate_* columns + the provenance file.
Paraphrases are NOT merged.
"""
from __future__ import annotations

import hashlib
import re
import shutil

import duckdb
import pandas as pd

from src.utils.common import DATA, DB_PATH, ROOT

OUT = DATA / "experiment" / "sprint3_task_input.csv"
PROV = DATA / "experiment" / "sprint3_duplicate_provenance.csv"
COLAB = ROOT / "colab" / "sprint3_task_input.csv"
_Q = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", " ": " "})


def norm_dup(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").translate(_Q).lower().strip()).rstrip(" .;:,")


def load(con) -> pd.DataFrame:
    return con.execute("""
        SELECT s.statement_id, s.posting_id, s.provider, s.statement_text, s.evidence_span, s.source_section,
               s.evidence_valid, s.confirming_provider, s.phase, s.filter_version,
               CASE WHEN s.confirming_provider = 'groq' THEN p.paired_statement_text END AS cross_provider_paired_text
        FROM task_pipeline.sprint3_task_input s
        JOIN task_pipeline.task_statements t ON t.task_statement_id = s.statement_id
        LEFT JOIN prototype.prototype_statements p ON p.statement_id = t.source_statement_id
        ORDER BY s.statement_id""").df()


def freeze(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    assert d.statement_id.is_unique, "statement_id not unique"
    assert (d.statement_text.fillna("").str.strip() != "").all(), "blank statement_text"
    d = d.sort_values("statement_id").reset_index(drop=True)
    d["normalized_text"] = d.statement_text.map(norm_dup)
    g = d.groupby("normalized_text", sort=False)
    d["duplicate_group_size"] = g.statement_id.transform("size")
    for col, src in (("duplicate_statement_ids", "statement_id"), ("duplicate_posting_ids", "posting_id"),
                     ("duplicate_providers", "provider")):
        d[col] = g[src].transform(lambda x: ";".join(x))
    keep = ~d.duplicated("normalized_text", keep="first")
    kept_id = d[keep].set_index("normalized_text").statement_id
    prov = d[~keep].assign(kept_statement_id=lambda x: x.normalized_text.map(kept_id))[
        ["statement_id", "posting_id", "provider", "statement_text", "normalized_text", "kept_statement_id"]]
    out = d[keep].reset_index(drop=True)
    stats = {"database_rows": len(d), "exact_duplicate_rows_removed": int((~keep).sum()),
             "duplicate_groups": int((d[keep].duplicate_group_size > 1).sum()),
             "cross_provider_duplicate_groups": int((out.duplicate_providers.str.split(";").map(set).map(len) > 1).sum()),
             "within_posting_duplicates": int(d.duplicated(["posting_id", "normalized_text"]).sum()),
             "final_rows": len(out), "unique_postings": int(out.posting_id.nunique()),
             "providers": out.provider.value_counts().to_dict(), "sections": out.source_section.value_counts().to_dict(),
             "rows_with_cross_provider_paired_text": int(out.cross_provider_paired_text.notna().sum())}
    return out, prov, stats


def main():
    con = duckdb.connect(str(DB_PATH), read_only=True)
    d = load(con)
    con.close()
    out, prov, stats = freeze(d)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False, encoding="utf-8", lineterminator="\n")
    prov.to_csv(PROV, index=False, encoding="utf-8", lineterminator="\n")
    sha = hashlib.sha256(OUT.read_bytes()).hexdigest()
    OUT.with_suffix(".sha256").write_text(f"{sha}  {OUT.name}\n", encoding="utf-8")
    COLAB.parent.mkdir(exist_ok=True)
    shutil.copyfile(OUT, COLAB)
    assert hashlib.sha256(COLAB.read_bytes()).hexdigest() == sha
    print({**stats, "sha256": sha})


if __name__ == "__main__":
    main()
