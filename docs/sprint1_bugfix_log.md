# Sprint 1 bug-fix log

## BUG-001 - entity-encoded HTML left literal tags in `description` (fixed 2026-09-25)

**Found during:** Sprint 2 section-detection design (heading inventory showed `ul`, `li`, `p`, `div`, `td` as "headings").

**Root cause.** Some Arbeitnow API records deliver HTML that is itself entity-encoded
(`&lt;div class=&quot;content-intro&quot;&gt;&lt;p&gt;...`, occasionally double/triple encoded).
`clean_text()` stripped tags *before* unescaping entities, so the decoded tags survived as literal
text. Raw files were never affected (they are preserved byte-for-byte).

**Verification.** Before the fix, 249 of 2,726 Arbeitnow canonical postings (9.1%) matched
`</?(ul|li|p|div|td|tr|br|strong|span)\b[^>]*>`; NYC 0. After the fix: 0 in either source.

**Fix.** `src/cleaning/canonical.py::clean_text` now alternates tag stripping and entity unescaping
until the text is stable (max 6 passes). Plain `<` / `&` in normal text are preserved
(unit-checked: `"salary < 50k"`, `"C++ & SQL"`).

**Effect on Sprint 1 metrics (re-run of `run_pipeline.py --offline`, same raw snapshots):**

| Metric | Before | After | Why |
|---|---|---|---|
| Raw postings | 5,595 | 5,595 | unchanged (raw data untouched) |
| Canonical postings | 5,497 | 5,497 | unchanged (posting_id depends on source ID only) |
| Unique postings (normalized) | 4,037 | **4,029** | 8 real duplicates were hidden because one copy was HTML-encoded and the other was not |
| Unique postings (exact) | 4,069 | 4,067 | same reason |
| Duplicate rate | 26.6% | **26.7%** | 1,460 -> 1,468 duplicates |
| Language en / de / fr | 4,211 / 1,225 / 61 | **4,207 / 1,229 / 61** | markup no longer dominates short German postings |
| Arbeitnow mean description words | 625.3 | 618.4 | tag tokens removed |
| O\*NET / ESCO table counts | - | unchanged | reference loading untouched |

`posting_id` values are unchanged for every posting (deterministic on `source|source_job_id`).
The Sprint 1 acceptance test still passes 21/21. `outputs/sprint1_summary.md` and the EDA notebook
were regenerated with the corrected values.
