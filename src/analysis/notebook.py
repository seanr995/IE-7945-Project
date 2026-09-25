"""Generate and execute notebooks/01_data_exploration.ipynb (reads the DuckDB database only)."""
from __future__ import annotations

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

from src.utils.common import ROOT, get_logger

log = get_logger("analysis.notebook")
NB = ROOT / "notebooks" / "01_data_exploration.ipynb"


def md(s):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s):
    return nbf.v4.new_code_cell(s.strip())


CELLS = [
    md("""
# Sprint 1 - Job-Posting Corpus: Data Exploration

**Course:** IE7945 Workforce Analytics capstone - Sprint 1 ("assemble a shared job-posting corpus and agree on a data contract").

This notebook reads **only** `database/taxonomy.duckdb`, which `python run_pipeline.py` builds from preserved raw downloads.
Every number below is computed live from the database.

* **Empirical corpus:** public job postings (`job_postings`, raw records in `job_postings_raw`).
* **Reference taxonomies:** O\\*NET (`onet_*`) and ESCO (`esco_*`). They are *not* mapped to postings in Sprint 1.
"""),
    code("""
import sys, pathlib
ROOT = pathlib.Path.cwd().parent if pathlib.Path.cwd().name == 'notebooks' else pathlib.Path.cwd()
sys.path.insert(0, str(ROOT))
import duckdb, pandas as pd, matplotlib.pyplot as plt
from src.analysis.eda import style, BLUE, ORANGE, SOURCE_COLORS, INK, INK2, missingness, desc_stats
%matplotlib inline
pd.set_option('display.max_colwidth', 80); pd.set_option('display.width', 160)
con = duckdb.connect(str(ROOT / 'database' / 'taxonomy.duckdb'), read_only=True)
q = lambda sql: con.execute(sql).df()
plt.rcParams.update({'font.size': 10, 'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb'})
q("SELECT schema_name, table_name, estimated_size AS approx_rows FROM duckdb_tables() ORDER BY schema_name, table_name")
"""),
    md("## 0. Sources and provenance\nEvery source, its version, licence/terms status, and whether it is final or a temporary bootstrap."),
    code("q('SELECT dataset_name, role, version, record_count, status, final_or_temporary FROM source_metadata')"),
    md("## 1-5. Raw vs canonical vs unique postings, duplicates\n"
       "* **raw** = records exactly as delivered by the source (`job_postings_raw`).\n"
       "* **canonical** = one row per `posting_id` (identical source IDs repeated inside one snapshot collapsed).\n"
       "* **unique** = distinct `normalized_dup_key` (normalized title + company + description). "
       "Duplicates are **flagged, never deleted**."),
    code("""
counts = q('''
SELECT (SELECT count(*) FROM job_postings_raw) AS raw_postings,
       count(*) AS canonical_postings,
       count(DISTINCT exact_dup_key) AS unique_exact,
       count(DISTINCT normalized_dup_key) AS unique_postings,
       sum(is_normalized_duplicate::INT) AS duplicate_count,
       round(100*avg(is_normalized_duplicate::INT),2) AS duplicate_pct
FROM job_postings''')
counts.T.rename(columns={0: 'value'})
"""),
    code("""
q('''SELECT r.source, r.raw, c.canonical, c.unique_postings, c.duplicates, round(100*c.duplicates/c.canonical,2) AS dup_pct
FROM (SELECT source, count(*) raw FROM job_postings_raw GROUP BY 1) r
JOIN (SELECT source, count(*) canonical, count(DISTINCT normalized_dup_key) unique_postings,
             sum(is_normalized_duplicate::INT) duplicates FROM job_postings GROUP BY 1) c USING (source) ORDER BY 1''')
"""),
    md("**Why NYC has so many duplicates:** the City posts most vacancies twice - once *Internal* and once *External* - with the same Job ID and text. "
       "They are kept as separate canonical rows (different `source_job_id`) but share a `duplicate_group_id`."),
    code("""
q('''SELECT count(*) AS job_ids_posted_both_internal_and_external FROM (
       SELECT split_part(source_job_id,'-',1) jid FROM job_postings WHERE source='nyc_jobs'
       GROUP BY 1 HAVING count(DISTINCT split_part(source_job_id,'-',2)) = 2)''')
"""),
    code("""
d = q('''SELECT source, count(DISTINCT normalized_dup_key) uniq, sum(is_normalized_duplicate::INT)::INTEGER dups FROM job_postings GROUP BY 1 ORDER BY 1''')
fig, ax = plt.subplots(figsize=(8, 2.6))
ax.barh(d.source, d.uniq, color=BLUE, height=0.55, label='unique')
ax.barh(d.source, d.dups, left=d.uniq + 8, color='#86b6ef', height=0.55, label='duplicate')
for y, (u, du) in enumerate(zip(d.uniq, d.dups)):
    ax.text(u + du + 20, y, f'{du:,} dup / {u+du:,}', va='center', color=INK2, fontsize=9)
ax.set_xlim(0, (d.uniq + d.dups).max() * 1.3)
ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2); style(ax, 'Unique vs duplicate postings'); plt.tight_layout(); plt.show()
"""),
    md("## 6. Missingness (share of NULL per canonical field, by source)\nNULL means *not provided by the source* - nothing was imputed."),
    code("missingness(con)"),
    md("## 7. Postings by source"),
    code("q('SELECT source, count(*) postings, min(date_collected) collected FROM job_postings GROUP BY 1 ORDER BY 1')"),
    md("## 8. Top job titles (unique postings, normalized titles)"),
    code("q('''SELECT source, normalized_job_title, count(*) n FROM job_postings_unique GROUP BY ALL ORDER BY n DESC LIMIT 20''')"),
    code("q('SELECT source, count(DISTINCT normalized_job_title) distinct_titles, count(*) postings FROM job_postings GROUP BY 1')"),
    md("## 9. Industry\n"
       "**No source provides an employer industry field** (`industry` is 100% NULL). We did **not** infer it.\n\n"
       "* NYC postings are all City agencies, so `industry_derived = 'Public administration (City of New York agency)'` (rule-based, labelled derived).\n"
       "* NYC's `Job Category` is an **occupational/functional** category, not an industry; it is kept as `job_category_source`.\n"
       "* Arbeitnow provides free-form `tags` (kept in `job_category_source`), not an industry."),
    code("q('SELECT source, count(industry) industry_non_null, count(industry_derived) industry_derived_non_null, count(job_category_source) job_category_non_null, count(*) n FROM job_postings GROUP BY 1')"),
    code("""
cat = q('''SELECT job_category_source, count(*) n FROM job_postings_unique WHERE source='nyc_jobs' GROUP BY 1 ORDER BY 2 DESC LIMIT 12''').iloc[::-1]
fig, ax = plt.subplots(figsize=(8, 4.2)); ax.barh(cat.job_category_source.fillna('(missing)'), cat.n, color=BLUE, height=0.6)
for y, v in enumerate(cat.n): ax.text(v, y, f' {v}', va='center', color=INK2, fontsize=9)
style(ax, 'NYC job category (functional, source-provided) - unique postings'); plt.tight_layout(); plt.show()
"""),
    md("## 10. Seniority\n`seniority` = source value mapped to a common scale when available (`seniority_basis='source'`), otherwise a transparent "
       "title-keyword rule (`'title_rule'`), otherwise NULL. Raw source value is kept in `seniority_source`, rule output in `seniority_derived`."),
    code("q('''SELECT source, coalesce(seniority,'(unknown)') AS seniority, coalesce(seniority_basis,'none') AS basis, count(*) n FROM job_postings GROUP BY ALL ORDER BY source, n DESC''')"),
    code("""
sen = q("SELECT source, coalesce(seniority,'(unknown)') AS s, count(*) n FROM job_postings GROUP BY ALL")
order = ['intern_student','entry','experienced','senior','manager','executive','(unknown)']
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for ax, (src, g) in zip(axes, sen.groupby('source')):
    g = g.set_index('s').reindex(order).fillna(0)
    ax.barh(order, g.n, color=SOURCE_COLORS.get(src, BLUE), height=0.6)
    for y, v in enumerate(g.n):
        if v: ax.text(v, y, f' {int(v):,}', va='center', color=INK2, fontsize=9)
    style(ax, f'{src}: seniority')
axes[0].invert_yaxis(); plt.tight_layout(); plt.show()
"""),
    md("## 11. Description length (characters and words)"),
    code("desc_stats(con)"),
    code("""
df = q('SELECT source, description_word_count w FROM job_postings WHERE description_word_count IS NOT NULL')
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharex=True)
for ax, (src, g) in zip(axes, df.groupby('source')):
    ax.hist(g.w.clip(upper=2000), bins=40, color=SOURCE_COLORS.get(src, BLUE), edgecolor='#fcfcfb', linewidth=1)
    ax.axvline(g.w.median(), color=INK, linewidth=1, linestyle='--')
    ax.text(g.w.median(), ax.get_ylim()[1]*0.92, f' median {int(g.w.median())} words', color=INK, fontsize=9)
    style(ax, f'{src}: description words', 'words (clipped at 2,000)', 'postings')
plt.tight_layout(); plt.show()
"""),
    code("q('SELECT source, count(*) FILTER (WHERE description_word_count < 50) under_50_words, count(*) FILTER (WHERE description IS NULL) no_description, round(avg(posting_text_word_count),1) avg_full_text_words FROM job_postings GROUP BY 1')"),
    md("## 12. Language mix (langdetect, deterministic seed; ISO 639-1)"),
    code("q('''SELECT coalesce(language,'(undetected)') AS language, source, count(*) n, round(avg(language_confidence),3) avg_conf FROM job_postings GROUP BY ALL ORDER BY n DESC''')"),
    md("## 13. Geography\nNYC `country`/`state` are derived from source scope (municipal employer); `city` is not parsed. Arbeitnow only provides `location_raw` - country is **not** inferred."),
    code("q('''SELECT source, location_raw, count(*) n FROM job_postings GROUP BY source, location_raw QUALIFY row_number() OVER (PARTITION BY source ORDER BY count(*) DESC) <= 8 ORDER BY source, n DESC''')"),
    code("q('SELECT source, count(DISTINCT location_raw) distinct_locations, count(*) FILTER (WHERE remote_flag) remote_postings FROM job_postings GROUP BY 1')"),
    md("## 14. Date coverage"),
    code("q('SELECT source, min(date_posted) first_posted, max(date_posted) last_posted, median(date_posted) median_posted, count(date_posted) dated, min(date_collected) collected FROM job_postings GROUP BY 1')"),
    code("""
dd = q("SELECT source, date_trunc('month', date_posted) m, count(*) n FROM job_postings WHERE date_posted IS NOT NULL GROUP BY ALL ORDER BY m")
fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
for ax, (src, g) in zip(axes, dd.groupby('source')):
    ax.bar(g.m, g.n, width=20, color=SOURCE_COLORS.get(src, BLUE)); style(ax, f'{src}: postings by month posted', '', 'postings')
    ax.tick_params(axis='x', rotation=45)
plt.tight_layout(); plt.show()
"""),
    md("## 15. Source comparison"),
    code("""
q('''SELECT source, count(*) postings, count(DISTINCT company_name) employers,
       round(100*avg((salary_min IS NOT NULL)::INT),1) pct_salary, round(100*avg((requirements IS NOT NULL)::INT),1) pct_requirements,
       round(100*avg((seniority_basis='source')::INT),1) pct_seniority_from_source, median(description_word_count) median_words,
       string_agg(DISTINCT language, ',') languages FROM job_postings GROUP BY 1''')
"""),
    code("q(\"SELECT salary_period, salary_currency, count(*) n, median(salary_min) median_min, median(salary_max) median_max FROM job_postings WHERE salary_min IS NOT NULL GROUP BY ALL ORDER BY n DESC\")"),
    md("## 16. O\\*NET reference tables\nO\\*NET 31.0 restructured the content model: *Skills* are now **Essential Skills (2.A)** + **Transferable Skills (2.B)** "
       "(both loaded into `onet_skills`, distinguished by `source_file`), and *Technology Skills* became **Software Skills (2.E)** (`onet_technology_skills`)."),
    code("q(\"SELECT table_name, files, row_count, status FROM reference_load_report WHERE table_name LIKE 'onet%'\")"),
    code("q('SELECT source_file, count(*) AS n_rows, count(DISTINCT element_id) elements, count(DISTINCT onetsoc_code) occupations FROM onet_skills GROUP BY 1')"),
    md("## 17. ESCO reference tables"),
    code("q(\"SELECT dataset_name, version, status, final_or_temporary FROM source_metadata WHERE dataset_name LIKE 'esco%'\")"),
    code("q(\"SELECT table_name, files, row_count, status FROM reference_load_report WHERE table_name LIKE 'esco%'\")"),
    md("## 18. Data-quality findings (computed)"),
    code("""
f = []
c = counts.iloc[0]
f.append(f"{int(c.duplicate_count):,} of {int(c.canonical_postings):,} canonical postings ({c.duplicate_pct}%) are normalized duplicates.")
r = q("SELECT source, sum(is_normalized_duplicate::INT) d, count(*) n FROM job_postings GROUP BY 1")
for x in r.itertuples(): f.append(f"{x.source}: {int(x.d):,}/{x.n:,} duplicates ({100*x.d/x.n:.1f}%).")
f.append(f"{int(c.raw_postings - c.canonical_postings)} raw records repeated an existing source ID in the same snapshot and were collapsed.")
m = missingness(con)
f.append('Fields 100% NULL in every source: ' + ', '.join(m.index[(m == 1).all(axis=1)]))
s = q("SELECT count(*) n FROM job_postings WHERE seniority IS NULL").n[0]
f.append(f"{s:,} postings have no seniority from source or title rule.")
sh = q("SELECT count(*) n FROM job_postings WHERE coalesce(description_word_count,0) < 50").n[0]
f.append(f"{sh:,} postings have a description under 50 words.")
nl = q("SELECT count(*) n FROM job_postings WHERE language IS NOT NULL AND language NOT IN ('en','de')").n[0]
f.append(f"{nl:,} postings detected in a language other than en/de.")
for x in f: print('-', x)
"""),
    md("""
## 19. Limitations
* Both job sources are **temporary bootstrap sources** - instructor/team approval required before final corpus selection.
* NYC Jobs is **public-sector only and one city**; Arbeitnow is a **Germany-centric aggregator**. Neither is representative of the US or EU labour market.
* Snapshots are **point-in-time** (currently open postings), not a historical panel.
* USAJOBS / Adzuna were not pulled (no credentials configured); see `docs/USAJOBS_API_SETUP.md`.
* No source supplies an employer **industry**; NYC `job_category` is functional, Arbeitnow `tags` are free-form.
* Seniority for a large share of Arbeitnow postings (see section 10) comes from a keyword rule on titles (labelled `title_rule`), which is approximate.
* ESCO is loaded from the official ESCO **Web Services API** (v1.2.1) as a temporary fallback until the official CSV package is downloaded manually.
* Language detection (langdetect) is probabilistic; short or mixed-language postings can be misclassified.

## 20. Recommendations for Sprint 2
1. Get instructor approval for the corpus; add **USAJOBS** credentials to broaden US public-sector coverage, and consider Adzuna for private sector.
2. Download the official **ESCO v1.2.1 CSV** package (see `docs/ESCO_MANUAL_DOWNLOAD_REQUIRED.md`) - the pipeline swaps it in automatically.
3. Use `job_postings_unique` (one row per duplicate group) as the default extraction input to avoid double counting.
4. Both teams key everything on `posting_id`; the task team extracts from `description` (+`requirements`), the skill team from `posting_text`.
5. Decide language policy: English-only for the first extraction pass, or translate/handle German postings explicitly.
"""),
    code("con.close()"),
]


def build_and_execute(timeout: int = 600) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = CELLS
    nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    NB.parent.mkdir(exist_ok=True)
    ep = ExecutePreprocessor(timeout=timeout, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(NB.parent)}})
    nbf.write(nb, NB)
    errors = [o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.get("output_type") == "error"]
    if errors:
        raise RuntimeError(f"Notebook produced {len(errors)} error output(s)")
    log.info("Notebook executed without errors: %s", NB)
