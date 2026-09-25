"""Generate docs/data_contract.md, docs/data_sources.md, outputs/sprint1_summary.md,
outputs/meeting_brief.md from the database + computed metrics (no hand-typed numbers)."""
from __future__ import annotations

import duckdb

from src.cleaning.canonical import CONTRACT_VERSION, SENIORITY_LEVELS
from src.utils.common import DB_PATH, DOCS, OUTPUTS, ROOT, read_manifest, utc_now

S, D, SD = "source-provided", "derived", "source-provided (normalized)"

FIELDS = [
    # name, type, nullable, provenance, description, normalization rule
    ("posting_id", "VARCHAR(20)", "NO", D, "Deterministic primary key shared by both teams.",
     "first 20 hex of sha256('<source>|<source_job_id>'); if no source id: sha256('<source>|<title>|<company>|<description>')"),
    ("source", "VARCHAR", "NO", D, "Source system key (`nyc_jobs`, `arbeitnow`, later `usajobs`, `adzuna`).", "lower snake_case constant per connector"),
    ("source_job_id", "VARCHAR", "YES", S, "Identifier given by the source.", "NYC: `<Job ID>-<internal|external>` (Job ID alone is shared by the internal and external posting); Arbeitnow: `slug`"),
    ("source_record_ref", "VARCHAR", "NO", D, "Pointer to the exact raw record (file + row / array index).", "project-relative path + `#row=` / `#data[i]`"),
    ("retrieval_id", "VARCHAR", "NO", D, "Raw snapshot folder the record came from.", "`retrieval_<UTC timestamp>`"),
    ("job_title", "VARCHAR", "YES", SD, "Posting title as published.", "HTML stripped, whitespace collapsed, Unicode NFC"),
    ("normalized_job_title", "VARCHAR", "YES", D, "Title for grouping/matching.", "NFKC, lower-case, German gender tags (m/w/d) removed, punctuation except - / & + # . removed"),
    ("company_name", "VARCHAR", "YES", SD, "Employer (NYC: hiring City agency).", "whitespace/HTML cleanup only"),
    ("industry", "VARCHAR", "YES", S, "Employer industry **as provided by the source**. NULL for all current sources.", "never inferred"),
    ("industry_derived", "VARCHAR", "YES", D, "Rule-based industry where unambiguous.", "NYC -> 'Public administration (City of New York agency)'; otherwise NULL"),
    ("job_category_source", "VARCHAR", "YES", S, "Source's functional category/tags (NOT industry).", "NYC `Job Category`; Arbeitnow `tags` joined with '; '"),
    ("location_raw", "VARCHAR", "YES", S, "Location string as published.", "whitespace cleanup only"),
    ("country", "VARCHAR(2)", "YES", D, "ISO 3166-1 alpha-2.", "only when implied by the source scope (NYC -> US); see `geo_basis`"),
    ("state", "VARCHAR", "YES", D, "State/region code.", "only when implied by source scope (NYC -> NY)"),
    ("city", "VARCHAR", "YES", D, "City.", "not parsed in Sprint 1 (NULL)"),
    ("geo_basis", "VARCHAR", "NO", D, "How country/state/city were obtained.", "free text per source"),
    ("remote_flag", "BOOLEAN", "YES", S, "Remote flag if the source provides one.", "Arbeitnow `remote`; NULL otherwise"),
    ("employment_type", "VARCHAR", "YES", SD, "Full/part time, permanent, contract...", "NYC F/P -> full_time/part_time; Arbeitnow employment-type values of `job_types`"),
    ("seniority_source", "VARCHAR", "YES", S, "Raw seniority value(s) from the source.", "NYC `Career Level`; Arbeitnow seniority-bearing `job_types` values"),
    ("seniority_derived", "VARCHAR", "YES", D, "Seniority from title keywords.", "ordered regex rules in `src/cleaning/canonical.py` (TITLE_RULES)"),
    ("seniority", "VARCHAR", "YES", D, "Harmonized level: " + ", ".join(SENIORITY_LEVELS) + ".", "mapped source value if available, else `seniority_derived`, else NULL"),
    ("seniority_basis", "VARCHAR", "YES", D, "`source` | `title_rule` | NULL.", "records which path filled `seniority`"),
    ("description", "VARCHAR", "YES", SD, "Main posting body text.", "safe HTML->text (BeautifulSoup), entity unescape, NFC, whitespace collapse, paragraph breaks kept"),
    ("responsibilities", "VARCHAR", "YES", S, "Separate responsibilities section if the source has one.", "NULL for current sources (not split from description)"),
    ("requirements", "VARCHAR", "YES", S, "Separate requirements section.", "NYC `Minimum Qual Requirements`"),
    ("preferred_skills", "VARCHAR", "YES", S, "Separate preferred-skills section.", "NYC `Preferred Skills`"),
    ("salary_min", "DOUBLE", "YES", S, "Lower bound in `salary_currency` per `salary_period`.", "numeric parse only; not annualized"),
    ("salary_max", "DOUBLE", "YES", S, "Upper bound.", "numeric parse only; not annualized"),
    ("salary_currency", "VARCHAR(3)", "YES", D, "ISO 4217.", "NYC -> USD when a salary is present; NULL when no salary"),
    ("salary_period", "VARCHAR", "YES", SD, "annual | hourly | daily ...", "lower-cased source value"),
    ("date_posted", "DATE", "YES", SD, "Posting date.", "ISO-8601 (YYYY-MM-DD); NYC MM/DD/YYYY parsed; Arbeitnow epoch -> UTC date"),
    ("date_collected", "DATE", "NO", D, "Date the raw snapshot was retrieved (UTC).", "from retrieval_metadata.json"),
    ("language", "VARCHAR(2)", "YES", D, "Detected language, ISO 639-1.", "langdetect (seed 0) on first 3,000 chars of description (fallback title); NULL if <20 chars"),
    ("language_confidence", "DOUBLE", "YES", D, "langdetect probability of top language.", "rounded to 4 dp"),
    ("source_url", "VARCHAR", "YES", S, "Link to the posting.", "NULL if the source gives none (NYC export has no URL)"),
    ("posting_text", "VARCHAR", "YES", D, "Concatenation used for extraction.", "title + description + requirements + preferred_skills, blank-line separated"),
    ("description_char_count", "INTEGER", "YES", D, "Characters in `description`.", "len() after cleaning"),
    ("description_word_count", "INTEGER", "YES", D, "Whitespace tokens in `description`.", "split() after cleaning"),
    ("posting_text_word_count", "INTEGER", "YES", D, "Whitespace tokens in `posting_text`.", "split()"),
    ("exact_dup_key", "VARCHAR(32)", "NO", D, "Exact-duplicate key.", "sha256(title|company|description) of cleaned text"),
    ("normalized_dup_key", "VARCHAR(32)", "NO", D, "Normalized-duplicate key.", "sha256 of lower-cased alphanumeric-only normalized title, company, description"),
    ("duplicate_group_id", "VARCHAR(20)", "NO", D, "posting_id of the first member of the normalized duplicate group.", "ordering: source, date_posted, posting_id"),
    ("is_exact_duplicate", "BOOLEAN", "NO", D, "True for 2nd+ member of an exact group.", ""),
    ("is_normalized_duplicate", "BOOLEAN", "NO", D, "True for 2nd+ member of a normalized group.", "`job_postings_unique` = rows where this is false"),
    ("is_source_id_repeat", "BOOLEAN", "NO", D, "Always false in `job_postings` (repeats of the same posting_id in one snapshot are collapsed; counted in pipeline_runs).", ""),
    ("contract_version", "VARCHAR", "NO", D, "Data-contract version.", f"currently {CONTRACT_VERSION}"),
]


def _example(con, field: str) -> str:
    try:
        v = con.execute(f"SELECT {field} FROM job_postings WHERE {field} IS NOT NULL LIMIT 1").fetchone()
    except Exception:
        return ""
    if not v:
        return "NULL (not available)"
    s = str(v[0]).replace("\n", " ").replace("|", "/")
    return (s[:57] + "...") if len(s) > 60 else s


def data_contract(con) -> None:
    esc = lambda x: x.replace("|", "\\|")  # keep pipes from splitting markdown table cells
    rows = "\n".join(f"| `{f}` | {t} | {n} | {p} | {esc(d)} | {esc(r)} | {_example(con, f)} |" for f, t, n, p, d, r in FIELDS)
    (DOCS / "data_contract.md").write_text(f"""# Canonical Job-Posting Data Contract (v{CONTRACT_VERSION})

Shared by the **Task team** and the **Skill team**. Physical table: `job_postings` in
`database/taxonomy.duckdb` (also `data/processed/job_postings_canonical.parquet`).
Generated {utc_now()} by `src/analysis/reports.py`; examples are real values from the database.

## Why one contract

```
TASK TEAM : posting_id -> task extraction  -> O*NET mapping (onet_tasks / onet_work_activities)
SKILL TEAM: posting_id -> skill extraction -> ESCO / Lightcast mapping (esco_skills / onet_skills / onet_technology_skills)
```

Both teams' outputs join back on the **same `posting_id`** over the **same canonical corpus**, so
task and skill results can be compared per posting, per occupation and per source. A posting that
exists for one team must exist, with identical text, for the other.

## Global conventions

| Topic | Rule |
|---|---|
| Encoding | UTF-8 everywhere (files, DB, exports). Unicode NFC for text fields. |
| Missing values | SQL `NULL` = not available from the source. No empty strings, no 'N/A', no imputation. |
| Dates | ISO-8601 `YYYY-MM-DD` (`DATE`); timestamps in UTC `YYYY-MM-DDTHH:MM:SSZ`. |
| Language | ISO 639-1 two-letter code (`en`, `de`, `fr`), detected - see `language`. |
| Salary | As published: `salary_min`/`salary_max` in `salary_currency` (ISO 4217) per `salary_period`. Not annualized, not converted. NULL when not published. |
| Raw preservation | Raw files under `data/raw/jobs/<source>/retrieval_<ts>/` are never modified; every canonical row points to its raw record (`source_record_ref`); the full raw record is in `job_postings_raw.raw_record_json`. |
| Source provenance | `data/data_manifest.csv` (and DB tables `data_manifest`, `source_metadata`): URL, version, retrieval time, SHA-256, licence/terms, status. |
| Derived fields | Anything not literally provided by the source is either in a `*_derived` column or documented as `derived` below with its rule; `seniority_basis` / `geo_basis` state how a value was obtained. |
| Duplicates | Flagged, never deleted. Exact key = cleaned title+company+description; normalized key = alphanumeric-lower-case version. `job_postings_unique` view keeps the first row of each normalized group. Default extraction input = `job_postings_unique`. |
| posting_id | `sha256('<source>\|<source_job_id>')[:20]`, or `sha256('<source>\|<title>\|<company>\|<description>')[:20]` if the source has no ID. Stable across re-runs for the same source record. |
| Seniority scale | {", ".join(f"`{s}`" for s in SENIORITY_LEVELS)} (+ NULL = unknown). |
| Versioning | Breaking changes bump `contract_version`; both teams must agree before a change. |

## Fields

| field_name | type | nullable | source-provided / derived | description | normalization rule | example |
|---|---|---|---|---|---|---|
{rows}

## Companion tables

* `job_postings_raw(raw_record_id, source, retrieval_id, source_record_ref, raw_record_json)` - every raw record exactly as delivered.
* `onet_*` - O\\*NET reference (codes `onetsoc_code`, `element_id`, `scale_id`, original `data_value`, `onet_version`).
* `esco_*` - ESCO reference (`concept_uri` keys, `esco_version`, `esco_source_mode`).
* Downstream team outputs should be tables keyed by `posting_id` (e.g. `task_extractions(posting_id, task_text, onet_task_id, ...)`,
  `skill_extractions(posting_id, skill_text, esco_skill_uri, ...)`).
""", encoding="utf-8")


def data_sources(m: dict) -> None:
    man = read_manifest()
    lim = {
        "jobs_nyc_jobs": "Public sector only; one city; point-in-time snapshot of open postings; most vacancies appear twice (internal + external); no per-posting URL; no industry field.",
        "jobs_arbeitnow": "Aggregator of mostly Germany-based postings (German/English); ~100 postings/page, live ordering can repeat records across pages; no salary, no country field; employers are third parties.",
        "onet_database": "US occupational taxonomy; O*NET 31.0 renames Skills -> Essential + Transferable Skills and Technology Skills -> Software Skills.",
        "esco_api_fallback": "Temporary: derived from the ESCO Web Services API, not the official CSV package; columns are a subset (labels, descriptions, codes, relations, broader links).",
        "esco_classification_en_csv": "Official package.",
    }
    temp = {"jobs_nyc_jobs": "TEMPORARY bootstrap - instructor/team approval required before final corpus selection.",
            "jobs_arbeitnow": "TEMPORARY bootstrap - instructor/team approval required before final corpus selection.",
            "onet_database": "Final (official reference)",
            "esco_api_fallback": "TEMPORARY until official ESCO CSV is downloaded manually",
            "esco_classification_en_csv": "Final (official reference)"}
    blocks = []
    for r in man:
        blocks.append(f"""## {r['dataset_name']}

| | |
|---|---|
| Provider | {r['provider']} |
| Dataset | {r['dataset_name']} ({r['format']}) |
| Official URL | {r['official_source_url']} |
| Download URL | `{r['download_url']}` |
| Retrieval date (UTC) | {r['retrieval_timestamp_utc']} |
| Version | {r['version']} |
| Local raw path | `{r['raw_file_path']}` |
| Local extracted path | `{r['extracted_path'] or 'n/a (raw files read directly)'}` |
| Record count | {r.get('record_count', '')} |
| SHA-256 | `{r['sha256']}` |
| Size (bytes) | {r['file_size_bytes']} |
| License / terms | {r['license_or_terms']} |
| Status | {r['status']} |
| Limitations | {lim.get(r['dataset_name'], '')} |
| Final vs temporary | {temp.get(r['dataset_name'], 'requires verification')} |
| Notes | {r['notes']} |
""")
    (DOCS / "data_sources.md").write_text(f"""# Data Sources (generated {utc_now()})

Generated from `data/data_manifest.csv`. For directory snapshots (API pages), SHA-256 is a
deterministic digest over the sorted list of (relative path, file SHA-256).

**ESCO status:** {m['esco_status']}

{''.join(blocks)}
## Sources considered but not used

| Source | Why not (today) |
|---|---|
| USAJOBS Search API | Approved source, but requires an API key - none configured. See `docs/USAJOBS_API_SETUP.md`. |
| Adzuna API | Requires `ADZUNA_APP_ID` / `ADZUNA_APP_KEY` - none configured. |
| The Muse public API | Its developer docs state "Registration is required for any use beyond testing"; not registered, so not used for bulk collection. |
| ESCO official CSV package | Requires human acceptance of a privacy statement + email; see `docs/ESCO_MANUAL_DOWNLOAD_REQUIRED.md`. |
""", encoding="utf-8")


def _pct(x):
    return f"{100 * x:.1f}%"


def summary(m: dict) -> None:
    tc = m["table_counts"]
    by = {r["source"]: r for r in m["by_source"]}
    lang = ", ".join(f"{k} {v['n']:,} ({_pct(v['pct'])})" for k, v in m["language_mix"].items())
    sen = ", ".join(f"{k} {v:,}" for k, v in m["seniority"].items())
    ds = {r["source"]: r for r in m["description_stats"]}
    dsl = "\n".join(
        f"| {s} | {int(r['n']):,} | {r['words_mean']} | {r['words_median']} | {r['words_std']} | {r['words_min']} | "
        f"{r['words_p25']} | {r['words_p75']} | {r['words_p95']} | {r['words_max']} | {r['chars_mean']} | {r['chars_median']} |"
        for s, r in ds.items())
    miss = "\n".join(f"| {r['field']} | " + " | ".join(_pct(r[s]) for s in sorted(by)) + " |" for r in m["missingness"])
    onet = "\n".join(f"| {t} | {n:,} |" for t, n in tc.items() if t.startswith("onet_"))
    esco = "\n".join(f"| {t} | {n:,} |" for t, n in tc.items() if t.startswith("esco_")) or "| (none loaded) | 0 |"
    alltab = "\n".join(f"| {t} | {n:,} |" for t, n in tc.items())
    dates = "\n".join(f"| {s} | {d['min_posted']} | {d['max_posted']} | {d['median_posted']} | {d['collected']} |" for s, d in m["date_coverage"].items())
    locs = "; ".join(f"**{s}**: " + ", ".join(f"{k} ({v})" for k, v in list(d.items())[:5]) for s, d in m["top_locations"].items())
    nyc, arb = by.get("nyc_jobs", {}), by.get("arbeitnow", {})
    findings = [
        f"Overall normalized duplicate rate is **{_pct(m['duplicate_rate'])}** ({m['duplicates']:,} of {m['canonical_postings']:,} canonical postings); "
        f"NYC {_pct(nyc.get('duplicate_rate', 0))} vs Arbeitnow {_pct(arb.get('duplicate_rate', 0))}.",
        f"NYC duplicates are structural: {m['nyc_internal_external_pairs']:,} Job IDs are posted both *Internal* and *External* with the same text.",
        f"{m['source_id_repeats']} raw records repeated an existing source ID within the same snapshot (NYC export repeats + Arbeitnow page overlap) and were collapsed.",
        "No source provides employer **industry** (100% NULL); NYC `job_category` is functional, not industry; NYC industry_derived = public administration by rule.",
        "`responsibilities` and `city` are 100% NULL: not separable/parsable without extraction; left NULL by design.",
        f"Seniority basis: {', '.join(f'{k} {v:,}' for k, v in m['seniority_basis'].items())} (NULL = no source value and no title keyword).",
        f"{m['empty_or_short_descriptions']:,} postings have descriptions under 50 words.",
        f"Language mix is driven by source: NYC is English; Arbeitnow mixes German and English ({lang}).",
    ]
    (OUTPUTS / "sprint1_summary.md").write_text(f"""# Sprint 1 Summary - Shared Job-Posting Corpus

Generated {utc_now()} from `database/taxonomy.duckdb` (all values computed; see `outputs/tables/sprint1_metrics.json`).

## Job sources
| Source | Raw | Canonical | Unique | Duplicates | Dup. rate | Status |
|---|---|---|---|---|---|---|
""" + "\n".join(f"| {s} | {r['raw']:,} | {r['canonical']:,} | {r['unique_postings']:,} | {int(r['duplicates']):,} | {_pct(r['duplicate_rate'])} | Temporary bootstrap - instructor/team approval required |" for s, r in by.items()) + f"""

* NYC Jobs: City of New York (DCAS) via NYC Open Data - https://data.cityofnewyork.us/City-Government/Jobs-NYC-Postings/kpav-sd4t
* Arbeitnow: free public Job Board API - https://www.arbeitnow.com/api/job-board-api
* USAJOBS / Adzuna: not collected (credentials not configured).

## Headline counts
| Metric | Value |
|---|---|
| Raw posting records | {m['raw_postings']:,} |
| Canonical postings (`job_postings`) | {m['canonical_postings']:,} |
| Unique postings (normalized) | {m['unique_postings']:,} |
| Unique postings (exact) | {m['unique_postings_exact']:,} |
| Duplicate count (normalized) | {m['duplicates']:,} |
| Duplicate rate (of canonical) | {_pct(m['duplicate_rate'])} |
| Duplicate rate (of raw, incl. collapsed repeats) | {_pct(m['raw_duplicate_rate'])} |
| Distinct normalized titles | {m['distinct_normalized_titles']:,} |

## Language mix
{lang}

## Seniority distribution (harmonized)
{sen}

NYC source `Career Level`: {', '.join(f'{k} {v:,}' for k, v in m['nyc_career_level'].items())}

## Description length (words; chars at right)
| Source | n | mean | median | std | min | p25 | p75 | p95 | max | chars mean | chars median |
|---|---|---|---|---|---|---|---|---|---|---|---|
{dsl}

## Geographic & date coverage
| Source | First posted | Last posted | Median posted | Collected |
|---|---|---|---|---|
{dates}

Top locations - {locs}. Arbeitnow remote-flagged postings: {m['remote_arbeitnow']:,}.

## Missingness (NULL share by source)
| Field | {' | '.join(sorted(by))} |
|---|{'---|' * len(by)}
{miss}

## O*NET
Version **{m['onet_version']}** (official O*NET Resource Center, CC BY 4.0).

| Table | Rows |
|---|---|
{onet}

## ESCO
Version **{m['esco_version']}** - {m['esco_status']} (mode: {m['esco_mode']}).

| Table | Rows |
|---|---|
{esco}

## All DuckDB tables
| Table | Rows |
|---|---|
{alltab}

## Data-quality findings
""" + "\n".join(f"- {f}" for f in findings) + """

## Known limitations
- Both job sources are **temporary bootstrap sources - instructor/team approval required before final corpus selection.**
- NYC = public sector, single city; Arbeitnow = Germany-centric aggregator. Not representative of any national labour market.
- Point-in-time snapshots of open postings, not a time series.
- No employer industry in any source; seniority partly rule-derived; city not parsed.
- ESCO tables come from the official ESCO Web Services API (temporary) until the official CSV package is downloaded manually.
- Licence/terms for both job sources are marked "requires verification".

## Recommendations for Sprint 2
1. Instructor sign-off on corpus; add USAJOBS key (and optionally Adzuna) and re-run `python run_pipeline.py`.
2. Download official ESCO v1.2.1 CSV into `data/reference/esco/raw/` - loaded automatically on next run.
3. Use `job_postings_unique` as extraction input; key all outputs on `posting_id`.
4. Agree an English-first extraction pass (or explicit German handling) given the language mix.
5. Task team: segment `description`/`requirements` into task sentences -> O*NET `onet_tasks`; Skill team: extract from `posting_text` -> `esco_skills` / `onet_technology_skills`.
""", encoding="utf-8")

    (OUTPUTS / "meeting_brief.md").write_text(f"""# Sprint 1 - Meeting Brief

**Corpus:** {m['canonical_postings']:,} canonical postings ({m['raw_postings']:,} raw) from NYC Open Data Jobs + Arbeitnow API -> **{m['unique_postings']:,} unique** (duplicate rate {_pct(m['duplicate_rate'])}).
**Languages:** {', '.join(f"{k} {_pct(v['pct'])}" for k, v in list(m['language_mix'].items())[:3])}.
**References:** O*NET {m['onet_version']} loaded ({tc.get('onet_occupations', 0):,} occupations, {tc.get('onet_tasks', 0):,} tasks); ESCO {m['esco_version']} via official API fallback ({tc.get('esco_occupations', 0):,} occupations, {tc.get('esco_skills', 0):,} skills).
**Contract:** `docs/data_contract.md` v{CONTRACT_VERSION} - one `posting_id` for both teams.

**Decisions needed:** (1) approve bootstrap sources or supply USAJOBS key; (2) someone downloads the official ESCO CSV; (3) English-only vs bilingual extraction.

## 60-second spoken summary
"This sprint we built the shared data foundation. We pulled {m['raw_postings']:,} real job-posting records from two public
sources - the City of New York's open-data job postings and the Arbeitnow public jobs API - kept every raw file untouched,
and standardized them into {m['canonical_postings']:,} canonical postings with a stable posting ID. After flagging duplicates -
mostly New York posting the same job internally and externally - we have {m['unique_postings']:,} unique postings, a
{_pct(m['duplicate_rate'])} duplicate rate. The corpus is about {_pct(m['language_mix'].get('en', {}).get('pct', 0))} English, with German from Arbeitnow.
We loaded O*NET {m['onet_version']} and ESCO {m['esco_version']} as reference taxonomies into one DuckDB database, and wrote a data
contract so the task team and the skill team extract from exactly the same postings and join on the same ID. Both job sources are
temporary until you approve them, and the official ESCO package needs one manual download. Next sprint the teams start
extraction and mapping."
""", encoding="utf-8")


def generate(m: dict) -> None:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        data_contract(con)
    finally:
        con.close()
    data_sources(m)
    summary(m)
