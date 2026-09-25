# Canonical Job-Posting Data Contract (v1.0)

Shared by the **Task team** and the **Skill team**. Physical table: `job_postings` in
`database/taxonomy.duckdb` (also `data/processed/job_postings_canonical.parquet`).
Generated 2026-09-25T17:32:39Z by `src/analysis/reports.py`; examples are real values from the database.

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
| Seniority scale | `intern_student`, `entry`, `experienced`, `senior`, `manager`, `executive` (+ NULL = unknown). |
| Versioning | Breaking changes bump `contract_version`; both teams must agree before a change. |

## Fields

| field_name | type | nullable | source-provided / derived | description | normalization rule | example |
|---|---|---|---|---|---|---|
| `posting_id` | VARCHAR(20) | NO | derived | Deterministic primary key shared by both teams. | first 20 hex of sha256('<source>\|<source_job_id>'); if no source id: sha256('<source>\|<title>\|<company>\|<description>') | 0473e8ccf7ec305122ac |
| `source` | VARCHAR | NO | derived | Source system key (`nyc_jobs`, `arbeitnow`, later `usajobs`, `adzuna`). | lower snake_case constant per connector | arbeitnow |
| `source_job_id` | VARCHAR | YES | source-provided | Identifier given by the source. | NYC: `<Job ID>-<internal\|external>` (Job ID alone is shared by the internal and external posting); Arbeitnow: `slug` | senior-architektin-schwerpunkt-entwurf-angebotsplanung-ma... |
| `source_record_ref` | VARCHAR | NO | derived | Pointer to the exact raw record (file + row / array index). | project-relative path + `#row=` / `#data[i]` | data/raw/jobs/arbeitnow/retrieval_20260925T052546Z/page_0... |
| `retrieval_id` | VARCHAR | NO | derived | Raw snapshot folder the record came from. | `retrieval_<UTC timestamp>` | retrieval_20260925T052546Z |
| `job_title` | VARCHAR | YES | source-provided (normalized) | Posting title as published. | HTML stripped, whitespace collapsed, Unicode NFC | Senior Architekt:in (m/w/d) – Schwerpunkt Entwurf: Angebo... |
| `normalized_job_title` | VARCHAR | YES | derived | Title for grouping/matching. | NFKC, lower-case, German gender tags (m/w/d) removed, punctuation except - / & + # . removed | senior architekt in schwerpunkt entwurf angebotsplanung /... |
| `company_name` | VARCHAR | YES | source-provided (normalized) | Employer (NYC: hiring City agency). | whitespace/HTML cleanup only | gropyus |
| `industry` | VARCHAR | YES | source-provided | Employer industry **as provided by the source**. NULL for all current sources. | never inferred | NULL (not available) |
| `industry_derived` | VARCHAR | YES | derived | Rule-based industry where unambiguous. | NYC -> 'Public administration (City of New York agency)'; otherwise NULL | Public administration (City of New York agency) |
| `job_category_source` | VARCHAR | YES | source-provided | Source's functional category/tags (NOT industry). | NYC `Job Category`; Arbeitnow `tags` joined with '; ' | Execution |
| `location_raw` | VARCHAR | YES | source-provided | Location string as published. | whitespace cleanup only | Berlin, Berlin |
| `country` | VARCHAR(2) | YES | derived | ISO 3166-1 alpha-2. | only when implied by the source scope (NYC -> US); see `geo_basis` | US |
| `state` | VARCHAR | YES | derived | State/region code. | only when implied by source scope (NYC -> NY) | NY |
| `city` | VARCHAR | YES | derived | City. | not parsed in Sprint 1 (NULL) | NULL (not available) |
| `geo_basis` | VARCHAR | NO | derived | How country/state/city were obtained. | free text per source | location_raw only; country not provided by source (not in... |
| `remote_flag` | BOOLEAN | YES | source-provided | Remote flag if the source provides one. | Arbeitnow `remote`; NULL otherwise | False |
| `employment_type` | VARCHAR | YES | source-provided (normalized) | Full/part time, permanent, contract... | NYC F/P -> full_time/part_time; Arbeitnow employment-type values of `job_types` | fulltime permanent |
| `seniority_source` | VARCHAR | YES | source-provided | Raw seniority value(s) from the source. | NYC `Career Level`; Arbeitnow seniority-bearing `job_types` values | berufserfahren |
| `seniority_derived` | VARCHAR | YES | derived | Seniority from title keywords. | ordered regex rules in `src/cleaning/canonical.py` (TITLE_RULES) | senior |
| `seniority` | VARCHAR | YES | derived | Harmonized level: intern_student, entry, experienced, senior, manager, executive. | mapped source value if available, else `seniority_derived`, else NULL | senior |
| `seniority_basis` | VARCHAR | YES | derived | `source` \| `title_rule` \| NULL. | records which path filled `seniority` | title_rule |
| `description` | VARCHAR | YES | source-provided (normalized) | Main posting body text. | safe HTML->text (BeautifulSoup), entity unescape, NFC, whitespace collapse, paragraph breaks kept | About The Company  GROPYUS is a technology-based construc... |
| `responsibilities` | VARCHAR | YES | source-provided | Separate responsibilities section if the source has one. | NULL for current sources (not split from description) | NULL (not available) |
| `requirements` | VARCHAR | YES | source-provided | Separate requirements section. | NYC `Minimum Qual Requirements` | 1. A baccalaureate degree from an accredited college and ... |
| `preferred_skills` | VARCHAR | YES | source-provided | Separate preferred-skills section. | NYC `Preferred Skills` | - Strong interpersonal and communication skills, to devel... |
| `salary_min` | DOUBLE | YES | source-provided | Lower bound in `salary_currency` per `salary_period`. | numeric parse only; not annualized | 68339.0 |
| `salary_max` | DOUBLE | YES | source-provided | Upper bound. | numeric parse only; not annualized | 97000.0 |
| `salary_currency` | VARCHAR(3) | YES | derived | ISO 4217. | NYC -> USD when a salary is present; NULL when no salary | USD |
| `salary_period` | VARCHAR | YES | source-provided (normalized) | annual \| hourly \| daily ... | lower-cased source value | annual |
| `date_posted` | DATE | YES | source-provided (normalized) | Posting date. | ISO-8601 (YYYY-MM-DD); NYC MM/DD/YYYY parsed; Arbeitnow epoch -> UTC date | 2026-09-20 |
| `date_collected` | DATE | NO | derived | Date the raw snapshot was retrieved (UTC). | from retrieval_metadata.json | 2026-09-25 |
| `language` | VARCHAR(2) | YES | derived | Detected language, ISO 639-1. | langdetect (seed 0) on first 3,000 chars of description (fallback title); NULL if <20 chars | de |
| `language_confidence` | DOUBLE | YES | derived | langdetect probability of top language. | rounded to 4 dp | 1.0 |
| `source_url` | VARCHAR | YES | source-provided | Link to the posting. | NULL if the source gives none (NYC export has no URL) | https://www.arbeitnow.com/jobs/companies/gropyus/senior-a... |
| `posting_text` | VARCHAR | YES | derived | Concatenation used for extraction. | title + description + requirements + preferred_skills, blank-line separated | Senior Architekt:in (m/w/d) – Schwerpunkt Entwurf: Angebo... |
| `description_char_count` | INTEGER | YES | derived | Characters in `description`. | len() after cleaning | 5071 |
| `description_word_count` | INTEGER | YES | derived | Whitespace tokens in `description`. | split() after cleaning | 634 |
| `posting_text_word_count` | INTEGER | YES | derived | Whitespace tokens in `posting_text`. | split() | 643 |
| `exact_dup_key` | VARCHAR(32) | NO | derived | Exact-duplicate key. | sha256(title\|company\|description) of cleaned text | 306073922775285ebdf418cda6a087c2 |
| `normalized_dup_key` | VARCHAR(32) | NO | derived | Normalized-duplicate key. | sha256 of lower-cased alphanumeric-only normalized title, company, description | 4a9995da61121269c239bca1ace7996a |
| `duplicate_group_id` | VARCHAR(20) | NO | derived | posting_id of the first member of the normalized duplicate group. | ordering: source, date_posted, posting_id | 0473e8ccf7ec305122ac |
| `is_exact_duplicate` | BOOLEAN | NO | derived | True for 2nd+ member of an exact group. |  | False |
| `is_normalized_duplicate` | BOOLEAN | NO | derived | True for 2nd+ member of a normalized group. | `job_postings_unique` = rows where this is false | False |
| `is_source_id_repeat` | BOOLEAN | NO | derived | Always false in `job_postings` (repeats of the same posting_id in one snapshot are collapsed; counted in pipeline_runs). |  | False |
| `contract_version` | VARCHAR | NO | derived | Data-contract version. | currently 1.0 | 1.0 |

## Companion tables

* `job_postings_raw(raw_record_id, source, retrieval_id, source_record_ref, raw_record_json)` - every raw record exactly as delivered.
* `onet_*` - O\*NET reference (codes `onetsoc_code`, `element_id`, `scale_id`, original `data_value`, `onet_version`).
* `esco_*` - ESCO reference (`concept_uri` keys, `esco_version`, `esco_source_mode`).
* Downstream team outputs should be tables keyed by `posting_id` (e.g. `task_extractions(posting_id, task_text, onet_task_id, ...)`,
  `skill_extractions(posting_id, skill_text, esco_skill_uri, ...)`).
