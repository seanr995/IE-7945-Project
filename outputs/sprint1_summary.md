# Sprint 1 Summary - Shared Job-Posting Corpus

Generated 2026-09-25T17:32:39Z from `database/taxonomy.duckdb` (all values computed; see `outputs/tables/sprint1_metrics.json`).

## Job sources
| Source | Raw | Canonical | Unique | Duplicates | Dup. rate | Status |
|---|---|---|---|---|---|---|
| arbeitnow | 2,800 | 2,726 | 2,625 | 101 | 3.7% | Temporary bootstrap - instructor/team approval required |
| nyc_jobs | 2,795 | 2,771 | 1,404 | 1,367 | 49.3% | Temporary bootstrap - instructor/team approval required |

* NYC Jobs: City of New York (DCAS) via NYC Open Data - https://data.cityofnewyork.us/City-Government/Jobs-NYC-Postings/kpav-sd4t
* Arbeitnow: free public Job Board API - https://www.arbeitnow.com/api/job-board-api
* USAJOBS / Adzuna: not collected (credentials not configured).

## Headline counts
| Metric | Value |
|---|---|
| Raw posting records | 5,595 |
| Canonical postings (`job_postings`) | 5,497 |
| Unique postings (normalized) | 4,029 |
| Unique postings (exact) | 4,067 |
| Duplicate count (normalized) | 1,468 |
| Duplicate rate (of canonical) | 26.7% |
| Duplicate rate (of raw, incl. collapsed repeats) | 28.0% |
| Distinct normalized titles | 3,413 |

## Language mix
en 4,207 (76.5%), de 1,229 (22.4%), fr 61 (1.1%)

## Seniority distribution (harmonized)
experienced 2,839, manager 735, (unknown) 645, entry 439, intern_student 414, senior 356, executive 69

NYC source `Career Level`: Experienced (non-manager) 2,078, Entry-Level 287, Manager 282, Student 83, Executive 41

## Description length (words; chars at right)
| Source | n | mean | median | std | min | p25 | p75 | p95 | max | chars mean | chars median |
|---|---|---|---|---|---|---|---|---|---|---|---|
| arbeitnow | 2,726 | 618.3 | 575.5 | 278.5 | 6 | 425.0 | 770.8 | 1098.3 | 2229 | 4425.9 | 4188.5 |
| nyc_jobs | 2,771 | 597.9 | 532.0 | 273.1 | 88 | 399.0 | 742.0 | 1143.0 | 1963 | 4165.0 | 3756.0 |
| ALL | 5,497 | 608.0 | 557.0 | 276.0 | 6 | 407.0 | 759.0 | 1123.4 | 2229 | 4294.4 | 3977.0 |

## Geographic & date coverage
| Source | First posted | Last posted | Median posted | Collected |
|---|---|---|---|---|
| arbeitnow | 2026-09-20 | 2026-09-25 | 2026-09-23 | 2026-09-25 |
| nyc_jobs | 2025-09-25 | 2026-09-21 | 2026-08-25 | 2026-09-25 |

Top locations - **arbeitnow**: Berlin (554), Munich (201), Hamburg (143), München (135), Germany (114); **nyc_jobs**: 30-30 Thomson Ave L I City Qns (341), 42-09 28th Street (204), 96-05 Horace Harding Expway (150), 1 Centre St., N.Y. (125), 55 Water St Ny Ny (119). Arbeitnow remote-flagged postings: 153.

## Missingness (NULL share by source)
| Field | arbeitnow | nyc_jobs |
|---|---|---|
| job_title | 0.0% | 0.0% |
| company_name | 0.0% | 0.0% |
| industry | 100.0% | 100.0% |
| industry_derived | 100.0% | 0.0% |
| location_raw | 2.1% | 0.0% |
| country | 100.0% | 0.0% |
| state | 100.0% | 0.0% |
| city | 100.0% | 100.0% |
| employment_type | 32.1% | 0.0% |
| seniority_source | 56.7% | 0.0% |
| seniority | 23.7% | 0.0% |
| description | 0.0% | 0.0% |
| responsibilities | 100.0% | 100.0% |
| requirements | 100.0% | 1.3% |
| preferred_skills | 100.0% | 43.5% |
| salary_min | 100.0% | 0.0% |
| salary_max | 100.0% | 0.0% |
| salary_currency | 100.0% | 0.0% |
| date_posted | 0.0% | 0.0% |
| language | 0.0% | 0.0% |
| source_url | 0.0% | 100.0% |

## O*NET
Version **31.0** (official O*NET Resource Center, CC BY 4.0).

| Table | Rows |
|---|---|
| onet_abilities | 94,640 |
| onet_alternate_titles | 54,269 |
| onet_content_model_reference | 3,006 |
| onet_job_zones | 923 |
| onet_knowledge | 60,060 |
| onet_occupations | 1,016 |
| onet_scales_reference | 33 |
| onet_skills | 63,700 |
| onet_tasks | 18,838 |
| onet_technology_skills | 31,821 |
| onet_work_activities | 74,702 |

## ESCO
Version **v1.2.1** - ESCO_STATUS = MANUAL_DOWNLOAD_REQUIRED (official ZIP); TEMPORARY_API_FALLBACK_LOADED (mode: api_fallback).

| Table | Rows |
|---|---|
| esco_occupation_hierarchy | 3,039 |
| esco_occupation_skill_relations | 126,051 |
| esco_occupations | 3,039 |
| esco_skill_hierarchy | 20,183 |
| esco_skills | 13,939 |

## All DuckDB tables
| Table | Rows |
|---|---|
| data_manifest | 4 |
| esco_occupation_hierarchy | 3,039 |
| esco_occupation_skill_relations | 126,051 |
| esco_occupations | 3,039 |
| esco_skill_hierarchy | 20,183 |
| esco_skills | 13,939 |
| job_postings | 5,497 |
| job_postings_raw | 5,595 |
| onet_abilities | 94,640 |
| onet_alternate_titles | 54,269 |
| onet_content_model_reference | 3,006 |
| onet_job_zones | 923 |
| onet_knowledge | 60,060 |
| onet_occupations | 1,016 |
| onet_scales_reference | 33 |
| onet_skills | 63,700 |
| onet_tasks | 18,838 |
| onet_technology_skills | 31,821 |
| onet_work_activities | 74,702 |
| pipeline_runs | 1 |
| reference_load_report | 16 |
| source_metadata | 5 |

## Data-quality findings
- Overall normalized duplicate rate is **26.7%** (1,468 of 5,497 canonical postings); NYC 49.3% vs Arbeitnow 3.7%.
- NYC duplicates are structural: 1,324 Job IDs are posted both *Internal* and *External* with the same text.
- 98 raw records repeated an existing source ID within the same snapshot (NYC export repeats + Arbeitnow page overlap) and were collapsed.
- No source provides employer **industry** (100% NULL); NYC `job_category` is functional, not industry; NYC industry_derived = public administration by rule.
- `responsibilities` and `city` are 100% NULL: not separable/parsable without extraction; left NULL by design.
- Seniority basis: none 645, source 3,952, title_rule 900 (NULL = no source value and no title keyword).
- 5 postings have descriptions under 50 words.
- Language mix is driven by source: NYC is English; Arbeitnow mixes German and English (en 4,207 (76.5%), de 1,229 (22.4%), fr 61 (1.1%)).

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
