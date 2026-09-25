# Data Sources (generated 2026-09-25T17:32:39Z)

Generated from `data/data_manifest.csv`. For directory snapshots (API pages), SHA-256 is a
deterministic digest over the sorted list of (relative path, file SHA-256).

**ESCO status:** ESCO_STATUS = MANUAL_DOWNLOAD_REQUIRED (official ZIP); TEMPORARY_API_FALLBACK_LOADED

## jobs_arbeitnow

| | |
|---|---|
| Provider | Arbeitnow (arbeitnow.com) free public Job Board API |
| Dataset | jobs_arbeitnow (JSON API pages) |
| Official URL | https://www.arbeitnow.com/blog/job-board-api |
| Download URL | `https://www.arbeitnow.com/api/job-board-api?page=<n>` |
| Retrieval date (UTC) | 2026-09-25T05:25:46Z |
| Version | snapshot retrieval_20260925T052546Z |
| Local raw path | `data/raw/jobs/arbeitnow/retrieval_20260925T052546Z` |
| Local extracted path | `n/a (raw files read directly)` |
| Record count | 2800 |
| SHA-256 | `22c300b9647abff4647ad3d4404e5d8d6c8e7b4737bd0e1af0d9ec7448222708` |
| Size (bytes) | 21433568 |
| License / terms | API response states: 'This is a free public API for jobs, please do not abuse. I would appreciate linking back to the site. By using the API, you agree to the terms of service present on Arbeitnow.com' - full ToS requires verification |
| Status | OK - Temporary bootstrap source - instructor/team approval required before final corpus selection. |
| Limitations | Aggregator of mostly Germany-based postings (German/English); ~100 postings/page, live ordering can repeat records across pages; no salary, no country field; employers are third parties. |
| Final vs temporary | TEMPORARY bootstrap - instructor/team approval required before final corpus selection. |
| Notes | Genuinely open job-board API (no key); mostly Germany-based postings in German/English, useful for language-mix testing. Aggregator content: employers are third parties. Temporary bootstrap source - instructor/team approval required before final corpus selection. |
## jobs_nyc_jobs

| | |
|---|---|
| Provider | City of New York - Department of Citywide Administrative Services (DCAS), via NYC Open Data |
| Dataset | jobs_nyc_jobs (CSV export + Socrata metadata JSON) |
| Official URL | https://data.cityofnewyork.us/City-Government/Jobs-NYC-Postings/kpav-sd4t |
| Download URL | `https://data.cityofnewyork.us/api/views/kpav-sd4t/rows.csv?accessType=DOWNLOAD` |
| Retrieval date (UTC) | 2026-09-25T05:25:37Z |
| Version | rowsUpdatedAt=1790103669 |
| Local raw path | `data/raw/jobs/nyc_jobs/retrieval_20260925T052537Z` |
| Local extracted path | `n/a (raw files read directly)` |
| Record count | 2795 |
| SHA-256 | `2a8345f7702877e5f2f6402d36b9ea381febd03f31ab6f4f90692a2a5b52a16c` |
| Size (bytes) | 18799416 |
| License / terms | NYC Open Data public dataset; governed by NYC Open Data Terms of Use (https://opendata.cityofnewyork.us/overview/#termsofuse) - requires verification |
| Status | OK - Temporary bootstrap source - instructor/team approval required before final corpus selection. |
| Limitations | Public sector only; one city; point-in-time snapshot of open postings; most vacancies appear twice (internal + external); no per-posting URL; no industry field. |
| Final vs temporary | TEMPORARY bootstrap - instructor/team approval required before final corpus selection. |
| Notes | Official municipal open-data export of current City of New York job postings (public sector only). Temporary bootstrap source - instructor/team approval required before final corpus selection. |
## esco_api_fallback

| | |
|---|---|
| Provider | European Commission (ESCO Web Services API) |
| Dataset | esco_api_fallback (gzip-compressed JSON API pages (sha256 over directory)) |
| Official URL | https://esco.ec.europa.eu/en/use-esco/use-esco-services-api/esco-web-service-api |
| Download URL | `https://ec.europa.eu/esco/api/search?type=<occupation|skill>&language=en&full=true&selectedVersion=v1.2.1` |
| Retrieval date (UTC) | 2026-09-25T05:34:36Z |
| Version | v1.2.1 |
| Local raw path | `data/reference/esco/raw/api_v1.2.1/retrieval_20260925T052824Z` |
| Local extracted path | `data/reference/esco/extracted/api_v1.2.1` |
| Record count | 3039 occupations; 13939 skills |
| SHA-256 | `73f843d25fa0ea5460a4438710cea008528e2fa0d3ed608c7501858fb91f68c4` |
| Size (bytes) | 81164845 |
| License / terms | requires verification (ESCO is published by the European Commission; confirm reuse terms at https://esco.ec.europa.eu before redistribution) |
| Status | OK (temporary API fallback) |
| Limitations | Temporary: derived from the ESCO Web Services API, not the official CSV package; columns are a subset (labels, descriptions, codes, relations, broader links). |
| Final vs temporary | TEMPORARY until official ESCO CSV is downloaded manually |
| Notes | Temporary Sprint 1 fallback: official ZIP needs manual privacy acceptance. Version explicitly requested via selectedVersion and accepted by the API (the API rejects invalid versions). |
## onet_database

| | |
|---|---|
| Provider | O*NET Resource Center (National Center for O*NET Development, sponsored by USDOL/ETA) |
| Dataset | onet_database (ZIP of CSV files) |
| Official URL | https://www.onetcenter.org/database.html |
| Download URL | `https://www.onetcenter.org/dl_files/database/db_31_0_csv.zip` |
| Retrieval date (UTC) | 2026-09-25T05:25:04Z |
| Version | 31.0 |
| Local raw path | `data/reference/onet/raw/onet_database_31.0_csv.zip` |
| Local extracted path | `data/reference/onet/extracted/31.0` |
| Record count | 45 csv files |
| SHA-256 | `55033fc68b4c13ec23e7f74dc6378660f6e854e75d55d6e333ae0a761d3987cd` |
| Size (bytes) | 16237378 |
| License / terms | Creative Commons Attribution 4.0 International (CC BY 4.0), as stated on the official O*NET database page / https://www.onetcenter.org/license_db.html; attribution to O*NET (USDOL/ETA) required |
| Status | OK |
| Limitations | US occupational taxonomy; O*NET 31.0 renames Skills -> Essential + Transferable Skills and Technology Skills -> Software Skills. |
| Final vs temporary | Final (official reference) |
| Notes | Complete CSV archive db_31_0_csv.zip served from the official dl_files directory (the page lists the individual CSVs and other ZIP formats). |

## Sources considered but not used

| Source | Why not (today) |
|---|---|
| USAJOBS Search API | Approved source, but requires an API key - none configured. See `docs/USAJOBS_API_SETUP.md`. |
| Adzuna API | Requires `ADZUNA_APP_ID` / `ADZUNA_APP_KEY` - none configured. |
| The Muse public API | Its developer docs state "Registration is required for any use beyond testing"; not registered, so not used for bulk collection. |
| ESCO official CSV package | Requires human acceptance of a privacy statement + email; see `docs/ESCO_MANUAL_DOWNLOAD_REQUIRED.md`. |
