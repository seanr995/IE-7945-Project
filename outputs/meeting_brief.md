# Sprint 1 - Meeting Brief

**Corpus:** 5,497 canonical postings (5,595 raw) from NYC Open Data Jobs + Arbeitnow API -> **4,029 unique** (duplicate rate 26.7%).
**Languages:** en 76.5%, de 22.4%, fr 1.1%.
**References:** O*NET 31.0 loaded (1,016 occupations, 18,838 tasks); ESCO v1.2.1 via official API fallback (3,039 occupations, 13,939 skills).
**Contract:** `docs/data_contract.md` v1.0 - one `posting_id` for both teams.

**Decisions needed:** (1) approve bootstrap sources or supply USAJOBS key; (2) someone downloads the official ESCO CSV; (3) English-only vs bilingual extraction.

## 60-second spoken summary
"This sprint we built the shared data foundation. We pulled 5,595 real job-posting records from two public
sources - the City of New York's open-data job postings and the Arbeitnow public jobs API - kept every raw file untouched,
and standardized them into 5,497 canonical postings with a stable posting ID. After flagging duplicates -
mostly New York posting the same job internally and externally - we have 4,029 unique postings, a
26.7% duplicate rate. The corpus is about 76.5% English, with German from Arbeitnow.
We loaded O*NET 31.0 and ESCO v1.2.1 as reference taxonomies into one DuckDB database, and wrote a data
contract so the task team and the skill team extract from exactly the same postings and join on the same ID. Both job sources are
temporary until you approve them, and the official ESCO package needs one manual download. Next sprint the teams start
extraction and mapping."
