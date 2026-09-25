# 5-minute talking points - Workhuman Taxonomy of Work progress

Numbers are from `outputs/tables/sprint1_metrics.json` (current verified run).

## Slide 1 - End goal (25 s)
**Say:** Our goal is a taxonomy of work grounded in real job postings: extract tasks and skills, organise them against standards, map tasks to skills, and ultimately answer "for this business initiative, which tasks and skills matter?"
**Technical detail:** Postings are the evidence; O*NET and ESCO are reference frames, not the answer.
**Likely question:** Why not just start from O*NET? **Answer:** O*NET is curated and lags the market; postings show current language, and we align to O*NET instead of copying it.

## Slide 2 - What we have completed (40 s)
**Say:** Sprint 1 is complete: 5,595 raw postings, 5,497 canonical, 4,029 unique, plus O*NET 31.0 and ESCO v1.2.1 in one DuckDB database, a shared data contract, an executed EDA notebook, and a one-command pipeline passing 21/21 checks.
**Technical detail:** The unique count was corrected from 4,037 to 4,029 after we found entity-encoded HTML in 249 Arbeitnow descriptions; fixing it exposed 8 hidden duplicates. Raw and canonical counts did not change.
**Likely question:** Is it reproducible? **Answer:** Yes: raw files are checksummed, reruns are idempotent, and `run_pipeline.py` rebuilds everything.

## Slide 3 - How the data fits together (45 s)
**Say:** Two public sources feed one canonical corpus. Every posting has a deterministic posting_id. Tasks are aligned to O*NET, skills to ESCO.
**Technical detail:** posting_id = sha256(source | source_job_id), so IDs are stable across reruns.
**Likely question:** Are these sources final? **Answer:** No. They are temporary bootstrap sources pending approval; USAJOBS can be added once a key is available.

## Slide 4 - What EDA taught us (45 s)
**Say:** 26.7% of postings are duplicates overall, 49.3% in NYC because the city posts most jobs internally and externally. 76.5% English, 22.4% German. No source provides industry, and we did not invent it.
**Technical detail:** Duplicates are flagged, not deleted; `job_postings_unique` is the default extraction input.
**Likely question:** Is German a problem? **Answer:** It is a decision: English-first extraction or explicit German handling. The prototype sample is English-only.

## Slide 5 - Shared data contract (40 s)
**Say:** Both teams extract from the same posting_id over the same text, so their outputs join later without rework.
**Technical detail:** NULL means not available; derived fields are named `*_derived` or carry a basis column.
**Likely question:** What if a team needs a new field? **Answer:** Contract changes are versioned and both teams agree first.

## Slide 6 - Advanced prototype direction (60 s)
**Say:** We are building an evidence-preserving extraction engine. Gemini and Groq extract independently; every statement must quote a verbatim span that exists in the posting, and disagreement between models routes items to human review. Provider layer, caching, sample, section detection and evidence gates are built and smoke-tested; the 200-posting experiment is paused mid-run, so we are not showing metrics yet.
**Technical detail:** The consensus score is deterministic (evidence 35%, cross-model agreement 40%, lexical support 15%, kind consistency 10%). We never use a model's self-reported confidence.
**Likely question:** Which model is better? **Answer:** We will not claim that. We will measure agreement, over- and under-splitting, and unsupported-evidence rates, then validate against human labels.

## Slide 7 - Collaboration path (45 s)
**Say:** The shared foundation feeds both teams. The task team goes from responsibilities to tasks, clusters and O*NET evaluation; the skill team from requirements to skills, clusters and ESCO evaluation. Joint work maps tasks to skills and then initiatives to both.
**Technical detail:** The join key everywhere is posting_id plus a verbatim evidence span.
**Likely question:** How do teams avoid duplicating work? **Answer:** Shared sample, shared gold set and shared evaluation script.

## Slide 8 - Status and next (30 s)
**Say:** The foundation is complete. Next: finish the dual-model extraction run, publish disagreement metrics, human-label the 100-posting gold set, and run O*NET/ESCO candidate matching.
**Technical detail:** The run resumes from cache, so completed API calls are not repeated.
**Likely question:** When will results be ready? **Answer:** The remaining extraction takes about an hour of API time; human labels depend on annotator time.
