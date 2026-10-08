# Architecture - posting -> tasks / skills -> reference taxonomies

```
APPROVED JOB POSTINGS  (NYC Open Data Jobs NYC Postings, Arbeitnow API - temporary bootstrap sources)
        |
        v
CANONICAL POSTING      main.job_postings / job_postings_unique        key: posting_id
        |
        +---------------------------+
        |                           |
        v                           v
TASK EXTRACTION                SKILL EXTRACTION            (skill team - not built in Section A)
task_pipeline.task_statements  (future skill_statements)
        |                           |
        v                           v
O*NET ALIGNMENT                ESCO ALIGNMENT
task_pipeline.task_onet_alignment  (future skill_esco_alignment)
        |                           |
        +-------- posting_id -------+
                     |
                     v
               TASK-SKILL MAP       (future: co-occurrence / evidence within the same posting)
                     |
                     v
      BUSINESS INITIATIVE -> TASKS -> SKILLS
```

## What each layer means

1. **NYC and Arbeitnow were canonicalized at the posting layer.** `src/cleaning/canonical.py` maps both
   sources onto one data contract (`docs/data_contract.md`): one `posting_id`, one `posting_text`, shared
   fields for title, seniority, language, duplicates. Nothing else is merged.
2. **Canonicalization does NOT mean O\*NET and ESCO share IDs.** The canonical layer is about job postings
   only. No common O\*NET/ESCO identifier exists in this project and none is invented.
3. **O\*NET and ESCO remain independent reference systems**, loaded as-is with native identifiers
   (`onet_tasks.task_id`, O\*NET-SOC `onetsoc_code`; ESCO `concept_uri`) and their own versions
   (O\*NET 31.0, ESCO v1.2.1).
4. **Extracted tasks align independently to O\*NET.** Tasks come only from posting evidence
   (`task_statements.evidence_span` is a verbatim substring of `job_postings.posting_text`); O\*NET is never
   used to generate them. Alignment is a separate step and a separate table, and "unmatched" is a valid result.
5. **Extracted skills align independently to ESCO** (skill team; same pattern, separate tables).
6. **`posting_id` is the empirical bridge.** A task and a skill are related when they are evidenced in the
   same posting (and, more strongly, in the same or neighbouring evidence span). This is what later supports a
   task-skill map: observed co-occurrence in real postings.
7. **A future O\*NET <-> ESCO occupation crosswalk would be a separate interoperability layer** (its own
   table with its own provenance), not a property of the canonical corpus or of either alignment table.
8. **An occupation crosswalk alone would not prove that an individual O\*NET task requires a particular ESCO
   skill.** Mapping occupation A (O\*NET) to occupation B (ESCO) says the occupations correspond; it says
   nothing about which task within A needs which skill within B. Task-skill relations need task-level evidence,
   i.e. the posting-level bridge in point 6 (or a separately validated resource).

## Section A data flow (task extraction)

| Step | Command | Output |
|---|---|---|
| Freeze 200-posting sample | `python -m src.prototype.freeze` | `data/experiment/task_extraction_sample_200.csv`, `task_pipeline.experiment_sample`, `outputs/task_extraction/sample_profile/` |
| Blind gold files | `python -m src.prototype.task_gold build` | `data/annotations/task_gold_100.csv`, `..._audit_20_annotator_b.csv`, `..._assignment.csv` |
| Extraction (baseline) | `python -m src.prototype.run_extraction` | `prototype.prototype_*` (raw gated outputs, agreement), `task_pipeline.task_statements` |
| Extraction (optional Claude arm) | `python -m src.prototype.run_extraction --provider anthropic` | `prototype.prototype_model_outputs_anthropic`, extra rows in `task_statements` (`experiment_arm='claude_third_provider'`) |
| O\*NET alignment | `python -m src.prototype.alignment tasks [--rerank]` | `task_pipeline.task_onet_alignment` |
| Evaluation (after human labels) | `python -m src.prototype.task_eval evaluate` | `outputs/task_evaluation/` |

Schemas: `main` = Sprint 1 corpus + references (rebuilt by `run_pipeline.py`); `prototype` = experiment
working tables; `task_pipeline` = Section A deliverables. `run_pipeline.py` preserves `prototype` and
`task_pipeline` when it rebuilds `main` (`src/ingestion/build_db.py::PRESERVED_SCHEMAS`).

### task_statements (key columns)

`task_statement_id`, `posting_id`, `atomic_task`, `source_section`, `evidence_span`, `evidence_start`/`evidence_end`
(offsets into `job_postings.posting_text`), `evidence_match_type` (exact | normalized | fuzzy | none),
`evidence_valid`, `provider`, `model`, `prompt_version`, `extraction_timestamp`, `confirming_provider`,
`agreement_status`, `disagreement_type`, `review_status`, `is_valid_task`, `experiment_arm`.
A model task whose evidence cannot be traced to the posting is kept with `evidence_valid = false` and a
`rejected_*` review status; it is never `is_valid_task`.

### task_onet_alignment (key columns)

`task_statement_id`, `candidate_rank`, `onet_task_id`, `onet_occupation_code`, `onet_task_statement`,
`similarity_score`, `top1_margin`, `mapping_method`, `decision` (accepted | review | unmatched |
accepted_after_rerank), `accepted` (rank-1 row only), `embedding_model` (BAAI/bge-small-en-v1.5, local),
`onet_version`, `thresholds`.

Decision rule (config `alignment`): top-1 cosine >= `accept_min_similarity` -> accepted; >=
`review_min_similarity` -> review; otherwise unmatched. With `--rerank`, review-band tasks are re-ranked by
Gemini and Groq independently, restricted to the retrieved candidates (`constrain_rerank`); a task is accepted
only if both put the same candidate first. The thresholds are **provisional** - there are no human O\*NET
mapping labels yet to calibrate them, so accepted mappings are candidates for review, not ground truth.
