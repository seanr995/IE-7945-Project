# Task annotation guidelines - human gold set (Section A, task extraction)

**Files**
| File | Who | Postings |
|---|---|---|
| `data/annotations/task_gold_100.csv` | Annotator A | all 100 gold postings |
| `data/annotations/task_gold_audit_20_annotator_b.csv` | Annotator B | blind audit subset: 20 of the same 100 |
| `data/annotations/task_gold_100_assignment.csv` | - | who labels which posting, and how the postings were chosen |

The files are built by `python -m src.prototype.task_gold build` from the frozen experiment sample
(`data/experiment/task_extraction_sample_200.csv`). They contain **only posting text**: no Gemini, Groq or
Claude output, no O\*NET tasks, no machine suggestions. Labels must be produced by people. **Never paste model
output into the gold columns and never use an LLM to fill them.** Annotator B must not open
`task_gold_100.csv` until both annotators have finished the 20 audit postings.

> This gold set covers **tasks only**. Skills are labelled separately by the skill team; the older combined
> task+skill file `data/annotation/sprint2_gold_100.csv` (`src/prototype/gold.py`) is superseded for tasks.

## File format (one row per gold task)

| Column | Fill in? | Content |
|---|---|---|
| `posting_id`, `source`, `job_title` | given | Copy `posting_id` to every row you add for the same posting. |
| `task_index` | yes | 1, 2, 3, ... within the posting. |
| `responsibilities_text` | given | Responsibilities sections found by heading rules (regex, not AI). May be incomplete - read the full text. |
| `full_posting_text` | given | `job_postings.posting_text`, the text evidence offsets refer to. |
| `gold_task_statement` | yes | One atomic task (see below), short imperative: verb + object. |
| `evidence_span` | yes | Text copied **verbatim** from `full_posting_text` that states this task. |
| `annotator` | yes | Your initials. |
| `adjudication_status` | yes | `not_started` -> `in_progress` -> `labelled`; `no_tasks` if the posting states no task; `adjudicated` after A/B disagreements are resolved. |
| `notes` | optional | Uncertainty, rule questions, anything to discuss. |

Each posting starts with one empty row. Add a row (same `posting_id`) for every further task; the long text
columns may stay blank on added rows. Set `adjudication_status` on at least one row of the posting - only
postings marked `labelled`, `no_tasks` or `adjudicated` are scored. `in_progress` postings are ignored.

## Definition

**An atomic task is one explicit work action performed by the worker.** It is something you could watch the
job holder do and say when it is finished: *verb + object (+ qualifier if needed to keep the meaning)*.

"Design, build and maintain ETL pipelines" ->
`Design ETL pipelines` | `Build ETL pipelines` | `Maintain ETL pipelines`
when each is an independently meaningful work action.

**Not tasks - never label them on their own:** technologies and tools (`Python`, `SQL`, `AWS`, `Excel`),
soft skills (`communication skills`), knowledge areas (`contract law`), degrees (`bachelor's degree`),
certifications or licences, years of experience, personal traits, benefits, company descriptions,
application instructions, schedule/residency/legal text.

## Evidence span (mandatory)

* Every gold task needs an `evidence_span` copied character for character from `full_posting_text`
  (copy-paste; do not retype, fix typos or change punctuation). The shortest clause or sentence that states
  the task.
* If you cannot point to a span, do not label the task.
* `python -m src.prototype.task_eval validate` checks every span against the posting text and lists the rows
  where it is not found.
* The same span may support several tasks (e.g. the three ETL tasks above share one span).

## Rules

1. **Compound actions (several verbs, one object).** Split when each verb is a distinct, independently
   meaningful action: "Prepare and review budgets" -> `Prepare budgets`, `Review budgets`. Do **not** split
   fixed expressions or near-synonyms used as one action: "monitor and track progress" -> `Monitor project progress`;
   "plan and organise events" may be split only if planning and organising are clearly separate work.
2. **One action, several objects.** Split when the objects are different work products:
   "Prepare budgets, invoices and payroll reports" -> 3 tasks. Keep one task when the objects form one unit of
   work: "Maintain the chart of accounts and general ledger" may stay `Maintain the general ledger and chart of accounts`
   if the annotator judges it one activity - write the reason in `notes`.
3. **One action expressed with different wording.** Label it once per posting ("Develop ETL pipelines" and
   later "build data pipelines for ETL" -> one task). Use the clearest wording; any one of the spans is enough.
4. **Repeated responsibilities / duplicated bullet points.** Label once per posting, even if repeated in
   several sections or bullets.
5. **Vague responsibilities.** Skip statements without a concrete action and object: "other duties as
   assigned", "support the team as needed", "contribute to our mission", "wear many hats". Keep vague-but-real
   work when an object is present: "Support the annual budget process" -> `Support the annual budget process`.
6. **Explicit vs implied tasks.** Label only what is written. Do not add tasks the job "obviously" involves
   (a "Data Analyst" posting that never says "clean data" has no `Clean data` task). Do not turn a goal or
   outcome into a task unless an action is stated ("drive revenue growth" is a goal; "negotiate contracts to
   drive revenue growth" -> `Negotiate contracts`).
7. **Managerial responsibilities.** Label them as tasks with their object: `Supervise field inspectors`,
   `Manage a team of five engineers`, `Approve vendor invoices`, `Set quarterly team goals`.
   "Leadership" or "management experience" alone is a qualification, not a task.
8. **Tools appearing inside tasks.** Keep the action, drop the tool unless it is the object of the work:
   "Build dashboards in Tableau" -> `Build dashboards`; "Administer the Salesforce CRM" -> `Administer the Salesforce CRM`
   (the system is what is being administered). Never label the tool alone.
9. **Qualifications written as responsibilities.** "Experience managing vendor contracts", "Ability to analyse
   large datasets", "Knowledge of procurement rules" are requirements, not tasks - do not label them, even
   if they appear under a responsibilities heading. Exception: an explicit duty written in requirement words
   inside the duties list ("You will be responsible for managing vendor contracts") -> `Manage vendor contracts`.
10. **Tasks outside a Responsibilities section.** Label by meaning, not by section. A task stated in the
    summary paragraph, "About the role" or a NYC `Minimum Qualifications` field is still a task if it describes
    work the job holder will do. Ignore benefits, how-to-apply, EEO and residency text.
11. **Statement wording.** Short imperative, present tense, <= ~12 words, keep the object specific
    ("Prepare monthly budget variance reports", not "Prepare reports"). Do not add words that are not supported
    by the span.
12. **Non-English fragments.** The sample is English-only; if a posting contains German text, label tasks only
    from English text and note it.

## Double annotation and adjudication (20-posting audit)

1. Annotator A labels all 100 postings in `task_gold_100.csv`.
2. Annotator B labels the 20 audit postings in `task_gold_audit_20_annotator_b.csv` **without seeing A's labels
   or any model output**.
3. Run `python -m src.prototype.task_eval agreement` - it compares A and B on the postings both marked as
   labelled, with the same matcher used for model evaluation, and writes
   `outputs/task_evaluation/inter_annotator_audit20/disagreements.csv`.
4. Discuss the disagreements, update these rules if a rule was unclear, correct `task_gold_100.csv` where the
   discussion changes the answer and set those postings to `adjudicated`. B's file is kept unchanged as the
   pre-adjudication record.

## Scoring (for reference)

`python -m src.prototype.task_eval evaluate` matches gold vs model tasks **within the same posting**
(normalized exact -> lexical -> semantic similarity, one-to-one), and writes precision, recall, F1, per-posting
metrics, false positive / false negative examples with heuristic error categories, and an
`ambiguous_review.csv` of borderline pairs for a human decision. See `src/prototype/task_eval.py`.
