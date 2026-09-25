# Annotation guidelines - Sprint 2 gold set (`data/annotation/sprint2_gold_100.csv`)

Status: **for human annotators**. The gold columns are intentionally empty; no model output may be
copied into them. The purpose is to measure extraction precision/recall (`python -m src.prototype.gold evaluate`).

## What to fill in
| Column | Content |
|---|---|
| `gold_tasks` | One task per line (or separated by ` \| `). |
| `gold_skills` | One skill per line (or separated by ` \| `). |
| `reviewer` | Your initials. |
| `review_status` | `not_started` -> `in_progress` -> `done` (use `skip` if the text is unusable and say why in `notes`). |
| `notes` | Ambiguities, disagreements, anything the team should discuss. |

Annotate only from `relevant_text` (the sections the extractors saw). Do not open the models' outputs first.

## 1. Task vs skill
* **TASK** = an observable work activity performed by the job holder: *verb + object*.
  "Prepare monthly budget reports", "Inspect construction sites", "Interview job candidates".
* **SKILL** = a capability needed to perform work: knowledge area, technology, tool, method,
  language, or interpersonal capability. "Python", "AutoCAD", "contract law", "stakeholder communication".
* Test: *Could you watch someone doing it and say when it is finished?* -> task. *Is it something a person
  has or knows?* -> skill.

## 2. Atomicity
* One activity or one capability per line. Split coordinated objects when they are separate work:
  "Prepare and review budgets" -> `Prepare budgets`, `Review budgets`.
* Do **not** split fixed expressions or a single activity with a compound object that is done as one unit:
  "Maintain the chart of accounts" stays one task.
* Lists of technologies are always split: "Python, SQL and Spark" -> 3 skills.

## 3. Explicit vs inferred
* Label **only what is written**. Do not add skills the job "obviously" needs.
  "Build dashboards in Tableau" -> task `Build dashboards`, skill `Tableau`; **not** `data visualization` unless written.
* Degree requirements alone ("Bachelor's degree"), years of experience alone, licences and certificates
  are **not** skills unless the text names a knowledge area (e.g. "degree in civil engineering" -> skill `civil engineering`).

## 4. Technologies and tools
* Named software, languages, platforms, frameworks, equipment are skills (`Microsoft Excel`, `Kubernetes`, `forklift`).
* Keep the vendor/product name as written; drop version numbers unless meaningful.

## 5. Soft skills
* Include only when stated as a requirement ("excellent written communication" -> `written communication`).
* Drop pure adjectives with no capability ("motivated", "passionate", "team player" is kept as `teamwork`
  only if phrased as a requirement).

## 6. Compound statements
* A sentence can yield both kinds: "Develop ETL pipelines using Python and SQL" -> task `Develop ETL pipelines`,
  skills `Python`, `SQL`.
* Responsibility phrased as a requirement ("Experience managing vendor contracts") -> skill `vendor contract management`;
  if it is in the duties section ("Manage vendor contracts") -> task.

## 7. Duplicate concepts
* List a concept once per posting even if repeated in several sections. Near-synonyms in the same posting
  ("MS Excel", "Excel") are one skill.

## 8. Vague language
* Skip statements with no concrete activity or capability ("support the team as needed",
  "other duties as assigned", "contribute to our mission").
* Keep vague-but-real work if an object is present ("Support the budget process" -> task).

## 9. Evidence spans
* Every gold item must be supported by a span in `relevant_text`. If you cannot point to a span, do not label it.
* Optional: after the statement add `  <= "exact source words"` to record the span; the evaluator ignores it.

## 10. Section handling
* Tasks normally come from responsibilities/duties; skills from requirements/qualifications/skills,
  but label by meaning, not by section.
* Ignore `excluded` material (benefits, how to apply, EEO text, residency, salary) - it is already removed.
* `[preferred_qualifications]` items are labelled like requirements (the evaluator does not separate them).

## Workflow
1. Two annotators label the same 20 postings independently, compare, and refine these rules.
2. Then split the remaining 80. Record disagreements in `notes`.
3. Run `python -m src.prototype.gold evaluate`.
