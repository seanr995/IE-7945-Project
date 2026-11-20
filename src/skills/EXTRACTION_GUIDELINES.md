# Skill extraction guidelines

Input: `outputs/skill_extraction/postings_188.csv`, column `skill_input_text` (labelled blocks:
`[TITLE]`, `[DESCRIPTION]`, `[REQUIREMENTS]`, `[PREFERRED SKILLS]`).

One output row per (posting, skill):

| field | rule |
|---|---|
| `posting_id` | copied from the input |
| `skill_text` | the skill exactly as written in the posting; must be a substring of `evidence_span` |
| `normalized_skill` | lower-case, singular, common English name (`python`, `project management`, `aws`). Same skill = same string across postings. Translate non-English skills to English here. |
| `type` | `tool` = a named software, platform, programming language, framework, standard or technology product. `concept` = everything else (methods, knowledge areas, competencies). |
| `evidence_span` | the sentence or bullet that names the skill, copied verbatim as ONE contiguous piece of one line. Several skills may share a span. |

Include: skills, knowledge areas, competencies, tools, technologies, spoken languages.
Exclude: degrees, years of experience, personality traits ("passionate", "self-starter"), benefits,
company descriptions, and plain task descriptions that name no skill ("attend meetings").
A soft skill counts only when the posting names it as a skill or ability (communication, leadership).
Do not invent skills. One row per distinct `normalized_skill` per posting.

Validation (Python, not the model): every `evidence_span` must be a substring of the posting's
`skill_input_text`, and `skill_text` must occur inside its span (case-insensitive).
