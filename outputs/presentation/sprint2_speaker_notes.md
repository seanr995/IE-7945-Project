# Sprint 2 — Speaker Notes (5 minutes)

Deck: `outputs/presentation/Sprint2_Task_Extraction_Results.pptx` (8 slides, about 35 seconds each)

---

**1. Sprint 2 Objective (~25s)**
This sprint turned raw job postings into atomic task statements, and then we measured how good that extraction is. Every task we produce is tied to evidence in the posting and scored against human labels.

**2. Extraction Architecture (~40s)**
The pipeline has five steps. We start from canonical postings, NYC Jobs and Arbeitnow. Rule-based parsing splits each posting into responsibilities, requirements and skills. Gemini and Groq extract tasks independently. Then comes the key step, the evidence gate: a task survives only if the span it cites actually appears in the posting. Everything lands in one DuckDB table, `task_statements`.

**3. What We Built (~30s)**
We ran it on 200 frozen postings and got 3,614 extracted tasks. 3,479 passed evidence validation, and the gate rejected 132: 106 had invalid evidence and 26 were unsupported. 683 tasks were extracted by both models independently. All 32 automated tests pass.

**4. Human Evaluation Method (~35s)**
We built a human-labeled evaluation subset: 43 postings with 129 human tasks. Labeling was blind, so annotators saw only the posting, with no model output and no O*NET. Each human task cites its evidence in the posting, and we checked those spans against the source. Human and model tasks are matched one-to-one within each posting: exact first, then lexical, then semantic at cosine 0.85. No LLM is used as a judge.

**5. Precision / Recall / F1 (~55s)**
Gemini scores precision 9.0%, recall 48.8%, F1 15.2%. Groq scores precision 11.9%, recall 38.0%, F1 18.1%.
Recall is the number to focus on. Gemini finds about half of the tasks the human marked as key under strict matching, and about 72% once you count close paraphrases.
Precision reads low for a structural reason. Annotators recorded the 3 key tasks per posting, but the models extract every task, about 16 per posting for Gemini. Most "false positives" are valid tasks the human simply didn't list, so precision is a lower bound.
Where both models agree, precision is highest, at 14.2%.

**6. Error Analysis (~45s)**
There are four error types:
- **Extra tasks** are the biggest group. These are real tasks outside the human's top 3, like "Deploy application updates".
- **Missed tasks** usually come from narrative paragraphs rather than bullet lists, like "Oversee design reviews".
- **Compound splitting and paraphrase** means the model splits or merges tasks differently from the human. "Lead high cost deals" versus "Close complex, high-value deals" scores 0.77, just under our threshold.
- **Task vs skill** is the last one. Gemini sometimes turns skill lines into tasks, like "Configure Microsoft Purview", and a few human labels are skills, like "Strong writing skills".

**7. O*NET Alignment (~35s)**
After extraction, we match each of the 3,479 tasks to its closest O*NET task. 286 are accepted candidate matches, 2,503 are review candidates, and 690 have no close O*NET equivalent. These are candidate mappings, not ground truth, and the thresholds are provisional. O*NET is a reference only: it never generates or filters our tasks.

**8. Final Takeaway (~25s)**
The evidence-backed task extraction pipeline works and is measurable. It takes real postings to tasks, evaluates them against humans, and aligns them to O*NET as a reference. Every number here is reproducible from cached outputs, with no new API calls.

---

*If asked about sample size:* "Current metrics reflect the completed human-labeled evaluation subset: 43 postings and 129 tasks, from NYC Jobs and Arbeitnow."
