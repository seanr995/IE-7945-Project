# Sprint 4 - run instructions (Gemini)

Sprint 4 uses **Gemini only** (`google-genai`, model from `GEMINI_MODEL`, default `gemini-3.1-flash-lite`). Claude makes zero Sprint 4 calls.

1. Upload this folder to Google Drive so the code lives at `MyDrive/CAPESTONE PROJECT/SPRINT4_READY/` (the existing Sprint 3 folders in
   `CAPESTONE PROJECT` stay untouched; inputs are found in the project folder or in `SPRINT4_READY`).
2. Create `.env` from `.env.example` in `SPRINT4_READY/` (primary) or in `CAPESTONE PROJECT/` (fallback; `SPRINT4_READY/.env` wins) and set
   `GEMINI_API_KEY`. Keys are never printed.
3. Open `Sprint4_Agentic_Labeling_Hierarchy_Colab.ipynb` -> **Runtime -> Run all**.

## Gemini call plan (normal 10, hard cap 12)
Leaf labeler 4 (20 topics per call) - hierarchy builder 1 - O*NET evaluator 4 (20 topics per call) - critic 1.
The 2 spare calls are for ONE batched repair of invalid records (max 2 per run) or an adaptive 20 -> 10 split; the cap of 12 real generation
requests is never exceeded. Native structured output (Pydantic schema) + local validation; Python writes every file.
Requests are spaced by `GEMINI_MIN_SECONDS_BETWEEN_CALLS`; HTTP 429 honours retry-after, backs off exponentially (max 2 retries) and a daily
quota stops the run with `GEMINI_QUOTA_EXHAUSTED`; a rejected key stops with `GEMINI_AUTHENTICATION_FAILED`. Nothing is lost:
every response is cached in `sprint4_results/cache/gemini/` and every attempt is in `sprint4_results/provenance/sprint4_ledger.json`, so re-running
Run all resumes with zero duplicate calls.

## Outputs (`sprint4_results/`)
`leaf_evidence.csv`, `leaf_topics_labeled.csv`, `task_hierarchy.csv`, `onet_top10_matches.csv`, `onet_evaluation.csv`, `agent_decisions.jsonl`,
`hierarchy_diagnostics.csv`, `sprint4_evaluation_report.md`, `hierarchy_stability.csv/.json` (5 seeds 101-505, 80% within-leaf subsampling, Ward 20/8,
Hungarian/Jaccard matching), `onet_gap_analysis.csv`, `topic_explainability_cards.csv` + `topic_explainability_cards/LT_001.md ... LT_080.md`,
`human_review_template.csv` (blank human columns), `api_usage.csv`, `api_ledger.jsonl`.

## Getting the results back
After the run download **`Sprint4_RETURN_TO_REPO.zip`** from `CAPESTONE PROJECT` and extract it into
`D:\Capstone IE7945\Taxonomy Project\CAPESTONE PROJECT\` (see `RETURN_TO_REPO_INSTRUCTIONS.txt` inside). It holds `sprint4_results/`, the notebook and this
README (no `.env`, no keys, no embedding cache). `Sprint4_Agentic_Results.zip` is the same results folder as a stand-alone archive.

## Notes
- Hierarchy search tests the average-linkage grid (L2 12/16/20 x L1 4/6/8) and a Ward grid; average linkage chains on this embedding space and fails
  the balance gates, so the numerically selected candidate is Ward. Every candidate is in `hierarchy_diagnostics.csv`.
- No dollar cost is claimed for Gemini; the report lists Gemini calls and tokens and keeps the historical Sprint 3 Claude spend estimate separately.
- Offline test mode (`SPRINT4_MOCK=1`) uses a mock Gemini and hashed embeddings, never touches the real ledger, and must not be set in Colab.
