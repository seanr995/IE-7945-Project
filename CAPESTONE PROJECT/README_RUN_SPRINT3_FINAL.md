# Sprint 3 final run (agentic taxonomy + matched benchmark)

1. Upload the contents of this folder as `/My Drive/CAPESTONE PROJECT/` (merge with the existing Drive folder so an
   existing `agentic_taxonomy_results/provenance/api_budget.json` is kept).
2. Create `/My Drive/CAPESTONE PROJECT/.env` from `.env.example` (ANTHROPIC_API_KEY, ANTHROPIC_MODEL).
3. Open `Sprint3_Final_Agentic_and_Benchmark_Colab.ipynb` and Run all.

The notebook stops on the first failure. Preflight makes no taxonomy call. The agentic stage never exceeds the
persisted 20-call budget (calls already made are preserved; a floor of 5 earlier calls is recorded in
`agentic_taxonomy_results/provenance/api_budget_prior_usage.json`). The benchmark makes zero Claude calls.
Requests contain only model, max_tokens, system and messages; replies are JSON text validated locally.
