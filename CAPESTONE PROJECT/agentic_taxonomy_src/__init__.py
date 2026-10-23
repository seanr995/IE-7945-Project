"""Sprint 3 final runtime: agentic taxonomy induction (Claude, JSON-text responses) + matched-granularity benchmark.

config          settings from <project>/.env, project paths, constants
provenance      logging, JSON I/O, hashing, secret-safe ZIP
budget          persisted hard API budget (never reset) + plan scaled to the REAL remaining calls
cache           response cache (identical request -> no API call)
schemas         pydantic response schemas + explicit JSON contracts + topic-count-hint guard
llm_client      plain Messages API (model, max_tokens, system, messages ONLY) + JSON parsing + one repair retry
semantic_index  validated embedding loading, diversity sampling, retrieval helpers
prompts         role prompts (each ends with its JSON contract)
registry        immutable-ID topic registry with provenance
agentic         the agentic induction pipeline
evaluation      shared evaluator (Sprint 3 metric definitions + persistence, alignment, co-association)
methods / granularity / decision / audit / report / benchmark   matched-granularity benchmark (no Claude)
"""
__version__ = "2.0.0"
