"""Router: the common operations both providers expose. PROTOTYPE.

    extract_statements(provider, posting_id, sections)   -> ExtractionResult
    validate_statements(provider, posting_id, text, items) -> ValidationResult
    adjudicate_disagreement(provider, posting_id, text, items) -> AdjudicationResult
    rerank_candidates(provider, batch)                    -> RerankResult (ids constrained to candidate list)

Prompts are versioned (config prompt_versions); changing a prompt requires bumping its version,
which also invalidates the cache for that operation only.
"""
from __future__ import annotations

import json

from src.llm.base import LLMProvider, LLMResult, load_config
from src.llm.gemini_provider import GeminiProvider
from src.llm.groq_provider import GroqProvider
from src.llm.schemas import AdjudicationResult, ExtractionResult, RerankResult, ValidationResult

EXTRACT_SYSTEM = """You extract ATOMIC work evidence from a job posting for a workforce taxonomy.

DEFINITIONS
- TASK: an observable work activity the job holder performs (verb + object), e.g. "Build ETL pipelines",
  "Prepare monthly reports", "Interview job candidates".
- SKILL: a capability, knowledge area, technology, tool, method or interpersonal capability required to do
  the work, e.g. "Python", "SQL", "statistical analysis", "stakeholder communication".
- One sentence may contain both: "Develop ETL pipelines using Python and SQL" -> TASK "Develop ETL pipelines";
  SKILLS "Python", "SQL".

RULES
1. Extract only what the posting states explicitly. Do not infer, generalise or add anything that is not written.
2. evidence_text MUST be copied VERBATIM (character for character) from the posting: the shortest sentence or
   clause that supports the statement. Never paraphrase evidence.
3. One statement = one activity or one capability. Split coordinated lists ("Python, SQL and Spark" -> 3 skills).
4. statement_text: tasks as short imperative verb phrases; skills as short noun phrases (<= 8 words).
5. Do NOT extract: benefits, salary, company descriptions, application instructions, legal/EEO text,
   residency or schedule requirements, years of experience alone, degrees alone, generic nouns.
6. evidence_section: the section the evidence comes from (responsibilities | requirements |
   preferred_qualifications | skills | general_description).
7. Return JSON only, matching the schema."""

VALIDATE_SYSTEM = """You audit extracted job-posting statements. For each numbered item decide:
supported = the evidence text explicitly supports the statement (no inference);
kind_correct = the TASK/SKILL label is right (TASK = observable work activity; SKILL = capability, knowledge,
technology, tool, method or interpersonal capability). Return JSON only."""

ADJUDICATE_SYSTEM = """Two independent extractors disagreed about the numbered candidate statements below.
Using ONLY the posting text, decide for each item: keep (explicitly supported and atomic) or reject
(unsupported, not atomic, benefit/boilerplate, or duplicate of another item), and the correct kind
(task = observable work activity; skill = capability/knowledge/technology/tool/method). Return JSON only."""

RERANK_SYSTEM = """You rank reference-taxonomy candidates for statements extracted from job postings.
For each statement you receive its source evidence and up to 5 candidate reference concepts with ids.
- ranked_candidate_ids: ids of candidates that genuinely describe the same activity/capability, best first.
  Use ONLY ids from that statement's candidate list. Never invent ids or concepts.
- reject_all: true if none of the candidates is an acceptable match (then ranked_candidate_ids is empty).
Return JSON only."""


class Router:
    def __init__(self, run_id: str | None = None, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        self.pv = self.cfg["prompt_versions"]
        self.providers: dict[str, LLMProvider] = {
            "gemini": GeminiProvider(self.cfg["gemini"], run_id=run_id),
            "groq": GroqProvider(self.cfg["groq"], run_id=run_id),
        }

    def _p(self, name: str) -> LLMProvider:
        return self.providers[name]

    @staticmethod
    def format_sections(sections: list[dict]) -> str:
        return "\n\n".join(f"### SECTION: {s['section']}\n{s['text']}" for s in sections)

    def extract_statements(self, provider: str, posting_id: str, title: str, sections: list[dict]) -> LLMResult:
        prompt = f"JOB TITLE: {title}\n\nPOSTING TEXT (sections detected deterministically):\n\n" \
                 f"{self.format_sections(sections)}"
        return self._p(provider).generate_json(system=EXTRACT_SYSTEM, prompt=prompt, schema_model=ExtractionResult,
                                               operation="extract", prompt_version=self.pv["extract"],
                                               posting_id=posting_id)

    def validate_statements(self, provider: str, posting_id: str, items: list[dict]) -> LLMResult:
        lines = [f"[{i}] kind={it['kind']} | statement: {it['statement_text']} | evidence: \"{it['evidence_text']}\""
                 for i, it in enumerate(items)]
        return self._p(provider).generate_json(system=VALIDATE_SYSTEM, prompt="\n".join(lines),
                                               schema_model=ValidationResult, operation="validate",
                                               prompt_version=self.pv["validate"], posting_id=posting_id)

    def adjudicate_disagreement(self, provider: str, posting_id: str, source_text: str, items: list[dict]) -> LLMResult:
        lines = [f"[{i}] proposed_by={it['proposed_by']} kind={it['kind']} | statement: {it['statement_text']} "
                 f"| evidence: \"{it['evidence_text']}\"" for i, it in enumerate(items)]
        prompt = f"POSTING TEXT:\n{source_text}\n\nCANDIDATE ITEMS:\n" + "\n".join(lines)
        return self._p(provider).generate_json(system=ADJUDICATE_SYSTEM, prompt=prompt,
                                               schema_model=AdjudicationResult, operation="adjudicate",
                                               prompt_version=self.pv["adjudicate"], posting_id=posting_id)

    def rerank_candidates(self, provider: str, batch: list[dict], reference: str) -> LLMResult:
        """batch: [{statement_text, evidence_text, kind, candidates:[{id, text}]}]"""
        blocks = []
        for i, b in enumerate(batch):
            cands = "\n".join(f"   - id={c['id']}: {c['text']}" for c in b["candidates"])
            blocks.append(f"[{i}] {b['kind'].upper()}: {b['statement_text']}\n   evidence: \"{b['evidence_text']}\"\n"
                          f"   candidates ({reference}):\n{cands}")
        return self._p(provider).generate_json(system=RERANK_SYSTEM, prompt="\n\n".join(blocks),
                                               schema_model=RerankResult, operation=f"rerank_{reference}",
                                               prompt_version=self.pv["rerank"])


def constrain_rerank(result: dict, batch: list[dict]) -> list[dict]:
    """Enforce: ids must come from the statement's own candidate list; reject_all empties ranking.
    Returns one normalized decision per batch index (missing decisions -> status 'no_decision')."""
    by_idx = {d["statement_index"]: d for d in result.get("decisions", [])}
    out = []
    for i, b in enumerate(batch):
        allowed = [c["id"] for c in b["candidates"]]
        d = by_idx.get(i)
        if d is None:
            out.append({"index": i, "ranked": [], "reject_all": None, "invalid_ids": [], "status": "no_decision"})
            continue
        ranked, invalid = [], []
        for cid in d.get("ranked_candidate_ids", []):
            (ranked if cid in allowed and cid not in ranked else invalid).append(cid)
        reject = bool(d.get("reject_all")) and not ranked
        out.append({"index": i, "ranked": [] if reject else ranked, "reject_all": reject,
                    "invalid_ids": invalid, "status": "ok"})
    return out
