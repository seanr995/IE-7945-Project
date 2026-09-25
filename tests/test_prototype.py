"""Unit tests for the Sprint 2 prototype. No real API calls: providers are mocked."""
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import duckdb
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.llm.usage as usage
from src.llm.base import LLMProvider, RawResponse, TransientError
from src.llm.cache import LLMCache
from src.llm.gemini_provider import parse_gemini_response
from src.llm.groq_provider import parse_groq_response
from src.llm.router import constrain_rerank
from src.llm.schemas import ExtractionResult
from src.prototype.alignment import topk
from src.prototype.evidence import consensus, locate_evidence, match_items, split_events, support_overlap
from src.prototype.sections import detect_sections, model_input


@pytest.fixture(autouse=True)
def _isolated_usage_log(tmp_path, monkeypatch):
    monkeypatch.setattr(usage, "USAGE_LOG", tmp_path / "usage.jsonl")


# ---------------------------------------------------------------- parsers

def test_gemini_parser_text_and_usage():
    resp = NS(text='{"items": []}', usage_metadata=NS(prompt_token_count=10, candidates_token_count=5,
                                                        thoughts_token_count=3), model_version="gemini-x")
    r = parse_gemini_response(resp)
    assert r.text == '{"items": []}' and (r.input_tokens, r.output_tokens, r.reasoning_tokens) == (10, 5, 3)


def test_gemini_parser_skips_thought_parts_and_handles_missing_usage():
    parts = [NS(text="thinking...", thought=True), NS(text='{"items": []}', thought=False)]
    resp = NS(text=None, candidates=[NS(content=NS(parts=parts))], usage_metadata=None)
    r = parse_gemini_response(resp)
    assert r.text == '{"items": []}' and r.input_tokens is None


def test_gemini_parser_empty_raises_transient():
    with pytest.raises(TransientError):
        parse_gemini_response(NS(text="", candidates=[], usage_metadata=None))


def test_groq_parser():
    resp = NS(choices=[NS(message=NS(content='{"items": []}'))], model="openai/gpt-oss-120b",
              usage=NS(prompt_tokens=100, completion_tokens=50, completion_tokens_details=NS(reasoning_tokens=7)))
    r = parse_groq_response(resp)
    assert (r.input_tokens, r.output_tokens, r.reasoning_tokens, r.model) == (100, 50, 7, "openai/gpt-oss-120b")
    with pytest.raises(TransientError):
        parse_groq_response(NS(choices=[NS(message=NS(content="  "))], usage=None))


# ---------------------------------------------------------------- cache + schema

class FakeProvider(LLMProvider):
    name = "fake"

    def __init__(self, texts, cache):
        super().__init__({"model": "fake-1", "max_retries": 2}, cache=cache)
        self.texts, self.calls = list(texts), 0

    def _call(self, model, system, prompt, schema, schema_name):
        self.calls += 1
        return RawResponse(text=self.texts.pop(0), input_tokens=1, output_tokens=1, model=model)


GOOD = '{"items": [{"kind": "skill", "statement_text": "SQL", "evidence_text": "SQL", "evidence_section": "skills"}]}'


def test_cache_second_identical_request_makes_zero_calls(tmp_path):
    p = FakeProvider([GOOD, GOOD], LLMCache(tmp_path))
    kw = dict(system="s", prompt="p", schema_model=ExtractionResult, operation="extract", prompt_version="v1")
    a = p.generate_json(**kw)
    b = p.generate_json(**kw)
    assert p.calls == 1 and not a.cache_hit and b.cache_hit and a.data == b.data
    c = p.generate_json(**{**kw, "prompt_version": "v2"})  # prompt version is part of the key
    assert p.calls == 2 and not c.cache_hit
    rows = usage.read_usage()
    assert [r["cache_hit"] for r in rows] == [False, True, False]


def test_invalid_json_is_not_cached_and_retried(tmp_path):
    p = FakeProvider(["not json", GOOD], LLMCache(tmp_path))
    r = p.generate_json(system="s", prompt="p", schema_model=ExtractionResult, operation="extract", prompt_version="v1")
    assert p.calls == 2 and r.retry_count == 1 and r.data["items"][0]["statement_text"] == "SQL"


def test_item_level_schema_validation_drops_only_bad_items():
    r = ExtractionResult.model_validate({"items": [
        {"kind": "task", "statement_text": "x" * 500, "evidence_text": "ab", "evidence_section": "requirements"},
        {"kind": "banana", "statement_text": "SQL", "evidence_text": "SQL", "evidence_section": "skills"},
        {"kind": "skill", "statement_text": "SQL", "evidence_text": "SQL", "evidence_section": "skills"}]})
    assert len(r.items) == 1 and r.schema_invalid_items == 2


# ---------------------------------------------------------------- evidence

SRC = "Responsibilities:\nDevelop ETL pipelines using Python and SQL.\nPrepare the team’s monthly reports."


def test_evidence_exact_normalized_fuzzy_none():
    e = locate_evidence(SRC, "Develop ETL pipelines using Python and SQL.")
    assert e["match_type"] == "exact" and SRC[e["start"]:e["end"]] == "Develop ETL pipelines using Python and SQL."
    n = locate_evidence(SRC, "prepare the team's   monthly reports")
    assert n["match_type"] == "normalized" and n["exact_text"] == "Prepare the team’s monthly reports"
    assert SRC[n["start"]:n["end"]] == n["exact_text"]
    f = locate_evidence(SRC, "Develop ETL pipeline using Python and SQL.")
    assert f["match_type"] == "fuzzy"
    assert locate_evidence(SRC, "Lead a team of 20 engineers")["match_type"] == "none"


def test_support_overlap_rejects_invented_statement():
    ev = "Develop ETL pipelines using Python and SQL."
    assert support_overlap("Develop ETL pipelines", ev) == 1.0
    assert support_overlap("Python", ev) == 1.0
    assert support_overlap("Kubernetes cluster administration", ev) < 0.34


# ---------------------------------------------------------------- agreement

def _it(kind, text, s=None, e=None):
    return {"kind": kind, "statement_text": text, "start": s, "end": e}


def test_agreement_matching_pairs_boundary_kind_conflict():
    g = [_it("task", "Develop ETL pipelines", 0, 40), _it("skill", "Python"), _it("skill", "SQL"),
         _it("task", "Prepare monthly reports", 50, 80)]
    q = [_it("task", "Develop ETL pipelines using Python and SQL", 0, 40), _it("skill", "Python"),
         _it("task", "SQL", 30, 33)]
    m = match_items(g, q)
    kinds = {g[i]["statement_text"]: mt for i, j, mt, _ in m["pairs"]}
    assert kinds["Python"] == "lexical"
    assert kinds["Develop ETL pipelines"] == "lexical"  # jaccard 3/5 >= 0.5
    assert m["gemini_unmatched"][2][0] == "kind_conflict"  # SQL skill vs SQL task
    assert m["gemini_unmatched"][3][0] == "missing_in_other"


def test_semantic_match_uses_similarity_when_lexical_fails():
    g, q = [_it("task", "Interview job candidates")], [_it("task", "Conduct applicant interviews")]
    assert match_items(g, q)["pairs"] == []
    assert match_items(g, q, sim={(0, 0): 0.91})["pairs"][0][2] == "semantic"


def test_split_events_detects_oversplitting():
    a = [_it("task", "Prepare budgets"), _it("task", "Review budgets")]
    b = [_it("task", "Prepare and review budgets")]
    assert split_events(a, b) == 1 and split_events(b, a) == 0


def test_consensus_single_model_never_high():
    assert consensus("exact", "lexical", 1.0, False) == (1.0, "high")
    s, tier = consensus("exact", "boundary", 1.0, False)
    assert tier != "high"
    assert consensus("exact", "single_model", 1.0, False)[1] == "medium"
    assert consensus("fuzzy", "single_model", 0.6, False)[1] == "low"


# ---------------------------------------------------------------- sections

def test_sections_heading_rules_and_source_fields():
    d = "We are a great company.\nWhat you'll do:\nBuild APIs.\nBenefits\nFree lunch.\nRequirements\nPython."
    r = detect_sections(d, requirements="Bachelor's degree", preferred_skills=None)
    secs = [(s["section"], s["method"]) for s in r["sections"]]
    assert ("responsibilities", "heading_rule") in secs and ("excluded", "heading_rule") in secs
    assert ("requirements", "source_field") in secs
    mi = model_input(r["sections"], 1000)
    assert all(s["section"] != "excluded" for s in mi) and mi[0]["section"] == "responsibilities"
    assert detect_sections("Just a paragraph without headings.")["method"] == "llm_in_extraction"


# ---------------------------------------------------------------- retrieval + rerank constraints

def test_topk_candidate_retrieval_order():
    m = np.eye(4, dtype=np.float32)
    q = np.array([[0.9, 0.1, 0.4, 0.0]], dtype=np.float32)
    idx, sims = topk(q, m, 3)
    assert idx[0].tolist() == [0, 2, 1] and sims[0][0] == pytest.approx(0.9)


def test_rerank_cannot_introduce_concepts_outside_candidates():
    batch = [{"candidates": [{"id": "c1"}, {"id": "c2"}]}, {"candidates": [{"id": "c1"}]}, {"candidates": [{"id": "c1"}]}]
    res = {"decisions": [{"statement_index": 0, "ranked_candidate_ids": ["c9", "c2", "c2", "c1"], "reject_all": False},
                         {"statement_index": 1, "ranked_candidate_ids": [], "reject_all": True}]}
    out = constrain_rerank(res, batch)
    assert out[0]["ranked"] == ["c2", "c1"] and out[0]["invalid_ids"] == ["c9", "c2"]
    assert out[1]["reject_all"] is True and out[1]["ranked"] == []
    assert out[2]["status"] == "no_decision"


# ---------------------------------------------------------------- initiative provenance

class FakeEmbedder:
    VOCAB = ["report", "data", "dashboard", "inspect", "building", "safety", "python", "sql", "customer", "support"]

    def embed(self, texts):
        m = np.array([[t.lower().count(w) + 0.01 for w in self.VOCAB] for t in texts], dtype=np.float32)
        return m / np.linalg.norm(m, axis=1, keepdims=True)

    def close(self):
        pass


def _mini_db(path):
    con = duckdb.connect(str(path))
    text = "Build data dashboards and reports in Python and SQL. Inspect building safety systems."
    con.execute("""CREATE TABLE job_postings AS SELECT 'p1' AS posting_id, ? AS posting_text, 'nyc_jobs' AS source,
                   'Data Analyst' AS job_title, 'Agency' AS company_name, NULL::VARCHAR AS source_url, 'r1' AS retrieval_id,
                   'raw#row=2' AS source_record_ref, DATE '2026-09-01' AS date_posted""", [text])
    con.execute("CREATE SCHEMA prototype")
    rows = [("t1", "task", "Build data dashboards and reports", "Build data dashboards and reports in Python and SQL."),
            ("s1", "skill", "Python", "Build data dashboards and reports in Python and SQL."),
            ("s2", "skill", "SQL", "Build data dashboards and reports in Python and SQL."),
            ("t2", "task", "Inspect building safety systems", "Inspect building safety systems.")]
    df = pd.DataFrame([{"statement_id": a, "posting_id": "p1", "kind": k, "statement_text": s, "evidence_text": e,
                        "evidence_section": "responsibilities", "evidence_start": text.find(e), "evidence_end": text.find(e) + len(e),
                        "agreement_status": "lexical_agree", "consensus_score": 1.0, "confidence_tier": "high",
                        "extractor_provider": "gemini", "extractor_model": "m", "validator_provider": "groq",
                        "paired_statement_text": s, "adjudication_decision": None} for a, k, s, e in rows])
    con.register("df", df)
    con.execute("CREATE TABLE prototype.prototype_statements AS SELECT * FROM df")
    con.execute("""CREATE TABLE prototype.prototype_task_onet_candidates AS SELECT 't1' AS statement_id, '15-2051.00' AS onet_soc_code,
                   '1' AS onet_reference_id, 'Prepare reports' AS onet_text, 'Data Scientists' AS onet_occupation_or_element,
                   'onet_task' AS reference_type, 1 AS rank, 0.8 AS embedding_similarity, 0.8 AS final_candidate_score,
                   'not_required' AS reranker_status, NULL::INT AS gemini_rank, NULL::INT AS groq_rank, false AS candidate_gap""")
    con.execute("""CREATE TABLE prototype.prototype_skill_esco_candidates AS SELECT 's1' AS statement_id, 'uri:py' AS esco_uri,
                   'Python (computer programming)' AS esco_label, 1 AS rank, 0.9 AS embedding_similarity, 0.9 AS final_candidate_score,
                   'not_required' AS reranker_status, NULL::INT AS gemini_rank, NULL::INT AS groq_rank, false AS candidate_gap""")
    return con


def test_initiative_results_all_have_verified_evidence(tmp_path, monkeypatch):
    import src.demo.initiative_to_work as itw
    monkeypatch.setattr(itw, "MIN_RELEVANCE", 0.3)
    con = _mini_db(tmp_path / "t.duckdb")
    out = itw.analyze("Improve data reporting dashboards", con=con, embedder=FakeEmbedder(), persist=True)
    assert out["status"] == "ok" and out["all_results_have_evidence"]
    top = out["results"][0]
    assert top["task_statement"] == "Build data dashboards and reports"
    assert {s["skill"] for s in top["skills"]} <= {"Python", "SQL"}  # skills only from the same posting's evidence
    assert all(e["provenance_verified"] and e["posting_id"] == "p1" for e in top["evidence"])
    assert top["onet_candidate"]["mapping_status"] == "candidate"
    assert set(top["explanation"]) == {"result", "why", "evidence_postings", "reference_alignment", "model_agreement", "limitations"}
    assert con.execute("SELECT count(*) FROM prototype.prototype_initiative_results").fetchone()[0] > 0


def test_initiative_without_evidence_returns_insufficient(tmp_path, monkeypatch):
    import src.demo.initiative_to_work as itw
    monkeypatch.setattr(itw, "MIN_RELEVANCE", 0.99)
    con = _mini_db(tmp_path / "t2.duckdb")
    out = itw.analyze("Launch a customer support desk", con=con, embedder=FakeEmbedder(), persist=False)
    assert out["status"] == "insufficient_evidence" and out["results"] == []
