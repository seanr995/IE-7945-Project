"""Unit tests for Section A (task extraction) additions. No API calls; synthetic data only (no gold labels)."""
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm.anthropic_provider import parse_anthropic_response
from src.llm.base import PermanentError, TransientError
from src.prototype import task_statements as ts
from src.prototype.freeze import title_group
from src.prototype.task_eval import fn_category, fp_category, match, prf
from src.prototype.task_gold import audit_subset, gold_tasks


class FakeSim:
    """Cosine from a fixed table; unknown pairs are dissimilar."""

    def __init__(self, table):
        self.t = {frozenset(k): v for k, v in table.items()}

    def __call__(self, a, b):
        return self.t.get(frozenset((a, b)), 0.1)


TH = {"lexical": 0.6, "semantic": 0.85, "review": 0.75}


def test_match_stages_and_one_to_one():
    gold = ["Build ETL pipelines", "Interview job candidates", "Prepare monthly budget reports"]
    pred = ["build ETL pipelines.", "Develop ETL pipelines", "Conduct applicant interviews", "Draft budget summaries"]
    sim = FakeSim({("Interview job candidates", "Conduct applicant interviews"): 0.90,
                   ("Prepare monthly budget reports", "Draft budget summaries"): 0.80})
    m = match(gold, pred, sim, TH)
    pairs = {(gold[i], pred[j]): how for i, j, how, _ in m["pairs"]}
    assert pairs[("Build ETL pipelines", "build ETL pipelines.")] == "normalized_exact"
    assert pairs[("Interview job candidates", "Conduct applicant interviews")] == "semantic"
    # one-to-one: "Develop ETL pipelines" cannot also take the already-matched gold task
    assert pred.index("Develop ETL pipelines") in m["fp"]
    assert [(gold[i], pred[j]) for i, j, _ in m["ambiguous"]] == [("Prepare monthly budget reports", "Draft budget summaries")]
    assert gold.index("Prepare monthly budget reports") in m["fn"]  # ambiguous is not a true positive


def test_prf_edge_cases():
    assert prf(2, 4, 2)["precision"] == 0.5 and prf(2, 4, 2)["recall"] == 1.0
    assert prf(0, 0, 3)["precision"] is None and prf(0, 2, 0)["recall"] is None
    assert prf(0, 2, 3)["f1"] == 0.0


def test_error_categories():
    g = ["Prepare and review budgets"]
    preds = [{"task": "Prepare budgets", "evidence_valid": True, "support_overlap": 1.0, "section": "responsibilities"},
             {"task": "Review budgets", "evidence_valid": True, "support_overlap": 1.0, "section": "responsibilities"},
             {"task": "Python", "evidence_valid": True, "support_overlap": 1.0, "section": "skills"},
             {"task": "Experience with vendors", "evidence_valid": True, "support_overlap": 1.0, "section": "responsibilities"},
             {"task": "Lead a team", "evidence_valid": False, "support_overlap": 0.2, "section": "responsibilities"}]
    pt = [p["task"] for p in preds]
    cats = [fp_category(p, pt, g) for p in preds]
    assert cats[0].startswith("compound_over_split") and cats[1].startswith("compound_over_split")
    assert cats[2] == cats[3] == "skill_or_qualification_as_task"
    assert cats[4].startswith("hallucinated_or_implied")
    assert fn_category("Prepare budgets", ["Prepare and review budgets"], ["Prepare budgets", "Review budgets"], False) \
        .startswith("compound_under_split")
    assert fn_category("Inspect sites", [], ["Inspect sites"], True) == "ambiguous_match_pending_review"
    assert fn_category("Inspect sites", [], ["Inspect sites"], False) == "missed_task"


def test_gold_reader_counts_only_labelled_postings():
    d = pd.DataFrame([
        {"posting_id": "p1", "gold_task_statement": "Inspect sites", "evidence_span": "Inspect sites", "annotator": "A",
         "adjudication_status": "labelled", "notes": ""},
        {"posting_id": "p1", "gold_task_statement": "Write reports", "evidence_span": "write reports", "annotator": "A",
         "adjudication_status": "", "notes": ""},
        {"posting_id": "p2", "gold_task_statement": "Draft only", "evidence_span": "", "annotator": "A",
         "adjudication_status": "in_progress", "notes": ""}])
    g = gold_tasks(d)
    assert g.posting_id.tolist() == ["p1", "p1"]  # p2 is still in progress


def test_audit_subset_deterministic_and_balanced():
    gold = pd.DataFrame({"posting_id": [f"p{i}" for i in range(100)],
                         "source": ["a"] * 50 + ["b"] * 50,
                         "title_group_analysis_only": [f"g{i % 7}" for i in range(100)]})
    x, y = audit_subset(gold, "seed"), audit_subset(gold, "seed")
    assert x == y and len(x) == 20 and len(set(x)) == 20
    assert sum(p in set(gold[gold.source == "a"].posting_id) for p in x) == 10
    first_a = gold.set_index("posting_id").loc[x[:7]]
    assert first_a.title_group_analysis_only.nunique() == 7  # title-group spread before filling


def test_title_group_is_keyword_only():
    assert title_group("Senior Data Engineer") == "software / data / IT"
    assert title_group("Assistant Corporation Counsel") == "legal / compliance"
    assert title_group("Zookeeper") == "unclassified"


def _row(**kw):
    base = {"evidence_match_type": "exact", "gate_status": "accepted", "experiment_arm": "baseline_gemini_groq",
            "disagreement_type": None, "adjudication_decision": None, "looks_like_qualification": False,
            "agreement_status": "lexical_agree"}
    return {**base, **kw}


def test_task_statement_review_status_rules():
    assert ts._status(_row()) == "accepted_cross_model_agreement"
    assert ts._status(_row(evidence_match_type="none", gate_status="rejected_evidence_not_found")) == "rejected_invalid_evidence"
    assert ts._status(_row(gate_status="rejected_unsupported_statement")) == "rejected_unsupported_statement"
    assert ts._status(_row(evidence_match_type="fuzzy")) == "review_required"
    assert ts._status(_row(disagreement_type="kind_conflict", agreement_status="gemini_only")) == "review_required"
    assert ts._status(_row(adjudication_decision="reject")) == "review_required"
    assert ts._status(_row(agreement_status="groq_only", disagreement_type="missing_in_other")) == "single_model_unconfirmed"
    assert ts._status(_row(agreement_status="not_compared")) == "single_model_not_compared"
    assert ts._status(_row(experiment_arm="claude_third_provider", agreement_status=None)) == "third_provider_experiment"
    assert ts.QUALIFICATION_LIKE.match("Experience managing vendor contracts")
    assert not ts.QUALIFICATION_LIKE.match("Manage vendor contracts")


def test_anthropic_parser():
    resp = NS(stop_reason="end_turn", model="claude-opus-5-5", usage=NS(input_tokens=12, output_tokens=7),
              content=[NS(type="thinking", thinking=""), NS(type="text", text='{"items": []}')])
    r = parse_anthropic_response(resp)
    assert r.text == '{"items": []}' and (r.input_tokens, r.output_tokens, r.model) == (12, 7, "claude-opus-5-5")
    with pytest.raises(PermanentError):
        parse_anthropic_response(NS(stop_reason="refusal", content=[], usage=None))
    with pytest.raises(TransientError):
        parse_anthropic_response(NS(stop_reason="max_tokens", content=[NS(type="text", text=" ")], usage=None))


def test_anthropic_provider_requires_env_key(monkeypatch):
    from src.llm.anthropic_provider import AnthropicProvider
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    p = AnthropicProvider({"model": "claude-opus-5-5"})
    with pytest.raises(PermanentError):
        _ = p.client
