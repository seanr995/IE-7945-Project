"""Tests for the rebuilt Sprint 3 runtime in CAPESTONE PROJECT_READY. No real Anthropic call is ever made."""
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agentic_mock_world import ORIGINAL, READY, make_world, mock_claude  # noqa: E402

from agentic_taxonomy_src import decision as dec  # noqa: E402
from agentic_taxonomy_src import evaluation as MX  # noqa: E402
from agentic_taxonomy_src import llm_client as lc  # noqa: E402
from agentic_taxonomy_src import methods as M  # noqa: E402
from agentic_taxonomy_src import prompts  # noqa: E402
from agentic_taxonomy_src.agentic import AgenticRun  # noqa: E402
from agentic_taxonomy_src.budget import ApiBudget, BudgetExceeded, allocate  # noqa: E402
from agentic_taxonomy_src.cache import ResponseCache  # noqa: E402
from agentic_taxonomy_src.config import StopRun  # noqa: E402
from agentic_taxonomy_src.granularity import matched_grid  # noqa: E402
from agentic_taxonomy_src.provenance import safe_zip  # noqa: E402
from agentic_taxonomy_src.registry import TopicRegistry  # noqa: E402
from agentic_taxonomy_src.schemas import SCHEMAS, TopicCountHintError, assert_no_topic_count_hint  # noqa: E402

ALLOWED = {"model", "max_tokens", "system", "messages"}
FORBIDDEN = {"temperature", "top_p", "top_k", "tools", "tool_choice"}
HISTORICAL = ["sprint3_task_input.csv", "sprint3_colab_experiment.py", "sprint3_advanced_challengers.py",
              "sprint3_results/run_manifest.json", "sprint3_results/evaluation/clustering_experiments.csv",
              "sprint3_results/embeddings/sentence_t5_embeddings.npy", "sprint3_results/embeddings/e5_embeddings.npy",
              "sprint3_advanced_results/run_manifest.json", "sprint3_advanced_results/advanced_challenger_report.md"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def budget(tmp, start=True):
    b = ApiBudget(tmp / "provenance" / "api_budget.json")
    if start:
        b.start_or_resume_run()
    return b


def client(tmp, transport=None, b=None):
    return lc.ClaudeClient(transport or lc.MockTransport(responder=mock_claude), b or budget(tmp), ResponseCache(tmp / "cache"),
                           "claude-sonnet-5-5", tmp / "provenance", 8000)


def disc_prompt():
    return prompts.discovery([{"code": "s0", "text": "Financial Reporting task 1"}, {"code": "s1", "text": "Financial Reporting task 2"}])


# ---------------------------------------------------------------- real transport, fake SDK client (kwargs captured)
class FakeMessages:
    def __init__(self, reply=None, exc=None):
        self.kwargs, self.reply, self.exc = [], reply, exc

    def create(self, **kw):
        self.kwargs.append(kw)
        if self.exc:
            raise self.exc
        return NS(content=[NS(type="text", text=self.reply)], stop_reason="end_turn", model=kw["model"],
                  usage=NS(input_tokens=10, output_tokens=5))


def real_transport(reply=None, exc=None):
    t = lc.AnthropicTransport("sk-ant-test-not-real-000000000000")      # SDK client is built; no network at init
    t.client = NS(messages=FakeMessages(reply, exc), models=NS(retrieve=lambda m: NS(id=m, display_name="x", created_at="")))
    return t


def test_01_to_06_request_kwargs_exactly_supported_fields():
    t = real_transport(reply=json.dumps({"work_types": []}))
    req = lc.build_request("claude-sonnet-5-5", 8000, "sys", [{"role": "user", "content": "x"}])
    t.create(req)
    sent = t.client.messages.kwargs[0]
    assert set(sent) == ALLOWED                                    # 1 only supported fields
    for k in FORBIDDEN:                                            # 2-6 temperature, top_p, top_k, tools, tool_choice
        assert k not in sent
        with pytest.raises(StopRun):
            t.create({**req, k: 0})                                 # transport refuses them even if someone adds one
    assert len(t.client.messages.kwargs) == 1


def test_full_mock_run_sends_only_supported_fields(tmp_path):
    t = lc.MockTransport(responder=mock_claude)
    AgenticRun(make_world(), client(tmp_path, t), tmp_path, allocate(15)).run()
    assert t.requests and all(set(r) == ALLOWED for r in t.requests)
    assert all(r["messages"][-1]["content"].rstrip().endswith("}") for r in t.requests)    # every prompt ends with the contract
    assert all("Return ONLY valid JSON" in r["messages"][-1]["content"] for r in t.requests)


def test_07_plain_fenced_and_trailing_comma_json_parse(tmp_path):
    body = json.dumps({"decisions": [{"topic_id": "AT_0001", "action": "keep"}]})
    for text in (body, f"```json\n{body}\n```", "Sure:\n" + body + "\nThanks", body.replace("}]}", "},]}")):
        obj, steps = lc.extract_json(text)
        assert SCHEMAS["critic"].model_validate(obj).decisions[0].action == "KEEP"
    c = client(tmp_path, real_transport(reply="```json\n" + json.dumps({"work_types": [], "statements_not_work": []}) + "\n```"))
    data, info = c.call("discovery", *disc_prompt(), "v0")
    assert data["work_types"] == [] and not info["cache_hit"]


def test_08_invalid_json_triggers_local_validation_and_one_repair(tmp_path):
    t = lc.MockTransport(queue=['{"work_types": [{"definition": "missing name"}]}',
                                json.dumps({"work_types": [], "statements_not_work": []})])
    c = client(tmp_path, t)
    c.call("discovery", *disc_prompt(), "v0")
    assert len(t.requests) == 2 and t.requests[1]["messages"][1]["role"] == "assistant"
    assert "did not match the required JSON structure" in t.requests[1]["messages"][2]["content"]
    ev = [e for e in c.budget.state["events"] if e.get("call")]
    assert [e["kind"] for e in ev] == ["normal", "repair"] and ev[0]["status"] == "invalid_json" and ev[1]["status"] == "ok"


def test_09_repair_respects_budget(tmp_path):
    b = ApiBudget(tmp_path / "b.json")
    b.state["calls_made"] = 13                                     # 7 remain -> too few for a plan
    with pytest.raises(BudgetExceeded):
        b.start_or_resume_run()
    b.state["calls_made"] = 12                                     # 8 remain -> plan 6 normal + 2 reserve
    b.start_or_resume_run()
    t = lc.MockTransport(responder=lambda r: "not json")
    c = lc.ClaudeClient(t, b, ResponseCache(tmp_path / "c"), "m", tmp_path, 8000)
    for i in range(2):
        with pytest.raises(BudgetExceeded):
            c.call("discovery", *prompts.discovery([{"code": "s0", "text": f"task {i}"}]), "v")
    with pytest.raises(BudgetExceeded):                            # reserve (2) used by the two repairs: no third repair
        c.call("consolidation", *prompts.consolidation([]), "v")
    assert b.state["calls_made"] == 12 + 5 <= 20 and b.summary()["run_reserve_used"] == 2
    assert not list((tmp_path / "c").rglob("*.json"))              # nothing invalid was cached


def test_10_api_400_surfaced_clearly(tmp_path):
    import anthropic
    import httpx
    err = anthropic.BadRequestError("bad", response=httpx.Response(400, request=httpx.Request("POST", "https://api.anthropic.com")),
                                    body={"error": {"type": "invalid_request_error", "message": "temperature is not supported"}})
    c = client(tmp_path, real_transport(exc=err))
    with pytest.raises(StopRun, match=r"rejected the request \(400\).*temperature is not supported"):
        c.call("discovery", *disc_prompt(), "v0")
    ev = [e for e in c.budget.state["events"] if e.get("call")]
    assert len(ev) == 1 and ev[0]["status"] == "error:bad_request:400"           # no retry on 400


def test_11_failed_request_does_not_corrupt_outputs(tmp_path):
    calls = {"n": 0}

    def flaky(req):
        calls["n"] += 1
        if calls["n"] == 3:
            raise lc.ClaudeAPIError(400, "bad_request", "boom")
        return mock_claude(req)
    c = client(tmp_path, lc.MockTransport(responder=flaky))
    with pytest.raises(StopRun):
        AgenticRun(make_world(), c, tmp_path, allocate(15)).run()
    assert not (tmp_path / "assignments" / "assignments_final.csv").exists()
    assert not (tmp_path / "agentic_summary.json").exists()
    assert len(list((tmp_path / "cache").rglob("*.json"))) == 2                  # only the 2 valid replies cached
    b = json.loads((tmp_path / "provenance" / "api_budget.json").read_text())
    assert b["calls_made"] == 3 and b["events"][-1]["status"].startswith("error")


def test_12_budget_persists_preserved_never_reset(tmp_path):
    p = tmp_path / "provenance" / "api_budget.json"
    p.parent.mkdir(parents=True)
    old = {"hard_limit": 20, "operational_limit": 16, "calls_made": 5, "calls_by_role": {"discovery": 5},
           "events": [{"call": i, "role": "discovery", "status": "sent"} for i in range(1, 6)]}   # earlier implementation
    p.write_text(json.dumps(old))
    b = ApiBudget(p, hard_limit=99)
    assert b.state["calls_made"] == 5 and b.remaining == 15 and b.state["hard_limit"] == 20
    run = b.start_or_resume_run()
    assert run["normal"] == 13 and run["reserve"] == 2
    b.authorize("discovery")
    b2 = ApiBudget(p)                                             # restart
    assert b2.state["calls_made"] == 6 and b2.start_or_resume_run()["run_id"] == run["run_id"]   # same plan resumed
    ro = ApiBudget(tmp_path / "x" / "api_budget.json", readonly=True)
    assert not (tmp_path / "x" / "api_budget.json").exists() and ro.remaining == 20
    (tmp_path / "y").mkdir()
    (tmp_path / "y" / "api_budget_prior_usage.json").write_text(json.dumps({"calls_already_made_floor": 5}))
    assert ApiBudget(tmp_path / "y" / "api_budget.json").state["calls_made"] == 5                # floor applied, no file
    assert ApiBudget(p).state["calls_made"] == 6                                                    # higher real count kept


def test_allocation_scales_to_real_remaining():
    a20, a15 = allocate(20), allocate(15)
    assert (a20["normal"], a20["reserve"]) == (16, 4) and (a15["normal"], a15["reserve"]) == (13, 2)
    assert sum(a15["role_caps"].values()) == 13 and a15["role_caps"]["discovery"] == 4
    assert allocate(9)["normal"] == 7
    with pytest.raises(BudgetExceeded):
        allocate(7)


def test_slot_committed_before_send(tmp_path):
    class Boom(lc.MockTransport):
        def create(self, req):
            raise KeyboardInterrupt
    c = client(tmp_path, Boom())
    with pytest.raises(KeyboardInterrupt):
        c.call("discovery", *disc_prompt(), "v0")
    assert ApiBudget(tmp_path / "provenance" / "api_budget.json").state["calls_made"] == 1


def test_cache_prevents_duplicate_requests(tmp_path):
    t = lc.MockTransport(responder=mock_claude)
    a, i1 = client(tmp_path, t).call("discovery", *disc_prompt(), "v0")
    b, i2 = client(tmp_path, t, ApiBudget(tmp_path / "provenance" / "api_budget.json")).call("discovery", *disc_prompt(), "v0")
    assert a == b and i2["cache_hit"] and len(t.requests) == 1


def test_model_validation_metadata_once(tmp_path):
    b = budget(tmp_path)
    t = real_transport()
    r1 = lc.model_validation(t, "claude-sonnet-5-5", b, tmp_path)
    r2 = lc.model_validation(t, "claude-sonnet-5-5", b, tmp_path)
    assert r1 == r2 and b.state["metadata_requests"] == 1 and b.state["calls_made"] == 0


# ---------------------------------------------------------------- taxonomy mechanics
def test_no_predefined_topic_count():
    for bad in ("Produce 100 topics.", "target_topics", "aim for 80-120 categories", "use k=100", "n_clusters"):
        with pytest.raises(TopicCountHintError):
            assert_no_topic_count_hint(bad)
    card = {"topic_id": "AT_0001", "topic_name": "X", "definition": "d"}
    for role, args in (("discovery", ([{"code": "s1", "text": "Lead 2 groups"}],)), ("consolidation", ([],)),
                       ("boundary_review", ([card],)), ("ambiguous", ([], [])), ("novelty", ([],)), ("critic", ([],)),
                       ("consistency", ([card],))):
        system, instr, data, contract = prompts.BUILDERS[role](*args)
        assert_no_topic_count_hint(system + instr + contract)
        assert "k-means" not in (system + instr).lower() and "o*net" not in (system + instr).lower()


def test_mock_run_recovers_structure_without_count(tmp_path):
    t = lc.MockTransport(responder=mock_claude)
    c = client(tmp_path, t)
    s = AgenticRun(make_world(), c, tmp_path, c.budget.run).run()
    assert s["N_AGENTIC"] == 6 and s["mode"] == "full" and s["stop_reason"]
    assert c.budget.summary()["run_normal_used"] <= 13
    n = len(t.requests)
    c2 = client(tmp_path, t, ApiBudget(tmp_path / "provenance" / "api_budget.json"))
    AgenticRun(make_world(), c2, tmp_path, c2.budget.start_or_resume_run()).run()                 # restart -> all cached
    assert len(t.requests) == n


def test_registry_ids_and_provenance():
    r = TopicRegistry()
    a, b, c = r.add("A", "a", supporting_statement_ids=["x1"]), r.add("A2", "a"), r.add("C", "c", supporting_statement_ids=["x2", "x3"])
    assert (a, b, c) == ("AT_0001", "AT_0002", "AT_0003") and r.merge(b, a, "consolidation")
    assert r.get(b)["merged_into"] == a and r.resolve(b) == a
    kids = r.split(c, [{"name": "C1", "supporting_statement_ids": ["x2"]}, {"name": "C2", "supporting_statement_ids": ["x3"]}], "critic")
    assert kids == ["AT_0004", "AT_0005"] and r.get(c)["status"] == "split" and r.add("N", "n") == "AT_0006"


def test_grid_and_fitters():
    assert matched_grid(37)["grid"] == [30, 37, 44] and matched_grid(5)["grid"] == [3, 5, 7]
    Z = make_world(n_per_type=20, dims=24, seed=3)["X_t5"].astype(np.float64)
    assert len(set(M.fit_kmeans(Z, 6)[1])) == 6 and M.fit_gmm(Z, 6)[2].shape[1] == 6
    g, _ = M.knn_graph(Z.astype(np.float32), 10)
    info, lab, T = M.leiden_nearest_count(g, 6)
    assert info["actual_count"] == len(set(lab)) and info["abs_difference"] == int(T.abs_diff.min())
    noisy = lambda Zm, a, b, c: (None, np.r_[np.zeros(5, int), -np.ones(len(Zm) - 5, int)], None)
    G, picks, _ = M.hdbscan_search(Z, [6], grid={"min_cluster_size": [5], "min_samples": [1], "selection": ["eom"]}, fit=noisy)
    assert G.rejected_excessive_noise.all() and picks[6]["feasible"] is False


def test_bertopic_reduced_is_labelled_post_hoc():
    class FakeBT:
        def fit(self, docs, embeddings=None, y=None):
            self.topics_ = list(y)

        def reduce_topics(self, docs, nr_topics):
            k = nr_topics - (1 if min(self.topics_) < 0 else 0)
            self.topics_ = [t % k if t >= 0 else -1 for t in self.topics_]
    nat = np.array([i % 8 for i in range(38)] + [-1, -1])
    red, info = M.bertopic_reduce([f"d{i}" for i in range(40)], np.zeros((40, 3)), nat, 4, model_factory=FakeBT)
    assert info["variant"] == "POST_HOC_REDUCED_TO_AGENTIC_GRANULARITY" and info["actual_count"] == 4
    assert M.bertopic_reduce(["d"] * 40, np.zeros((40, 3)), nat, 12, model_factory=FakeBT)[1]["applicable"] is False


def test_claims_and_decision():
    C = pd.DataFrame({"label": ["A", "B"], "low_confidence_rate": [0.1, 0.3]})
    good = {"l": {**dec.superlative(C, "low_confidence_rate", "min"), "table": "t"}}
    assert dec.verify_claims(good, {"t": C})[0]["consistent"]
    assert not dec.verify_claims({"l": {**good["l"], "winners": ["B"]}}, {"t": C})[0]["consistent"]
    base = dict(coverage_work=1.0, stability_ari=0.6, persistence_share_j50=0.6, dominant_share=0.05, tiny_cluster_share=0,
                noise_share=0, complexity_rank=1, human_preference_share=np.nan)
    res, _ = dec.decide(pd.DataFrame([{**base, "family": "leiden", "label": "le"}], index=["LE"]), False)
    assert res["outcome"] == "LEIDEN SELECTED" and res["freeze_ready"] is False


def test_zip_excludes_env_and_aborts_on_secret(tmp_path):
    d = tmp_path / "r"
    (d / "a").mkdir(parents=True)
    (d / ".env").write_text("ANTHROPIC_API_KEY=sk-ant-REALLYSECRET1234567890")
    (d / "a" / "ok.csv").write_text("x\n1\n")
    safe_zip(d, tmp_path / "z.zip")
    assert zipfile.ZipFile(tmp_path / "z.zip").namelist() == ["a/ok.csv"]
    (d / "a" / "leak.md").write_text("sk-ant-REALLYSECRET1234567890")
    with pytest.raises(StopRun):
        safe_zip(d, tmp_path / "z2.zip")


# ---------------------------------------------------------------- CLI + integration
def synthetic_data():
    w = make_world(n_per_type=30, dims=48, seed=7)
    rng = np.random.default_rng(1)
    X = w["X_t5"]
    rows = np.arange(0, len(X), 6)
    P = X[rows] + 0.05 * rng.normal(size=(len(rows), X.shape[1])).astype(np.float32)
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    X5 = X + 0.08 * rng.normal(size=X.shape).astype(np.float32)
    X5 /= np.linalg.norm(X5, axis=1, keepdims=True)
    return {**w, "P_t5": P, "X_e5": X5, "P_e5": X5[rows], "input_sha256": "synthetic", "hashes": {}, "checks": {}, "manifest": {},
            "paired": pd.DataFrame({"row": rows, "statement_id": w["df"].statement_id.values[rows], "paired_text": "p"})}


def _cli(name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, READY / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_agentic_cli_requires_preflight_then_runs(tmp_path, monkeypatch):
    import agentic_taxonomy_src.semantic_index as si
    data = synthetic_data()
    mt = lc.MockTransport(responder=mock_claude)
    monkeypatch.setattr(lc, "AnthropicTransport", lambda key, **k: mt)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy-key-not-real-0000000")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
    cli = _cli("sprint3_agentic_taxonomy")
    monkeypatch.setattr(si, "load_project", lambda project, need_e5=False: data)
    assert cli.main(["--project-dir", str(tmp_path)]) == 1 and not mt.requests               # no PASS preflight -> refuses
    import agentic_taxonomy_src
    from agentic_taxonomy_src.provenance import code_fingerprint, write_json
    write_json(tmp_path / "agentic_taxonomy_results" / "provenance" / "preflight_status.json",
               {"status": "PASS", "model": "claude-sonnet-5-5",
                "code_fingerprint": code_fingerprint(Path(agentic_taxonomy_src.__file__).parent)})
    (tmp_path / "agentic_taxonomy_results" / "provenance" / "api_budget_prior_usage.json").write_text(
        json.dumps({"calls_already_made_floor": 5}))
    assert cli.main(["--project-dir", str(tmp_path)]) == 0
    out = tmp_path / "agentic_taxonomy_results"
    b = json.loads((out / "provenance" / "api_budget.json").read_text())
    assert b["calls_made"] == 5 + len(mt.requests) <= 20 and all(set(r) == ALLOWED for r in mt.requests)
    n = len(mt.requests)
    assert cli.main(["--project-dir", str(tmp_path)]) == 0 and len(mt.requests) == n        # completed run is not re-run
    names = zipfile.ZipFile(tmp_path / "Sprint3_Agentic_Taxonomy_Results.zip").namelist()
    assert "agentic_summary.json" in names and not any(".env" in x for x in names)
    blob = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in out.rglob("*") if p.suffix in (".json", ".jsonl", ".md", ".csv"))
    assert "sk-ant-dummy" not in blob


def test_integration_agentic_then_benchmark(tmp_path, monkeypatch):
    before = {p: sha(READY / p) for p in HISTORICAL}
    data = synthetic_data()
    ag = tmp_path / "agentic"
    c = client(ag)
    AgenticRun(data, c, ag, c.budget.run).run()
    from agentic_taxonomy_src import benchmark as BM
    monkeypatch.setattr(BM, "load_project", lambda project, need_e5=True: data)
    man = BM.Benchmark(READY, ag, tmp_path / "final").run()
    out = tmp_path / "final"
    G = pd.read_csv(out / "evaluation" / "matched_granularity_comparison.csv")
    assert set(G.grid_point) >= {"N_LOW", "N_AGENTIC", "N_HIGH"}
    for f in ("final_method_comparison.csv", "cross_method_agreement.csv", "topic_persistence.csv", "human_review_sample.csv",
              "api_usage.csv", "paraphrase_consistency.csv", "provider_robustness.csv"):
        assert (out / "evaluation" / f).exists()
    H = pd.read_csv(out / "evaluation" / "human_review_sample.csv", keep_default_na=False)
    assert (H[["human_best_topic", "human_assignment_correct", "human_notes"]] == "").all().all()
    assert json.loads((out / "audit" / "report_consistency_check.json").read_text())["all_consistent"] is True
    assert man["claude_calls_in_benchmark"] == 0 and man["decision"]["freeze_ready"] is False
    assert {p: sha(READY / p) for p in before} == before


def test_historical_files_identical_in_ready_and_original():
    for p in HISTORICAL:
        assert sha(READY / p) == sha(ORIGINAL / p), p
