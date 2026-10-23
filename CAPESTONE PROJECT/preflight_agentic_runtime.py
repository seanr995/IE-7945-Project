#!/usr/bin/env python
"""Preflight for the agentic runtime. Makes NO taxonomy (Messages) call.

    python preflight_agentic_runtime.py --project-dir "/content/drive/MyDrive/CAPESTONE PROJECT"
    python preflight_agentic_runtime.py --project-dir ... --offline     # skip the live model check (local use)

Checks: files + frozen input + embedding alignment, historical results, .env + key + model string, Anthropic SDK,
request construction (exactly model/max_tokens/system/messages; no temperature/top_p/top_k/tools/tool_choice anywhere
in the runtime source), schemas on mocked replies, mocked client round-trip (cache + repair + budget), persisted
budget (never reset; plan feasible), writable output folders, and - unless --offline - ONE model validation request
(models.retrieve, logged separately as metadata; skipped if this model was already validated).
Writes <project>/agentic_taxonomy_results/provenance/preflight_status.json. Exit code 1 on any failure.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
FORBIDDEN_SOURCE = re.compile(r"\b(temperature|top_p|top_k|tool_choice|tools)\s*=|[\"'](temperature|top_p|top_k|tool_choice|tools)[\"']\s*:")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-dir", required=True)
    ap.add_argument("--offline", action="store_true", help="skip the live model validation request")
    a = ap.parse_args()
    project = Path(a.project_dir)
    checks = []

    def check(name, fn):
        try:
            detail = fn()
            checks.append({"check": name, "status": "PASS", "detail": detail})
            print(f"PASS  {name}" + (f" - {detail}" if detail else ""), flush=True)
            return True
        except Exception as e:
            checks.append({"check": name, "status": "FAIL", "detail": f"{type(e).__name__}: {e}"})
            print(f"FAIL  {name} - {type(e).__name__}: {e}", flush=True)
            return False

    from agentic_taxonomy_src import config
    from agentic_taxonomy_src.provenance import code_fingerprint, utc, write_json
    paths = config.Paths(project)
    prov = paths.agentic / "provenance"
    state = {}

    def files():
        from agentic_taxonomy_src.semantic_index import load_project
        d = load_project(project, need_e5=True)
        state["data"] = d
        return f"{len(d['df'])} statements, T5 {d['X_t5'].shape}, E5 {'present' if d['X_e5'] is not None else 'absent'}, " \
               f"{len(d['paired'])} paired wordings, input SHA-256 frozen"
    check("input CSV + Sentence-T5 embeddings + mapping alignment", files)

    def historical():
        need = [paths.sprint3_results / "run_manifest.json", paths.sprint3_results / "evaluation" / "clustering_experiments.csv",
                paths.advanced_results / "run_manifest.json",
                paths.advanced_results / "evaluation" / "advanced_challenger_comparison.csv", project / config.SPRINT3_SCRIPT]
        miss = [str(p) for p in need if not p.exists()]
        if miss:
            raise FileNotFoundError(", ".join(miss))
        from agentic_taxonomy_src.provenance import sha256_file
        if sha256_file(project / config.SPRINT3_SCRIPT) != config.SPRINT3_SCRIPT_SHA256:
            raise ValueError("sprint3_colab_experiment.py differs from the frozen version")
        return "Sprint 3 + advanced results present; metric-definition script is the frozen version"
    check("historical results", historical)

    def env():
        if not config.load_env(project):
            raise FileNotFoundError(f"{paths.env_file} not found")
        s = config.settings()
        if not s.api_key_present:
            raise ValueError("ANTHROPIC_API_KEY missing in .env")
        if not s.model:
            raise ValueError("ANTHROPIC_MODEL missing in .env")
        state["settings"] = s
        note = "" if s.configured_output_tokens >= config.MIN_OUTPUT_TOKENS else \
            f"; AGENTIC_MAX_OUTPUT_TOKENS={s.configured_output_tokens} raised to the safe ceiling {s.max_output_tokens}"
        return f"key present (not shown); model '{s.model}'; hard limit {s.hard_limit}{note}"
    env_ok = check(".env + ANTHROPIC_API_KEY + ANTHROPIC_MODEL", env)

    def sdk():
        import anthropic
        c = anthropic.Anthropic(api_key="preflight-not-a-key")
        assert callable(c.messages.create) and callable(c.models.retrieve)
        return f"anthropic {anthropic.__version__}"
    check("Anthropic SDK import + messages/models API", sdk)

    def request():
        from agentic_taxonomy_src.llm_client import ALLOWED_REQUEST_KEYS, build_request, check_request
        req = build_request("m", 100, "sys", [{"role": "user", "content": "x"}])
        assert set(req) == set(ALLOWED_REQUEST_KEYS) == {"model", "max_tokens", "system", "messages"}
        for bad in ("temperature", "top_p", "top_k", "tools", "tool_choice"):
            try:
                check_request({**req, bad: 0})
                raise AssertionError(f"{bad} was not rejected")
            except config.StopRun:
                pass
        hits = []
        for p in sorted((HERE / "agentic_taxonomy_src").glob("*.py")) + [HERE / "sprint3_agentic_taxonomy.py"]:
            for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
                if FORBIDDEN_SOURCE.search(line) and "FORBIDDEN" not in line:
                    hits.append(f"{p.name}:{i}")
        if hits:
            raise AssertionError(f"forbidden request parameter in source: {hits}")
        return "request kwargs = model, max_tokens, system, messages; no sampling/tool parameter anywhere in the runtime"
    check("request construction (no temperature/top_p/top_k/tools/tool_choice)", request)

    def schemas():
        from agentic_taxonomy_src.llm_client import extract_json
        from agentic_taxonomy_src.schemas import SCHEMAS
        samples = mocked_replies()
        for role, S in SCHEMAS.items():
            obj, _ = extract_json(samples[role])
            S.model_validate(obj)
            obj, steps = extract_json("Here you go:\n```json\n" + samples[role] + "\n```")
            S.model_validate(obj)
        try:
            extract_json("not json at all")
            raise AssertionError("invalid text was accepted")
        except ValueError:
            pass
        return f"{len(SCHEMAS)} role schemas parse mocked plain and fenced JSON; invalid text rejected"
    check("schemas parse mocked Claude replies", schemas)

    def roundtrip():
        from agentic_taxonomy_src.budget import ApiBudget
        from agentic_taxonomy_src.cache import ResponseCache
        from agentic_taxonomy_src.llm_client import ClaudeClient, MockTransport
        from agentic_taxonomy_src import prompts
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            b = ApiBudget(td / "b.json")
            b.start_or_resume_run()
            t = MockTransport(queue=["oops not json", mocked_replies()["discovery"]])
            c = ClaudeClient(t, b, ResponseCache(td / "cache"), "mock", td, 8000)
            s, i, d, k = prompts.discovery([{"code": "s0", "text": "Prepare monthly financial reports"}])
            c.call("discovery", s, i, d, k, "v0")
            c.call("discovery", s, i, d, k, "v0")                       # cache hit
            assert len(t.requests) == 2 and b.state["calls_made"] == 2 and all(set(r) == {"model", "max_tokens", "system",
                                                                                         "messages"} for r in t.requests)
            assert t.requests[0]["messages"][-1]["content"].rstrip().endswith("}")   # prompt ends with the JSON contract
        return "invalid reply -> one repair (reserve) -> valid; identical request -> cache hit; budget counted"
    check("mocked client round-trip (repair + cache + budget)", roundtrip)

    def budget():
        from agentic_taxonomy_src.budget import ApiBudget, allocate
        s = state.get("settings") or config.settings()
        b = ApiBudget(prov / "api_budget.json", s.hard_limit, readonly=True)   # never writes
        run = b.run or {}
        if run.get("status") == "in_progress":
            plan = run
            msg = f"resuming run {run['run_id']} with its persisted plan"
        else:
            plan = allocate(b.remaining)
            msg = "new run plan"
        state["budget"] = b.summary()
        return (f"calls already made {b.state['calls_made']} of {b.state['hard_limit']} (never reset); remaining "
                f"{b.remaining}; {msg}: normal {plan['normal']} + reserve {plan['reserve']} {plan['role_caps']}")
    check("API budget (persisted, preserved, plan feasible)", budget)

    def writable():
        for d in (paths.agentic, paths.final, prov):
            d.mkdir(parents=True, exist_ok=True)
            t = d / ".preflight_write_test"
            t.write_text("ok")
            t.unlink()
        return f"{paths.agentic} and {paths.final} writable"
    check("output directories writable", writable)

    if a.offline:
        checks.append({"check": "live model validation", "status": "SKIPPED", "detail": "--offline"})
        print("SKIP  live model validation (--offline)")
    elif env_ok:
        def model():
            from agentic_taxonomy_src.budget import ApiBudget
            from agentic_taxonomy_src.llm_client import AnthropicTransport, model_validation
            s = state["settings"]
            b = ApiBudget(prov / "api_budget.json", s.hard_limit)
            r = model_validation(AnthropicTransport(config.api_key()), s.model, b, prov)
            return f"model '{r['model']}' available ({r['info'].get('display_name', '')}); metadata request, not a taxonomy call"
        check("live model validation (metadata only)", model)
    failed = [c for c in checks if c["status"] == "FAIL"]
    status = "FAIL" if failed else ("PASS_OFFLINE" if a.offline else "PASS")
    s = state.get("settings")
    write_json(prov / "preflight_status.json", {"status": status, "utc": utc(), "checks": checks, "offline": a.offline,
                                                "model": s.model if s else None,
                                                "code_fingerprint": code_fingerprint(HERE / "agentic_taxonomy_src")})
    print(f"\nPREFLIGHT {status}" + (f" - {len(failed)} failed check(s)" if failed else ""), flush=True)
    sys.exit(1 if failed else 0)


def mocked_replies() -> dict:
    return {
        "discovery": json.dumps({"work_types": [{"candidate_topic_name": "Financial Reporting", "definition": "Prepare financial reports.",
                                                 "core_action": "prepare", "object_or_outcome": "financial reports",
                                                 "inclusion_rules": ["reports"], "exclusion_rules": ["audits"],
                                                 "supporting_statement_ids": ["s0"], "representative_statement_ids": ["s0"],
                                                 "confidence": 0.9}], "unlisted_work_types": [], "statements_not_work": []}),
        "consolidation": json.dumps({"decisions": [{"pair_id": "P0_0", "decision": "keep_separate", "reason": "different action"}]}),
        "boundary_review": json.dumps({"decisions": [], "registry_consistent": True}),
        "ambiguous": json.dumps({"decisions": [{"statement_id": "s3", "decision": "ASSIGN", "primary_topic_id": "AT_0001"}]}),
        "novelty": json.dumps({"decisions": [{"group_id": "G0", "decision": "REJECT", "reason": "not coherent"}]}),
        "critic": json.dumps({"decisions": [{"topic_id": "AT_0001", "action": "KEEP"}]}),
        "consistency": json.dumps({"decisions": [], "registry_consistent": True}),
    }


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)
