#!/usr/bin/env python
"""Sprint 3 FINAL - agentic taxonomy induction (no predefined topic count).

    python sprint3_agentic_taxonomy.py --project-dir "/content/drive/MyDrive/CAPESTONE PROJECT"

Requires a passing preflight (preflight_agentic_runtime.py) for the exact current code and model. Uses the persisted
API budget (never reset; hard limit 20) and a run plan scaled to the REAL remaining calls. Completed calls are cached,
so re-running after a disconnect resumes the same run without spending budget. A run that already completed is not
re-run (use --force-new-run to start another one with whatever budget remains).
Outputs: <project>/agentic_taxonomy_results/  ZIP: <project>/Sprint3_Agentic_Taxonomy_Results.zip
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
STALE = ("topic_registry", "assignments", "refinement", "smoke", "agentic_summary.json", "agentic_report.md", "run_manifest.json")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-dir", required=True)
    ap.add_argument("--output", default=None, help="default: <project>/agentic_taxonomy_results")
    ap.add_argument("--force-new-run", action="store_true")
    ap.add_argument("--skip-preflight-check", action="store_true", help=argparse.SUPPRESS)   # offline tests only
    a = ap.parse_args(argv)

    from agentic_taxonomy_src import config
    from agentic_taxonomy_src.agentic import AgenticRun
    from agentic_taxonomy_src.budget import ApiBudget, BudgetExceeded
    from agentic_taxonomy_src.cache import ResponseCache
    from agentic_taxonomy_src import llm_client
    from agentic_taxonomy_src.provenance import code_fingerprint, log, read_json, safe_zip, utc, write_json
    from agentic_taxonomy_src.semantic_index import load_project

    project = Path(a.project_dir)
    out = Path(a.output) if a.output else config.Paths(project).agentic
    prov = out / "provenance"
    budget = None
    try:
        config.load_env(project)
        s = config.settings()
        if not a.skip_preflight_check:
            pf = read_json(prov / "preflight_status.json", {})
            fp = code_fingerprint(HERE / "agentic_taxonomy_src")
            if pf.get("status") != "PASS" or pf.get("code_fingerprint") != fp or pf.get("model") != s.model:
                raise config.StopRun("preflight has not PASSED for this exact code and model - run "
                                     "preflight_agentic_runtime.py first (no taxonomy call was made)")
        budget = ApiBudget(prov / "api_budget.json", s.hard_limit)
        run = budget.run or {}
        if run.get("status") == "completed" and (out / "agentic_summary.json").exists() and not a.force_new_run:
            log(f"agentic run {run['run_id']} already completed - reusing its results (no API call)")
            return 0
        if run.get("status") != "in_progress":                       # new run: archive outputs of earlier attempts
            stale = [n for n in STALE if (out / n).exists()]
            if stale:
                dst = out / "_superseded" / utc().replace(":", "").replace("-", "")[:15]
                dst.mkdir(parents=True, exist_ok=True)
                for n in stale:
                    shutil.move(str(out / n), str(dst / n))
                log(f"archived outputs of an earlier attempt to {dst} (budget and cache kept)")
        data = load_project(project)
        log(f"validated frozen input: {len(data['df'])} statements, T5 embeddings {data['X_t5'].shape}")
        plan = budget.start_or_resume_run()
        log(f"budget: {budget.state['calls_made']} of {budget.state['hard_limit']} calls already made; run {plan['run_id']} "
            f"plan = {plan['normal']} normal + {plan['reserve']} reserve {plan['role_caps']}")
        transport = llm_client.AnthropicTransport(config.api_key())
        client = llm_client.ClaudeClient(transport, budget, ResponseCache(out / "cache"), s.model, prov, s.max_output_tokens)
        summary = AgenticRun(data, client, out, plan, s.max_refinement_rounds).run()
        if summary.get("N_AGENTIC") is None:
            raise config.StopRun(f"agentic run produced no taxonomy: {summary.get('stop_reason')}")
        budget.finish_run("completed")
        summary.update({"model": s.model, "budget": budget.summary(), "run_id": plan["run_id"]})
        write_json(out / "agentic_summary.json", summary)
        manifest = {"created_utc": utc(), "model": s.model, "request_style": "Messages API: model, max_tokens, system, messages; "
                    "JSON text parsed + validated locally; no tool calling; no sampling parameters",
                    "max_output_tokens": s.max_output_tokens, "input_sha256": data["input_sha256"],
                    "embedding_hashes": data["hashes"], "predefined_topic_count": None, "onet_used": False,
                    "kmeans_used_for_sampling": False, "run_plan": plan, "budget": budget.summary(),
                    "cache": {"hits": client.cache.hits, "misses": client.cache.misses},
                    "N_AGENTIC": summary["N_AGENTIC"], "stop_reason": summary["stop_reason"],
                    "code_fingerprint": code_fingerprint(HERE / "agentic_taxonomy_src")}
        write_json(out / "run_manifest.json", manifest)
        write_report(out, summary, manifest)
        z = safe_zip(out, project / "Sprint3_Agentic_Taxonomy_Results.zip")
        log(f"ZIP {z['zip']} ({z['files']} files)")
        log(f"DONE  N_AGENTIC={summary['N_AGENTIC']} | coverage {summary['coverage']:.3f} | {summary['stop_reason']} | "
            f"calls {budget.state['calls_made']}/{budget.state['hard_limit']}")
        return 0
    except (config.StopRun, BudgetExceeded, llm_client.ClaudeAPIError) as e:
        if budget is not None and isinstance(e, config.StopRun) and "rejected the request (400)" in str(e):
            budget.finish_run("failed_api_400")
        print(f"\nSTOP: {e}\n", file=sys.stderr, flush=True)
        return 1


def write_report(out, summary, manifest):
    import pandas as pd
    reg = pd.read_csv(out / "topic_registry" / "topic_registry_final.csv")
    act = reg[reg.status == "active"].sort_values("primary_members", ascending=False)
    L = ["# Agentic taxonomy induction", "", f"- model `{manifest['model']}` | request style: {manifest['request_style']}",
         f"- **N_AGENTIC = {summary['N_AGENTIC']}** (no topic count was given to Claude)",
         f"- stop reason: {summary['stop_reason']}",
         f"- coverage {summary['coverage']:.3f}; ambiguity (low-confidence automatic) {summary['ambiguity_rate']:.3f}; "
         f"sources {summary['assignment_sources']}",
         f"- Claude calls: {manifest['budget']['calls_made']} of {manifest['budget']['hard_limit']} total; this run "
         f"{manifest['budget']['run_normal_used']} normal + {manifest['budget']['run_reserve_used']} reserve; cache hits "
         f"{manifest['cache']['hits']}", f"- skipped decisions: {len(summary.get('skipped_decisions') or [])}", "",
         "| topic_id | name | members | definition |", "|---|---|---:|---|"]
    L += [f"| {r.topic_id} | {r.topic_name} | {r.primary_members} | {str(r.definition)[:160]} |" for r in act.itertuples()]
    (out / "agentic_report.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
