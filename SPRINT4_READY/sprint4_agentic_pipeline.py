"""Sprint 4 entry point (used by the Colab notebook).  All paths are relative to the project base folder
(on Colab: /content/drive/MyDrive/CAPESTONE PROJECT/).

    from sprint4_agentic_pipeline import preflight, run, validate, show
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from sprint4_src.pipeline import OUT, RETURN_ZIP, ZIP_NAME, make_settings, preflight, run, safe_config, validate  # noqa: E402,F401


def show(base):
    base = Path(base)
    R = base / "sprint4_results"
    summ = json.loads((R / "run_summary.json").read_text(encoding="utf-8"))
    led, onet = summ["ledger"], summ["onet"]
    print("=" * 72)
    print(f"SPRINT 4 {'(MOCK RUN) ' if summ['mock'] else ''}SUMMARY")
    print("=" * 72)
    print(f"Hierarchy (L1 / L2 / leaves) : {summ['level1']} / {summ['level2']} / {summ['leaves']}")
    print(f"O*NET tasks embedded         : {summ['onet_tasks']:,}")
    print(f"O*NET coverage / purity / NN : {onet['coverage_meaningful_alignment']:.1%} / {onet['purity_mean']:.3f} / {onet['nn_agreement_top10_mean']:.3f}")
    print(f"Alignment counts             : {onet['alignment_counts']}")
    print(f"LLM provider / model         : {led['provider']} / {led['model']}")
    print(f"Sprint 4 Claude calls        : {led['claude_sprint4_calls']}")
    print(f"Sprint 4 Gemini calls        : {led['gemini_real_calls']} (normal {led['normal_calls']}, repair {led['repair_calls']}, retry {led['retry_calls']}, split {led['split_calls']}; cap {led['hard_cap']}; cache hits {led['cache_hits']})")
    print(f"Sprint 4 Gemini tokens in/out: {led['input_tokens']:,} / {led['output_tokens']:,}")
    print(f"Sprint 3 Claude historical   : ${summ['historical_claude']['estimated_cost_usd']:.4f} (estimate, informational)")
    print(f"Convergence                  : {summ['convergence']['stop_reason']}")
    print(f"Outputs                      : {R}")
    sb, gp = summ["stability"], summ["gap"]["counts"]
    print(f"Hierarchy persistence L1 / L2: {sb['mean_level1_persistence']} / {sb['mean_level2_persistence']}  (ARI {sb['level1_ari_pairwise_mean']} / {sb['level2_ari_pairwise_mean']})")
    print(f"O*NET coverage classes       : {gp}")
    print(f"Zip (all results)            : {base / ZIP_NAME}")
    print(f"RETURN ZIP (download this)   : {base / RETURN_ZIP}")
    return summ


if __name__ == "__main__":
    b = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
    preflight(b)
    run(b)
    print(validate(b)["status"])
    show(b)
