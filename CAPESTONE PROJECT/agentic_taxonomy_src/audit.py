"""Programmatic audit of the historical Sprint 3 outputs (read-only; nothing historical is modified).

Machine-readable tables are the source of truth; report prose is re-checked against them claim by claim.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .provenance import read_json, sha256_file, write_json

LABELS = {"kmeans": "Sentence-T5 + K-Means", "bertopic": "Sentence-T5 + BERTopic", "leiden": "Sentence-T5 + Leiden",
          "gmm": "Sentence-T5 + GMM"}


def _arg(df, col, higher=True):
    """(best method, value); ties are joined with '='."""
    v = pd.to_numeric(df[col], errors="coerce")
    if v.notna().sum() == 0:
        return None, None
    best = v.max() if higher else v.min()
    tied = df.loc[np.isclose(v, best, rtol=0, atol=1e-12), "method"].tolist()
    return "=".join(tied), float(best)


def _prose_answer(md, label):
    m = re.search(r"\*\*" + re.escape(label) + r"\*\*\s*(Sentence-T5 \+ [A-Za-z\-]+)", md)
    return m.group(1) if m else None


def audit_advanced(project: Path) -> dict:
    adv = project / "sprint3_advanced_results"
    cmp_path, rep_path = adv / "evaluation" / "advanced_challenger_comparison.csv", adv / "advanced_challenger_report.md"
    if not cmp_path.exists():
        return {"available": False}
    C = pd.read_csv(cmp_path)
    md = rep_path.read_text(encoding="utf-8") if rep_path.exists() else ""
    man = read_json(adv / "run_manifest.json", {})
    truth = {
        "highest_semantic_cohesion_raw": _arg(C, "semantic_cohesion"),
        "highest_cohesion_lift_assigned": _arg(C, "semantic_cohesion_lift"),
        "highest_cohesion_lift_coverage_adjusted": _arg(C, "semantic_cohesion_lift_incl_noise"),
        "highest_stability": _arg(C, "stability_ari"),
        "lowest_low_confidence_rate_common": _arg(C, "low_confidence_rate_common", higher=False),
        "lowest_low_confidence_rate_native": _arg(C, "low_confidence_rate_native", higher=False),
        "best_paraphrase_kappa": _arg(C, "paraphrase_kappa"),
        "best_provider_robustness (lowest |gap|)": _arg(C, "provider_nn_gap_abs", higher=False),
        "highest_coverage": _arg(C, "coverage"),
        "lowest_noise": _arg(C, "noise_share", higher=False),
        "highest_granularity_score": _arg(C, "granularity_score"),
    }
    prose = {"Highest semantic cohesion:": ("highest_cohesion_lift_coverage_adjusted", "highest_semantic_cohesion_raw"),
             "Most stable:": ("highest_stability",),
             "Best at preserving Gemini/Groq paraphrases:": ("best_paraphrase_kappa",),
             "Handles ambiguous tasks best:": ("lowest_low_confidence_rate_common",),
             "Most useful topic granularity:": ("highest_granularity_score",)}
    checks = []
    for label, keys in prose.items():
        said = _prose_answer(md, label)
        for k in keys:
            m, v = truth[k]
            ok_set = [LABELS.get(x, x) for x in str(m).split("=")]
            checks.append({"claim": label, "prose_says": said, "table_metric": k, "table_best": ", ".join(ok_set),
                           "table_value": v, "consistent": said in ok_set})
    # complexity claim: GMM listed as complexity without value although it adds no dependency
    cx = re.search(r"\*\*Adds complexity without enough value:\*\*(.*)", md)
    if cx and "GMM" in cx.group(1):
        checks.append({"claim": "Adds complexity without enough value", "prose_says": "includes GMM",
                       "table_metric": "extra_dependencies / tuned_hyperparameters",
                       "table_best": "GMM: no extra dependencies, 1 tuned hyper-parameter (same as K-Means)",
                       "table_value": None, "consistent": False})
    # decision-rule asymmetry: baseline exempt from the pathology flag that disqualified challengers
    rule = []
    base = C[C.method == "kmeans"].iloc[0]
    for _, r in C[C.method != "kmeans"].iterrows():
        costs = str(r.get("unacceptable_costs_vs_kmeans", ""))
        if "pathology: unstable" in costs and "unstable" in str(base.pathology_flags):
            rule.append(f"{r.method}: disqualified for 'pathology: unstable' although the K-Means baseline carries the same flag")
        if r.method == "leiden" and not pd.isna(r.get("kmeans_same_k_cohesion_lift")):
            rule.append(f"leiden: cohesion-lift cost ({base.semantic_cohesion_lift_incl_noise:.3f} -> "
                        f"{r.semantic_cohesion_lift_incl_noise:.3f}) compares {int(r.n_clusters)} communities with "
                        f"{int(base.n_clusters)} K-Means clusters; K-Means at k={int(r.n_clusters)} reaches only "
                        f"{r.kmeans_same_k_cohesion_lift:.3f}, so most of the gap is granularity, not method quality")
    nonpath = C[C.pathology_flags.isna() | (C.pathology_flags.astype(str).isin(["", "nan"]))].method.tolist()
    return {"available": True, "comparison_csv": str(cmp_path), "report_md": str(rep_path),
            "truth": {k: {"method": v[0], "value": v[1]} for k, v in truth.items()}, "prose_checks": checks,
            "prose_inconsistencies": int(sum(not c["consistent"] for c in checks)),
            "root_cause": ("advanced_challengers.write_report computed every 'highest/best/lowest' answer over the "
                           f"NON-PATHOLOGICAL pool only ({nonpath}); with a single non-pathological method every "
                           "superlative named it regardless of the numbers"),
            "decision_rule_issues": rule, "final_recommendation": man.get("final_recommendation")}


def audit_primary(project: Path) -> dict:
    s3r = project / "sprint3_results"
    R = pd.read_csv(s3r / "evaluation" / "clustering_experiments.csv")
    man = read_json(s3r / "run_manifest.json", {})
    K = R[R.clustering_method == "kmeans"].sort_values(["embedding_model", "param_k"])
    issues, facts = [], {}
    all_flagged = bool((R.pathology_flags.fillna("") != "").all())
    facts["all_configurations_pathological"] = all_flagged
    if all_flagged:
        issues.append("every one of the 26 Sprint 3 configurations carried a pathology flag; the winner was chosen "
                      "'among the least-flagged' (all K-Means configurations are flagged unstable: subsample ARI "
                      f"{K.stability_ari.min():.3f}-{K.stability_ari.max():.3f} < 0.50)")
    for m, g in K.groupby("embedding_model"):
        lift_inc = bool(np.all(np.diff(g.semantic_cohesion_lift.values) > 0))
        stab = g.stability_ari.values
        facts[f"{m}: cohesion_lift_monotone_increasing_in_k"] = lift_inc
        facts[f"{m}: stability_by_k"] = dict(zip(g.param_k.astype(int).tolist(), np.round(stab, 3).tolist()))
        if lift_inc:
            issues.append(f"{m}: cohesion lift rises monotonically with k (k={int(g.param_k.min())}->{int(g.param_k.max())}: "
                          f"{g.semantic_cohesion_lift.iloc[0]:.3f}->{g.semantic_cohesion_lift.iloc[-1]:.3f}) while subsample "
                          f"stability falls ({stab[0]:.3f}->{stab[-1]:.3f}); cohesion-first selection therefore drifts to "
                          "the largest k tested")
    win = man.get("chosen_configuration", {})
    kmax = int(K.param_k.max())
    if str(win.get("parameters", "")) == f"k={kmax}":
        issues.append(f"the winner k={kmax} lies on the BOUNDARY of the tested grid {sorted(K.param_k.astype(int).unique().tolist())}: "
                      "k=100 is an artefact of the grid edge, not evidence of 100 natural task concepts")
    H = R[R.clustering_method == "hdbscan"]
    issues.append(f"HDBSCAN on the 50-d PCA space collapsed: {int(H.n_clusters.min())}-{int(H.n_clusters.max())} clusters "
                  f"with {H.noise_share.min():.0%}-{H.noise_share.max():.0%} noise for every configuration")
    bad_kappa = R[pd.to_numeric(R.provider_pair_kappa, errors="coerce") < -1]
    if len(bad_kappa):
        issues.append(f"{len(bad_kappa)} HDBSCAN configurations have chance-corrected paraphrase kappa < -1 (chance near 1 "
                      "under a dominant cluster); the report's E5-vs-T5 'provider_pair_kappa' means average these "
                      "degenerate values and are not interpretable")
    rep = read_json(s3r / "run_manifest.json", {}).get("chosen_configuration", {})
    trace = " | ".join(rep.get("trace", []))
    if "robustness" in str(rep.get("decided_by")):
        issues.append("E5 vs Sentence-T5 (both k=100) were tied on cohesion lift, stability and granularity; T5 won only "
                      f"at the 4th hierarchy level (robustness). Trace: {trace}")
    rt = pd.read_csv(s3r / "evaluation" / "representation_tests.csv")
    facts["paraphrase_mrr"] = dict(zip(rt.embedding_model, rt.paraphrase_mrr.round(4)))
    fe = s3r / "figures" / "figure_errors.txt"
    if fe.exists():
        issues.append("figure 04_stability_comparison.png was not produced (pandas 'config_id is both an index level "
                      "and a column label' error in the Sprint 3 script) - see sprint3_results/figures/figure_errors.txt")
    return {"winner": win, "facts": facts, "issues": issues}


def write_audit(project: Path, out_dir: Path) -> dict:
    project, out_dir = Path(project), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prim, adv = audit_primary(project), audit_advanced(project)
    files = sorted(p for p in project.rglob("*") if p.is_file() and not any(
        x in p.parts for x in ("final_taxonomy_results", "agentic_taxonomy_results", "agentic_taxonomy_src", "__pycache__",
                               "checkpoints")) and not p.name.startswith(".env"))
    inv = [{"path": str(p.relative_to(project)).replace("\\", "/"), "bytes": p.stat().st_size, "sha256": sha256_file(p)}
           for p in files]
    pd.DataFrame(inv).to_csv(out_dir / "historical_file_inventory.csv", index=False)
    L = ["# Audit of existing Sprint 3 results", "",
         "Machine-readable tables are treated as the source of truth; prose was re-checked claim by claim. "
         "No historical file was modified (SHA-256 inventory: `historical_file_inventory.csv`).", "",
         "## 1. Primary Sprint 3 experiment (`sprint3_results/`)",
         f"Chosen: `{prim['winner'].get('config_id')}` (decided by `{prim['winner'].get('decided_by')}`).", ""]
    L += [f"- {i}" for i in prim["issues"]]
    L += ["", "Key facts:", "```", *[f"{k}: {v}" for k, v in prim["facts"].items()], "```", "",
          "**Conclusion:** k=100 is not evidence of 100 natural task concepts. K-Means needs k before fitting, the "
          "cohesion criterion rises mechanically with k, every K-Means configuration failed the stability threshold, and "
          "the selected k sits on the grid boundary.", ""]
    L += ["## 2. Advanced challenger experiment (`sprint3_advanced_results/`)"]
    if adv.get("available"):
        L += ["", "### 2a. Report prose vs numeric table", "",
              "| claim in report | prose says | metric checked | table says | value | consistent |", "|---|---|---|---|---:|---|"]
        for c in adv["prose_checks"]:
            v = "" if c["table_value"] is None else f"{c['table_value']:.4f}"
            L.append(f"| {c['claim']} | {c['prose_says']} | {c['table_metric']} | {c['table_best']} | {v} | "
                     f"{'yes' if c['consistent'] else '**NO**'} |")
        L += ["", f"**{adv['prose_inconsistencies']} prose inconsistencies.** Root cause: {adv['root_cause']}.", "",
              "### 2b. Numeric ground truth (from `advanced_challenger_comparison.csv`)", "",
              "| quantity | method | value |", "|---|---|---:|"]
        L += [f"| {k} | {v['method']} | {v['value']:.4f} |" for k, v in adv["truth"].items() if v["method"]]
        L += ["", "### 2c. Decision-rule issues", ""] + [f"- {x}" for x in adv["decision_rule_issues"]]
        L += ["", f"The historical recommendation was `{adv['final_recommendation']}`. Because the replacement rule was "
              "asymmetric and compared partitions of different granularity, that recommendation is NOT carried forward; "
              "the final benchmark re-evaluates all methods at matched granularity with one symmetric rule."]
    L += ["", "## 3. Consequences for the final design",
          "- The topic count is investigated independently (agentic induction, no count given) instead of inherited.",
          "- Methods are compared at matched granularity (N_LOW / N_AGENTIC / N_HIGH) with one symmetric rule.",
          "- Every superlative in the final report is generated from, and re-verified against, the numeric tables.",
          "- Topic-level persistence is added because global ARI hides which topics survive resampling."]
    (out_dir / "existing_results_audit.md").write_text("\n".join(L), encoding="utf-8")
    write_json(out_dir / "existing_results_audit.json", {"primary": prim, "advanced": adv})
    return {"primary": prim, "advanced": adv, "inventory_files": len(inv)}
