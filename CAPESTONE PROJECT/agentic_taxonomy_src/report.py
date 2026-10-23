"""Final Sprint 3 report. Every superlative is a `claim` computed from the numeric tables, rendered from the claim
object, and independently re-verified (decision.verify_claims). A failed verification is stamped at the top."""
from __future__ import annotations

import json
import traceback

import numpy as np
import pandas as pd

from . import decision as dec
from .provenance import log, write_json


def f(x, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}"
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def pct(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * float(x):.1f}%"


def tbl(df, cols):
    s = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.to_dict("records"):
        s.append("| " + " | ".join(f(r.get(c)) if isinstance(r.get(c), (float, np.floating, int, np.integer))
                                   and not isinstance(r.get(c), bool) else str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(s)


def claim(C, table, metric, direction, label_col="label"):
    s = dec.superlative(C, metric, direction, label_col)
    s.update({"table": table, "label_col": label_col})
    return s


def say(c):
    return (" = ".join(c["winners"]) + f" ({c['metric']} {f(c['value'])})") if c["winners"] else f"n/a ({c['metric']} unavailable)"


def write_final_report(b, R, budget, figs):
    try:
        return _write(b, R, budget, figs)
    except Exception as e:
        log(f"report failed: {e}\n{traceback.format_exc()}")
        (b.out / "final_sprint3_report.md").write_text(f"# Report generation failed\n\n```\n{traceback.format_exc()}\n```",
                                                       encoding="utf-8")
        return {"error": str(e)}


def _write(b, R, budget, figs):
    g, C, D, S = b.grid, b.C, b.decision, b.summary
    n = g["N_AGENTIC"]
    tables = {"final": C, "all": R}
    claims = {"q12_global_stability": claim(C, "final", "stability_ari", "max"),
              "q13_topic_persistence": claim(C, "final", "persistence_share_j50", "max"),
              "q13b_persistence_median": claim(C, "final", "persistence_median_jaccard", "max"),
              "q14_paraphrase": claim(C, "final", "paraphrase_kappa", "max"),
              "cohesion_lift": claim(C, "final", "semantic_cohesion_lift_incl_unassigned", "max"),
              "lowest_ambiguity": claim(C, "final", "low_confidence_rate", "min"),
              "clearest_boundaries": claim(C, "final", "centroid_inconsistency_rate", "min")}
    if b.human:
        claims["q15_human"] = claim(C, "final", "human_preference_share", "max")
    check = dec.verify_claims(claims, tables)
    # the decision itself is re-derived independently
    redo, _ = dec.decide(C.drop(columns=["gate_failures", "selected"], errors="ignore"), b.human is not None)
    check.append({"claim": "decision", "metric": "evidence hierarchy",
                  "stated_winners": [D.get("selected") or D.get("provisional_leader")],
                  "consistent": (redo.get("selected") or redo.get("provisional_leader")) == (D.get("selected") or D.get("provisional_leader"))})
    ok = all(c["consistent"] for c in check)
    write_json(b.out / "audit" / "report_consistency_check.json", {"all_consistent": ok, "checks": check, "claims": claims})

    sel = D.get("selected") or D.get("provisional_leader")
    W = C.loc[sel]
    km = R[(R.family == "kmeans") & (R.embedding == "t5") & R.partition_id.str.contains("@")].sort_values("target_count")
    gm = R[R.family == "gmm"].sort_values("target_count")
    le = R[(R.family == "leiden") & R.partition_id.str.contains("@")].sort_values("target_count")
    bt = R[R.family.str.startswith("bertopic")]
    ag, hy = R.loc["AGENTIC_FULL"], R.loc["HYBRID_PROTOTYPE"]
    km_n = R.loc[f"KMEANS_T5@{n}"] if f"KMEANS_T5@{n}" in R.index else None
    conv = pd.DataFrame(S.get("convergence") or [])
    audit_issues = b.audit["primary"]["issues"]
    q = b.summary.get("queue") or {}
    disc = pd.read_csv(b.agentic_dir / "provenance" / "discovery_calls.csv") if (b.agentic_dir / "provenance" / "discovery_calls.csv").exists() else pd.DataFrame()
    saturated = int(disc.saturated.sum()) if "saturated" in disc else 0

    # Q6: E5 vs T5 at matched granularity
    e5_lines, votes = [], {"t5": 0, "e5": 0, "tie": 0}
    for t in g["grid"]:
        a, c5 = f"KMEANS_T5@{t}", f"KMEANS_E5@{t}"
        if a in R.index and c5 in R.index:
            for m, tol in (("stability_ari", 0.05), ("persistence_share_j50", 0.05), ("paraphrase_kappa", 0.05),
                           ("semantic_cohesion_lift_incl_unassigned", 0.02)):
                d = (R.loc[a, m] or 0) - (R.loc[c5, m] or 0)
                w = "t5" if d > tol else ("e5" if d < -tol else "tie")
                votes[w] += 1
            e5_lines.append(f"k={t}: T5 stability {f(R.loc[a, 'stability_ari'])} vs E5 {f(R.loc[c5, 'stability_ari'])}; "
                            f"persistence(J>=.5) {pct(R.loc[a, 'persistence_share_j50'])} vs {pct(R.loc[c5, 'persistence_share_j50'])}; "
                            f"paraphrase kappa {f(R.loc[a, 'paraphrase_kappa'])} vs {f(R.loc[c5, 'paraphrase_kappa'])}; "
                            f"cohesion lift {f(R.loc[a, 'semantic_cohesion_lift_incl_unassigned'])} vs {f(R.loc[c5, 'semantic_cohesion_lift_incl_unassigned'])}")
    e5_verdict = ("no E5 embeddings available" if not e5_lines else
                  "T5 better" if votes["t5"] > votes["e5"] + votes["tie"] / 2 else
                  "E5 better" if votes["e5"] > votes["t5"] + votes["tie"] / 2 else "no material difference (within tolerances)")
    X = b.X_agree.sort_values("ari", ascending=False)
    cons = b.cons.sort_values("size", ascending=False)
    core_all = float((b.cb.member_class == "CORE").mean())
    hyb_vs_ag = {m: (hy.get(m), ag.get(m)) for m in ("stability_ari", "persistence_share_j50", "paraphrase_kappa",
                                                      "semantic_cohesion_lift_incl_unassigned", "low_confidence_rate", "coverage_work")}

    L = []
    if not ok:
        L += ["> **REPORT CONSISTENCY CHECK FAILED** - see `audit/report_consistency_check.json`. Treat prose with caution.", ""]
    L += ["# Final Sprint 3 Report - agentic taxonomy induction + matched-granularity benchmark", "",
          "All numbers are generated from the result tables; every superlative below was re-verified programmatically "
          f"(`audit/report_consistency_check.json`: all_consistent={ok}). No metric here is an accuracy: no topic-level "
          "gold labels exist. O*NET was not used (Sprint 4).", "", "> **Known limitation.** " + b.s3.LIMITATION, "",
          "## Decision summary",
          f"- **Outcome: {D['outcome']}**" + (f" - `{sel}` ({W['label']})" if sel else ""),
          f"- Recommended topic count: **{f(W['n_clusters'])}** (coverage of work statements {pct(W['coverage_work'])})",
          f"- Freeze status: **{'READY TO FREEZE' if D['freeze_ready'] else 'NOT READY TO FREEZE'}**" +
          ("" if D["freeze_ready"] else f" - blockers: {'; '.join(D['freeze_blockers'])}"),
          f"- Agentic Claude calls: {budget.get('calls_made')} of hard limit {budget.get('hard_limit')}; benchmark: 0", "",
          "## Candidate comparison (final_method_comparison.csv)",
          tbl(C.reset_index(drop=True), ["label", "n_clusters", "coverage_work", "stability_ari", "persistence_share_j50",
                                          "paraphrase_kappa", "centroid_inconsistency_rate", "granularity_score",
                                          "low_confidence_rate", "provider_nn_gap_abs", "semantic_cohesion_lift_incl_unassigned",
                                          "silhouette", "complexity_rank", "gate_failures"]), "",
          "Decision trace (gates, then lexicographic hierarchy with tolerances):", *[f"- {t}" for t in D["trace"]], "",
          "## Answers", "",
          "**1. Why was k=100 not considered ground truth?** " + " ".join(f"({i + 1}) {x}." for i, x in enumerate(audit_issues[:4])) +
          " K-Means needs k before fitting, so k=100 was an input, not a finding.",
          f"**2. How many topics did the agentic system discover naturally?** N_AGENTIC = **{n}** active topics with members "
          f"(registry: {S.get('topics_created_total')} IDs created; status counts {S.get('status_counts')}). No topic count "
          f"was given to Claude. Discovery calls flagged saturated: {saturated} of {len(disc)}" +
          (" - the per-call output limit may have bound discovery; N may be an underestimate." if saturated else "."),
          f"**3. Why did it converge at that number?** {S.get('stop_reason')}. Convergence trace:\n\n" +
          (tbl(conv, [c for c in ("round", "topics_before", "topics_after", "assignment_change_rate", "assignment_noise_floor",
                                  "merges", "splits", "new_topics", "duplicate_flags") if c in conv]) if len(conv) else "(no refinement rounds)"),
          f"\n**4. N_LOW / N_AGENTIC / N_HIGH:** {g['N_LOW']} / {g['N_AGENTIC']} / {g['N_HIGH']} ({g['rule']}). "
          "This is a reference granularity for a controlled comparison, not ground truth.",
          "**5. K-Means around that range:**\n\n" + tbl(km, ["target_count", "stability_ari", "persistence_share_j50",
                                                          "paraphrase_kappa", "semantic_cohesion_lift_incl_unassigned",
                                                          "low_confidence_rate", "silhouette"]) +
          f"\n\nBest K-Means count by the hierarchy: **k={b.best_counts.get('kmeans_t5', {}).get('best_count')}** "
          f"(decided by {b.best_counts.get('kmeans_t5', {}).get('decided_by')})" +
          ("" if b.best_counts.get("kmeans_t5", {}).get("best_count") == n else
           " - algorithmic quality is better at a nearby count than at N_AGENTIC."),
          f"**6. Did E5/T5 conclusions change after controlling granularity?** {e5_verdict} (votes {votes}). " + " | ".join(e5_lines) +
          " Historically T5 won only on the 4th-level robustness tie-break.",
          "**7. GMM:**\n\n" + tbl(gm, ["partition_id", "actual_count", "empty_components", "bic", "aic", "stability_ari",
                                     "persistence_share_j50", "paraphrase_kappa", "posterior_entropy_norm", "low_confidence_rate",
                                     "semantic_cohesion_lift_incl_unassigned"]),
          "\n**8. Could Leiden naturally reproduce similar granularity?**\n\n" + tbl(le, ["target_count", "actual_count",
                                                                                     "abs_difference", "resolution", "k_neighbors",
                                                                                     "close_match", "modularity"]),
          "\n**9. Leiden stability at comparable granularity:** " + "; ".join(
              f"target {int(r.target_count)}: Leiden {f(r.stability_ari)} vs K-Means {f(R.loc[f'KMEANS_T5@{int(r.target_count)}', 'stability_ari'])}"
              for r in le.itertuples() if f"KMEANS_T5@{int(r.target_count)}" in R.index) + ".",
          "**10. BERTopic natural vs reduced:**\n\n" + (tbl(bt, ["label", "n_clusters", "noise_share", "coverage_work", "stability_ari",
                                                               "persistence_share_j50", "paraphrase_kappa",
                                                               "semantic_cohesion_lift_incl_unassigned"]) if len(bt) else
                                                         f"BERTopic stage unavailable: {b.bt_info}") +
          f"\n\nReduction: {json.dumps({k: v for k, v in b.bt_info.items() if k != 'natural_to_reduced'})}. The reduced "
          "variant is a post-hoc reduction, not BERTopic's natural discovery.",
          "**11. Could HDBSCAN obtain comparable granularity without excessive noise?** " + "; ".join(
              (f"target {t}: {'nearest feasible ' + str(p['actual_count']) + ' clusters at ' + pct(p['noise_share']) + ' noise' + (' (close)' if p.get('close_match') else ' (NOT close)') if p.get('feasible') else 'infeasible - ' + p.get('reason', '')}")
              for t, p in b.hdb_picks.items()) + " (configurations above 50% noise rejected).",
          f"**12. Strongest global stability:** {say(claims['q12_global_stability'])}.",
          f"**13. Strongest topic-level persistence:** {say(claims['q13_topic_persistence'])}; median best-Jaccard: "
          f"{say(claims['q13b_persistence_median'])}.",
          f"**14. Best paraphrase preservation:** {say(claims['q14_paraphrase'])} (chance-corrected kappa).",
          "**15. Clearest human-interpretable concepts:** " + (f"{say(claims['q15_human'])} from the human review." if b.human else
           "cannot be determined numerically - the human review sample (`evaluation/human_review_sample.csv`) is not yet "
           "labelled. Structurally, only the agentic/hybrid topics carry explicit names, definitions and inclusion/exclusion "
           "rules; algorithmic topics have keywords only."),
          "**16. Where did methods agree?** Highest pairwise agreement: " + "; ".join(
              f"{r.method_a} vs {r.method_b} ARI {f(r.ari)} / matched Jaccard {f(r.matched_jaccard_mean)}" for r in X.head(3).itertuples()) +
          f". {pct(core_all)} of statements are cross-method CORE members; topics with the largest core share: " + ", ".join(
              f"{r.agentic_topic_name} ({pct(r.core_share)})" for r in cons.sort_values('core_share', ascending=False).head(5).itertuples()) + ".",
          "**17. Where did methods disagree?** Lowest agreement: " + "; ".join(
              f"{r.method_a} vs {r.method_b} ARI {f(r.ari)}" for r in X.tail(3).itertuples()) + ". Topics with the most "
          "method disagreement: " + ", ".join(f"{r.agentic_topic_name} ({pct(r.disagreement_share)})" for r in
                                             cons.sort_values('disagreement_share', ascending=False).head(5).itertuples()) + ".",
          (f"**18. Did agentic reasoning improve topic boundaries?** Versus K-Means at the same count: centroid inconsistency "
           f"{f(ag.centroid_inconsistency_rate)} vs {f(km_n.centroid_inconsistency_rate)}, low-confidence rate "
           f"{f(ag.low_confidence_rate)} vs {f(km_n.low_confidence_rate)}, paraphrase kappa {f(ag.paraphrase_kappa)} vs "
           f"{f(km_n.paraphrase_kappa)}. Semantic boundary quality itself needs the human review." if km_n is not None else
           "**18.** K-Means at N_AGENTIC unavailable."),
          f"**19. Did agentic reasoning introduce instability?** Agentic stability {f(ag.stability_ari)} (Claude decisions frozen), "
          f"hybrid {f(hy.stability_ari)}" + (f", K-Means at k={n} {f(km_n.stability_ari)}" if km_n is not None else "") +
          ". Run-to-run variance of Claude discovery itself was NOT measured (one budgeted run); repeated runs would be needed.",
          "**20. Is a hybrid system better?** Hybrid vs agentic: " + ", ".join(f"{m} {f(a)} vs {f(c)}" for m, (a, c) in hyb_vs_ag.items()) +
          f". In the evidence hierarchy the hybrid ranks {'first' if sel == 'HYBRID_PROTOTYPE' else 'behind ' + str(sel)}.",
          f"**21. Recommended final taxonomy method:** {D['outcome']}" + (f" - {W['label']}" if sel else "") + ".",
          f"**22. Recommended final topic count:** {f(W['n_clusters'])}.",
          f"**23. Why?** Decided at: {D.get('decided_by')}. Candidates passing all gates: {D.get('passed_gates')}. "
          f"Gate failures: " + "; ".join(f"{i}: {r}" for i, r in C.gate_failures.items() if r) + ".",
          "**24. Limitations:** " + " ".join([
              "(a) " + b.s3.LIMITATION,
              "(b) No topic-level gold labels; nothing here is accuracy.",
              "(c) Agentic discovery ran once under a 20-call budget; its run-to-run variance is unmeasured and its "
              "stability figure freezes Claude decisions.",
              f"(d) {saturated} discovery call(s) hit the output ceiling." if saturated else "(d) No discovery call hit the output ceiling.",
              f"(e) Ambiguous queue: {q.get('queued')} statements, {q.get('reviewed_by_claude')} reviewed by Claude, "
              f"{q.get('propagated')} propagated, {q.get('left_low_confidence')} left as low-confidence automatic assignments.",
              "(f) Paraphrase pairs come from Sprint 2 cross-model agreement and favour easier paraphrases.",
              "(g) Human review criterion is " + ("available." if b.human else "PENDING.")]),
          f"**25. Is Sprint 3 ready to freeze before Sprint 4?** **{'YES' if D['freeze_ready'] else 'NO'}**" +
          ("" if D["freeze_ready"] else f" - {'; '.join(D['freeze_blockers'])}. Label `evaluation/human_review_sample.csv`, then "
           "re-run the benchmark with `--human-review` (no Claude calls; checkpoints make it fast)."), "",
          "## Files", "- `evaluation/final_method_comparison.csv`, `matched_granularity_comparison.csv`, "
          "`cross_method_agreement.csv`, `topic_persistence.csv`, `paraphrase_consistency.csv`, `provider_robustness.csv`, "
          "`human_review_sample.csv`, `api_usage.csv`", "- `consensus/` co-association core / boundary members",
          "- `audit/existing_results_audit.md` historical audit; `audit/report_consistency_check.json`",
          f"- figures: {', '.join(figs.get('created', []))}" + (f" (failed: {figs['failed']})" if figs.get("failed") else "")]
    (b.out / "final_sprint3_report.md").write_text("\n".join(L), encoding="utf-8")
    return {"all_consistent": ok, "checks": len(check)}


def make_figures(b):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    R = pd.DataFrame(b.rows.values())
    fd, done, failed = b.out / "figures", [], []

    def fig(name, fn):
        try:
            fn()
            plt.tight_layout()
            plt.savefig(fd / name, dpi=150)
            done.append(name)
        except Exception as e:
            failed.append(f"{name}: {e}")
            log(f"figure {name} failed: {e}")
        finally:
            plt.close("all")

    def f1():
        _, ax = plt.subplots(1, 3, figsize=(15, 4))
        for (fam, emb), d in R[R.partition_id.str.contains("@")].groupby(["family", "embedding"]):
            d = d.sort_values("n_clusters")
            for a, m in zip(ax, ("stability_ari", "persistence_share_j50", "semantic_cohesion_lift_incl_unassigned")):
                a.plot(d.n_clusters, d[m], "o-", label=f"{fam} {emb}")
        for a, t in zip(ax, ("subsample stability ARI", "topics persisting (J>=0.5)", "cohesion lift")):
            a.axvline(b.grid["N_AGENTIC"], ls="--", c="gray")
            a.set_title(t)
            a.set_xlabel("actual topic count")
        ax[0].legend(fontsize=7)
    fig("01_matched_granularity_curves.png", f1)

    def f2():
        p = b.out / "matched_benchmark" / "leiden" / "resolution_search.csv"
        T = pd.read_csv(p)
        _, ax = plt.subplots(figsize=(7, 4))
        for (t, k), d in T.groupby(["target", "k_neighbors"]):
            d = d.sort_values("resolution")
            ax.plot(d.resolution, d.communities, ".-", label=f"target {t}, k={k}")
        ax.set_xscale("log")
        ax.set_xlabel("resolution")
        ax.set_ylabel("communities")
        ax.legend(fontsize=6)
        ax.set_title("Leiden adaptive resolution search")
    fig("02_leiden_resolution_search.png", f2)

    def f3():
        conv = pd.DataFrame(b.summary.get("convergence") or [])
        _, ax = plt.subplots(figsize=(7, 4))
        ax.plot(conv["round"], conv.topics_after, "o-", label="active topics")
        ax2 = ax.twinx()
        ax2.plot(conv["round"], conv.assignment_change_rate, "s--", c="tab:red", label="assignment change")
        ax2.plot(conv["round"], conv.assignment_noise_floor, "^:", c="gray", label="noise floor")
        ax.set_xlabel("refinement round")
        ax.set_title("Agentic convergence")
        ax2.legend(fontsize=7)
    fig("03_agentic_convergence.png", f3)

    def f4():
        ids = b.X_agree
        names = sorted(set(ids.method_a) | set(ids.method_b))
        Mx = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
        for r in ids.itertuples():
            Mx.loc[r.method_a, r.method_b] = Mx.loc[r.method_b, r.method_a] = r.ari
        _, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(Mx.values.astype(float), vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(names)), names, rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(len(names)), names, fontsize=7)
        plt.colorbar(im)
        ax.set_title("Cross-method ARI")
    fig("04_cross_method_ari.png", f4)

    def f5():
        _, ax = plt.subplots(figsize=(7, 4))
        ax.hist(b.cb.mean_coassociation.dropna(), bins=30)
        ax.axvline(2 / 3, ls="--", c="g")
        ax.axvline(1 / 3, ls="--", c="r")
        ax.set_title("Co-association of statements with their agentic topic-mates")
    fig("05_coassociation_distribution.png", f5)

    def f6():
        C = b.C
        metrics = ["coverage_work", "stability_ari", "persistence_share_j50", "paraphrase_kappa"]
        _, ax = plt.subplots(figsize=(12, 4.5))
        w = 0.8 / len(metrics)
        for i, m in enumerate(metrics):
            ax.bar(np.arange(len(C)) + i * w, pd.to_numeric(C[m], errors="coerce"), w, label=m)
        ax.set_xticks(np.arange(len(C)) + 0.4, C.partition_id, rotation=45, ha="right", fontsize=7)
        ax.legend(fontsize=7)
        ax.set_title("Final candidates")
    fig("06_final_candidates.png", f6)
    return {"created": done, "failed": failed}
