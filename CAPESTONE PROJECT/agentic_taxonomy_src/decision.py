"""Evidence-hierarchy decision logic + claim verification (no winner is hard-coded).

Gates (pre-declared; reuse Sprint 3 pathology thresholds + topic persistence): a candidate failing any gate cannot be
selected. Survivors are compared lexicographically with tolerances in the user-specified order:
 1 human semantic coherence (PENDING until the human review sample is labelled) 2 coverage of work statements
 3 global stability 4 topic persistence 5 paraphrase consistency 6 clear boundaries 7 granularity 8 low ambiguity
 9 provider robustness 10 cohesion / silhouette (supporting) 11 complexity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GATES = {"coverage_work": (">=", 0.80), "stability_ari": (">=", 0.50), "persistence_share_j50": (">=", 0.50),
         "dominant_share": ("<=", 0.20), "tiny_cluster_share": ("<=", 0.25), "noise_share": ("<=", 0.50)}
HIERARCHY = [("human_preference_share", 0.05, "max", "1 human semantic coherence"),
             ("coverage_work", 0.05, "max", "2 coverage"),
             ("stability_ari", 0.05, "max", "3 stability"),
             ("persistence_share_j50", 0.05, "max", "4 topic persistence"),
             ("paraphrase_kappa", 0.05, "max", "5 paraphrase consistency"),
             ("centroid_inconsistency_rate", 0.03, "min", "6 clear / non-redundant boundaries"),
             ("granularity_score", 0.25, "max", "7 sensible granularity"),
             ("low_confidence_rate", 0.05, "min", "8 low ambiguity"),
             ("provider_nn_gap_abs", 0.05, "min", "9 provider robustness"),
             ("semantic_cohesion_lift_incl_unassigned", 0.02, "max", "10a cohesion (supporting)"),
             ("silhouette", 0.01, "max", "10b silhouette (supporting)"),
             ("complexity_rank", 0.0, "min", "11 complexity / cost")]
COMPLEXITY_RANK = {"kmeans": 1, "gmm": 1, "leiden": 2, "hdbscan": 2, "bertopic": 3, "bertopic_reduced": 4,
                   "hybrid": 3, "agentic": 4}
OUTCOMES = {"agentic": "AGENTIC TAXONOMY SELECTED", "hybrid": "HYBRID TAXONOMY RECOMMENDED",
            "kmeans": "KMEANS MATCHED TO AGENTIC GRANULARITY SELECTED", "leiden": "LEIDEN SELECTED"}


def gate_failures(row) -> list[str]:
    out = []
    for col, (op, thr) in GATES.items():
        v = row.get(col)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            if col in ("coverage_work", "stability_ari", "persistence_share_j50"):
                out.append(f"{col}=n/a")
            continue
        if (op == ">=" and v < thr) or (op == "<=" and v > thr):
            out.append(f"{col}={v:.3f} {'<' if op == '>=' else '>'} {thr}")
    return out


def lexicographic(df: pd.DataFrame, hierarchy=HIERARCHY):
    """Keep candidates within tolerance of the best at each level; NaN is worst. Returns (id, trace, decided_by)."""
    pool = df.copy()
    trace, decided = [], None
    for col, tol, direction, label in hierarchy:
        if col not in pool or pool[col].isna().all():
            trace.append(f"{label} [{col}]: not available for any candidate -> PENDING / no effect")
            continue
        v = pd.to_numeric(pool[col], errors="coerce")
        v = v.fillna(-np.inf) if direction == "max" else v.fillna(np.inf)
        best = v.max() if direction == "max" else v.min()
        keep = (v >= best - tol) if direction == "max" else (v <= best + tol)
        trace.append(f"{label} [{col}]: best={best:.4f} tol={tol} -> kept {', '.join(map(str, pool.index[keep.values]))}")
        pool = pool[keep.values]
        if len(pool) == 1:
            decided = label
            break
    if len(pool) > 1:
        decided = "tie after all levels -> lowest complexity, then highest stability"
        pool = pool.sort_values(["complexity_rank", "stability_ari"], ascending=[True, False])
    return pool.index[0], trace, decided


def decide(cands: pd.DataFrame, human_available: bool) -> dict:
    C = cands.copy()
    C["gate_failures"] = [";".join(gate_failures(r)) for r in C.to_dict("records")]
    ok = C[C.gate_failures == ""]
    res = {"gates": GATES, "hierarchy": [h[3] for h in HIERARCHY], "candidates": int(len(C)),
           "passed_gates": ok.index.tolist(), "human_review": "available" if human_available else "PENDING"}
    if ok.empty:
        # still report the provisional leader among all candidates, but nothing can be frozen
        lead, trace, dec = lexicographic(C)
        res.update({"outcome": "INSUFFICIENT EVIDENCE TO FREEZE", "provisional_leader": lead, "trace": trace,
                    "decided_by": dec, "freeze_ready": False,
                    "freeze_blockers": [f"no candidate passed the gates ({'; '.join(f'{i}: {g}' for i, g in C.gate_failures.items())})"]})
        return res, C
    win, trace, dec = lexicographic(ok)
    fam = C.loc[win, "family"]
    outcome = OUTCOMES.get(fam, f"ANOTHER EXISTING METHOD SELECTED: {C.loc[win, 'label']}")
    rest = ok.drop(index=win)
    second = lexicographic(rest)[0] if len(rest) else None
    blockers = [] if human_available else ["human review sample not yet labelled (criterion 1: human semantic coherence)"]
    res.update({"outcome": outcome, "selected": win, "second": second, "trace": trace, "decided_by": dec,
                "freeze_ready": not blockers, "freeze_blockers": blockers})
    return res, C


# ---------------------------------------------------------------- claims (prose must equal numbers)
def superlative(df: pd.DataFrame, metric: str, direction: str, label_col="label"):
    v = pd.to_numeric(df[metric], errors="coerce")
    if v.notna().sum() == 0:
        return {"metric": metric, "direction": direction, "winners": [], "value": None}
    best = v.max() if direction == "max" else v.min()
    winners = df.loc[np.isclose(v, best, rtol=0, atol=1e-12), label_col].tolist()
    return {"metric": metric, "direction": direction, "winners": winners, "value": float(best)}


def verify_claims(claims: dict, tables: dict) -> list[dict]:
    """Independent re-computation (numpy argsort, not superlative()) of every superlative claim."""
    out = []
    for cid, c in claims.items():
        if "metric" not in c:
            continue
        df = tables[c["table"]]
        vals = pd.to_numeric(df[c["metric"]], errors="coerce").to_numpy(dtype=float)
        if np.all(np.isnan(vals)):
            ok = not c["winners"]
        else:
            arr = np.where(np.isnan(vals), -np.inf if c["direction"] == "max" else np.inf, vals)
            order = np.argsort(-arr if c["direction"] == "max" else arr, kind="stable")
            best = arr[order[0]]
            truth = sorted(df[c.get("label_col", "label")].to_numpy()[np.isclose(arr, best, rtol=0, atol=1e-12)].tolist())
            ok = sorted(c["winners"]) == truth and np.isclose(best, c["value"])
        out.append({"claim": cid, "metric": c["metric"], "stated_winners": c["winners"], "consistent": bool(ok)})
    return out
