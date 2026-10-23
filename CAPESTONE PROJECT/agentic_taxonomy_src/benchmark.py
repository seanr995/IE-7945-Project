"""Matched-granularity benchmark (VIEW B) + agentic vs algorithmic vs consensus comparison. ZERO Claude calls.

Reads the converged agentic taxonomy, derives N_LOW / N_AGENTIC / N_HIGH, refits algorithms around that granularity on
the cached embeddings, evaluates everything with one evaluator, aligns topics across methods, builds a co-association
consensus, samples disagreements for human review and applies the evidence-hierarchy decision.
"""
from __future__ import annotations

import json
import pickle
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import decision as dec
from . import methods as M
from . import evaluation as MX
from .audit import write_audit
from .config import SEED, StopRun, load_sprint3_module
from .provenance import log, read_json, write_json
from .semantic_index import load_project
from .granularity import matched_grid
from .registry import TopicRegistry

SEEDS = [101, 202, 303, 404, 505]


def _h(*p):
    import hashlib
    return hashlib.sha256("|".join(map(str, p)).encode()).hexdigest()


class Benchmark:
    def __init__(self, project, agentic_dir, out, human_review=None, use_e5=True, allow_smoke=False):
        self.project, self.agentic_dir, self.out = Path(project), Path(agentic_dir), Path(out)
        self.human_review, self.use_e5, self.allow_smoke = human_review, use_e5, allow_smoke
        for d in ("audit", "agentic", "matched_benchmark/kmeans", "matched_benchmark/gmm", "matched_benchmark/leiden",
                  "matched_benchmark/bertopic", "matched_benchmark/hdbscan", "matched_benchmark/representation_comparison",
                  "evaluation/overlap_matrices", "consensus", "figures", "checkpoints"):
            (self.out / d).mkdir(parents=True, exist_ok=True)
        self.rows, self.labels, self.pers, self.store = {}, {}, [], {}
        self.timing = {}

    # ------------------------------------------------------------------ setup
    def load(self):
        self.s3, self.s3_sha = load_sprint3_module(self.project)
        self.data = load_project(self.project, need_e5=self.use_e5)
        df = self.df = self.data["df"]
        self.N = len(df)
        self.audit = write_audit(self.project, self.out / "audit")
        summ = read_json(self.agentic_dir / "agentic_summary.json")
        if summ is None:
            raise StopRun(f"{self.agentic_dir}/agentic_summary.json not found - run the agentic notebook first")
        if summ.get("mode") != "full" and not self.allow_smoke:
            raise StopRun(f"agentic results are from mode='{summ.get('mode')}'; the benchmark needs the FULL agentic run")
        if not summ.get("N_AGENTIC"):
            raise StopRun("agentic run did not produce a topic count (N_AGENTIC)")
        self.summary = summ
        self.reg = TopicRegistry.load(self.agentic_dir / "topic_registry" / "registry_final.json")
        A = pd.read_csv(self.agentic_dir / "assignments" / "assignments_final.csv", dtype=str, keep_default_na=False)
        if A.statement_id.tolist() != df.statement_id.tolist():
            raise StopRun("agentic assignments are not aligned with the frozen input statement order")
        self.A = A
        for sub in ("topic_registry", "assignments", "refinement", "provenance", "cache"):
            src = self.agentic_dir / sub
            if src.exists():
                shutil.copytree(src, self.out / "agentic" / sub, dirs_exist_ok=True)
        for f in ("agentic_summary.json", "agentic_report.md", "run_manifest.json"):
            if (self.agentic_dir / f).exists():
                shutil.copy2(self.agentic_dir / f, self.out / "agentic" / f)
        self.grid = matched_grid(int(summ["N_AGENTIC"]), self.N)
        log(f"N_AGENTIC={self.grid['N_AGENTIC']} -> grid {self.grid['grid']} ({self.grid['rule']})")
        adv = read_json(self.project / "sprint3_advanced_results" / "run_manifest.json", {})
        self.T_common = adv.get("common_low_confidence_margin")
        # spaces
        self.space = {}
        pca_t5, Z_t5, pinfo = self.s3.controlled_pca(self.data["X_t5"])
        self.space["t5"] = {"X": self.data["X_t5"], "Z": Z_t5, "P": self.data["P_t5"], "pca": pca_t5,
                            "nn": self.s3.provider_nn_pairs(self.data["X_t5"], df.provider.values), "pca_info": pinfo}
        if self.T_common is None:
            _, lab = M.fit_kmeans(Z_t5, 100)
            ids, C = self.s3.centroids(self.data["X_t5"], lab)
            mem, _ = self.s3.membership(self.data["X_t5"], lab, ids, C, None, "other", None)
            self.T_common = float(np.nanquantile(mem.gap[lab >= 0], 0.10))
        if self.use_e5 and self.data.get("X_e5") is not None:
            pca_e5, Z_e5, pinfo5 = self.s3.controlled_pca(self.data["X_e5"])
            self.space["e5"] = {"X": self.data["X_e5"], "Z": Z_e5, "P": self.data["P_e5"], "pca": pca_e5,
                                "nn": self.s3.provider_nn_pairs(self.data["X_e5"], df.provider.values), "pca_info": pinfo5}
        self.paired_rows = self.data["paired"].row.values
        self.subs = self.s3.subsample_indices(self.N)
        self.non_work = (A.assignment_source == "non_work_flagged_by_claude").values
        self.claude_rows = (A.assignment_source == "claude").values
        self.ag_labels, self.ag_ids = MX.to_int_labels(A.primary_topic.tolist())
        self._prototypes()

    def _prototypes(self):
        """Agentic concept prototypes = Claude-evidenced statements per surviving topic (registry support resolved
        through merge chains + Claude-decided assignments)."""
        row_of = {s: i for i, s in enumerate(self.df.statement_id)}
        ev = {t: set() for t in self.ag_ids}
        for tid, t in self.reg.topics.items():
            tgt = self.reg.resolve(tid)
            if tgt in ev:
                ev[tgt] |= {row_of[s] for s in t.get("supporting_statement_ids", []) if s in row_of}
        for i in np.where(self.claude_rows)[0]:
            t = self.A.primary_topic.iat[i]
            if t in ev:
                ev[t].add(int(i))
        self.evidence = {t: sorted(r for r in rows if not self.non_work[r]) for t, rows in ev.items()}

    def proto_assign(self, rows=None, evidence_filter=None):
        X = self.space["t5"]["X"]
        ids, P = [], []
        for k, t in enumerate(self.ag_ids):
            ev = self.evidence[t] if evidence_filter is None else [r for r in self.evidence[t] if r in evidence_filter]
            if ev:
                ids.append(k)
                P.append(X[ev].mean(0))
        P = np.vstack(P)
        P /= np.linalg.norm(P, axis=1, keepdims=True)
        rows = np.arange(self.N) if rows is None else np.asarray(rows)
        S = X[rows] @ P.T
        o = np.argsort(-S, axis=1)
        lab = np.array(ids)[o[:, 0]]
        margin = S[np.arange(len(rows)), o[:, 0]] - (S[np.arange(len(rows)), o[:, 1]] if P.shape[0] > 1 else -1)
        return lab, margin, (np.array(ids), P)

    # ------------------------------------------------------------------ evaluation with checkpoints
    def evaluate(self, pid, family, label, model, labels, fit_sub, native_conf=None, pred_native=None, seed_runs=None,
                 extra=None, protocol="refit on the Sprint 3 5 x 80% subsamples"):
        ck = self.out / "checkpoints" / f"{_h(pid)[:16]}.pkl"
        if ck.exists():
            with open(ck, "rb") as f:
                row, lab, pers_df, keep = pickle.load(f)
            log(f"  {pid}: loaded from checkpoint")
        else:
            t = time.time()
            sp = self.space[model]
            lab = np.asarray(labels, dtype=int)
            r, arr = MX.evaluate_partition(self.s3, lab, sp["X"], sp["Z"], sp["P"], self.paired_rows, sp["nn"],
                                           self.df.provider.values, self.T_common, native_conf, pred_native, family)
            runs = MX.subsample_runs(fit_sub, self.N, self.subs) if fit_sub is not None else None
            stab = MX.stability_from_runs(self.s3, runs) if runs else {}
            pers_df, pers = MX.topic_persistence(lab, runs) if runs else (pd.DataFrame(), {})
            seed = MX.seed_ari(self.s3, seed_runs() if callable(seed_runs) else seed_runs) if seed_runs is not None else {}
            work = ~self.non_work
            row = {"partition_id": pid, "family": family, "label": label, "embedding": model, **r, **stab, **pers, **seed,
                   "coverage_work": float((lab[work] >= 0).mean()), "stability_protocol": protocol,
                   "complexity_rank": dec.COMPLEXITY_RANK.get(family, 3), "fit_seconds": round(time.time() - t, 1),
                   **(extra or {})}
            keep = {k: arr.get(k) for k in ("own_sim", "own_sim_incl", "sil", "para_same", "para_valid")}
            with open(ck, "wb") as f:
                pickle.dump((row, lab, pers_df, keep), f)
            log(f"  {pid}: topics={row.get('n_clusters')} cov={row['coverage_work']:.3f} "
                f"stab={row.get('stability_ari', float('nan')):.3f} pers50={row.get('persistence_share_j50') or 0:.2f} "
                f"kappa={row.get('paraphrase_kappa') or 0:.3f} lift={row.get('semantic_cohesion_lift_incl_unassigned') or 0:.3f}")
        self.rows[pid], self.labels[pid], self.store[pid] = row, lab, keep
        if len(pers_df):
            self.pers.append(pers_df.assign(partition_id=pid, label=label))
        return row

    # ------------------------------------------------------------------ partitions
    def run_agentic(self):
        P = self.space["t5"]["P"]
        ids, Pm = self.proto_assign()[2]
        pred = ids[np.argmax(P @ Pm.T, axis=1)] if len(P) else None
        _, margin, _ = self.proto_assign()
        frozen = self.ag_labels.copy()
        claude = self.claude_rows

        def agentic_sub(idx):
            lab, _, _ = self.proto_assign(idx, evidence_filter=set(idx.tolist()))
            out = lab.copy()
            fz = claude[idx] | (frozen[idx] < 0)               # Claude decisions + unresolved stay as decided
            out[fz] = frozen[idx][fz]
            out[self.non_work[idx]] = -1
            return out

        self.evaluate("AGENTIC_FULL", "agentic", "Agentic taxonomy (Claude concepts + decisions)", "t5", frozen,
                      agentic_sub, margin, pred, extra={"target_count": self.grid["N_AGENTIC"]},
                      protocol="prototypes rebuilt from 80% of the Claude evidence; Claude-decided rows frozen "
                               "(discovery-run variance NOT measured)")
        hyb, hmargin, _ = self.proto_assign()
        hyb = hyb.copy()
        hyb[self.non_work] = -1

        def hybrid_sub(idx):
            lab, _, _ = self.proto_assign(idx, evidence_filter=set(idx.tolist()))
            lab[self.non_work[idx]] = -1
            return lab
        self.evaluate("HYBRID_PROTOTYPE", "hybrid", "Hybrid: agentic concepts + deterministic T5 assignment", "t5", hyb,
                      hybrid_sub, hmargin, pred, extra={"target_count": self.grid["N_AGENTIC"]},
                      protocol="prototypes rebuilt from 80% of the Claude evidence; every row re-assigned by retrieval")

    def run_kmeans(self, model, counts, tag=""):
        sp = self.space[model]
        for n in counts:
            km, lab = M.fit_kmeans(sp["Z"], n)
            conf = self.s3.kmeans_ratio_confidence(km, sp["Z"])
            pred = km.predict(sp["pca"].transform(sp["P"]).astype(np.float64)) if len(sp["P"]) else None
            Zm = sp["Z"]
            self.evaluate(f"KMEANS_{model.upper()}@{n}{tag}", "kmeans", f"K-Means ({model.upper()}) k={n}", model, lab,
                          lambda idx, n=n, Zm=Zm: M.fit_kmeans(Zm[idx], n)[1], conf, pred,
                          seed_runs=lambda n=n, Zm=Zm: [M.fit_kmeans(Zm, n, s)[1] for s in SEEDS],
                          extra={"target_count": n, "actual_count": n, "abs_difference": 0, "param": f"k={n}"})

    def run_gmm(self, counts, tag=""):
        sp = self.space["t5"]
        for n in counts:
            gm, lab, post = M.fit_gmm(sp["Z"], n)
            ent = -(post * np.log(np.clip(post, 1e-300, None))).sum(1)
            pred = gm.predict(sp["pca"].transform(sp["P"]).astype(np.float64)) if len(sp["P"]) else None
            self.gmm_post = getattr(self, "gmm_post", {})
            self.gmm_post[n] = post
            self.evaluate(f"GMM_T5@{n}{tag}", "gmm", f"GMM (diag) n={n}", "t5", lab,
                          lambda idx, n=n: M.fit_gmm(sp["Z"][idx], n)[1], post.max(1), pred,
                          seed_runs=lambda n=n: [M.fit_gmm(sp["Z"], n, s)[1] for s in SEEDS],
                          extra={"target_count": n, "actual_count": int(len(set(lab.tolist()))), "abs_difference": int(n - len(set(lab.tolist()))),
                                 "empty_components": int(n - len(set(lab.tolist()))), "bic": float(gm.bic(sp["Z"])),
                                 "aic": float(gm.aic(sp["Z"])), "posterior_entropy_norm": float((ent / np.log(n)).mean()),
                                 "param": f"n_components={n}"})

    def run_leiden(self, targets):
        sp = self.space["t5"]
        X = sp["X"]
        searches, picks = [], {}
        graphs = {k: M.knn_graph(X, k) for k in M.LEIDEN_K_NEIGHBORS}
        for t in targets:
            best = None
            for k, (g, W) in graphs.items():
                info, lab, T = M.leiden_nearest_count(g, t)
                searches.append(T.assign(target=t, k_neighbors=k))
                key = (info["abs_difference"], -info["modularity"])
                if best is None or key < best[0]:
                    best = (key, k, info, lab)
            picks[t] = best
        pd.concat(searches).to_csv(self.out / "matched_benchmark" / "leiden" / "resolution_search.csv", index=False)
        rows = []
        for t, (_, k, info, lab) in picks.items():
            g, W = graphs[k]
            share = M.community_shares(W, lab)
            pred = M.knn_vote(sp["P"], X, lab, k) if len(sp["P"]) else None
            res = info["resolution"]

            def sub(idx, k=k, res=res):
                gs, _ = M.knn_graph(X[idx], k)
                return M.leiden(gs, res)[0]
            row = self.evaluate(f"LEIDEN_T5@{t}", "leiden", f"Leiden k_nn={k} res={res:.4g} (target {t})", "t5", lab, sub,
                                share, pred, seed_runs=lambda g=g, res=res: [M.leiden(g, res, s)[0] for s in SEEDS],
                                extra={**{a: b for a, b in info.items()}, "k_neighbors": k, "param": f"k_nn={k},res={res:.4g}",
                                       "boundary_node_rate": float(np.nanmean(share < 0.5))})
            rows.append(row)
        return picks

    def run_hdbscan(self, targets):
        sp = self.space["t5"]
        G, picks, labs = M.hdbscan_search(sp["Z"], targets)
        G.to_csv(self.out / "matched_benchmark" / "hdbscan" / "focused_grid.csv", index=False)
        write_json(self.out / "matched_benchmark" / "hdbscan" / "nearest_feasible.json", picks)
        done = set()
        for t, p in picks.items():
            if not p.get("feasible"):
                continue
            key = (p["min_cluster_size"], p["min_samples"], p["selection"])
            cl, lab, prob = M.fit_hdbscan(sp["Z"], *key)
            import hdbscan
            pred, _ = hdbscan.approximate_predict(cl, sp["pca"].transform(sp["P"]).astype(np.float64)) if len(sp["P"]) else (None, None)
            self.evaluate(f"HDBSCAN_T5@{t}", "hdbscan", f"HDBSCAN mcs={key[0]} ms={key[1]} {key[2]} (target {t})", "t5",
                          lab, lambda idx, key=key: M.fit_hdbscan(sp["Z"][idx], *key)[1], prob, pred,
                          extra={**{a: b for a, b in p.items()}, "param": f"mcs={key[0]},ms={key[1]},{key[2]}"})
            done.add(key)
        return picks

    def run_bertopic(self, n_target):
        sp = self.space["t5"]
        X, docs = sp["X"], self.df.statement_text.values
        nat = M.bertopic_natural(X, docs)
        import hdbscan
        pred_nat = hdbscan.approximate_predict(nat["hdbscan"], np.asarray(nat["umap"].transform(sp["P"]), dtype=np.float64))[0] \
            if len(sp["P"]) and nat["labels"].max() >= 0 else None

        def nat_sub(idx):
            return M.bertopic_natural(X[idx], docs[idx])["labels"]
        kw = M.bertopic_keywords(docs, nat["labels"])
        pd.DataFrame([{"topic": k, "keywords_for_inspection_only": v} for k, v in kw.items()]).to_csv(
            self.out / "matched_benchmark" / "bertopic" / "natural_keywords.csv", index=False)
        n_nat = int(len(set(nat["labels"][nat["labels"] >= 0].tolist())))
        self.evaluate("BERTOPIC_NATURAL", "bertopic", f"BERTopic natural ({n_nat} topics)", "t5", nat["labels"], nat_sub,
                      nat["prob"], pred_nat, extra={"variant": "NATURAL", "target_count": None, "actual_count": n_nat,
                                                    "param": json.dumps(M.BERTOPIC_NATURAL)})
        red, info = M.bertopic_reduce(docs, X, nat["labels"], n_target)
        write_json(self.out / "matched_benchmark" / "bertopic" / "post_hoc_reduction.json", info)
        if red is not None:
            mp = info["natural_to_reduced"]
            pred_red = np.array([mp.get(int(p), -1) if p >= 0 else -1 for p in pred_nat]) if pred_nat is not None else None

            def red_sub(idx):
                nl = M.bertopic_natural(X[idx], docs[idx])["labels"]
                r, _ = M.bertopic_reduce(docs[idx], X[idx], nl, n_target)
                return r if r is not None else nl
            kw = M.bertopic_keywords(docs, red)
            pd.DataFrame([{"topic": k, "keywords_for_inspection_only": v} for k, v in kw.items()]).to_csv(
                self.out / "matched_benchmark" / "bertopic" / "reduced_keywords.csv", index=False)
            self.evaluate(f"BERTOPIC_REDUCED@{n_target}", "bertopic_reduced",
                          f"BERTopic {M.POST_HOC_LABEL} ({info['actual_count']} topics)", "t5", red, red_sub, None, pred_red,
                          extra={**{k: v for k, v in info.items() if k != "natural_to_reduced"}})
        return info

    # ------------------------------------------------------------------ comparisons
    def tables(self):
        R = pd.DataFrame(self.rows.values()).set_index("partition_id", drop=False)
        R.to_csv(self.out / "evaluation" / "all_partitions.csv", index=False)
        matched = R[R.partition_id.str.contains("@")].copy()
        matched["grid_point"] = [("N_LOW" if r.target_count == self.grid["N_LOW"] else "N_HIGH" if r.target_count == self.grid["N_HIGH"]
                                  else "N_AGENTIC" if r.target_count == self.grid["N_AGENTIC"] else "historical")
                                 if not pd.isna(r.target_count) else "" for r in matched.itertuples()]
        matched.to_csv(self.out / "evaluation" / "matched_granularity_comparison.csv", index=False)
        if self.pers:
            pd.concat(self.pers).to_csv(self.out / "evaluation" / "topic_persistence.csv", index=False)
        cols_p = ["partition_id", "label", "paraphrase_consistency", "paraphrase_kappa", "paraphrase_pairs",
                  "paraphrase_wording_unassigned", "paraphrase_chance", "paraphrase_consistency_centroid", "paraphrase_kappa_centroid"]
        R.reindex(columns=cols_p).to_csv(self.out / "evaluation" / "paraphrase_consistency.csv", index=False)
        cols_v = ["partition_id", "label", "nn_same_topic_rate_same_provider", "nn_same_topic_rate_cross_provider",
                  "nn_cross_minus_same_provider_same_topic", "provider_nn_gap_abs", "provider_cramers_v", "provider_chi2_p"]
        R.reindex(columns=cols_v).to_csv(self.out / "evaluation" / "provider_robustness.csv", index=False)
        rep = R[R.family == "kmeans"].copy()
        rep.to_csv(self.out / "matched_benchmark" / "representation_comparison" / "kmeans_e5_vs_t5.csv", index=False)
        for fam, d in (("kmeans", "kmeans"), ("gmm", "gmm"), ("leiden", "leiden"), ("hdbscan", "hdbscan")):
            R[R.family == fam].to_csv(self.out / "matched_benchmark" / d / f"{fam}_partitions.csv", index=False)
        R[R.family.str.startswith("bertopic")].to_csv(self.out / "matched_benchmark" / "bertopic" / "bertopic_partitions.csv", index=False)
        return R

    def best_count_per_family(self, R):
        out = {}
        for fam, emb in (("kmeans", "t5"), ("kmeans", "e5"), ("gmm", "t5"), ("leiden", "t5"), ("hdbscan", "t5")):
            d = R[(R.family == fam) & (R.embedding == emb) & R.partition_id.str.contains("@") & ~R.partition_id.str.endswith("_HIST")]
            if d.empty:
                continue
            win, trace, decided = dec.lexicographic(d)
            out[f"{fam}_{emb}"] = {"best_partition": win, "best_count": int(R.loc[win, "n_clusters"]),
                                   "target_count": None if pd.isna(R.loc[win, "target_count"]) else int(R.loc[win, "target_count"]),
                                   "decided_by": decided, "trace": trace}
        return out

    def cross_method(self, ids):
        rows, matches = [], {}
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                res, match, (A, B, J) = MX.align(self.labels[a], self.labels[b])
                rows.append({"method_a": a, "method_b": b, **res})
                matches[(a, b)] = match
                if a == "AGENTIC_FULL":
                    pd.DataFrame(J, index=[f"{a}:{x}" for x in A], columns=[f"{b}:{y}" for y in B]).round(4).to_csv(
                        self.out / "evaluation" / "overlap_matrices" / f"{a}__vs__{b}.csv")
        X = pd.DataFrame(rows)
        X.to_csv(self.out / "evaluation" / "cross_method_agreement.csv", index=False)
        return X, matches

    def consensus(self, ids):
        parts = {p: self.labels[p] for p in ids}
        A = MX.coassociation(parts, self.N)
        cb = MX.core_boundary(A, self.ag_labels)
        cb["statement_id"] = self.df.statement_id.values
        cb["statement_text"] = self.df.statement_text.values
        cb["agentic_topic"] = [self.ag_ids[t] if t >= 0 else "" for t in cb.topic]
        cb["agentic_topic_name"] = [self.reg.get(t)["topic_name"] if t else "" for t in cb.agentic_topic]
        summ = cb[cb.topic >= 0].groupby(["agentic_topic", "agentic_topic_name"]).agg(
            size=("row", "size"), mean_coassociation=("mean_coassociation", "mean"),
            core_share=("member_class", lambda s: float((s == "CORE").mean())),
            boundary_share=("member_class", lambda s: float((s == "BOUNDARY").mean())),
            disagreement_share=("member_class", lambda s: float((s == "METHOD_DISAGREEMENT").mean()))).reset_index()
        summ.sort_values("size", ascending=False).to_csv(self.out / "consensus" / "coassociation_summary.csv", index=False)
        cb[cb.member_class == "CORE"].to_csv(self.out / "consensus" / "stable_core_members.csv", index=False)
        cb[cb.member_class.isin(["BOUNDARY", "METHOD_DISAGREEMENT"])].to_csv(self.out / "consensus" / "boundary_members.csv", index=False)
        write_json(self.out / "consensus" / "consensus_methods.json",
                   {"partitions": ids, "rule": "CORE: mean co-association with topic-mates >= 2/3; BOUNDARY 1/3-2/3; "
                                               "METHOD_DISAGREEMENT < 1/3 (methods where either statement is unassigned skipped)"})
        self.cb = cb
        return summ

    def human_sample(self, matches):
        df = self.df
        ag = self.ag_labels
        name = lambda t: f"{self.ag_ids[t]}: {self.reg.get(self.ag_ids[t])['topic_name']}" if t >= 0 else "UNASSIGNED"
        cols = {"kmeans": f"KMEANS_T5@{self.grid['N_AGENTIC']}", "gmm": f"GMM_T5@{self.grid['N_AGENTIC']}",
                "leiden": f"LEIDEN_T5@{self.grid['N_AGENTIC']}", "bertopic": "BERTOPIC_NATURAL"}

        def lab(m, i):
            p = cols[m]
            if p not in self.labels:
                return "n/a"
            v = self.labels[p][i]
            return f"{m.upper()}:{v}" if v >= 0 else f"{m.upper()}:OUTLIER"

        def agrees(m, i):
            p = cols[m]
            key = ("AGENTIC_FULL", p)
            if p not in self.labels or key not in matches or ag[i] < 0:
                return None
            mt = matches[key].get(ag[i])
            return mt is not None and self.labels[p][i] == mt[0]
        order = sorted(range(self.N), key=lambda i: _h(SEED, "human", df.statement_id.iat[i]))
        src = self.A.assignment_source.values
        prov = {t for t, v in self.reg.topics.items() if any(p.get("action") in ("absorbed", "split_into") or p.get("source") == "split"
                                                             for p in v["provenance"])}
        novel = {t for t, v in self.reg.topics.items() if v["created_round"] in ("novelty", "novelty_gate")}
        gpost = getattr(self, "gmm_post", {}).get(self.grid["N_AGENTIC"])
        cls = dict(zip(self.cb.row, self.cb.member_class)) if hasattr(self, "cb") else {}
        cats = [("agentic/K-Means agreement", 12, lambda i: agrees("kmeans", i) is True),
                ("agentic/K-Means disagreement", 20, lambda i: agrees("kmeans", i) is False),
                ("Leiden disagreement with agentic", 12, lambda i: agrees("leiden", i) is False),
                ("GMM ambiguous (low max posterior)", 10, lambda i: gpost is not None and gpost[i].max() < 0.6),
                ("BERTopic outlier", 10, lambda i: "BERTOPIC_NATURAL" in self.labels and self.labels["BERTOPIC_NATURAL"][i] < 0),
                ("agentic low-confidence assignment", 12, lambda i: src[i] == "auto_low_confidence"),
                ("cross-method stable core", 10, lambda i: cls.get(i) == "CORE"),
                ("cross-method boundary / disagreement", 12, lambda i: cls.get(i) in ("BOUNDARY", "METHOD_DISAGREEMENT")),
                ("member of a merged/split agentic topic", 10, lambda i: ag[i] >= 0 and self.ag_ids[ag[i]] in prov),
                ("member of a new (novelty-gate) agentic topic", 8, lambda i: ag[i] >= 0 and self.ag_ids[ag[i]] in novel),
                ("flagged non-work by Claude", 6, lambda i: src[i] == "non_work_flagged_by_claude")]
        taken, rows = set(), []
        for reason, quota, f in cats:
            k = 0
            for i in order:
                if k >= quota:
                    break
                if i in taken or not f(i):
                    continue
                taken.add(i)
                k += 1
                pat = ", ".join(f"{m}{'=' if agrees(m, i) else '≠' if agrees(m, i) is False else '?'}AG" for m in cols)
                rows.append({"statement_id": df.statement_id.iat[i], "statement_text": df.statement_text.iat[i],
                             "agentic_topic": name(ag[i]), "kmeans_topic": lab("kmeans", i), "gmm_topic": lab("gmm", i),
                             "leiden_topic": lab("leiden", i), "bertopic_topic": lab("bertopic", i),
                             "agreement_pattern": pat, "review_reason": reason, "human_best_topic": "",
                             "human_assignment_correct": "", "human_notes": ""})
        H = pd.DataFrame(rows)
        H.to_csv(self.out / "evaluation" / "human_review_sample.csv", index=False, encoding="utf-8-sig")
        return H

    def human_preferences(self):
        p = Path(self.human_review) if self.human_review else self.out / "evaluation" / "human_review_sample.csv"
        if not p.exists():
            return None
        H = pd.read_csv(p, dtype=str, keep_default_na=False)
        lab = H[H.human_best_topic.str.strip() != ""]
        if lab.empty:
            return None
        fams = {"agentic": "agentic", "kmeans": "kmeans", "gmm": "gmm", "leiden": "leiden", "bertopic": "bertopic"}
        share = {f: float(lab.human_best_topic.str.lower().str.contains(k).mean()) for k, f in fams.items()}
        corr = H.human_assignment_correct.str.strip().str.lower()
        share["agentic_assignment_correct_rate"] = float(corr[corr != ""].isin(["1", "yes", "true", "y"]).mean()) if (corr != "").any() else None
        share["labelled_rows"] = int(len(lab))
        return share

    # ------------------------------------------------------------------ orchestration
    def run_historical(self):
        sp = self.space["t5"]
        km, lab = M.fit_kmeans(sp["Z"], 100)
        self.evaluate("HIST_KMEANS_T5_k100", "kmeans", "Historical winner: K-Means (T5) k=100", "t5", lab,
                      lambda idx: M.fit_kmeans(sp["Z"][idx], 100)[1], self.s3.kmeans_ratio_confidence(km, sp["Z"]),
                      km.predict(sp["pca"].transform(sp["P"]).astype(np.float64)) if len(sp["P"]) else None,
                      extra={"target_count": None, "actual_count": 100, "param": "k=100", "historical": True})
        gm, lab, post = M.fit_gmm(sp["Z"], 140)
        self.evaluate("HIST_GMM_T5_140", "gmm", "Historical: GMM (diag) n=140", "t5", lab,
                      lambda idx: M.fit_gmm(sp["Z"][idx], 140)[1], post.max(1),
                      gm.predict(sp["pca"].transform(sp["P"]).astype(np.float64)) if len(sp["P"]) else None,
                      extra={"target_count": None, "actual_count": int(len(set(lab.tolist()))), "param": "n_components=140", "historical": True})
        g, W = M.knn_graph(sp["X"], 20)
        lab, mod = M.leiden(g, 2.0)

        def sub(idx):
            gs, _ = M.knn_graph(sp["X"][idx], 20)
            return M.leiden(gs, 2.0)[0]
        self.evaluate("HIST_LEIDEN_T5_k20_r2.0", "leiden", "Historical: Leiden k_nn=20 res=2.0", "t5", lab, sub,
                      M.community_shares(W, lab), M.knn_vote(sp["P"], sp["X"], lab, 20) if len(sp["P"]) else None,
                      extra={"target_count": None, "actual_count": int(len(set(lab.tolist()))), "modularity": mod,
                             "param": "k_nn=20,res=2.0", "historical": True})

    def api_usage(self):
        rows = []
        p = self.agentic_dir / "provenance" / "api_calls.jsonl"
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    d = json.loads(line)
                    u = d.get("usage") or {}
                    rows.append({"stage": "agentic", "role": d.get("role"), "cache_hit": d.get("cache_hit"),
                                 "real_call": not d.get("cache_hit"), "call_no": d.get("call_no"),
                                 "input_tokens": u.get("input_tokens"), "output_tokens": u.get("output_tokens"),
                                 "note": d.get("note"), "utc": d.get("utc")})
        rows.append({"stage": "matched_benchmark", "role": "none", "cache_hit": None, "real_call": False, "call_no": None,
                     "input_tokens": 0, "output_tokens": 0, "note": "matched benchmark makes ZERO Claude calls", "utc": None})
        U = pd.DataFrame(rows)
        U.to_csv(self.out / "evaluation" / "api_usage.csv", index=False)
        budget = read_json(self.agentic_dir / "provenance" / "api_budget.json", {})
        return U, budget

    def run(self):
        t0 = time.time()
        self.load()
        g = self.grid
        log("=== agentic + hybrid ===")
        self.run_agentic()
        log("=== K-Means (T5" + (", E5" if "e5" in self.space else "") + ") ===")
        self.run_kmeans("t5", g["grid"])
        if "e5" in self.space:
            self.run_kmeans("e5", g["grid"])
        log("=== GMM ===")
        self.run_gmm(g["grid"])
        log("=== Leiden (adaptive resolution search) ===")
        self.leiden_picks = self.run_leiden(g["grid"])
        log("=== HDBSCAN (focused search, noise-capped) ===")
        self.hdb_picks = self.run_hdbscan(g["grid"])
        log("=== BERTopic natural + post-hoc reduced ===")
        try:
            self.bt_info = self.run_bertopic(g["N_AGENTIC"])
        except Exception as e:                       # BERTopic failures must not discard the rest of the benchmark
            log(f"BERTopic stage failed: {type(e).__name__}: {e}")
            self.bt_info = {"error": f"{type(e).__name__}: {e}"}
        log("=== historical winners under the same evaluator ===")
        self.run_historical()
        R = self.tables()
        self.best_counts = self.best_count_per_family(R)
        n = g["N_AGENTIC"]
        cross_ids = [p for p in ("AGENTIC_FULL", "HYBRID_PROTOTYPE", f"KMEANS_T5@{n}", f"GMM_T5@{n}", f"LEIDEN_T5@{n}",
                                 "BERTOPIC_NATURAL", f"BERTOPIC_REDUCED@{n}", f"HDBSCAN_T5@{n}") if p in self.labels]
        self.X_agree, self.matches = self.cross_method(cross_ids)
        self.cons = self.consensus([p for p in cross_ids if p != "HYBRID_PROTOTYPE"])
        self.H = self.human_sample(self.matches)
        human = self.human_preferences()
        cands = ["AGENTIC_FULL", "HYBRID_PROTOTYPE"] + [v["best_partition"] for v in self.best_counts.values()] + \
                [p for p in ("BERTOPIC_NATURAL", f"BERTOPIC_REDUCED@{n}") if p in self.rows] + \
                [p for p in self.rows if p.startswith("HIST_")]
        C = R.loc[list(dict.fromkeys(cands))].copy()
        fam_h = {"agentic": "agentic", "hybrid": "agentic", "kmeans": "kmeans", "gmm": "gmm", "leiden": "leiden",
                 "bertopic": "bertopic", "bertopic_reduced": "bertopic", "hdbscan": None}
        C["human_preference_share"] = [human.get(fam_h.get(f)) if human and fam_h.get(f) else np.nan for f in C.family]
        self.decision, C = dec.decide(C, human is not None)
        C["selected"] = C.index == self.decision.get("selected", self.decision.get("provisional_leader"))
        C.to_csv(self.out / "evaluation" / "final_method_comparison.csv", index=False)
        self.C, self.human = C, human
        U, budget = self.api_usage()
        self.timing["total_seconds"] = round(time.time() - t0, 1)
        from .report import write_final_report, make_figures
        figs = make_figures(self)
        check = write_final_report(self, R, budget, figs)
        manifest = {"created_utc": pd.Timestamp.now("UTC").isoformat(), "project_dir": str(self.project),
                    "agentic_results": str(self.agentic_dir), "input_sha256": self.data["input_sha256"],
                    "embedding_hashes": self.data["hashes"], "validation": self.data["checks"],
                    "sprint3_script_sha256": self.s3_sha, "grid": self.grid, "T_common_low_confidence_margin": self.T_common,
                    "partitions": list(self.rows), "best_count_per_family": self.best_counts,
                    "leiden_search": {int(t): v[2] for t, v in self.leiden_picks.items()},
                    "hdbscan_search": self.hdb_picks, "bertopic": {k: v for k, v in self.bt_info.items() if k != "natural_to_reduced"},
                    "decision": self.decision, "human_review": human, "claude_calls_in_benchmark": 0,
                    "agentic_api_budget": {k: budget.get(k) for k in ("hard_limit", "calls_made", "remaining_calls",
                                                                      "reserve_calls_used", "calls_by_role")},
                    "report_consistency": check, "runtime_seconds": self.timing, "seeds": {"global": SEED, "stability": SEEDS},
                    "subsample_protocol": "Sprint 3 subsample_indices (rng 42, 5 x 80%)",
                    "known_input_quality_limitation": self.s3.LIMITATION, "onet_used": False}
        write_json(self.out / "run_manifest.json", manifest)
        return manifest
