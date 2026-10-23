"""Agentic taxonomy induction (no predefined topic count).

3015 tasks -> Sentence-T5 embeddings -> diversity sample -> Claude discovery -> registry -> consolidation
-> boundary review -> high-confidence automatic assignment -> ambiguous queue to Claude -> novelty grouping +
Claude validation -> diagnostics -> critic (merge / split / refine / reassign) -> convergence -> consistency -> N_AGENTIC

- Claude does only semantic reasoning; similarity, retrieval, sampling, diagnostics and assignment are local.
- Topic prototypes come only from Claude-evidenced statements; every other statement is assigned by retrieval.
- Thresholds are derived from observed distributions (LOO calibration on Claude-labelled anchors, quantiles of
  diagnostics, the corpus nearest-neighbour similarity). AUTO_PRECISION_TARGET is a declared quality target.
- The number of calls per stage comes from the run plan (scaled to the real remaining budget). Unused calls stay unused.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import prompts
from . import semantic_index as si
from .budget import BudgetExceeded
from .config import SEED
from .provenance import log, write_json
from .registry import BAD_NAME, TopicRegistry

AUTO_PRECISION_TARGET = 0.90
DISCOVERY_STATEMENT_TOKENS_PER_CALL = 9000
MAX_INPUT_TOKENS_PER_CALL = 24000
JSON_TARGET_TOKENS = 3000                 # planned JSON size per reply (max_tokens is only a ceiling)
EST_OUTPUT_TOKENS = {"work_type": 95, "consolidation": 70, "registry": 45, "ambiguous": 34, "novelty": 110, "critic": 120}
SATURATION_SHARE = 0.85


def code(i):
    return f"s{int(i)}"


def parse_code(c, n):
    try:
        i = int(str(c).strip().lstrip("sS"))
        return i if 0 <= i < n else None
    except Exception:
        return None


def capacity(kind):
    return max(1, (JSON_TARGET_TOKENS - 150) // EST_OUTPUT_TOKENS[kind])


class AgenticRun:
    def __init__(self, data, client, out_dir, plan: dict, max_rounds: int = 2):
        self.df, self.X, self.N = data["df"], data["X_t5"], len(data["df"])
        self.client, self.out, self.caps, self.max_rounds = client, Path(out_dir), dict(plan["role_caps"]), max_rounds
        self.reg = TopicRegistry()
        self.stage_log, self.convergence = [], []
        self.non_work, self.claude_primary, self.claude_secondary, self.unresolved, self.propagated = set(), {}, {}, {}, {}
        self.queue_info, self.stop_reason, self.thresholds, self.deferred_pairs = None, None, {}, []
        self.failures = []
        S = self.X @ self.X.T
        np.fill_diagonal(S, -np.inf)
        self.tau_prop = float(np.median(S.max(1)))         # typical nearest-neighbour similarity of the corpus
        del S
        self.row_of = {s: i for i, s in enumerate(self.df.statement_id)}
        for d in ("topic_registry/history", "assignments", "refinement", "provenance"):
            (self.out / d).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ helpers
    def text(self, i):
        return self.df.statement_text.iat[int(i)]

    def sid(self, i):
        return self.df.statement_id.iat[int(i)]

    def can(self, role):
        return self.client.budget.can(role)

    def snapshot(self, label):
        self.reg.save(self.out / "topic_registry" / "history" / f"registry_v{self.reg.version:04d}_{label}.json")

    def log_stage(self, stage, **kw):
        kw = {"stage": stage, "active_topics": len(self.reg.active()), "registry_version": self.reg.version,
              "budget": self.client.budget.summary(), **kw}
        self.stage_log.append(kw)
        write_json(self.out / "provenance" / "stage_log.json", self.stage_log)
        b = kw["budget"]
        log(f"[{stage}] active topics={kw['active_topics']} | calls made {b['calls_made']}/{b['hard_limit']} "
            f"(this run {b['run_normal_used']} normal + {b['run_reserve_used']} reserve) | "
            + json.dumps({k: v for k, v in kw.items() if k not in ('stage', 'budget', 'active_topics', 'registry_version')},
                         default=str)[:400])

    def ask(self, role, args, note):
        """One Claude decision. An unrepairable reply / exhausted reserve skips this decision (logged) instead of
        aborting the run; API 400 / authentication errors still stop the run (raised as StopRun)."""
        system, instr, data, contract = prompts.BUILDERS[role](*args)
        try:
            out, info = self.client.call(role, system, instr, data, contract, self.reg.fingerprint(), note=note)
        except BudgetExceeded as e:
            self.failures.append({"role": role, "note": note, "error": str(e)})
            write_json(self.out / "provenance" / "skipped_decisions.json", self.failures)
            log(f"  {role}: decision skipped ({e})")
            return None, None
        with open(self.out / "refinement" / "claude_decisions.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"role": role, "note": note, "registry_version": self.reg.version, "cache_hit": info["cache_hit"],
                                "output": out}, default=str) + "\n")
        return out, info

    def evidence_rows(self, tid):
        rows = {self.row_of[s] for s in self.reg.get(tid)["supporting_statement_ids"] if s in self.row_of}
        rows |= {r for r, t in self.claude_primary.items() if self.reg.resolve(t) == tid}
        return sorted(r for r in rows if r not in self.non_work)

    def prototypes(self, rng=None, frac=0.8):
        ids, P = [], []
        for t in self.reg.active():
            rows = self.evidence_rows(t)
            if rng is not None and len(rows) > 1:
                rows = sorted(rng.choice(rows, size=max(1, int(round(frac * len(rows)))), replace=False).tolist())
            if rows:
                ids.append(t)
                P.append(self.X[rows].mean(0))
        return ids, (si.normalize(np.vstack(P)) if P else np.zeros((0, self.X.shape[1])))

    def examples(self, tid, k=3):
        rows = self.evidence_rows(tid)
        if not rows:
            return []
        c = si.normalize(self.X[rows].mean(0))
        return [f"{code(r)}: {self.text(r)}" for r in sorted(rows, key=lambda r: -float(self.X[r] @ c))[:k]]

    # ------------------------------------------------------------------ discovery
    def discover(self, sample_rows):
        regions = si.balanced_regions(self.X, sample_rows, self.caps["discovery"])
        cap, sat = capacity("work_type"), []
        for r, rows in enumerate(regions):
            if not self.can("discovery"):
                break
            block = [{"code": code(i), "text": self.text(i)} for i in rows]
            out, info = self.ask("discovery", (block,), f"region {r + 1}/{len(regions)} ({len(rows)} statements)")
            if out is None:
                continue
            allowed, n_types, halluc = set(int(i) for i in rows), 0, 0
            for wt in out["work_types"]:
                sup = [parse_code(c, self.N) for c in wt["supporting_statement_ids"]]
                halluc += sum(1 for s in sup if s is None or s not in allowed)
                sup = sorted({s for s in sup if s is not None and s in allowed})
                rep = [s for s in (parse_code(c, self.N) for c in wt["representative_statement_ids"]) if s in allowed]
                tid = self.reg.add(wt["candidate_topic_name"], wt["definition"], wt["core_action"], wt["object_or_outcome"],
                                   wt["inclusion_rules"], wt["exclusion_rules"], [self.sid(s) for s in rep],
                                   [self.sid(s) for s in sup], "discovery",
                                   {"source": "discovery", "region": r + 1, "call_key": info["key"]}, wt.get("confidence"))
                n_types += 1
                if not sup:
                    self.reg.reject(tid, "discovery", "no valid supporting statement")
                elif BAD_NAME.match(wt["candidate_topic_name"]):
                    self.reg.reject(tid, "discovery", "catch-all / invalid topic name")
            for u in out["unlisted_work_types"]:
                sup = sorted({s for s in (parse_code(c, self.N) for c in u["supporting_statement_ids"]) if s in allowed})
                if len(sup) >= 2 and not BAD_NAME.match(u["name"]):
                    self.reg.add(u["name"], u["name"], supporting_statement_ids=[self.sid(s) for s in sup],
                                 created_round="discovery", provenance={"source": "discovery_unlisted", "region": r + 1},
                                 definition_pending=True)
            for c in out["statements_not_work"]:
                s = parse_code(c, self.N)
                if s is not None and s in allowed:
                    self.non_work.add(s)
            sat.append({"region": r + 1, "statements": len(rows), "work_types": n_types, "unlisted": len(out["unlisted_work_types"]),
                        "hallucinated_ids": halluc, "est_capacity": cap, "cache_hit": info["cache_hit"],
                        "saturated": bool(info.get("stop_reason") == "max_tokens" or n_types >= SATURATION_SHARE * cap
                                          or out["unlisted_work_types"])})
        pd.DataFrame(sat).to_csv(self.out / "provenance" / "discovery_calls.csv", index=False)
        self.snapshot("discovery")
        self.log_stage("discovery", regions=len(regions), calls=len(sat), saturated=sum(s["saturated"] for s in sat),
                       non_work_flagged=len(self.non_work))

    # ------------------------------------------------------------------ consolidation + boundary review
    def overlap_pairs(self):
        ids, P = self.prototypes()
        P_by, S = dict(zip(ids, P)), P @ P.T
        sup = {t: set(self.evidence_rows(t)) for t in ids}
        raw = {t: (float(np.mean(self.X[list(sup[t])] @ P_by[t])) if sup[t] else 1.0) for t in ids}
        solid = [raw[t] for t in ids if len(sup[t]) >= 3]
        typical = float(np.median(solid)) if solid else float(np.median(list(raw.values()) or [1.0]))
        spread = {t: raw[t] if len(sup[t]) >= 3 else typical for t in ids}
        tok = {t: set(self.reg.get(t)["topic_name"].lower().split()) for t in ids}
        pairs = []
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                ta, tb = ids[a], ids[b]
                shared = len(sup[ta] & sup[tb])
                nj = len(tok[ta] & tok[tb]) / max(len(tok[ta] | tok[tb]), 1)
                if S[a, b] >= min(spread[ta], spread[tb]) or shared > 0 or nj >= 0.5:
                    pairs.append((float(S[a, b]), ta, tb, shared))
        return sorted(pairs, key=lambda p: (-p[0], p[1], p[2]))

    def consolidate(self):
        pairs, cap = self.overlap_pairs(), capacity("consolidation")
        stats = {"MERGE": 0, "REFINE_BOUNDARY": 0, "KEEP_SEPARATE": 0, "NEEDS_MORE_EVIDENCE": 0}
        reviewed, pos = 0, 0
        while pos < len(pairs) and self.can("consolidation"):
            batch = pairs[pos:pos + cap]
            pos += cap
            live = [p for p in batch if self.reg.resolve(p[1]) and self.reg.resolve(p[2]) and
                    self.reg.resolve(p[1]) != self.reg.resolve(p[2])]
            if not live:
                continue
            payload = [{"pair_id": f"P{pos}_{j}", "similarity": round(s, 3), "shared_supporting_statements": sh,
                        "topic_a": prompts.topic_card(self.reg.get(a), self.examples(a)),
                        "topic_b": prompts.topic_card(self.reg.get(b), self.examples(b))} for j, (s, a, b, sh) in enumerate(live)]
            out, _ = self.ask("consolidation", (payload,), f"{len(live)} pairs")
            if out is None:
                continue
            by_id = {p["pair_id"]: (a, b) for p, (_, a, b, _) in zip(payload, live)}
            for d in out["decisions"]:
                if d["pair_id"] not in by_id:
                    continue
                a, b = by_id[d["pair_id"]]
                reviewed += 1
                stats[d["decision"]] += 1
                if d["decision"] == "MERGE":
                    keep = d.get("keep_topic_id") if d.get("keep_topic_id") in (a, b) else min(a, b)
                    self.reg.merge(b if keep == a else a, keep, "consolidation", d.get("reason", ""))
                elif d["decision"] == "REFINE_BOUNDARY":
                    for rd in d.get("refined", []):
                        t = self.reg.resolve(rd["topic_id"]) if rd["topic_id"] in (a, b) else None
                        if t:
                            self.reg.refine(t, "consolidation", d.get("reason", ""), definition=rd["definition"],
                                            inclusion_rules=rd["inclusion_rules"], exclusion_rules=rd["exclusion_rules"])
        self.deferred_pairs = [(a, b) for _, a, b, _ in pairs[pos:]]
        self.snapshot("consolidation")
        self.log_stage("consolidation", candidate_pairs=len(pairs), reviewed=reviewed, deferred=len(self.deferred_pairs),
                       decisions=stats)

    def registry_review(self, role, note):
        if not self.can(role):
            self.log_stage(role, skipped="no budget in plan")
            return [], 0
        cards = [prompts.topic_card(self.reg.get(t), compact=True) for t in self.reg.active()]
        out, _ = self.ask(role, (cards,), note)
        if out is None:
            return [], 0
        ev0 = len(self.reg.events)
        flagged = self.apply_registry_decisions(out["decisions"], role)
        self.snapshot(role)
        self.log_stage(role, decisions=len(out["decisions"]), registry_consistent=out["registry_consistent"],
                       structural_changes=len(self.reg.events) - ev0)
        return flagged, len(self.reg.events) - ev0

    def apply_registry_decisions(self, decisions, round_name):
        flagged = []
        for d in decisions:
            ids = [t for t in dict.fromkeys(self.reg.resolve(t) for t in d["topic_ids"] if t in self.reg.topics) if t]
            if not ids:
                continue
            if d["action"] == "MERGE" and len(ids) >= 2:
                keep = self.reg.resolve(d.get("keep_topic_id")) if d.get("keep_topic_id") in self.reg.topics else None
                keep = keep if keep in ids else min(ids)
                for t in ids:
                    if t != keep:
                        self.reg.merge(t, keep, round_name, d.get("reason", ""))
            elif d["action"] == "REJECT_NON_TASK":
                for t in ids:
                    self.reg.reject(t, round_name, d.get("reason", "") or "not a type of work")
            elif d["action"] == "REFINE_BOUNDARY" and d.get("definition"):
                self.reg.refine(ids[0], round_name, d.get("reason", ""), definition=d["definition"])
            elif d["action"] == "FLAG_SPLIT":
                flagged += ids
        return flagged

    # ------------------------------------------------------------------ assignment
    def calibrate(self, ids, P):
        """LOO nearest-prototype accuracy on Claude-labelled anchors -> margin + low-similarity thresholds."""
        idx = {t: k for k, t in enumerate(ids)}
        ev = {t: self.evidence_rows(t) for t in ids}
        sums = {t: self.X[ev[t]].sum(0) for t in ids}
        rows = []
        for t in ids:
            for r in ev[t]:
                Pl = P.copy()
                if len(ev[t]) > 1:
                    Pl[idx[t]] = si.normalize(sums[t] - self.X[r])
                s = Pl @ self.X[r]
                o = np.argsort(-s)
                rows.append({"row": r, "topic": t, "pred": ids[o[0]], "top1": float(s[o[0]]),
                             "margin": float(s[o[0]] - s[o[1]]) if len(o) > 1 else 1.0})
        c = pd.DataFrame(rows)
        if c.empty:
            return {"margin_threshold": 0.0, "low_similarity_threshold": -1.0, "anchors": 0}
        c["correct"] = c.pred == c.topic
        m_star = next((float(m) for m in sorted(c.margin.unique())
                       if len(c[c.margin >= m]) >= max(10, 0.1 * len(c)) and c[c.margin >= m].correct.mean() >= AUTO_PRECISION_TARGET),
                      float(c.margin.max()) + 1e-9)
        ok = c[c.correct]
        c.to_csv(self.out / "assignments" / "loo_calibration.csv", index=False)
        return {"margin_threshold": m_star,
                "low_similarity_threshold": float(np.percentile(ok.top1, 5)) if len(ok) else float(c.top1.min()),
                "anchors": int(len(c)), "loo_accuracy_all": float(c.correct.mean()),
                "auto_precision_at_threshold": float(c[c.margin >= m_star].correct.mean()) if (c.margin >= m_star).any() else None,
                "target": AUTO_PRECISION_TARGET,
                "rule": "smallest LOO margin with precision >= target on Claude-labelled anchors; low-similarity = P5 of "
                        "correct anchors' top-1 similarity"}

    def deterministic(self, ids, P, thr):
        S = self.X @ P.T
        o = np.argsort(-S, axis=1)
        top1 = S[np.arange(self.N), o[:, 0]]
        top2 = S[np.arange(self.N), o[:, 1]] if len(ids) > 1 else np.full(self.N, -1.0)
        margin = top1 - top2
        reason = np.where(margin < thr["margin_threshold"], "LOW_MARGIN", "")
        reason = np.where(top1 < thr["low_similarity_threshold"], "LOW_SIMILARITY", reason)
        return pd.DataFrame({"row": np.arange(self.N), "auto_topic": [ids[j] for j in o[:, 0]], "top1_sim": top1,
                             "top2_sim": top2, "margin": margin, "queue_reason": reason,
                             "candidates": [[ids[j] for j in o[i, :5]] for i in range(self.N)]})

    def assignment_pass(self):
        ids, P = self.prototypes()
        self.thresholds = self.calibrate(ids, P)
        return self.deterministic(ids, P, self.thresholds)

    def resolve_ambiguous(self, frame):
        q = frame[(frame.queue_reason != "") & ~frame.row.isin(list(self.claude_primary) + list(self.non_work))]
        cap = capacity("ambiguous")
        start = int(q.sort_values("margin").row.iloc[0]) if len(q) else None
        order = si.fps_within(self.X, q.row.values, start) if len(q) else np.array([], dtype=int)
        cand_of = dict(zip(frame.row, frame.candidates))
        reviewed, novelty, pos, calls = set(), [], 0, 0
        stats = {"ASSIGN": 0, "SECONDARY_ASSIGNMENT": 0, "NEW_TOPIC_CANDIDATE": 0, "UNCERTAIN": 0, "invalid_topic": 0}
        while pos < len(order) and self.can("ambiguous"):
            batch = order[pos:pos + cap]
            cat_ids = sorted({t for r in batch for t in cand_of[r]})
            catalog = [prompts.topic_card(self.reg.get(t), self.examples(t, 2)) for t in cat_ids]
            tasks = [{"statement_id": code(r), "task": self.text(r), "candidates": cand_of[r]} for r in batch]
            while len(tasks) > 1 and si.est_tokens(json.dumps(catalog + tasks)) > MAX_INPUT_TOKENS_PER_CALL:
                tasks, batch = tasks[:-1], batch[:-1]
            pos += len(batch)
            out, _ = self.ask("ambiguous", (tasks, catalog), f"batch {calls + 1}: {len(tasks)} tasks")
            calls += 1
            if out is None:
                continue
            allowed = {int(r) for r in batch}
            for d in out["decisions"]:
                r = parse_code(d["statement_id"], self.N)
                if r not in allowed:
                    continue
                reviewed.add(r)
                dec = d["decision"]
                if dec in ("ASSIGN", "SECONDARY_ASSIGNMENT") and d.get("primary_topic_id") not in cand_of[r]:
                    stats["invalid_topic"] += 1
                    dec = "UNCERTAIN"
                stats[dec] += 1
                if dec in ("ASSIGN", "SECONDARY_ASSIGNMENT"):
                    self.claude_primary[r] = d["primary_topic_id"]
                    if dec == "SECONDARY_ASSIGNMENT" and d.get("secondary_topic_id") in cand_of[r] and \
                            d["secondary_topic_id"] != d["primary_topic_id"]:
                        self.claude_secondary[r] = d["secondary_topic_id"]
                elif dec == "NEW_TOPIC_CANDIDATE":
                    novelty.append({"row": r, "name": d.get("new_topic_name") or "", "definition": d.get("new_topic_definition") or ""})
                else:
                    self.unresolved[r] = "claude_uncertain"
        rev = np.array(sorted(r for r in reviewed if r in self.claude_primary))
        if len(rev):                                   # very similar unseen queue items inherit the nearest decision
            for r in order[pos:]:
                s = self.X[rev] @ self.X[r]
                j = int(np.argmax(s))
                t = self.claude_primary[int(rev[j])]
                if s[j] >= self.tau_prop and t in cand_of[r]:
                    self.propagated[int(r)] = t
        self.queue_info = {"queued": int(len(q)), "reviewed_by_claude": len(reviewed), "propagated": len(self.propagated),
                           "left_low_confidence": int(len(order) - len(reviewed) - len(self.propagated)), "calls": calls,
                           "per_call_capacity": cap, "decisions": stats, "propagation_similarity": self.tau_prop}
        q.assign(reviewed=q.row.isin(reviewed)).drop(columns=["candidates"]).to_csv(
            self.out / "assignments" / "ambiguous_queue.csv", index=False)
        self.log_stage("ambiguous", **self.queue_info)
        return novelty

    def novelty_gate(self, novelty):
        """Group novelty candidates; only coherent groups (>= min support) are considered; Claude validates them."""
        if not novelty:
            self.log_stage("novelty", candidates=0, created=0)
            return
        rows = np.array([n["row"] for n in novelty])
        S = self.X[rows] @ self.X[rows].T
        comp, c = -np.ones(len(rows), dtype=int), 0
        for i in range(len(rows)):
            if comp[i] >= 0:
                continue
            stack, comp[i] = [i], c
            while stack:
                k = stack.pop()
                for j in np.where((S[k] >= self.tau_prop) & (comp < 0))[0]:
                    comp[j] = c
                    stack.append(j)
            c += 1
        sizes = [len(self.evidence_rows(t)) for t in self.reg.active()]
        m_min = max(3, int(np.percentile(sizes, 10))) if sizes else 3
        groups = []
        for g in range(c):
            members = np.where(comp == g)[0]
            if len(members) < m_min:
                for k in members:
                    self.unresolved[int(rows[k])] = "novelty_isolated"
                continue
            groups.append(members)
        created, decided = [], {}
        if groups and self.can("novelty"):
            ids, P = self.prototypes()
            payload = []
            for gi, members in enumerate(groups):
                cent = si.normalize(self.X[rows[members]].mean(0))
                near = [ids[j] for j in np.argsort(-(P @ cent))[:3]] if len(ids) else []
                payload.append({"group_id": f"G{gi}", "tasks": [f"{code(rows[k])}: {self.text(rows[k])}" for k in members[:8]],
                                "proposed_names": sorted({novelty[k]["name"] for k in members if novelty[k]["name"]})[:4],
                                "nearest_existing_topics": [prompts.topic_card(self.reg.get(t), compact=True) for t in near]})
            out, _ = self.ask("novelty", (payload,), f"{len(groups)} groups")
            decided = {d["group_id"]: d for d in out["decisions"]} if out else {}
        for gi, members in enumerate(groups):
            d = decided.get(f"G{gi}")
            grow = [int(rows[k]) for k in members]
            if d is None:                              # no validation budget: keep them unresolved (never auto-create)
                for r in grow:
                    self.unresolved[r] = "novelty_unvalidated"
                continue
            if d["decision"] == "CREATE_TOPIC" and d.get("topic_name"):
                tid = self.reg.add(d["topic_name"], d.get("definition") or d["topic_name"], d.get("core_action", ""),
                                   d.get("object_or_outcome", ""), d.get("inclusion_rules"), d.get("exclusion_rules"),
                                   supporting_statement_ids=[self.sid(r) for r in grow], created_round="novelty",
                                   provenance={"source": "novelty_validated_by_claude", "group_size": len(grow), "min_support": m_min})
                for r in grow:
                    self.claude_primary[r] = tid
                created.append(tid)
            elif d["decision"] == "ASSIGN_EXISTING" and self.reg.resolve(d.get("existing_topic_id") or ""):
                for r in grow:
                    self.claude_primary[r] = d["existing_topic_id"]
            else:
                for r in grow:
                    self.unresolved[r] = "novelty_rejected"
        pd.DataFrame([{**n, "group": int(comp[i])} for i, n in enumerate(novelty)]).to_csv(
            self.out / "assignments" / "novelty_candidates.csv", index=False)
        self.snapshot("novelty")
        self.log_stage("novelty", candidates=len(novelty), coherent_groups=len(groups), min_support=m_min, created=created)

    def final_labels(self, frame):
        prim, sec, src = [], [], []
        for r in range(self.N):
            p, s, so = "", "", None
            if r in self.non_work:
                so = "non_work_flagged_by_claude"
            elif r in self.claude_primary and self.reg.resolve(self.claude_primary[r]):
                p = self.reg.resolve(self.claude_primary[r])
                s2 = self.reg.resolve(self.claude_secondary[r]) if r in self.claude_secondary else None
                s, so = (s2 if s2 and s2 != p else ""), "claude"
            elif r in self.unresolved:
                so = self.unresolved[r]
            elif r in self.propagated and self.reg.resolve(self.propagated[r]):
                p, so = self.reg.resolve(self.propagated[r]), "propagated_from_claude"
            else:
                p = frame.auto_topic.iat[r]
                so = "auto_high_confidence" if frame.queue_reason.iat[r] == "" else "auto_low_confidence"
            prim.append(p), sec.append(s), src.append(so)
        return np.array(prim, dtype=object), np.array(sec, dtype=object), np.array(src, dtype=object)

    # ------------------------------------------------------------------ diagnostics + critic
    def diagnostics(self, prim, frame):
        from sklearn.cluster import KMeans
        ids = sorted({p for p in prim if p})
        if not ids:
            return pd.DataFrame()
        cents = {t: si.normalize(self.X[prim == t].mean(0)) for t in ids}
        C = np.vstack([cents[t] for t in ids])
        CC = C @ C.T
        np.fill_diagonal(CC, -np.inf)
        mthr = self.thresholds.get("margin_threshold", 0.0)
        m_min = max(3, int(np.percentile([len(self.evidence_rows(t)) for t in ids], 10)))
        rows = []
        for k, t in enumerate(ids):
            m = prim == t
            Xm = self.X[m]
            sims = Xm @ cents[t]
            j = int(np.argmax(CC[k])) if len(ids) > 1 else k
            gain = 0.0
            if len(Xm) >= 2 * m_min:
                lab = KMeans(2, n_init=10, random_state=SEED).fit_predict(Xm)
                if min((lab == 0).sum(), (lab == 1).sum()) >= m_min:
                    within = np.mean([float((Xm[lab == g] @ si.normalize(Xm[lab == g].mean(0))).mean()) for g in (0, 1)])
                    gain = float(within - sims.mean())
            rows.append({"topic_id": t, "topic_name": self.reg.get(t)["topic_name"], "task_count": int(m.sum()),
                         "semantic_cohesion": float(sims.mean()), "median_similarity": float(np.median(sims)),
                         "nearest_competing_topic": ids[j] if len(ids) > 1 else "",
                         "between_topic_similarity": float(CC[k, j]) if len(ids) > 1 else np.nan,
                         "low_confidence_rate": float((frame.margin.values[m] < mthr).mean()),
                         "secondary_membership_rate": float(np.mean([r in self.claude_secondary for r in np.where(m)[0]])),
                         "posting_diversity": int(self.df.posting_id[m].nunique()),
                         "provider_diversity": int(self.df.provider[m].nunique()),
                         "source_diversity": int(self.df.source_section[m].nunique()) if "source_section" in self.df else 1,
                         "split_gain": gain})
        D = pd.DataFrame(rows)
        q = lambda col, p: float(np.nanpercentile(D[col], p))
        flags = []
        for r in D.itertuples():
            f = []
            if r.semantic_cohesion < q("semantic_cohesion", 10):
                f.append("LOW_COHESION")
            if r.task_count > q("task_count", 90) and r.semantic_cohesion < q("semantic_cohesion", 50):
                f.append("OVER_BROAD")
            if not np.isnan(r.between_topic_similarity) and r.between_topic_similarity >= r.semantic_cohesion:
                f.append("OVERLAPPING")
                other = D[D.topic_id == r.nearest_competing_topic]
                mu = self.X[(prim == r.topic_id) | (prim == r.nearest_competing_topic)]
                if len(other) and float((mu @ si.normalize(mu.mean(0))).mean()) >= min(r.semantic_cohesion,
                                                                                       float(other.semantic_cohesion.iloc[0])):
                    f.append("POSSIBLE_MERGE")
                na = set(self.reg.get(r.topic_id)["topic_name"].lower().split())
                nb = set(self.reg.get(r.nearest_competing_topic)["topic_name"].lower().split())
                if na and nb and len(na & nb) / len(na | nb) >= 0.5:
                    f.append("DUPLICATE")
            if r.split_gain > 0 and r.split_gain >= q("split_gain", 90):
                f.append("POSSIBLE_SPLIT")
            if r.low_confidence_rate > 0 and r.low_confidence_rate > q("low_confidence_rate", 90):
                f.append("HIGH_AMBIGUITY")
            if r.task_count < m_min or r.posting_diversity < 2:
                f.append("UNDER_SUPPORTED")
            flags.append(";".join(f))
        D["diagnostic_flags"] = flags
        return D

    def critic(self, D, prim, max_calls, rnd, extra_split=()):
        from sklearn.cluster import KMeans
        fl = D[(D.diagnostic_flags != "") | D.topic_id.isin(list(extra_split))].copy()
        if fl.empty or max_calls <= 0:
            return {"flagged": int(len(fl)), "calls": 0}
        fl = fl.assign(_sev=fl.diagnostic_flags.str.count(";") + (fl.diagnostic_flags != "")).sort_values(
            ["_sev", "task_count"], ascending=[False, False])
        previews, payload = {}, []
        for r in fl.itertuples():
            rows = np.where(prim == r.topic_id)[0]
            c = si.normalize(self.X[rows].mean(0))
            item = {"topic": prompts.topic_card(self.reg.get(r.topic_id)), "flags": r.diagnostic_flags or "FLAG_SPLIT",
                    "diagnostics": {k: (round(getattr(r, k), 3) if isinstance(getattr(r, k), float) else getattr(r, k))
                                    for k in ("task_count", "semantic_cohesion", "median_similarity", "between_topic_similarity",
                                              "low_confidence_rate", "posting_diversity")},
                    "nearest_competing_topic": prompts.topic_card(self.reg.get(r.nearest_competing_topic), compact=True)
                    if r.nearest_competing_topic else None,
                    "members": [f"{code(i)}: {self.text(i)}" for i in rows[np.argsort(-(self.X[rows] @ c))][:5]]}
            if ("POSSIBLE_SPLIT" in r.diagnostic_flags or r.topic_id in extra_split) and len(rows) >= 4:
                lab = KMeans(2, n_init=10, random_state=SEED).fit_predict(self.X[rows])
                previews[r.topic_id] = {g: rows[lab == gi] for gi, g in enumerate("AB")}
                for g, gr in previews[r.topic_id].items():
                    cg = si.normalize(self.X[gr].mean(0))
                    item[f"preview_group_{g}"] = [f"{code(i)}: {self.text(i)}" for i in gr[np.argsort(-(self.X[gr] @ cg))][:4]]
            payload.append(item)
        cap, calls = capacity("critic"), 0
        stats = {"KEEP": 0, "MERGE": 0, "SPLIT": 0, "REFINE_BOUNDARY": 0, "REASSIGN": 0, "invalid": 0}
        pos = 0
        while pos < len(payload) and calls < max_calls and self.can("critic"):
            batch = payload[pos:pos + cap]
            while len(batch) > 1 and si.est_tokens(json.dumps(batch)) > MAX_INPUT_TOKENS_PER_CALL:
                batch = batch[:-1]
            pos += len(batch)
            out, _ = self.ask("critic", (batch,), f"round {rnd} batch {calls + 1}: {len(batch)} topics")
            calls += 1
            if out is None:
                break
            shown = {b["topic"]["topic_id"] for b in batch}
            listed = shown | {b["nearest_competing_topic"]["topic_id"] for b in batch if b["nearest_competing_topic"]}
            shown_rows = {parse_code(m.split(":")[0], self.N) for b in batch for k in ("members", "preview_group_A", "preview_group_B")
                          for m in b.get(k, [])}
            rn = f"critic_round_{rnd}"
            for d in out["decisions"]:
                t = d["topic_id"]
                if t not in shown or not self.reg.resolve(t):
                    stats["invalid"] += 1
                    continue
                stats[d["action"]] += 1
                if d["action"] == "MERGE" and d.get("merge_into") in listed:
                    self.reg.merge(t, d["merge_into"], rn, d.get("reason", ""))
                elif d["action"] == "SPLIT" and t in previews and len(d["split_children"]) >= 2 and \
                        {c["preview_group"] for c in d["split_children"][:2]} == {"A", "B"}:
                    kids = [{"name": ch["name"], "definition": ch["definition"],
                             "supporting_statement_ids": [self.sid(i) for i in previews[t][ch["preview_group"]]]}
                            for ch in d["split_children"][:2]]
                    for kid, ch in zip(self.reg.split(t, kids, rn, d.get("reason", "")), d["split_children"][:2]):
                        for i in previews[t][ch["preview_group"]]:
                            if self.claude_primary.get(int(i)) == t:
                                self.claude_primary[int(i)] = kid
                elif d["action"] == "REFINE_BOUNDARY":
                    self.reg.refine(t, rn, d.get("reason", ""), definition=d.get("definition"),
                                    inclusion_rules=d.get("inclusion_rules"), exclusion_rules=d.get("exclusion_rules"))
                elif d["action"] == "REASSIGN":
                    for ra in d.get("reassign", []):
                        i = parse_code(ra["statement_id"], self.N)
                        if i in shown_rows and ra["topic_id"] in listed and self.reg.resolve(ra["topic_id"]):
                            self.claude_primary[int(i)] = ra["topic_id"]
        return {"flagged": int(len(fl)), "calls": calls, "decisions": stats}

    def noise_floor(self, base_auto):
        rng, rates = np.random.default_rng(SEED), []
        for _ in range(3):
            ids, P = self.prototypes(rng=rng)
            if len(ids) >= 2:
                rates.append(float((np.array([ids[j] for j in np.argmax(self.X @ P.T, axis=1)], dtype=object) != base_auto).mean()))
        return float(np.mean(rates)) if rates else 0.0

    # ------------------------------------------------------------------ orchestration
    def run(self):
        sample = si.choose_sample(self.X, self.df.statement_text.tolist(), max(1, self.caps["discovery"]),
                                  DISCOVERY_STATEMENT_TOKENS_PER_CALL)
        sample_rows = np.asarray(sample["indices"])
        pd.DataFrame({"n": np.arange(1, len(sample["radius_curve"]) + 1), "coverage_radius": sample["radius_curve"]}).to_csv(
            self.out / "provenance" / "sampling_curve.csv", index=False)
        pd.DataFrame({"fps_rank": np.arange(len(sample_rows)), "row": sample_rows,
                      "statement_id": self.df.statement_id.values[sample_rows],
                      "statement_text": self.df.statement_text.values[sample_rows]}).to_csv(
            self.out / "provenance" / "discovery_sample.csv", index=False)
        self.log_stage("sampling", **{k: v for k, v in sample.items() if k not in ("indices", "radius_curve")})
        self.discover(sample_rows)
        if not self.reg.active():
            self.stop_reason = "discovery produced no valid topics"
            return self.finish(None)
        self.consolidate()
        split_flags, _ = self.registry_review("boundary_review", "global boundary review")
        frame = self.assignment_pass()
        self.novelty_gate(self.resolve_ambiguous(frame))
        frame = self.assignment_pass()
        prim, sec, src = self.final_labels(frame)
        prev_prim, prev_count = prim.copy(), len(self.reg.active())
        for rnd in range(1, self.max_rounds + 1):
            D = self.diagnostics(prim, frame)
            D.to_csv(self.out / "refinement" / f"round_{rnd}_diagnostics.csv", index=False)
            n_flags = int((D.diagnostic_flags != "").sum()) if len(D) else 0
            left = self.caps["critic"] - self.client.budget._used("critic")
            if n_flags == 0 and not split_flags:
                self.stop_reason = f"round {rnd}: no diagnostic flags - nothing to refine"
                break
            if left <= 0 or not self.can("critic"):
                self.stop_reason = f"round {rnd}: critic calls in the plan are used up"
                break
            ev0 = len(self.reg.events)
            cr = self.critic(D, prim, max(1, math.ceil(left / (self.max_rounds - rnd + 1))), rnd, split_flags)
            split_flags = []
            kinds = pd.Series([e["kind"] for e in self.reg.events[ev0:]]).value_counts().to_dict()
            frame = self.assignment_pass()
            prim, sec, src = self.final_labels(frame)
            changed, floor = float((prim != prev_prim).mean()), self.noise_floor(frame.auto_topic.values.astype(object))
            D2 = self.diagnostics(prim, frame)
            row = {"round": rnd, "topics_before": prev_count, "topics_after": len(self.reg.active()),
                   "topic_count_change": len(self.reg.active()) - prev_count, "assignment_change_rate": changed,
                   "assignment_noise_floor": floor, "new_topics": kinds.get("add", 0), "merges": kinds.get("merge", 0),
                   "splits": kinds.get("split", 0), "refines": kinds.get("refine", 0), "rejects": kinds.get("reject", 0),
                   "low_confidence_rate": float((frame.queue_reason != "").mean()),
                   "duplicate_flags": int(D2.diagnostic_flags.str.contains("DUPLICATE").sum()) if len(D2) else 0,
                   "flags_before": n_flags, "flags_after": int((D2.diagnostic_flags != "").sum()) if len(D2) else 0,
                   "critic_calls": cr["calls"], "critic_decisions": cr.get("decisions")}
            self.convergence.append(row)
            pd.DataFrame(self.convergence).to_csv(self.out / "refinement" / "convergence.csv", index=False)
            self.snapshot(f"critic_round_{rnd}")
            self.log_stage(f"critic_round_{rnd}", **{k: v for k, v in row.items() if k != "critic_decisions"})
            if abs(row["topic_count_change"]) <= 1 and changed <= floor and row["duplicate_flags"] == 0:
                self.stop_reason = (f"converged after round {rnd}: topic-count change {row['topic_count_change']}, assignment "
                                    f"change {changed:.3f} <= prototype-noise floor {floor:.3f}, no duplicate flags")
                break
            prev_prim, prev_count = prim.copy(), len(self.reg.active())
        else:
            last = self.convergence[-1] if self.convergence else {}
            unmet = [m for m, ok in (("topic-count change <= 1", abs(last.get("topic_count_change", 99)) <= 1),
                                     ("assignment change <= noise floor",
                                      last.get("assignment_change_rate", 1) <= last.get("assignment_noise_floor", 0)),
                                     ("no duplicate flags", last.get("duplicate_flags", 1) == 0)) if not ok]
            self.stop_reason = f"refinement round limit ({self.max_rounds}) reached; unmet: {', '.join(unmet) or 'none'}"
        _, changes = self.registry_review("consistency", "final consistency")
        if changes:
            frame = self.assignment_pass()
            prim, sec, src = self.final_labels(frame)
            self.stop_reason += f"; final consistency review applied {changes} registry change(s)"
        return self.finish((prim, sec, src, frame))

    def finish(self, result):
        self.reg.save(self.out / "topic_registry" / "registry_final.json")
        summary = {"mode": "full", "stop_reason": self.stop_reason, "thresholds": self.thresholds, "queue": self.queue_info,
                   "convergence": self.convergence, "non_work_flagged": len(self.non_work),
                   "topics_created_total": len(self.reg.topics), "registry_version": self.reg.version,
                   "status_counts": pd.Series([v["status"] for v in self.reg.topics.values()]).value_counts().to_dict(),
                   "budget": self.client.budget.summary(), "deferred_consolidation_pairs": len(self.deferred_pairs),
                   "skipped_decisions": self.failures}
        reg_rows = [{k: (json.dumps(v[k]) if isinstance(v[k], (list, dict)) else v[k]) for k in
                     ("topic_id", "topic_name", "definition", "core_action", "object_or_outcome", "inclusion_rules",
                      "exclusion_rules", "representative_statement_ids", "status", "created_round", "merged_into",
                      "children", "parent", "definition_pending", "provenance")} for v in self.reg.topics.values()]
        if result is not None:
            prim, sec, src, frame = result
            A = pd.DataFrame({"statement_id": self.df.statement_id, "posting_id": self.df.posting_id, "provider": self.df.provider,
                              "statement_text": self.df.statement_text, "primary_topic": prim, "secondary_topic": sec,
                              "assignment_source": src, "top1_similarity": frame.top1_sim.values, "margin": frame.margin.values,
                              "queue_reason": frame.queue_reason.values})
            A.to_csv(self.out / "assignments" / "assignments_final.csv", index=False)
            counts = pd.Series(prim[prim != ""]).value_counts()
            for r in reg_rows:
                r["primary_members"] = int(counts.get(r["topic_id"], 0))
            assigned = prim != ""
            summary.update({"N_AGENTIC": int((counts > 0).sum()), "assigned": int(assigned.sum()),
                            "coverage": float(assigned.mean()), "secondary_memberships": int((sec != "").sum()),
                            "ambiguity_rate": float((src == "auto_low_confidence").mean()),
                            "assignment_sources": pd.Series(src).value_counts().to_dict()})
            write_json(self.out / "assignments" / "assignment_thresholds.json", self.thresholds)
        pd.DataFrame(reg_rows).to_csv(self.out / "topic_registry" / "topic_registry_final.csv", index=False)
        write_json(self.out / "agentic_summary.json", summary)
        return summary
