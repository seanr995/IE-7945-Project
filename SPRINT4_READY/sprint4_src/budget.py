"""Call ledger for Sprint 4 Gemini generation requests (call-count budget; no dollar claims) plus the historical
Sprint 3 Claude spend estimate, kept separately for the report only."""
from __future__ import annotations

from .config import (HIST_PRICE_IN, HIST_PRICE_OUT, PLANNED_NORMAL_CALLS, PRIOR_PROVENANCE_DIRS, CallBudgetError, Settings, read_json, utc,
                     write_json)


def historical_claude_spend(s: Settings) -> dict:
    """Sprint 3 Claude spend estimate rebuilt from the old api_budget.json token logs (informational; Sprint 4 makes no Claude calls)."""
    best = None
    for root in s.roots:
        for rel in PRIOR_PROVENANCE_DIRS:
            b = read_json(root / rel / "api_budget.json")
            if not b:
                continue
            floor = (read_json(root / rel / "api_budget_prior_usage.json") or {}).get("calls_already_made_floor", 0)
            ev = [e for e in b.get("events", []) if e.get("call") and (e.get("usage") or {}).get("input_tokens") is not None]
            tin = sum(e["usage"]["input_tokens"] or 0 for e in ev)
            tout = sum(e["usage"]["output_tokens"] or 0 for e in ev)
            calls_made = max(int(b.get("calls_made", 0)), int(floor))
            unreported = max(0, calls_made - len(ev))
            mi = max([e["usage"]["input_tokens"] or 0 for e in ev] or [0])
            mo = max([e["usage"]["output_tokens"] or 0 for e in ev] or [0])
            cost = (tin + unreported * mi) * HIST_PRICE_IN / 1e6 + (tout + unreported * mo) * HIST_PRICE_OUT / 1e6
            rec = {"source": f"{rel}/api_budget.json", "claude_calls_made": calls_made, "calls_with_token_usage": len(ev),
                   "input_tokens": tin, "output_tokens": tout, "estimated_cost_usd": round(cost, 4),
                   "note": "historical Sprint 3 Claude estimate at $2/$10 per MTok; calls without token counts priced at the largest observed call"}
            if best is None or rec["estimated_cost_usd"] > best["estimated_cost_usd"]:
                best = rec
    return best or {"source": "none found", "claude_calls_made": 0, "estimated_cost_usd": 0.0}


class Ledger:
    """Persisted BEFORE every real request leaves the machine. Rows: real attempts and cache hits (cached=True)."""

    def __init__(self, s: Settings):
        self.s = s
        self.path = s.prov / ("sprint4_ledger_mock.json" if s.mock else "sprint4_ledger.json")
        self.d = read_json(self.path) or {"rows": []}
        self.plan_total = PLANNED_NORMAL_CALLS

    def save(self):
        write_json(self.path, self.d)

    # ---- counters
    def real_rows(self):
        return [r for r in self.d["rows"] if not r["cached"]]

    @property
    def real_used(self) -> int:
        return len(self.real_rows())

    def kind_count(self, *kinds) -> int:
        return sum(1 for r in self.real_rows() if r["kind"] in kinds)

    @property
    def normal_done(self) -> int:
        return self.kind_count("normal")

    @property
    def planned_remaining(self) -> int:
        return max(0, self.plan_total - self.normal_done)

    def can_extra(self, n: int = 1, other_planned: int | None = None) -> bool:
        """True if n extra (repair/retry/split) calls fit under the hard cap while the remaining planned normal calls stay affordable."""
        pr = self.planned_remaining if other_planned is None else other_planned
        return self.s.max_real_calls - self.real_used - pr >= n

    # ---- rows
    def authorize(self, role, batch_id, topic_count, kind) -> int:
        if self.real_used >= self.s.max_real_calls:
            raise CallBudgetError(f"GEMINI_CALL_CAP_REACHED: {self.real_used}/{self.s.max_real_calls} real generation calls already used "
                                  f"(rerun resumes from cache; raise GEMINI_MAX_REAL_CALLS only deliberately)")
        if kind in ("repair",) and self.kind_count("repair") >= self.s.max_repair_calls:
            raise CallBudgetError("repair call limit (2) reached")
        n = self.real_used + 1
        self.d["rows"].append({"call_number": n, "provider": self.s.provider, "model": self.s.model, "agent_role": role, "batch_id": batch_id,
                               "topic_count": topic_count, "input_tokens": None, "output_tokens": None, "status": "sent", "cached": False,
                               "error_type": "", "timestamp": utc(), "kind": kind})
        self.save()
        return n

    def finish(self, n, status, usage=None, error_type=""):
        r = self.real_rows()[n - 1]
        r["status"], r["error_type"] = status, error_type
        if usage:
            r["input_tokens"], r["output_tokens"] = usage.get("input_tokens"), usage.get("output_tokens")
        self.save()

    def cache_hit(self, role, batch_id, topic_count):
        self.d["rows"].append({"call_number": None, "provider": self.s.provider, "model": self.s.model, "agent_role": role, "batch_id": batch_id,
                               "topic_count": topic_count, "input_tokens": 0, "output_tokens": 0, "status": "cache_hit", "cached": True,
                               "error_type": "", "timestamp": utc(), "kind": "cache"})
        self.save()

    def summary(self) -> dict:
        rr = self.real_rows()
        ok = [r for r in rr if r["status"] == "ok"]
        return {"provider": self.s.provider, "model": self.s.model, "gemini_real_calls": len(rr), "gemini_successful_calls": len(ok),
                "normal_calls": self.kind_count("normal"), "normal_successful_calls": sum(1 for r in ok if r["kind"] == "normal"), "repair_calls": self.kind_count("repair"),
                "retry_calls": self.kind_count("retry"), "split_calls": self.kind_count("split"),
                "cache_hits": sum(1 for r in self.d["rows"] if r["cached"]), "hard_cap": self.s.max_real_calls,
                "input_tokens": sum(r["input_tokens"] or 0 for r in rr), "output_tokens": sum(r["output_tokens"] or 0 for r in rr),
                "claude_sprint4_calls": 0}
