"""Persisted hard Claude API budget.

- The budget file is NEVER reset. An existing agentic_taxonomy_results/provenance/api_budget.json (including one
  written by an earlier implementation) is read and its calls_made preserved. A recorded prior-usage floor
  (api_budget_prior_usage.json) is applied with max(), so calls made before this runtime are never forgotten.
- hard_limit <= 20 always.
- authorize() commits the call slot to disk BEFORE the request is sent (a crash can never refund a call).
- A run plan is scaled to the REAL remaining budget when a run starts and persisted; a resumed run reuses the same
  plan so its prompts (and cache keys) are identical and completed calls replay from the cache for free.
- Normal work draws from the plan's role caps; repair / transient retries draw from the plan's reserve.
- Metadata requests (model validation) are logged separately and never use taxonomy-call budget.
"""
from __future__ import annotations

import uuid
from pathlib import Path

from .config import HARD_CEILING, PRIOR_USAGE_FILE
from .provenance import read_json, utc, write_json

# (role, minimum, maximum) - minimums give a meaningful pipeline; extra calls are added in priority order
ROLE_RANGE = {"discovery": (2, 4), "consolidation": (1, 2), "boundary_review": (0, 1), "ambiguous": (1, 4),
              "novelty": (0, 1), "critic": (1, 3), "consistency": (1, 2)}
EXTRA_PRIORITY = ["discovery", "ambiguous", "consolidation", "critic", "boundary_review", "novelty", "discovery",
                  "ambiguous", "critic", "ambiguous", "consistency"]
MIN_NORMAL = sum(v[0] for v in ROLE_RANGE.values())          # 6
MIN_RESERVE = 2
MAX_RESERVE = 4
NORMAL_CAP = 16


class BudgetExceeded(RuntimeError):
    pass


def allocate(remaining: int) -> dict:
    """Plan for one run, scaled to the real remaining calls. Raises BudgetExceeded if too few remain."""
    reserve = min(MAX_RESERVE, max(MIN_RESERVE, remaining - NORMAL_CAP))
    normal = min(NORMAL_CAP, remaining - reserve)
    if normal < MIN_NORMAL:
        raise BudgetExceeded(f"only {remaining} Claude calls remain; a meaningful agentic run needs at least "
                             f"{MIN_NORMAL + MIN_RESERVE} ({MIN_NORMAL} normal + {MIN_RESERVE} reserve)")
    caps = {r: lo for r, (lo, _) in ROLE_RANGE.items()}
    spare = normal - sum(caps.values())
    while spare > 0:
        progressed = False
        for r in EXTRA_PRIORITY:
            if spare and caps[r] < ROLE_RANGE[r][1]:
                caps[r] += 1
                spare -= 1
                progressed = True
        if not progressed:
            break
    return {"normal": sum(caps.values()), "reserve": reserve, "role_caps": caps, "remaining_at_start": remaining}


class ApiBudget:
    def __init__(self, path, hard_limit: int = HARD_CEILING, readonly: bool = False):
        """readonly=True: inspect the budget without writing (preflight must never create or overwrite the file)."""
        self.path, self.readonly = Path(path), readonly
        state = read_json(self.path)
        if state is None:
            state = {"hard_limit": HARD_CEILING, "calls_made": 0, "events": [], "metadata_requests": 0, "created_utc": utc()}
        state["hard_limit"] = min(int(state.get("hard_limit", HARD_CEILING)), int(hard_limit), HARD_CEILING)
        state.setdefault("events", [])
        state.setdefault("metadata_requests", 0)
        prior = read_json(self.path.parent / PRIOR_USAGE_FILE)
        if prior and int(prior.get("calls_already_made_floor", 0)) > int(state.get("calls_made", 0)):
            floor = int(prior["calls_already_made_floor"])
            state["events"].append({"call": None, "kind": "prior_usage_floor", "utc": utc(),
                                    "note": f"calls_made raised {state.get('calls_made', 0)} -> {floor}: {prior.get('source', '')}"})
            state["calls_made"] = floor
        self.state = state
        self._save()

    # ------------------------------------------------------------------ run plan
    @property
    def remaining(self) -> int:
        return max(0, self.state["hard_limit"] - self.state["calls_made"])

    @property
    def run(self) -> dict | None:
        return self.state.get("active_run")

    def start_or_resume_run(self) -> dict:
        r = self.run
        if r and r.get("status") == "in_progress":
            r["resumed_utc"] = r.get("resumed_utc", []) + [utc()]
            self._save()
            return r
        plan = allocate(self.remaining)
        r = {"run_id": uuid.uuid4().hex[:12], "status": "in_progress", "started_utc": utc(), **plan}
        if self.state.get("active_run"):
            self.state.setdefault("previous_runs", []).append(self.state["active_run"])
        self.state["active_run"] = r
        self._save()
        return r

    def finish_run(self, status="completed"):
        if self.run:
            self.run["status"] = status
            self.run["finished_utc"] = utc()
            self._save()

    def _used(self, role=None, kinds=("normal",)):
        rid = self.run["run_id"] if self.run else None
        return sum(1 for e in self.state["events"] if e.get("run_id") == rid and e.get("kind") in kinds
                   and (role is None or e.get("role") == role))

    def can(self, role) -> bool:
        if not self.run or self.remaining <= 0:
            return False
        return (self._used(role) < self.run["role_caps"].get(role, 0) and self._used() < self.run["normal"])

    def can_reserve(self) -> bool:
        return bool(self.run) and self.remaining > 0 and self._used(kinds=("repair", "retry")) < self.run["reserve"]

    def authorize(self, role, kind="normal", note="") -> int:
        if self.remaining <= 0:
            raise BudgetExceeded(f"hard limit {self.state['hard_limit']} reached ({self.state['calls_made']} calls made)")
        if kind == "normal" and not self.can(role):
            raise BudgetExceeded(f"plan cap for '{role}' reached")
        if kind in ("repair", "retry") and not self.can_reserve():
            raise BudgetExceeded("reserve exhausted")
        self.state["calls_made"] += 1
        n = self.state["calls_made"]
        self.state["events"].append({"call": n, "role": role, "kind": kind, "run_id": self.run["run_id"] if self.run else None,
                                     "note": note, "utc": utc(), "status": "sent"})
        self._save()                                              # committed BEFORE the request is sent
        return n

    def record(self, call_no, status, usage=None):
        for e in reversed(self.state["events"]):
            if e.get("call") == call_no:
                e["status"] = status
                if usage:
                    e["usage"] = usage
                break
        self._save()

    def note_metadata(self, what):
        self.state["metadata_requests"] += 1
        self.state["events"].append({"call": None, "kind": "metadata", "note": what, "utc": utc()})
        self._save()

    def summary(self) -> dict:
        r = self.run or {}
        return {"hard_limit": self.state["hard_limit"], "calls_made": self.state["calls_made"],
                "remaining_calls": self.remaining, "metadata_requests": self.state["metadata_requests"],
                "run_id": r.get("run_id"), "run_status": r.get("status"), "run_plan": r.get("role_caps"),
                "run_normal_used": self._used() if r else 0,
                "run_reserve_used": self._used(kinds=("repair", "retry")) if r else 0}

    def _save(self):
        if self.readonly:
            return
        self.state["remaining_calls"] = self.remaining
        self.state["updated_utc"] = utc()
        write_json(self.path, self.state)
