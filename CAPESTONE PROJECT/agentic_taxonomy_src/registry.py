"""Topic registry with immutable, auditable IDs (AT_0001, AT_0002, ...).

IDs are never reused or renumbered. Names / definitions may evolve (history kept). A merged topic keeps its ID with
status='merged' and merged_into=<survivor>; a split topic keeps its ID with status='split' and children=[...].
Every mutation increments `version` and appends a provenance event.
"""
from __future__ import annotations

import copy
import re
from datetime import datetime, timezone

from .provenance import read_json, stable_hash, write_json

ACTIVE = "active"
BAD_NAME = re.compile(r"^\s*(other|others|misc|miscellaneous|general|various|unclassified|n/?a|none)\b", re.I)


class TopicRegistry:
    def __init__(self):
        self.topics: dict[str, dict] = {}
        self.next_id = 1
        self.version = 0
        self.events: list[dict] = []

    # ------------------------------------------------------------------ core
    def _event(self, kind, **kw):
        self.version += 1
        self.events.append({"version": self.version, "kind": kind, "utc": datetime.now(timezone.utc).isoformat(), **kw})

    def new_id(self) -> str:
        tid = f"AT_{self.next_id:04d}"
        self.next_id += 1
        return tid

    def add(self, name, definition, core_action="", object_or_outcome="", inclusion_rules=None, exclusion_rules=None,
            representative_statement_ids=None, supporting_statement_ids=None, created_round="discovery",
            provenance=None, confidence=None, definition_pending=False) -> str:
        tid = self.new_id()
        self.topics[tid] = {
            "topic_id": tid, "topic_name": name.strip(), "definition": (definition or "").strip(),
            "core_action": core_action, "object_or_outcome": object_or_outcome,
            "inclusion_rules": list(inclusion_rules or []), "exclusion_rules": list(exclusion_rules or []),
            "representative_statement_ids": list(representative_statement_ids or []),
            "supporting_statement_ids": list(supporting_statement_ids or []),
            "created_round": created_round, "status": ACTIVE, "merged_into": None, "children": [], "parent": None,
            "confidence": confidence, "definition_pending": bool(definition_pending),
            "provenance": [dict(provenance or {}, action="created")], "history": []}
        self._event("add", topic_id=tid, name=name)
        return tid

    def active(self) -> list[str]:
        return [t for t, v in self.topics.items() if v["status"] == ACTIVE]

    def get(self, tid) -> dict:
        return self.topics[tid]

    def resolve(self, tid) -> str | None:
        """Follow merged_into chains to the surviving active topic (None if rejected / split / unknown)."""
        seen = set()
        while tid in self.topics and tid not in seen:
            seen.add(tid)
            t = self.topics[tid]
            if t["status"] == ACTIVE:
                return tid
            if t["status"] == "merged" and t["merged_into"]:
                tid = t["merged_into"]
                continue
            return None
        return None

    # ------------------------------------------------------------------ mutations
    def refine(self, tid, round_name, reason="", **fields):
        t = self.topics[tid]
        before = {k: copy.deepcopy(t[k]) for k in fields if k in t}
        changed = {k: v for k, v in fields.items() if v not in (None, "", []) and t.get(k) != v}
        if not changed:
            return False
        t["history"].append({"round": round_name, "before": before, "reason": reason})
        t.update(changed)
        if "definition" in changed:
            t["definition_pending"] = False
        t["provenance"].append({"action": "refined", "round": round_name, "fields": sorted(changed), "reason": reason})
        self._event("refine", topic_id=tid, fields=sorted(changed))
        return True

    def merge(self, absorbed, survivor, round_name, reason="") -> bool:
        a, s = self.resolve(absorbed), self.resolve(survivor)
        if a is None or s is None or a == s:
            return False
        ta, ts = self.topics[a], self.topics[s]
        ta["status"], ta["merged_into"] = "merged", s
        ta["provenance"].append({"action": "merged_into", "target": s, "round": round_name, "reason": reason})
        ts["supporting_statement_ids"] = sorted(set(ts["supporting_statement_ids"]) | set(ta["supporting_statement_ids"]))
        ts["representative_statement_ids"] = list(dict.fromkeys(ts["representative_statement_ids"] +
                                                                ta["representative_statement_ids"]))[:6]
        ts["provenance"].append({"action": "absorbed", "source": a, "round": round_name, "reason": reason})
        self._event("merge", absorbed=a, survivor=s, reason=reason)
        return True

    def split(self, tid, children: list[dict], round_name, reason="") -> list[str]:
        """children: [{name, definition, supporting_statement_ids, ...}] -> new IDs; parent keeps its ID (status split)."""
        p = self.topics[tid]
        if p["status"] != ACTIVE or len(children) < 2:
            return []
        new = []
        for c in children:
            cid = self.add(c["name"], c.get("definition", ""), p.get("core_action", ""), p.get("object_or_outcome", ""),
                           c.get("inclusion_rules"), c.get("exclusion_rules"), c.get("representative_statement_ids"),
                           c.get("supporting_statement_ids"), created_round=round_name,
                           provenance={"source": "split", "parent": tid, "reason": reason})
            self.topics[cid]["parent"] = tid
            new.append(cid)
        p["status"], p["children"] = "split", new
        p["provenance"].append({"action": "split_into", "children": new, "round": round_name, "reason": reason})
        self._event("split", parent=tid, children=new, reason=reason)
        return new

    def reject(self, tid, round_name, reason=""):
        t = self.topics[tid]
        if t["status"] != ACTIVE:
            return False
        t["status"] = "rejected"
        t["provenance"].append({"action": "rejected", "round": round_name, "reason": reason})
        self._event("reject", topic_id=tid, reason=reason)
        return True

    # ------------------------------------------------------------------ persistence
    def fingerprint(self) -> str:
        """Registry content hash used as the cache 'topic_registry_version'."""
        return f"v{self.version}-" + stable_hash({t: {k: v[k] for k in ("topic_name", "definition", "status",
                                                                         "merged_into", "inclusion_rules",
                                                                         "exclusion_rules")}
                                                   for t, v in sorted(self.topics.items())})[:12]

    def to_dict(self):
        return {"next_id": self.next_id, "version": self.version, "topics": self.topics, "events": self.events}

    @classmethod
    def from_dict(cls, d):
        r = cls()
        r.next_id, r.version, r.topics, r.events = d["next_id"], d["version"], d["topics"], d["events"]
        return r

    def save(self, path):
        write_json(path, self.to_dict())

    @classmethod
    def load(cls, path):
        d = read_json(path)
        return cls.from_dict(d) if d else cls()
