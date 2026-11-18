"""Gemini agents: short per-role system prompts, compact payloads and per-record business validators.
The model never changes leaf membership, never searches O*NET and never writes project files."""
from __future__ import annotations

import json

VAGUE = {"general tasks", "miscellaneous", "misc", "other", "others", "various", "general work", "work activities", "tasks", "general",
         "mixed tasks", "other tasks", "various tasks", "general activities", "miscellaneous tasks"}

SYS_LEAF = ("[ROLE:leaf_labeler] Label work-task topics. topic_name: verb-led WORK ACTIVITY, 2-8 words, Title Case; not an occupation, "
            "not a bare skill/tool, not vague. description <=35 words; include_if/exclude_if <=25 words; reason <=30 words; "
            "confidence 0-1. Input keys: id, n=size, core=most typical tasks, edge=least typical tasks, kw=keywords. If core and edge "
            "show several concepts set mixed_topic=true and coherent=false. Never move tasks. Be concise.")
SYS_HIER = ("[ROLE:hierarchy_builder] Name a task hierarchy given a fixed grouping. families: work-activity families (name 2-8 words, "
            "description <=30 words). domains: broad work domains over the families. List in misplaced_child_ids any child leaf (families) or "
            "child family (domains) that clearly does not belong; you only recommend, grouping is fixed. reason <=30 words; confidence 0-1. "
            "Input keys: leaves[id,name,act=core action,d=short description]. Return one entry per input id.")
SYS_ONET = ("[ROLE:onet_evaluator] Judge semantic alignment between a work topic and ALREADY-RETRIEVED O*NET task candidates (cands: oid, "
            "t=task, occ=occupation, cos=cosine). alignment: STRONG_MATCH (same work activity), PARTIAL_MATCH (related/narrower/broader), "
            "NO_MATCH (none fits). best_onet_task_ids only from the given oid values (empty for NO_MATCH). dominant_onet_concept: short "
            "phrase. semantic_alignment_confidence 0-1; reason <=30 words. Do not use outside knowledge to pick other tasks.")
SYS_CRITIC = ("[ROLE:critic] Review ONLY the listed problem cases; one decision per case id. Actions: ACCEPT, RENAME(new_name 2-8 words), "
              "REFINE_DESCRIPTION(new_description), MOVE_PARENT(leaf only; move_to_parent_id from l2 ids), MERGE_PARENT(parent only; "
              "merge_with_parent_id same level), SPLIT_PARENT(level-2 parent only), FLAG_UNRESOLVED. Prefer ACCEPT unless the evidence clearly "
              "justifies a change. Moving a leaf changes only the hierarchy; never tasks. reason <=30 words; confidence 0-1.")

INSTR = {"leaf_labeler": "Label each topic.", "hierarchy_builder": "Name every family and domain and flag misplaced children.",
         "onet_evaluator": "Judge each topic against its retrieved candidates.", "critic": "Decide on each flagged case."}


def user_prompt(role: str, payload: dict) -> str:
    return f"{INSTR[role]}\nINPUT_JSON:\n{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"


def _words(s: str) -> int:
    return len(str(s).split())


def _s(o, k, maxw=None, minw=1):
    v = getattr(o, k, None) if not isinstance(o, dict) else o.get(k)
    if not isinstance(v, str) or not v.strip():
        raise ValueError(f"{k} must be a non-empty string")
    if _words(v) < minw or (maxw and _words(v) > maxw):
        raise ValueError(f"{k} must be {minw}-{maxw or 'n'} words (got {_words(v)})")
    return v.strip()


def _conf(v, k="confidence"):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= float(v) <= 1:
        raise ValueError(f"{k} must be in [0,1]")
    return float(v)


def _name(v, k):
    n = _s({k: v}, k, 8, 2)
    if n.lower().strip(" .") in VAGUE:
        raise ValueError(f"{k} '{n}' is vague")
    return n


def leaf_validate(r: dict, tid: str):
    return {"topic_id": tid, "topic_name": _name(r.get("topic_name"), "topic_name"), "description": _s(r, "description", 35),
            "core_action": _s(r, "core_action", 12), "object_or_outcome": _s(r, "object_or_outcome", 20),
            "include_if": _s(r, "include_if", 25), "exclude_if": _s(r, "exclude_if", 25),
            "coherent": _bool(r, "coherent"), "mixed_topic": _bool(r, "mixed_topic"), "confidence": _conf(r.get("confidence")),
            "reason": _s(r, "reason", 30)}


def _bool(r, k):
    if not isinstance(r.get(k), bool):
        raise ValueError(f"{k} must be boolean")
    return r[k]


def parent_validate(r: dict, pid: str, children: list):
    mis = r.get("misplaced_child_ids")
    if not isinstance(mis, list) or not set(map(str, mis)) <= set(children):
        raise ValueError("misplaced_child_ids must be a subset of the given children")
    return {"parent_id": pid, "parent_name": _name(r.get("parent_name"), "parent_name"), "parent_description": _s(r, "parent_description", 30),
            "confidence": _conf(r.get("confidence")), "misplaced_child_ids": [str(x) for x in mis], "reason": _s(r, "reason", 30)}


def onet_validate(r: dict, tid: str, allowed: list):
    al = r.get("alignment")
    if al not in ("STRONG_MATCH", "PARTIAL_MATCH", "NO_MATCH"):
        raise ValueError("alignment invalid")
    b = [int(x) for x in r.get("best_onet_task_ids", [])]
    if not set(b) <= set(allowed):
        raise ValueError(f"best_onet_task_ids must come from the given candidates {allowed}")
    if al != "NO_MATCH" and not b:
        raise ValueError(f"{al} requires at least one best_onet_task_id")
    c = r.get("dominant_onet_concept")
    if not isinstance(c, str) or (al != "NO_MATCH" and not c.strip()):
        raise ValueError("dominant_onet_concept required")
    return {"topic_id": tid, "alignment": al, "best_onet_task_ids": b, "dominant_onet_concept": (c or "").strip(),
            "semantic_alignment_confidence": _conf(r.get("semantic_alignment_confidence"), "semantic_alignment_confidence"),
            "reason": _s(r, "reason", 30)}


def critic_validate(r: dict, tid: str, case_type: str, l2_ids: set, l1_ids: set):
    act, tt = r.get("action"), r.get("target_type")
    if tt != case_type:
        raise ValueError(f"target_type must be {case_type}")
    if act not in ("ACCEPT", "RENAME", "REFINE_DESCRIPTION", "MOVE_PARENT", "MERGE_PARENT", "SPLIT_PARENT", "FLAG_UNRESOLVED"):
        raise ValueError("action invalid")
    if act == "RENAME":
        _name(r.get("new_name"), "new_name")
    if act == "REFINE_DESCRIPTION":
        _s(r, "new_description", 40)
    if act == "MOVE_PARENT" and (case_type != "leaf" or r.get("move_to_parent_id") not in l2_ids):
        raise ValueError("MOVE_PARENT needs a leaf and move_to_parent_id from l2 ids")
    if act in ("MERGE_PARENT", "SPLIT_PARENT") and case_type != "parent":
        raise ValueError(f"{act} applies to a parent")
    if act == "MERGE_PARENT":
        same = l2_ids if tid in l2_ids else l1_ids
        if r.get("merge_with_parent_id") not in same or r["merge_with_parent_id"] == tid:
            raise ValueError("MERGE_PARENT needs merge_with_parent_id of the same level")
    if act == "SPLIT_PARENT" and tid not in l2_ids:
        raise ValueError("SPLIT_PARENT only for level-2 parents")
    return {"target_type": tt, "target_id": tid, "action": act, "new_name": r.get("new_name"), "new_description": r.get("new_description"),
            "move_to_parent_id": r.get("move_to_parent_id"), "merge_with_parent_id": r.get("merge_with_parent_id"),
            "confidence": _conf(r.get("confidence")), "reason": _s(r, "reason", 30)}


def by_id(lst, key):
    out = {}
    for e in lst:
        d = e if isinstance(e, dict) else e.model_dump()
        k = d.get(key)
        if k in out:
            raise ValueError(f"duplicate id {k}")
        out[k] = d
    return out
