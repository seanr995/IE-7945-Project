"""MOCK Gemini responder for local/offline testing ONLY. Produces schema-valid JSON from the request payload; no network call and
no SDK use; selected only when SPRINT4_MOCK=1."""
from __future__ import annotations

import hashlib
import json
import re


def _payload(user: str) -> dict:
    return json.loads(user.split("INPUT_JSON:\n", 1)[1])


def _h(x: str, mod: int) -> int:
    return int(hashlib.md5(x.encode()).hexdigest(), 16) % mod


def mock_responder(system: str, user: str) -> str:
    role = re.search(r"\[ROLE:(\w+)\]", system).group(1)
    p = _payload(user)
    if role == "leaf_labeler":
        out = []
        for t in p["topics"]:
            kw = [k for k in t["kw"] if " " not in k][:2] or ["work"]
            conf = round(0.45 + _h(t["id"], 50) / 100, 2)
            mixed = conf < 0.55
            out.append({"topic_id": t["id"], "topic_name": f"Coordinate {kw[0].title()} {kw[-1].title()} Work",
                        "description": f"Activities centred on {', '.join(kw)}.", "core_action": f"coordinate {kw[0]}",
                        "object_or_outcome": kw[-1], "include_if": f"task concerns {kw[0]}", "exclude_if": "task is about an unrelated activity",
                        "coherent": not mixed, "mixed_topic": mixed, "confidence": conf, "reason": "Mock label from keywords."})
        return json.dumps({"topics": out})
    if role == "hierarchy_builder":
        fams, doms = [], []
        for f in p["families"]:
            first = f["leaves"][0]["name"].split()
            ids = [x["id"] for x in f["leaves"]]
            mis = [ids[-1]] if len(ids) > 3 and _h(f["id"], 3) == 0 else []
            fams.append({"parent_id": f["id"], "parent_name": f"Manage {first[1] if len(first) > 1 else 'Core'} Activities",
                         "parent_description": f"Grouping of {len(ids)} related leaves.", "confidence": round(0.5 + _h(f["id"], 45) / 100, 2),
                         "misplaced_child_ids": mis, "reason": "Mock grouping review."})
        for d in p["domains"]:
            nm = d["leaf_names"][0].split() if d["leaf_names"] else ["x", "Core"]
            doms.append({"parent_id": d["id"], "parent_name": f"Operate {nm[1] if len(nm) > 1 else 'Core'} Domain",
                         "parent_description": f"Broad domain over {len(d['families'])} families.", "confidence": 0.7,
                         "misplaced_child_ids": [], "reason": "Mock domain review."})
        return json.dumps({"families": fams, "domains": doms})
    if role == "onet_evaluator":
        out = []
        for t in p["topics"]:
            ids = [c["oid"] for c in t["cands"]]
            k = _h(t["id"], 6)
            al = "NO_MATCH" if k == 0 else "PARTIAL_MATCH" if k in (1, 2) else "STRONG_MATCH"
            out.append({"topic_id": t["id"], "alignment": al, "best_onet_task_ids": [] if al == "NO_MATCH" else ids[:2],
                        "dominant_onet_concept": "" if al == "NO_MATCH" else "mock shared activity", "semantic_alignment_confidence": 0.7,
                        "reason": "Mock evaluation."})
        return json.dumps({"evaluations": out})
    if role == "critic":
        out, li = [], 0
        for c in p["cases"]:
            act, extra = "ACCEPT", {}
            if c["type"] == "leaf":
                k = li % 5
                li += 1
                if k == 0:
                    act, extra = "RENAME", {"new_name": c["name"] + " Support"}
                elif k == 1:
                    act, extra = "REFINE_DESCRIPTION", {"new_description": c["d"] + " (refined)"}
                elif k == 2 and c["alt"]:
                    act, extra = "MOVE_PARENT", {"move_to_parent_id": c["alt"][0][0]}
                elif k == 3:
                    act = "FLAG_UNRESOLVED"
            else:
                if _h(c["id"], 4) == 0:
                    act, extra = "MERGE_PARENT", {"merge_with_parent_id": c["near"][0]}
                elif _h(c["id"], 4) == 1:
                    act, extra = "RENAME", {"new_name": c["name"] + " Group"}
            out.append({"target_type": c["type"], "target_id": c["id"], "action": act, "new_name": extra.get("new_name"),
                        "new_description": extra.get("new_description"), "move_to_parent_id": extra.get("move_to_parent_id"),
                        "merge_with_parent_id": extra.get("merge_with_parent_id"), "confidence": 0.7, "reason": "Mock critic decision."})
        return json.dumps({"decisions": out})
    raise AssertionError(role)
