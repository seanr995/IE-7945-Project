"""Synthetic task world + rule-based mock Claude (JSON TEXT replies) for offline tests. No API calls ever.

mock_claude(request) receives the exact request dict the runtime would send (model, max_tokens, system, messages)
and returns JSON text, like the real Messages API with a JSON-only prompt.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

READY = Path(__file__).resolve().parents[1] / "CAPESTONE PROJECT_READY"
ORIGINAL = Path(__file__).resolve().parents[1] / "CAPESTONE PROJECT"
sys.path.insert(0, str(READY))

TYPES = ["Pipeline Development", "Pipeline Monitoring", "Financial Reporting", "Customer Analysis", "Model Evaluation",
         "Vendor Negotiation"]


def make_world(n_per_type=40, dims=64, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(len(TYPES), dims))
    rows, X = [], []
    for t, name in enumerate(TYPES):
        for j in range(n_per_type):
            X.append(centers[t] + 0.35 * rng.normal(size=dims))
            rows.append({"statement_id": f"id_{t}_{j}", "posting_id": f"p{(t * 7 + j) % 23}",
                         "provider": "groq" if j % 9 == 0 else "gemini", "statement_text": f"{name} task {j}",
                         "source_section": "responsibilities", "true_type": name})
    X = np.asarray(X, dtype=np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return {"df": pd.DataFrame(rows), "X_t5": X, "P_t5": X[:0],
            "paired": pd.DataFrame(columns=["row", "statement_id", "paired_text"])}


def _type_of(text):
    return next((n for n in TYPES if text.startswith(n)), None)


def _between(s, a, b="\n\nOUTPUT FORMAT"):
    return s.split(a, 1)[1].split(b, 1)[0]


def mock_claude(req):
    user = req["messages"][-1]["content"]
    if "previous reply" in user:                      # repair request: answer with the valid empty shape
        return json.dumps({"decisions": [], "registry_consistent": True})
    if user.startswith("What distinct types of work"):
        groups = {}
        for line in user.splitlines():
            m = re.match(r"^(s\d+): (.*)$", line)
            if m:
                groups.setdefault(_type_of(m.group(2)), []).append(m.group(1))
        return json.dumps({"work_types": [{"candidate_topic_name": t, "definition": f"Work on {t.lower()}.",
                                           "core_action": t.split()[-1], "object_or_outcome": t.split()[0],
                                           "inclusion_rules": [t], "exclusion_rules": [], "supporting_statement_ids": ids[:8],
                                           "representative_statement_ids": ids[:3], "confidence": 0.9}
                                          for t, ids in groups.items() if t],
                           "unlisted_work_types": [], "statements_not_work": []})
    if user.startswith("Each item is a pair"):
        pairs = json.loads(_between(user, "CANDIDATE PAIRS:\n"))
        return json.dumps({"decisions": [{"pair_id": p["pair_id"],
                                          "decision": "MERGE" if p["topic_a"]["name"] == p["topic_b"]["name"] else "KEEP_SEPARATE",
                                          "keep_topic_id": p["topic_a"]["topic_id"], "reason": "mock"} for p in pairs]})
    if user.startswith("Embedding retrieval could not assign"):
        cat = json.loads(_between(user, "TOPIC CATALOG:\n", "\n\nTASKS"))
        name_to_id = {c["name"]: c["topic_id"] for c in cat}
        tasks = json.loads(_between(user, "(candidates = allowed topic IDs):\n"))
        out = []
        for t in tasks:
            tid = name_to_id.get(_type_of(t["task"]))
            ok = tid in t["candidates"]
            out.append({"statement_id": t["statement_id"], "decision": "ASSIGN" if ok else "UNCERTAIN",
                        "primary_topic_id": tid if ok else None})
        return json.dumps({"decisions": out})
    if user.startswith("Each group collects"):
        groups = json.loads(_between(user, "NOVELTY GROUPS:\n"))
        return json.dumps({"decisions": [{"group_id": g["group_id"], "decision": "REJECT"} for g in groups]})
    if user.startswith("Each topic below was flagged"):
        flagged = json.loads(_between(user, "FLAGGED TOPICS:\n"))
        return json.dumps({"decisions": [{"topic_id": f["topic"]["topic_id"], "action": "KEEP"} for f in flagged]})
    return json.dumps({"decisions": [], "registry_consistent": True})
