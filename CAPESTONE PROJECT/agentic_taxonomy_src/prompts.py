"""Role prompts. Each builder returns (system, instruction, data, contract); the user message is
instruction + data + contract, so every prompt ENDS with its explicit JSON contract.

No prompt states, suggests or rewards a number of topics; none mentions K-Means, k=100 or O*NET.
"""
from __future__ import annotations

import json

from .schemas import json_contract

SYSTEM = (
    "You are a careful taxonomy analyst building a taxonomy of WORK from job-posting task statements. "
    "A valid topic is an observable, coherent TYPE OF WORK: a recognisable action applied to an object or outcome "
    "(e.g. 'Data Pipeline Development', 'Data Quality Validation', 'Customer Behavior Analysis', 'Financial Reporting', "
    "'Model Evaluation'). Invalid topics: skills, tools or knowledge areas ('Python', 'Communication'), qualifications "
    "('Bachelor's Degree') and catch-alls ('General', 'Other', 'Miscellaneous'). Action semantics matter: the same "
    "object does not make the same work ('Develop pipelines' and 'Monitor pipelines' are different work). Let the "
    "evidence decide how many distinct types exist; never pad or compress the taxonomy to reach any size. Refer to "
    "statements and topics only by the IDs given. Keep text fields brief. You always answer with JSON only.")


def statements_block(rows) -> str:
    return "\n".join(f"{r['code']}: {r['text']}" for r in rows)


def topic_card(t, examples=None, compact=False) -> dict:
    card = {"topic_id": t["topic_id"], "name": t["topic_name"], "definition": t["definition"]}
    if not compact:
        card.update({"core_action": t.get("core_action", ""), "object_or_outcome": t.get("object_or_outcome", ""),
                     "include": t.get("inclusion_rules", [])[:3], "exclude": t.get("exclusion_rules", [])[:3]})
    if examples:
        card["examples"] = examples
    return card


def discovery(rows):
    instr = ("What distinct types of work are represented by these task statements? Identify every distinct type of "
             "work that the statements support, and only those. For each type give a name, a one-sentence definition, "
             "the core action, the object or outcome, short inclusion and exclusion rules, the IDs of supporting "
             "statements and the most representative IDs. Do not invent types the statements do not evidence. If space "
             "runs short, list remaining distinct types under unlisted_work_types (name + supporting IDs) instead of "
             "dropping them. List IDs of statements that describe no work at all (skills, qualifications, conditions, "
             "company language) under statements_not_work.")
    return SYSTEM, instr, "TASK STATEMENTS:\n" + statements_block(rows), json_contract("discovery")


def consolidation(pairs):
    instr = ("Each item is a pair of candidate topics whose embeddings or evidence overlap. Decide per pair: "
             "KEEP_SEPARATE (different work, even if the object is shared), MERGE (the same type of work; give "
             "keep_topic_id), REFINE_BOUNDARY (both valid but overlapping; give sharper definitions/rules in refined), "
             "or NEEDS_MORE_EVIDENCE. Judge by the action performed, not by shared nouns.")
    return SYSTEM, instr, "CANDIDATE PAIRS:\n" + json.dumps(pairs, ensure_ascii=False), json_contract("consolidation")


def boundary_review(cards):
    instr = ("Review this topic registry as one taxonomy of work. Report ONLY topics that need action: MERGE "
             "(duplicates of the same work; give keep_topic_id), REJECT_NON_TASK (a skill, tool, qualification or "
             "catch-all), REFINE_BOUNDARY (give a sharper definition) or FLAG_SPLIT (one topic mixing clearly different "
             "work). Unmentioned topics are kept. Set registry_consistent to true if nothing needs action.")
    return SYSTEM, instr, "TOPIC REGISTRY:\n" + json.dumps(cards, ensure_ascii=False), json_contract("boundary_review")


def ambiguous(tasks, catalog):
    instr = ("Embedding retrieval could not assign these tasks confidently. For each task choose among ITS OWN "
             "candidate topic IDs only: ASSIGN (primary_topic_id), SECONDARY_ASSIGNMENT (primary + secondary, only when "
             "the task genuinely spans two distinct types of work), NEW_TOPIC_CANDIDATE (no candidate fits: give "
             "new_topic_name and a one-sentence new_topic_definition) or UNCERTAIN. Never force a fit.")
    return (SYSTEM, instr, "TOPIC CATALOG:\n" + json.dumps(catalog, ensure_ascii=False) +
            "\n\nTASKS (candidates = allowed topic IDs):\n" + json.dumps(tasks, ensure_ascii=False), json_contract("ambiguous"))


def novelty(groups):
    instr = ("Each group collects semantically similar tasks that did not fit any existing topic. For each group decide: "
             "CREATE_TOPIC (the group shows a coherent type of work the existing topics do not cover: give topic_name, "
             "definition, core_action, object_or_outcome and short rules), ASSIGN_EXISTING (an existing listed topic "
             "does cover it: give existing_topic_id) or REJECT (not a coherent type of work). Create a topic only with "
             "clear evidence.")
    return SYSTEM, instr, "NOVELTY GROUPS:\n" + json.dumps(groups, ensure_ascii=False), json_contract("novelty")


def critic(flagged):
    instr = ("Each topic below was flagged by quantitative diagnostics (flags, members and, for split candidates, two "
             "preview sub-groups A and B). Decide per topic: KEEP (the flag is not a real problem), MERGE (merge_into "
             "another listed topic id), SPLIT (one child per preview group; only if A and B are clearly different work), "
             "REFINE_BOUNDARY (sharper definition / rules) or REASSIGN (move listed statement IDs to another listed "
             "topic id). Prefer KEEP unless the evidence clearly shows a problem.")
    return SYSTEM, instr, "FLAGGED TOPICS:\n" + json.dumps(flagged, ensure_ascii=False), json_contract("critic")


def consistency(cards):
    instr = ("Final consistency review of the converged taxonomy of work. Report ONLY remaining problems: MERGE (true "
             "duplicates; give keep_topic_id), REJECT_NON_TASK (not a type of work) or REFINE_BOUNDARY (clearer "
             "definition). Do not restructure valid topics. Set registry_consistent to true if nothing needs action.")
    return SYSTEM, instr, "FINAL REGISTRY:\n" + json.dumps(cards, ensure_ascii=False), json_contract("consistency")


BUILDERS = {"discovery": discovery, "consolidation": consolidation, "boundary_review": boundary_review,
            "ambiguous": ambiguous, "novelty": novelty, "critic": critic, "consistency": consistency}
