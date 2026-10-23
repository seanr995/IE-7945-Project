"""Response schemas (validated locally with pydantic), explicit JSON contracts, and the topic-count-hint guard.

No Anthropic tool calling is used: every prompt ends with a JSON contract, the reply text is parsed as JSON and
validated here. Unknown extra fields are ignored; enum values are case-normalised.
"""
from __future__ import annotations

import json
import re
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

PROMPT_VERSIONS = {"discovery": "disc-json-v2", "consolidation": "cons-json-v2", "boundary_review": "bound-json-v2",
                   "ambiguous": "assign-json-v2", "novelty": "novel-json-v2", "critic": "crit-json-v2",
                   "consistency": "final-json-v2"}


class _M(BaseModel):
    model_config = ConfigDict(extra="ignore")


def _upper(v):
    return v.strip().upper().replace(" ", "_") if isinstance(v, str) else v


def _str_list(v):
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    return [str(x) for x in v]


class WorkType(_M):
    candidate_topic_name: str
    definition: str
    core_action: str = ""
    object_or_outcome: str = ""
    inclusion_rules: List[str] = Field(default_factory=list)
    exclusion_rules: List[str] = Field(default_factory=list)
    supporting_statement_ids: List[str] = Field(default_factory=list)
    representative_statement_ids: List[str] = Field(default_factory=list)
    confidence: float = 0.5
    norm_lists = field_validator("inclusion_rules", "exclusion_rules", "supporting_statement_ids",
                             "representative_statement_ids", mode="before")(_str_list)


class UnlistedType(_M):
    name: str
    supporting_statement_ids: List[str] = Field(default_factory=list)
    norm_l = field_validator("supporting_statement_ids", mode="before")(_str_list)


class DiscoveryResponse(_M):
    work_types: List[WorkType]
    unlisted_work_types: List[UnlistedType] = Field(default_factory=list)
    statements_not_work: List[str] = Field(default_factory=list)
    norm_l = field_validator("statements_not_work", mode="before")(_str_list)


class RefinedDef(_M):
    topic_id: str
    definition: str
    inclusion_rules: List[str] = Field(default_factory=list)
    exclusion_rules: List[str] = Field(default_factory=list)
    norm_l = field_validator("inclusion_rules", "exclusion_rules", mode="before")(_str_list)


class ConsolidationDecision(_M):
    pair_id: str
    decision: Literal["KEEP_SEPARATE", "MERGE", "REFINE_BOUNDARY", "NEEDS_MORE_EVIDENCE"]
    keep_topic_id: Optional[str] = None
    refined: List[RefinedDef] = Field(default_factory=list)
    reason: str = ""
    norm_u = field_validator("decision", mode="before")(_upper)


class ConsolidationResponse(_M):
    decisions: List[ConsolidationDecision]


class AssignmentDecision(_M):
    statement_id: str
    decision: Literal["ASSIGN", "SECONDARY_ASSIGNMENT", "NEW_TOPIC_CANDIDATE", "UNCERTAIN"]
    primary_topic_id: Optional[str] = None
    secondary_topic_id: Optional[str] = None
    new_topic_name: Optional[str] = None
    new_topic_definition: Optional[str] = None
    confidence: float = 0.5
    norm_u = field_validator("decision", mode="before")(_upper)


class AssignmentResponse(_M):
    decisions: List[AssignmentDecision]


class NoveltyDecision(_M):
    group_id: str
    decision: Literal["CREATE_TOPIC", "ASSIGN_EXISTING", "REJECT"]
    topic_name: Optional[str] = None
    definition: Optional[str] = None
    core_action: str = ""
    object_or_outcome: str = ""
    inclusion_rules: List[str] = Field(default_factory=list)
    exclusion_rules: List[str] = Field(default_factory=list)
    existing_topic_id: Optional[str] = None
    reason: str = ""
    norm_u = field_validator("decision", mode="before")(_upper)
    norm_l = field_validator("inclusion_rules", "exclusion_rules", mode="before")(_str_list)


class NoveltyResponse(_M):
    decisions: List[NoveltyDecision]


class SplitChild(_M):
    name: str
    definition: str
    preview_group: Literal["A", "B"]
    norm_u = field_validator("preview_group", mode="before")(_upper)


class Reassign(_M):
    statement_id: str
    topic_id: str


class CriticDecision(_M):
    topic_id: str
    action: Literal["KEEP", "MERGE", "SPLIT", "REFINE_BOUNDARY", "REASSIGN"]
    merge_into: Optional[str] = None
    split_children: List[SplitChild] = Field(default_factory=list)
    definition: Optional[str] = None
    inclusion_rules: List[str] = Field(default_factory=list)
    exclusion_rules: List[str] = Field(default_factory=list)
    reassign: List[Reassign] = Field(default_factory=list)
    reason: str = ""
    norm_u = field_validator("action", mode="before")(_upper)
    norm_l = field_validator("inclusion_rules", "exclusion_rules", mode="before")(_str_list)


class CriticResponse(_M):
    decisions: List[CriticDecision]


class RegistryDecision(_M):
    topic_ids: List[str]
    action: Literal["KEEP", "MERGE", "REJECT_NON_TASK", "REFINE_BOUNDARY", "FLAG_SPLIT"]
    keep_topic_id: Optional[str] = None
    definition: Optional[str] = None
    reason: str = ""
    norm_u = field_validator("action", mode="before")(_upper)
    norm_l = field_validator("topic_ids", mode="before")(_str_list)


class ConsistencyResponse(_M):
    decisions: List[RegistryDecision] = Field(default_factory=list)
    registry_consistent: bool = True


SCHEMAS = {"discovery": DiscoveryResponse, "consolidation": ConsolidationResponse, "boundary_review": ConsistencyResponse,
           "ambiguous": AssignmentResponse, "novelty": NoveltyResponse, "critic": CriticResponse,
           "consistency": ConsistencyResponse}

# One illustrative item per list; the shape is the contract, not the length.
EXAMPLES = {
    "discovery": {"work_types": [{"candidate_topic_name": "string", "definition": "one sentence", "core_action": "verb",
                                  "object_or_outcome": "string", "inclusion_rules": ["short phrase"],
                                  "exclusion_rules": ["short phrase"], "supporting_statement_ids": ["s12"],
                                  "representative_statement_ids": ["s12"], "confidence": 0.8}],
                  "unlisted_work_types": [{"name": "string", "supporting_statement_ids": ["s40"]}],
                  "statements_not_work": ["s77"]},
    "consolidation": {"decisions": [{"pair_id": "P0_0", "decision": "KEEP_SEPARATE | MERGE | REFINE_BOUNDARY | NEEDS_MORE_EVIDENCE",
                                     "keep_topic_id": "AT_0001 or null",
                                     "refined": [{"topic_id": "AT_0001", "definition": "string", "inclusion_rules": [],
                                                  "exclusion_rules": []}], "reason": "short"}]},
    "boundary_review": {"decisions": [{"topic_ids": ["AT_0001"], "action": "MERGE | REJECT_NON_TASK | REFINE_BOUNDARY | FLAG_SPLIT",
                                       "keep_topic_id": "AT_0001 or null", "definition": "string or null", "reason": "short"}],
                        "registry_consistent": False},
    "ambiguous": {"decisions": [{"statement_id": "s12", "decision": "ASSIGN | SECONDARY_ASSIGNMENT | NEW_TOPIC_CANDIDATE | UNCERTAIN",
                                 "primary_topic_id": "AT_0001 or null", "secondary_topic_id": "AT_0002 or null",
                                 "new_topic_name": "string or null", "new_topic_definition": "string or null", "confidence": 0.7}]},
    "novelty": {"decisions": [{"group_id": "G0", "decision": "CREATE_TOPIC | ASSIGN_EXISTING | REJECT", "topic_name": "string or null",
                               "definition": "string or null", "core_action": "verb", "object_or_outcome": "string",
                               "inclusion_rules": [], "exclusion_rules": [], "existing_topic_id": "AT_0001 or null",
                               "reason": "short"}]},
    "critic": {"decisions": [{"topic_id": "AT_0001", "action": "KEEP | MERGE | SPLIT | REFINE_BOUNDARY | REASSIGN",
                              "merge_into": "AT_0002 or null",
                              "split_children": [{"name": "string", "definition": "string", "preview_group": "A"}],
                              "definition": "string or null", "inclusion_rules": [], "exclusion_rules": [],
                              "reassign": [{"statement_id": "s12", "topic_id": "AT_0003"}], "reason": "short"}]},
    "consistency": {"decisions": [{"topic_ids": ["AT_0001", "AT_0004"], "action": "MERGE | REJECT_NON_TASK | REFINE_BOUNDARY",
                                   "keep_topic_id": "AT_0001 or null", "definition": "string or null", "reason": "short"}],
                    "registry_consistent": True},
}


def json_contract(role: str) -> str:
    return ("OUTPUT FORMAT - Return ONLY valid JSON. Do not use Markdown or code fences. Do not include any explanation "
            "outside the JSON. Use exactly this structure (lists may hold any number of items, including none):\n"
            + json.dumps(EXAMPLES[role], ensure_ascii=False))


# ---------------------------------------------------------------- no predefined topic count, ever
_COUNT_HINT = re.compile(
    r"(\b\d+\s*(?:-|to|–)?\s*\d*\s*(?:topics?|categories|clusters?|concepts?|work[ -]?types?|groups?|themes?)\b)"
    r"|(target[_ ]?topics?)|(\bk\s*=\s*\d+)|(n[_ ]?clusters?)|(number of (?:topics|clusters|categories) (?:should|must|to))",
    re.I)


class TopicCountHintError(ValueError):
    pass


def assert_no_topic_count_hint(text: str) -> None:
    m = _COUNT_HINT.search(text)
    if m:
        raise TopicCountHintError(f"prompt contains a topic-count hint: {m.group(0)!r}")
