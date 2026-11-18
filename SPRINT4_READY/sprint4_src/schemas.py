"""Pydantic response schemas. Passed to Gemini as native structured-output schemas AND re-validated locally."""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel


class LeafLabel(BaseModel):
    topic_id: str
    topic_name: str
    description: str
    core_action: str
    object_or_outcome: str
    include_if: str
    exclude_if: str
    coherent: bool
    mixed_topic: bool
    confidence: float
    reason: str


class LeafBatch(BaseModel):
    topics: List[LeafLabel]


class ParentDecision(BaseModel):
    parent_id: str
    parent_name: str
    parent_description: str
    confidence: float
    misplaced_child_ids: List[str]
    reason: str


class HierarchyResult(BaseModel):
    families: List[ParentDecision]
    domains: List[ParentDecision]


class OnetEval(BaseModel):
    topic_id: str
    alignment: Literal["STRONG_MATCH", "PARTIAL_MATCH", "NO_MATCH"]
    best_onet_task_ids: List[int]
    dominant_onet_concept: str
    semantic_alignment_confidence: float
    reason: str


class OnetBatch(BaseModel):
    evaluations: List[OnetEval]


class CriticDecision(BaseModel):
    target_type: Literal["leaf", "parent"]
    target_id: str
    action: Literal["ACCEPT", "RENAME", "REFINE_DESCRIPTION", "MOVE_PARENT", "MERGE_PARENT", "SPLIT_PARENT", "FLAG_UNRESOLVED"]
    new_name: Optional[str] = None
    new_description: Optional[str] = None
    move_to_parent_id: Optional[str] = None
    merge_with_parent_id: Optional[str] = None
    confidence: float
    reason: str


class CriticBatch(BaseModel):
    decisions: List[CriticDecision]
