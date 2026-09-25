"""Structured-output schemas (pydantic) shared by all providers. PROTOTYPE.

The same JSON Schema is sent to Gemini (response_json_schema) and Groq
(response_format=json_schema, strict) and every response is re-validated
locally with pydantic - a provider's claim of schema compliance is not trusted.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, ValidationError, model_validator
from pydantic.json_schema import SkipJsonSchema

SCHEMA_VERSION = "stmt-schema-v1"

SECTIONS = ["responsibilities", "requirements", "preferred_qualifications", "skills", "general_description"]


class ExtractedItem(BaseModel):
    kind: Literal["task", "skill"]
    statement_text: str = Field(min_length=2, max_length=300)
    evidence_text: str = Field(min_length=2, max_length=1500)
    evidence_section: Literal["responsibilities", "requirements", "preferred_qualifications", "skills",
                              "general_description"]


class ExtractionResult(BaseModel):
    """Item-level validation: one malformed item (e.g. an over-long statement) is dropped and
    counted in `schema_invalid_items` instead of discarding the whole response. The counter is
    excluded from the JSON schema sent to providers."""
    items: list[ExtractedItem] = Field(max_length=120)
    schema_invalid_items: SkipJsonSchema[int] = 0

    @model_validator(mode="before")
    @classmethod
    def _drop_invalid_items(cls, data):
        if isinstance(data, dict) and isinstance(data.get("items"), list):
            good, bad = [], 0
            for it in data["items"]:
                try:
                    good.append(ExtractedItem.model_validate(it))
                except ValidationError:
                    bad += 1
            data = {**data, "items": good, "schema_invalid_items": int(data.get("schema_invalid_items", 0)) + bad}
        return data


class ValidationVerdict(BaseModel):
    item_index: int
    supported: bool
    kind_correct: bool
    note: str = Field(default="", max_length=200)


class ValidationResult(BaseModel):
    verdicts: list[ValidationVerdict]


class AdjudicationDecision(BaseModel):
    item_index: int
    decision: Literal["keep", "reject"]
    kind: Literal["task", "skill"]
    reason: str = Field(default="", max_length=200)


class AdjudicationResult(BaseModel):
    decisions: list[AdjudicationDecision]


class RerankDecision(BaseModel):
    statement_index: int
    ranked_candidate_ids: list[str] = Field(max_length=5)
    reject_all: bool


class RerankResult(BaseModel):
    decisions: list[RerankDecision]


def json_schema(model: type[BaseModel]) -> dict:
    """Strict JSON schema acceptable to both providers (no defaults, all required,
    additionalProperties false, no min/max length keywords that some providers reject)."""
    s = model.model_json_schema()
    defs = s.pop("$defs", {})

    def fix(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return fix(dict(defs[node["$ref"].split("/")[-1]]))
            node = {k: fix(v) for k, v in node.items()
                    if k not in ("title", "default", "minLength", "maxLength", "maxItems", "minItems")
                    and not (k == "description" and isinstance(v, str))}  # docstrings are not part of the contract
            if node.get("type") == "object":
                node["additionalProperties"] = False
                node["required"] = list(node.get("properties", {}).keys())
            return node
        if isinstance(node, list):
            return [fix(x) for x in node]
        return node

    return fix(s)
