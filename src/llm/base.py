"""Provider-agnostic LLM base class. PROTOTYPE.

Every call goes through `generate_json`, which:
  1. builds a cache key and returns a cached, schema-valid response if present (0 API calls);
  2. otherwise calls the provider with a timeout, retrying transient failures with
     exponential backoff + jitter (and walking a configurable fallback-model chain);
  3. parses the raw response with a provider-specific pure function (unit-testable);
  4. validates it against the pydantic schema - invalid output is NEVER cached;
  5. logs one observability row (tokens NULL if not reported).
"""
from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, ValidationError

from src.llm.cache import LLMCache, cache_key, input_hash
from src.llm.schemas import SCHEMA_VERSION, json_schema
from src.llm.usage import log_usage
from src.utils.common import ROOT

CONFIG_PATH = ROOT / "config" / "llm_config.json"


def load_config() -> dict:
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if os.environ.get("GEMINI_MODEL"):
        cfg["gemini"]["model"] = os.environ["GEMINI_MODEL"]
    if os.environ.get("GEMINI_FALLBACK_MODELS"):
        cfg["gemini"]["fallback_models"] = [m.strip() for m in os.environ["GEMINI_FALLBACK_MODELS"].split(",") if m.strip()]
    if os.environ.get("GROQ_MODEL"):
        cfg["groq"]["model"] = os.environ["GROQ_MODEL"]
    if os.environ.get("EMBEDDING_MODEL"):
        cfg["embedding"]["model"] = os.environ["EMBEDDING_MODEL"]
    return cfg


class TransientError(Exception):
    """Retryable: timeouts, 429, 5xx, connection errors."""

    def __init__(self, msg: str, retry_after: float | None = None, kind: str = "transient"):
        super().__init__(msg)
        self.retry_after = retry_after
        self.kind = kind


class PermanentError(Exception):
    """Not retryable with the same model (400 invalid request, 404 model gone, auth)."""

    def __init__(self, msg: str, kind: str = "permanent"):
        super().__init__(msg)
        self.kind = kind


class SchemaError(Exception):
    pass


@dataclass
class RawResponse:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    model: str | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class LLMResult:
    data: dict
    provider: str
    model: str
    cache_hit: bool
    latency_ms: float | None
    retry_count: int
    input_tokens: int | None
    output_tokens: int | None
    request_id: str


def extract_json_text(text: str) -> str:
    """Strip markdown fences / leading prose a model may add despite JSON mode."""
    t = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if m:
        t = m.group(1).strip()
    if not t.startswith("{"):
        i, j = t.find("{"), t.rfind("}")
        if i >= 0 and j > i:
            t = t[i:j + 1]
    return t


def _scrub(msg: str) -> str:
    """Remove anything that looks like a credential from an error message before logging."""
    return re.sub(r"(AIza[0-9A-Za-z_\-]{20,}|gsk_[0-9A-Za-z]{20,}|key=[^&\s]+)", "[REDACTED]", str(msg))[:300]


class LLMProvider:
    name = "base"

    def __init__(self, cfg: dict, cache: LLMCache | None = None, run_id: str | None = None):
        self.cfg = cfg
        self.cache = cache or LLMCache()
        self.run_id = run_id
        self.models = [cfg["model"]] + list(cfg.get("fallback_models", []))

    # -- provider specific -------------------------------------------------------
    def _call(self, model: str, system: str, prompt: str, schema: dict, schema_name: str) -> RawResponse:
        raise NotImplementedError

    # -- common ------------------------------------------------------------------
    def generate_json(self, *, system: str, prompt: str, schema_model: type[BaseModel], operation: str,
                      prompt_version: str, posting_id: str | None = None) -> LLMResult:
        schema = json_schema(schema_model)
        ih = input_hash({"system": system, "prompt": prompt, "schema": schema})
        primary = self.models[0]
        key = cache_key(self.name, primary, operation, prompt_version, SCHEMA_VERSION, ih)
        hit = self.cache.get(self.name, key)
        if hit is not None:
            row = log_usage(provider=self.name, model=hit.get("model_used", primary), operation=operation,
                            prompt_version=prompt_version, schema_version=SCHEMA_VERSION, posting_id=posting_id,
                            input_tokens=None, output_tokens=None, latency_ms=0.0, cache_hit=True,
                            retry_count=0, status="ok", error_type=None, run_id=self.run_id)
            return LLMResult(hit["data"], self.name, hit.get("model_used", primary), True, 0.0, 0,
                             hit.get("input_tokens"), hit.get("output_tokens"), row["request_id"])

        retries, last_err = 0, None
        max_retries = int(self.cfg.get("max_retries", 4))
        for model in self.models:
            attempt = 0
            while attempt <= max_retries:
                t0 = time.perf_counter()
                try:
                    raw = self._call(model, system, prompt, schema, schema_model.__name__)
                    latency = raw.extra.get("api_latency_ms", (time.perf_counter() - t0) * 1000)
                    try:
                        data = schema_model.model_validate_json(extract_json_text(raw.text)).model_dump()
                    except (ValidationError, ValueError) as e:
                        raise SchemaError(str(e)[:300])
                    model_used = raw.model or model
                    row = log_usage(provider=self.name, model=model_used, operation=operation,
                                    prompt_version=prompt_version, schema_version=SCHEMA_VERSION,
                                    posting_id=posting_id, input_tokens=raw.input_tokens,
                                    output_tokens=raw.output_tokens, reasoning_tokens=raw.reasoning_tokens,
                                    latency_ms=round(latency, 1), cache_hit=False, retry_count=retries,
                                    status="ok", error_type=None, run_id=self.run_id)
                    self.cache.put(self.name, key, {
                        "provider": self.name, "model_requested": primary, "model_used": model_used,
                        "operation": operation, "prompt_version": prompt_version, "schema_version": SCHEMA_VERSION,
                        "input_hash": ih, "data": data, "input_tokens": raw.input_tokens,
                        "output_tokens": raw.output_tokens, "created_at": row["timestamp"]})
                    return LLMResult(data, self.name, model_used, False, latency, retries,
                                     raw.input_tokens, raw.output_tokens, row["request_id"])
                except SchemaError as e:
                    last_err = e
                    self._log_fail(model, operation, prompt_version, posting_id, retries, "schema_invalid", t0)
                    attempt += 1
                    retries += 1
                    if attempt > 1:  # one re-ask, then next model
                        break
                except TransientError as e:
                    last_err = e
                    self._log_fail(model, operation, prompt_version, posting_id, retries, e.kind, t0)
                    attempt += 1
                    retries += 1
                    wait = e.retry_after if e.retry_after else min(60.0, 2 ** attempt) + random.uniform(0, 1)
                    if e.kind == "unavailable" and attempt >= 3 and model != self.models[-1]:
                        break  # overloaded model: move down the fallback chain sooner
                    time.sleep(wait)
                except PermanentError as e:
                    last_err = e
                    self._log_fail(model, operation, prompt_version, posting_id, retries, e.kind, t0)
                    break
        raise RuntimeError(f"{self.name} {operation} failed after {retries} retries: {_scrub(last_err)}")

    def _log_fail(self, model, operation, prompt_version, posting_id, retries, kind, t0):
        log_usage(provider=self.name, model=model, operation=operation, prompt_version=prompt_version,
                  schema_version=SCHEMA_VERSION, posting_id=posting_id, input_tokens=None, output_tokens=None,
                  latency_ms=round((time.perf_counter() - t0) * 1000, 1), cache_hit=False, retry_count=retries,
                  status="error", error_type=kind, run_id=self.run_id)
