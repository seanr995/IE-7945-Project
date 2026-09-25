"""LLM observability: one row per logical request (cache hits included). PROTOTYPE.

Rows are appended to cache/llm/usage.jsonl (thread-safe) and loaded into the DuckDB
table prototype.llm_usage. Token fields are NULL when a provider does not report them.
"""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone

from src.llm.cache import CACHE_DIR

USAGE_LOG = CACHE_DIR / "usage.jsonl"
_lock = threading.Lock()

FIELDS = ["request_id", "provider", "model", "operation", "prompt_version", "schema_version", "posting_id",
          "input_tokens", "output_tokens", "reasoning_tokens", "latency_ms", "cache_hit", "retry_count",
          "status", "error_type", "timestamp", "run_id"]


def log_usage(**kw) -> dict:
    row = {k: kw.get(k) for k in FIELDS}
    row["request_id"] = row["request_id"] or uuid.uuid4().hex
    row["timestamp"] = row["timestamp"] or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    USAGE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with _lock, open(USAGE_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    return row


def read_usage() -> list[dict]:
    if not USAGE_LOG.exists():
        return []
    with open(USAGE_LOG, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
