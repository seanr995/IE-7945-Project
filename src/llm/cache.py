"""Content-addressed disk cache for every LLM / embedding request. PROTOTYPE.

Key = sha256(provider, model, operation, prompt_version, schema_version, sha256(input)).
Entries are JSON files under cache/llm/<provider>/<key[:2]>/<key>.json (git-ignored).
Only successful, schema-valid responses are cached; no API key is ever part of an entry.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path

from src.utils.common import ROOT

CACHE_DIR = ROOT / "cache" / "llm"
_lock = threading.Lock()


def input_hash(payload) -> str:
    s = payload if isinstance(payload, str) else json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def cache_key(provider: str, model: str, operation: str, prompt_version: str, schema_version: str,
              inp_hash: str) -> str:
    raw = "|".join([provider, model, operation, prompt_version, schema_version, inp_hash])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class LLMCache:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else CACHE_DIR

    def _path(self, provider: str, key: str) -> Path:
        return self.root / provider / key[:2] / f"{key}.json"

    def get(self, provider: str, key: str) -> dict | None:
        p = self._path(provider, key)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def put(self, provider: str, key: str, entry: dict) -> None:
        p = self._path(provider, key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        with _lock:
            tmp.write_text(json.dumps(entry, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, p)
