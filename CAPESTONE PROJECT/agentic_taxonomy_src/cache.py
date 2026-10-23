"""Response cache: identical request -> cache hit -> no API call, no budget use. Only validated responses are stored.

key = sha256(model | role | prompt_version | input_hash | topic_registry_version)
"""
from __future__ import annotations

from pathlib import Path

from .provenance import read_json, sha256_text, write_json


def cache_key(model, role, prompt_version, input_hash, registry_version) -> str:
    return sha256_text("|".join(map(str, (model, role, prompt_version, input_hash, registry_version))))


class ResponseCache:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.hits = self.misses = 0

    def get(self, role, key):
        v = read_json(self.root / role / f"{key}.json")
        if v is None:
            self.misses += 1
        else:
            self.hits += 1
        return v

    def put(self, role, key, entry):
        write_json(self.root / role / f"{key}.json", entry)
