"""Cached text embeddings. PROTOTYPE.

provider = 'local'  : open-source ONNX model via fastembed (default; see config _selection_note)
provider = 'gemini' : Gemini embedding API (gemini-embedding-001)
Embeddings are NOT generated with chat models. Each vector is cached by
sha256(model|dims|task_type|text) in cache/embeddings/embeddings.duckdb, so a rerun
re-embeds nothing. Vectors are L2-normalised (required after dimension truncation).
"""
from __future__ import annotations

import hashlib
import os
import time

import duckdb
import numpy as np

from src.llm.base import load_config
from src.llm.usage import log_usage
from src.utils.common import ROOT

EMB_DB = ROOT / "cache" / "embeddings" / "embeddings.duckdb"


def _key(model: str, dims: int, task: str, text: str) -> str:
    return hashlib.sha256(f"{model}|{dims}|{task}|{text}".encode("utf-8")).hexdigest()


class Embedder:
    def __init__(self, cfg: dict | None = None, run_id: str | None = None):
        self.cfg = (cfg or load_config())["embedding"]
        self.model, self.dims, self.task = self.cfg["model"], int(self.cfg["dimensions"]), self.cfg["task_type"]
        self.run_id = run_id
        EMB_DB.parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(EMB_DB))
        self.con.execute("CREATE TABLE IF NOT EXISTS emb (k VARCHAR PRIMARY KEY, model VARCHAR, dims INTEGER, "
                         "task VARCHAR, v FLOAT[])")
        self._client = None
        self._local = None
        self.provider = self.cfg.get("provider", "gemini")
        self.api_calls = 0
        self.cache_hits = 0

    @property
    def client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        return self._client

    def _local_embed(self, texts: list[str]) -> list[list[float]]:
        if self._local is None:
            from fastembed import TextEmbedding
            self._local = TextEmbedding(model_name=self.model, cache_dir=str(ROOT / "cache" / "models"))
        t0 = time.perf_counter()
        vecs = [v.tolist() for v in self._local.embed(texts, batch_size=int(self.cfg.get("batch_size", 128)))]
        log_usage(provider="local", model=self.model, operation="embed", prompt_version="n/a", schema_version="n/a",
                  posting_id=None, input_tokens=None, output_tokens=None,
                  latency_ms=round((time.perf_counter() - t0) * 1000, 1), cache_hit=False, retry_count=0,
                  status="ok", error_type=None, run_id=self.run_id)
        self.api_calls += 1
        return vecs

    def _api(self, texts: list[str]) -> list[list[float]]:
        if self.provider == "local":
            return self._local_embed(texts)
        from google.genai import types
        for attempt in range(8):
            t0 = time.perf_counter()
            try:
                r = self.client.models.embed_content(
                    model=self.model, contents=texts,
                    config=types.EmbedContentConfig(task_type=self.task, output_dimensionality=self.dims))
                vecs = [e.values for e in r.embeddings]
                if len(vecs) != len(texts):
                    raise RuntimeError(f"embedding count mismatch {len(vecs)} != {len(texts)}")
                log_usage(provider="gemini", model=self.model, operation="embed", prompt_version="n/a",
                          schema_version="n/a", posting_id=None, input_tokens=None, output_tokens=None,
                          latency_ms=round((time.perf_counter() - t0) * 1000, 1), cache_hit=False,
                          retry_count=attempt, status="ok", error_type=None, run_id=self.run_id)
                self.api_calls += 1
                return vecs
            except Exception as e:
                code = getattr(e, "code", None)
                kind = "rate_limit" if code == 429 else ("unavailable" if code in (500, 503) else type(e).__name__)
                log_usage(provider="gemini", model=self.model, operation="embed", prompt_version="n/a",
                          schema_version="n/a", posting_id=None, input_tokens=None, output_tokens=None,
                          latency_ms=round((time.perf_counter() - t0) * 1000, 1), cache_hit=False,
                          retry_count=attempt, status="error", error_type=kind, run_id=self.run_id)
                if code in (400, 401, 403, 404):
                    raise
                time.sleep(min(90, 5 * 2 ** attempt))
        raise RuntimeError("embedding failed after retries")

    def embed(self, texts: list[str]) -> np.ndarray:
        keys = [_key(self.model, self.dims, self.task, t) for t in texts]
        uniq = list(dict.fromkeys(keys))
        have = {}
        for i in range(0, len(uniq), 5000):
            chunk = uniq[i:i + 5000]
            rows = self.con.execute("SELECT k, v FROM emb WHERE k IN (SELECT unnest(?))", [chunk]).fetchall()
            have.update({k: v for k, v in rows})
        missing = [(k, t) for k, t in dict(zip(keys, texts)).items() if k not in have]
        self.cache_hits += len(uniq) - len(missing)
        bs = int(self.cfg.get("batch_size", 100)) * (40 if self.provider == "local" else 1)
        for i in range(0, len(missing), bs):
            chunk = missing[i:i + bs]
            vecs = self._api([t for _, t in chunk])
            self.con.executemany("INSERT OR REPLACE INTO emb VALUES (?, ?, ?, ?, ?)",
                                 [(k, self.model, self.dims, self.task, v) for (k, _), v in zip(chunk, vecs)])
            have.update({k: v for (k, _), v in zip(chunk, vecs)})
        m = np.array([have[k] for k in keys], dtype=np.float32)
        n = np.linalg.norm(m, axis=1, keepdims=True)
        return m / np.where(n == 0, 1, n)

    def close(self):
        self.con.close()
