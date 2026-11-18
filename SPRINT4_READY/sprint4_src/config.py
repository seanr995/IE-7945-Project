"""Paths, settings, small IO helpers. No absolute/Windows paths: everything is relative to the project folder(s)."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent          # folder that holds sprint4_src (on Colab: .../SPRINT4_READY or the project root)
PLAN = {"leaf_labeler": 4, "hierarchy_builder": 1, "onet_evaluator": 4, "critic": 1}   # normal call plan = 10
PLANNED_NORMAL_CALLS = sum(PLAN.values())
LEAF_BATCH, ONET_BATCH = 20, 20
MIN_SPLIT_BATCH = 10
N_LEAVES = 80
HIST_PRICE_IN, HIST_PRICE_OUT = 2.0, 10.0        # historical Sprint 3 Claude estimate only (USD / MTok); never used for Sprint 4
FROZEN_TOPICS = "final_taxonomy_results/final_deliverable/final_task_topics_leiden80.csv"
FROZEN_MEMBERS = "final_taxonomy_results/final_deliverable/final_task_topic_membership_leiden80.csv"
FROZEN_REPORT = "final_taxonomy_results/final_deliverable/final_leiden80_stability_report.md"
T5_EMB = "sprint3_results/embeddings/sentence_t5_embeddings.npy"
T5_MAP = "sprint3_results/embeddings/sentence_t5_embedding_mapping.csv"
T5_META = "sprint3_results/embeddings/sentence_t5_meta.json"
ONET_DIR_CANDIDATES = ["data/reference/onet/extracted/31.0/db_31_0_csv", "onet/db_31_0_csv", "db_31_0_csv"]
PRIOR_PROVENANCE_DIRS = ["final_taxonomy_results/agentic/provenance", "agentic_taxonomy_results/provenance"]


class StopRun(RuntimeError):
    pass


class GeminiAuthError(StopRun):
    pass


class GeminiQuotaError(StopRun):
    pass


class CallBudgetError(StopRun):
    pass


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def read_json(p, default=None):
    p = Path(p)
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, p)


def append_jsonl(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")


def log(msg: str):
    print(msg, flush=True)


def _parse_env(p: Path) -> dict:
    out = {}
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and v != "":
                out[k] = v
    return out


def env_search_paths(base: Path) -> list:
    """Primary: <runtime code folder>/.env (SPRINT4_READY/.env). Fallback: <project folder>/.env."""
    primary = CODE_DIR / ".env"
    if (Path(base) / "SPRINT4_READY" / "sprint4_src").exists():
        primary = Path(base) / "SPRINT4_READY" / ".env"
    fallback = Path(base) / ".env"
    return [primary] if primary == fallback else [primary, fallback]


def load_env(base: Path) -> list:
    """Load .env files. Fallback is applied first, then the primary overrides it; both override stale process variables so
    that editing .env takes effect. Returns key NAMES only (values are never printed)."""
    paths = env_search_paths(base)
    merged = {}
    for p in reversed(paths):                 # fallback first, primary wins
        merged.update(_parse_env(p))
    for k, v in merged.items():
        os.environ[k] = v
    return sorted(merged)


@dataclass
class Settings:
    base: Path
    mock: bool = False
    provider: str = "gemini"
    model: str = "gemini-3.1-flash-lite"
    max_real_calls: int = 12
    max_retries: int = 2
    min_seconds: float = 8.0
    max_repair_calls: int = 2
    max_output_tokens: int = 8192
    sleep = staticmethod(lambda s: None)      # replaced at runtime; tests inject a fake

    @property
    def results(self) -> Path:
        return self.base / "sprint4_results"

    @property
    def cache(self) -> Path:
        return self.results / "cache" / ("gemini_mock" if self.mock else "gemini")

    @property
    def cache_root(self) -> Path:
        return self.results / "cache"

    @property
    def prov(self) -> Path:
        return self.results / "provenance"

    @property
    def roots(self) -> list:
        r = [Path(self.base)]
        if CODE_DIR != Path(self.base):
            r.append(CODE_DIR)
        sub = Path(self.base) / "SPRINT4_READY"
        if sub.exists() and sub not in r:
            r.append(sub)
        return r

    def find(self, rel: str) -> Path:
        """First existing location of a project-relative input among the project folder and the runtime-code folder(s)."""
        for r in self.roots:
            if (r / rel).exists():
                return r / rel
        return Path(self.base) / rel

    @classmethod
    def from_env(cls, base, mock=None):
        e = os.environ
        mock = (e.get("SPRINT4_MOCK", "") == "1") if mock is None else mock
        provider = (e.get("LLM_PROVIDER", "gemini") or "gemini").lower()
        if provider != "gemini":
            raise StopRun(f"Sprint 4 supports LLM_PROVIDER=gemini only (got '{provider}')")
        return cls(base=Path(base), mock=mock, provider=provider,
                   model=e.get("GEMINI_MODEL", "gemini-3.1-flash-lite") or "gemini-3.1-flash-lite",
                   max_real_calls=min(12, int(e.get("GEMINI_MAX_REAL_CALLS", "12"))),
                   max_retries=min(2, int(e.get("GEMINI_MAX_RETRIES", "2"))),
                   min_seconds=float(e.get("GEMINI_MIN_SECONDS_BETWEEN_CALLS", "8")),
                   max_output_tokens=int(e.get("GEMINI_MAX_OUTPUT_TOKENS", "8192")))
