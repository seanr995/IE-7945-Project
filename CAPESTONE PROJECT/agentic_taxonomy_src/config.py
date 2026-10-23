"""Settings, project paths and constants. All paths derive from --project-dir (no platform-specific paths)."""
from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path

SEED = 42
HARD_CEILING = 20                      # absolute: configuration can lower it, never raise it
MIN_OUTPUT_TOKENS = 8000               # max_tokens is a ceiling (billing is per generated token); prevents JSON truncation
T5_MODEL = "sentence-transformers/sentence-t5-base"
E5_MODEL = "intfloat/e5-base-v2"
FROZEN_INPUT_SHA256 = "010795fed2864f9963f159204a857f2198a2246206009ab0dbbbb786799f4551"
SPRINT3_SCRIPT = "sprint3_colab_experiment.py"
SPRINT3_SCRIPT_SHA256 = "c2cbc1bf58126876106b19538769927e2f6cc6b687e9d014818ffb6e7c747a12"
PRIOR_USAGE_FILE = "api_budget_prior_usage.json"


class StopRun(Exception):
    """Validation / runtime failure that must stop the run with a clear message."""


@dataclass(frozen=True)
class Paths:
    project: Path

    @property
    def input_csv(self):
        return self.project / "sprint3_task_input.csv"

    @property
    def sprint3_results(self):
        return self.project / "sprint3_results"

    @property
    def embeddings(self):
        return self.sprint3_results / "embeddings"

    @property
    def advanced_results(self):
        return self.project / "sprint3_advanced_results"

    @property
    def agentic(self):
        return self.project / "agentic_taxonomy_results"

    @property
    def final(self):
        return self.project / "final_taxonomy_results"

    @property
    def env_file(self):
        return self.project / ".env"


@dataclass(frozen=True)
class Settings:
    model: str
    max_output_tokens: int
    configured_output_tokens: int
    hard_limit: int
    max_refinement_rounds: int
    api_key_present: bool


def load_env(project_dir) -> bool:
    """Load <project>/.env (python-dotenv if available, else a minimal parser). Values are never printed."""
    p = Path(project_dir) / ".env"
    if not p.exists():
        return False
    try:
        from dotenv import load_dotenv
        load_dotenv(p, override=False)
    except ImportError:
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    return True


def _int(name, default):
    v = os.environ.get(name, "").strip()
    try:
        return int(v) if v else default
    except ValueError:
        raise StopRun(f"{name} must be an integer")


def settings() -> Settings:
    configured = _int("AGENTIC_MAX_OUTPUT_TOKENS", MIN_OUTPUT_TOKENS)
    return Settings(model=os.environ.get("ANTHROPIC_MODEL", "").strip(),
                    max_output_tokens=max(configured, MIN_OUTPUT_TOKENS), configured_output_tokens=configured,
                    hard_limit=min(_int("AGENTIC_MAX_API_CALLS", HARD_CEILING), HARD_CEILING),
                    max_refinement_rounds=_int("AGENTIC_MAX_REFINEMENT_ROUNDS", 2),
                    api_key_present=bool(os.environ.get("ANTHROPIC_API_KEY", "").strip()))


def api_key() -> str:
    k = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not k:
        raise StopRun("ANTHROPIC_API_KEY is not set (expected in <project>/.env)")
    return k


def load_sprint3_module(project_dir):
    """Import the frozen Sprint 3 script so every metric uses the identical historical definition."""
    from .provenance import sha256_file
    path = Path(project_dir) / SPRINT3_SCRIPT
    if not path.exists():
        raise StopRun(f"{path} not found - it supplies the Sprint 3 metric definitions")
    spec = importlib.util.spec_from_file_location("sprint3_primary", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.lazy_imports()
    return mod, sha256_file(path)
