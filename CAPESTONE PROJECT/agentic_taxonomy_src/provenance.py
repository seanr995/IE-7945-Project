"""Logging, atomic JSON, hashing, code fingerprint and secret-safe ZIP."""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import StopRun

T0 = time.time()
SECRET = re.compile(r"(sk-ant-[A-Za-z0-9_\-]{10,}|ANTHROPIC_API_KEY\s*=\s*\S{10,})")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:6.0f}s] {msg}", flush=True)


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def stable_hash(obj) -> str:
    return sha256_text(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str))


def _default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_json(path, obj) -> None:
    """Atomic write (tmp + replace): an interrupted session never leaves a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_default), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def append_jsonl(path, row) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"utc": utc(), **row}, default=_default) + "\n")


def code_fingerprint(package_dir) -> str:
    """Hash of the runtime source, so preflight results are only valid for the exact code that was checked."""
    h = hashlib.sha256()
    for p in sorted(Path(package_dir).glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def safe_zip(src_dir, zpath, arc_prefix="") -> dict:
    """ZIP a results folder. Never includes .env*, checkpoints or caches of bytecode; aborts on secret-like text."""
    src_dir, zpath = Path(src_dir), Path(zpath)
    skipped, n = [], 0
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(src_dir.rglob("*")):
            if not p.is_file():
                continue
            if p.name.startswith(".env") or p.suffix == ".tmp" or "__pycache__" in p.parts or "checkpoints" in p.parts:
                skipped.append(str(p.relative_to(src_dir)).replace("\\", "/"))
                continue
            if p.suffix.lower() in (".json", ".jsonl", ".csv", ".md", ".txt", ".py") and p.stat().st_size < 50_000_000:
                if SECRET.search(p.read_text(encoding="utf-8", errors="ignore")):
                    raise StopRun(f"secret-like string found in {p} - ZIP aborted")
            z.write(p, arcname=(Path(arc_prefix) / p.relative_to(src_dir)).as_posix())
            n += 1
    return {"zip": str(zpath), "files": n, "skipped": skipped, "mb": round(zpath.stat().st_size / 1e6, 2)}
