#!/usr/bin/env python
"""Sprint 3 FINAL - VIEW B: matched-granularity benchmark + agentic vs algorithmic vs consensus. ZERO Claude calls.

    !python "/content/drive/MyDrive/CAPESTONE PROJECT/sprint3_final_matched_benchmark.py" \
      --project-dir "/content/drive/MyDrive/CAPESTONE PROJECT" \
      --agentic-results "/content/drive/MyDrive/CAPESTONE PROJECT/agentic_taxonomy_results" \
      --output "/content/drive/MyDrive/CAPESTONE PROJECT/final_taxonomy_results"

After labelling final_taxonomy_results/evaluation/human_review_sample.csv, re-run with
    --human-review ".../final_taxonomy_results/evaluation/human_review_sample.csv"
(checkpoints make the re-run fast; still zero Claude calls). The human file is read, never overwritten, when given.
"""
from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-dir", required=True)
    ap.add_argument("--agentic-results", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--human-review", default=None)
    ap.add_argument("--no-e5", action="store_true")
    ap.add_argument("--skip-install", action="store_true")
    a = ap.parse_args()
    deps = [("numpy", "numpy"), ("pandas", "pandas"), ("scipy", "scipy"), ("sklearn", "scikit-learn"),
            ("matplotlib", "matplotlib"), ("pydantic", "pydantic"), ("hdbscan", "hdbscan"), ("umap", "umap-learn"),
            ("igraph", "python-igraph"), ("leidenalg", "leidenalg"), ("bertopic", "bertopic")]
    missing = [p for m, p in deps if importlib.util.find_spec(m) is None]
    if missing and not a.skip_install:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing])
        importlib.invalidate_caches()
    from agentic_taxonomy_src.benchmark import Benchmark
    from agentic_taxonomy_src.config import StopRun
    from agentic_taxonomy_src.provenance import log, safe_zip
    out = Path(a.output)
    human = a.human_review
    if human and Path(human).resolve().parent == (out / "evaluation").resolve():
        keep = out / "evaluation" / "human_review_sample_LABELLED_INPUT.csv"
        shutil.copy2(human, keep)                     # the benchmark rewrites human_review_sample.csv; keep the labels
        human = str(keep)
    try:
        m = Benchmark(a.project_dir, a.agentic_results, out, human_review=human, use_e5=not a.no_e5).run()
        z = safe_zip(out, Path(a.project_dir) / "Sprint3_Final_Taxonomy_Results.zip", arc_prefix="final_taxonomy_results")
        log(f"ZIP: {z}")
        log(f"DONE: {m['decision']['outcome']} | freeze_ready={m['decision']['freeze_ready']} | Claude calls: 0")
    except StopRun as e:
        print(f"\nSTOP: {e}\n", file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
