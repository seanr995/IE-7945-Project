"""TASK / SKILL / OTHER classification gate before Sprint 3. SECTION A - TASK EXTRACTION.

    python -m src.prototype.task_classifier validate            # classify the 100 human-audited rows, score them
    python -m src.prototype.task_classifier apply               # ONLY if validation TASK precision >= 0.95
    python -m src.prototype.task_classifier validate --dry-run  # build prompts, no API call

Classifier: Claude via the existing AnthropicProvider (key only from ANTHROPIC_API_KEY in .env, never logged;
responses cached by src.llm.cache, so a rerun makes 0 calls). Input per statement: statement_text,
evidence_span, source_section. Output: task_class in {TASK, SKILL, OTHER} + a one-line reason.
The prompt is written from the label definitions only - the human audit labels are never shown to the model.

validate -> outputs/task_evaluation/task_classifier_validation.csv, task_classifier_metrics.json
apply    -> task_pipeline.sprint3_task_classification (every sprint3_task_input row + its class)
            task_pipeline.sprint3_task_input_clean     (TASK rows only)
            sprint3_task_input and task_statements are only read.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import duckdb
import pandas as pd
from pydantic import BaseModel, Field

from src.llm.base import load_config
from src.utils.common import DB_PATH, OUTPUTS, load_env

OUT = OUTPUTS / "task_evaluation"
AUDIT = OUT / "task_purity_audit_100.csv"
VALIDATION = OUT / "task_classifier_validation.csv"
METRICS = OUT / "task_classifier_metrics.json"
LABELS = ["TASK", "SKILL", "OTHER"]
PRECISION_GATE = 0.95
PROMPT_VERSION = "task-class-v1"
BATCH = 20
EVIDENCE_CHARS = 1200

SYSTEM = """You label statements that an extraction model pulled out of job postings. For each statement decide,
using its evidence span and source section, which ONE class it is:

TASK  - an observable work action actually performed as part of the job.
        e.g. "Build ETL pipelines", "Analyze customer behavior", "Prepare reports", "Monitor data quality".
SKILL - a capability, tool, knowledge area or ability the worker must have.
        e.g. "Python", "SQL", "communication skills", "GIS knowledge".
OTHER - degree/experience requirements, work conditions, generic expectations or personal traits, benefits,
        company/team/mission language, and any non-work statement.
        e.g. "Bachelor's degree required", "Work with frequent interruptions", "Learn from your peers",
        "Three years of experience".

Rules:
- Judge what the EVIDENCE says, not how the statement is phrased. The statement may have been rewritten as a verb
  phrase; do not accept an invented task.
- A requirement turned into a verb phrase is NOT a task: evidence "Strong Python skills" -> SKILL (not "Use Python");
  evidence "Ability to prioritize" -> SKILL (not "Set priorities"). It is TASK only if the evidence explicitly
  describes that activity as work the job holder will do.
- Evidence describing the team or organisation rather than the job holder's own work -> OTHER.
- Generic expectations or attitudes with no concrete work object -> OTHER.
- Return exactly one result per statement_id, with a short reason (max 20 words)."""


class ClassItem(BaseModel):
    statement_id: str
    task_class: Literal["TASK", "SKILL", "OTHER"]
    reason: str = Field(max_length=300)


class ClassBatch(BaseModel):
    results: list[ClassItem]


def build_prompt(rows: list[dict]) -> str:
    items = [{"statement_id": r["statement_id"], "statement_text": r["statement_text"],
              "source_section": r.get("source_section") or "unknown",
              "evidence_span": (r.get("evidence_span") or "")[:EVIDENCE_CHARS]} for r in rows]
    return "Classify each statement.\n\n" + json.dumps(items, ensure_ascii=False, indent=1)


def batches(rows: list[dict], n: int = BATCH) -> list[list[dict]]:
    rows = sorted(rows, key=lambda r: r["statement_id"])  # deterministic batches -> stable cache keys
    return [rows[i:i + n] for i in range(0, len(rows), n)]


def collect(batch: list[dict], data: dict) -> list[dict]:
    """Map a model response back to the batch; a missing / duplicated / foreign id is an error, not a guess."""
    got = {}
    for it in data["results"]:
        if it["statement_id"] in got:
            raise ValueError(f"duplicate statement_id in response: {it['statement_id']}")
        got[it["statement_id"]] = it
    ids = [r["statement_id"] for r in batch]
    missing, extra = set(ids) - set(got), set(got) - set(ids)
    if missing or extra:
        raise ValueError(f"response ids mismatch: missing={sorted(missing)[:5]} extra={sorted(extra)[:5]}")
    return [got[i] for i in ids]


def metrics(y_true: list[str], y_pred: list[str]) -> dict:
    n = len(y_true)
    cm = {t: {p: sum(1 for a, b in zip(y_true, y_pred) if a == t and b == p) for p in LABELS} for t in LABELS}
    tp = cm["TASK"]["TASK"]
    pred_task, true_task = sum(cm[t]["TASK"] for t in LABELS), sum(cm["TASK"].values())
    p = tp / pred_task if pred_task else None
    r = tp / true_task if true_task else None
    f1 = 2 * p * r / (p + r) if p and r else 0.0
    return {"n": n, "accuracy": sum(a == b for a, b in zip(y_true, y_pred)) / n if n else None,
            "task_precision": p, "task_recall": r, "task_f1": f1, "confusion_matrix_true_x_pred": cm}


def gate_passed(m: dict) -> bool:
    return m.get("task_precision") is not None and m["task_precision"] >= PRECISION_GATE


class Classifier:
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.cfg = load_config()["anthropic"]
        if not dry_run:
            load_env()
            if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
                raise SystemExit("ANTHROPIC_API_KEY is not set in .env - no classification was run.")
            from src.llm.anthropic_provider import AnthropicProvider
            self.p = AnthropicProvider(self.cfg, run_id="task_classifier")
        self.model = self.cfg["model"]

    def classify(self, rows: list[dict]) -> pd.DataFrame:
        bs = batches(rows)
        if self.dry_run:
            print(f"[dry-run] {len(rows)} statements -> {len(bs)} batches of <= {BATCH}; model {self.model}\n")
            print(SYSTEM, "\n\n", build_prompt(bs[0])[:3000])
            raise SystemExit(0)

        def one(b):
            res = self.p.generate_json(system=SYSTEM, prompt=build_prompt(b), schema_model=ClassBatch,
                                       operation="task_classification", prompt_version=PROMPT_VERSION)
            return [{**x, "classification_model": res.model, "cache_hit": res.cache_hit} for x in collect(b, res.data)]

        with ThreadPoolExecutor(int(self.cfg.get("max_concurrency", 1))) as ex:
            out = [x for part in ex.map(one, bs) for x in part]
        return pd.DataFrame(out).rename(columns={"reason": "classification_reason"})


def _sections(con) -> pd.DataFrame:
    return con.execute("SELECT task_statement_id AS statement_id, source_section FROM task_pipeline.task_statements").df()


def validate(dry_run: bool) -> dict:
    a = pd.read_csv(AUDIT, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    a["human_label"] = a.human_label.str.strip().str.upper()
    assert len(a) == 100 and a.human_label.isin(LABELS).all(), "audit must have 100 rows labelled TASK/SKILL/OTHER"
    con = duckdb.connect(str(DB_PATH), read_only=True)
    a = a.merge(_sections(con), on="statement_id", how="left")
    in_scope = set(con.execute("SELECT statement_id FROM task_pipeline.sprint3_task_input").df().statement_id)
    con.close()
    rows = a.drop(columns="human_label").to_dict("records")  # labels never reach the model
    pred = Classifier(dry_run).classify(rows)
    v = a.merge(pred, on="statement_id", how="left")
    v["correct"] = v.human_label == v.task_class
    v["in_sprint3_input"] = v.statement_id.isin(in_scope)
    v.to_csv(VALIDATION, index=False, encoding="utf-8-sig")
    res = {"prompt_version": PROMPT_VERSION, "model": pred.classification_model.iloc[0],
           "gate": f"task_precision >= {PRECISION_GATE}",
           "all_100": metrics(v.human_label.tolist(), v.task_class.tolist()),
           "sprint3_input_subset": metrics(v[v.in_sprint3_input].human_label.tolist(),
                                           v[v.in_sprint3_input].task_class.tolist())}
    res["gate_passed"] = gate_passed(res["all_100"])
    METRICS.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    return res


def apply(dry_run: bool) -> None:
    if not METRICS.exists():
        raise SystemExit("Run `validate` first.")
    m = json.loads(METRICS.read_text(encoding="utf-8"))
    if not gate_passed(m["all_100"]) or m.get("prompt_version") != PROMPT_VERSION:
        raise SystemExit(f"Gate not passed (TASK precision {m['all_100']['task_precision']}) - dataset NOT filtered.")
    con = duckdb.connect(str(DB_PATH), read_only=True)
    base = con.execute("SELECT * FROM task_pipeline.sprint3_task_input").df()
    con.close()
    pred = Classifier(dry_run).classify(base[["statement_id", "statement_text", "evidence_span",
                                              "source_section"]].to_dict("records"))
    cls = base.merge(pred[["statement_id", "task_class", "classification_model", "classification_reason"]],
                     on="statement_id", how="left", validate="one_to_one")
    assert cls.task_class.notna().all() and len(cls) == len(base)
    con = duckdb.connect(str(DB_PATH))
    con.register("cls", cls.assign(prompt_version=PROMPT_VERSION))
    con.execute("CREATE OR REPLACE TABLE task_pipeline.sprint3_task_classification AS SELECT * FROM cls")
    con.execute("""CREATE OR REPLACE TABLE task_pipeline.sprint3_task_input_clean AS
        SELECT statement_id, posting_id, provider, statement_text, evidence_span, source_section, evidence_valid,
               task_class, classification_model, classification_reason, prompt_version
        FROM task_pipeline.sprint3_task_classification WHERE task_class = 'TASK' ORDER BY posting_id, statement_id""")
    print(con.execute("SELECT task_class, count(*) FROM task_pipeline.sprint3_task_classification GROUP BY 1").fetchall())
    print("sprint3_task_input_clean:", con.execute("SELECT count(*) FROM task_pipeline.sprint3_task_input_clean").fetchone()[0])
    con.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["validate", "apply"])
    ap.add_argument("--dry-run", action="store_true", help="build prompts only, no API call")
    a = ap.parse_args()
    (validate if a.command == "validate" else apply)(a.dry_run)


if __name__ == "__main__":
    main()
