"""Production task_statements table. SECTION A - TASK EXTRACTION.

    python -m src.prototype.task_statements      # rebuild task_pipeline.task_statements from prototype tables

Built only from model outputs that were gated against the real posting text (no O*NET input).
Sources (all in schema `prototype`, written by src.prototype.run_extraction):
  baseline_gemini_groq   prototype_statements (kind='task')  - deduplicated across Gemini/Groq, with agreement
                         prototype_model_outputs (kind='task', evidence/support gate REJECTED) - kept as invalid rows
  claude_third_provider  prototype_model_outputs_anthropic (kind='task') - optional experiment arm, if it exists

review_status (first rule that applies):
  rejected_invalid_evidence       evidence could not be traced to job_postings.posting_text
  rejected_unsupported_statement  statement words not supported by its evidence span
  review_required                 fuzzy-only evidence | models disagree on task-vs-skill | boundary disagreement |
                                  AI adjudicator rejected | statement reads like a qualification
  accepted_cross_model_agreement  Gemini and Groq independently extracted it (lexical/semantic match)
  single_model_unconfirmed        dual-model phase, only one model extracted it
  single_model_not_compared       Gemini-primary phase, no second model run (quality rules did not trigger)
  third_provider_experiment       Claude arm (not part of the baseline)
is_valid_task = evidence_valid AND not rejected. Raw source text is never modified; offsets index posting_text.
"""
from __future__ import annotations

import re

import duckdb
import pandas as pd

from src.prototype.freeze import SCHEMA
from src.utils.common import DB_PATH, get_logger

log = get_logger("prototype.task_statements")

QUALIFICATION_LIKE = re.compile(
    r"^(experience|experienced|knowledge|degree|bachelor|master|phd|proficien|familiar|certif|licen[cs]e|"
    r"fluen|ability|able to|strong|excellent|demonstrated|proven|understanding|background|minimum|\d+\+? years)"
    r"|years of experience", re.I)

COLUMNS = ["task_statement_id", "posting_id", "atomic_task", "source_section", "evidence_span", "evidence_start",
           "evidence_end", "evidence_offset_basis", "evidence_match_type", "evidence_valid", "support_overlap",
           "provider", "model", "prompt_version", "schema_version", "extraction_timestamp", "confirming_provider",
           "agreement_status", "disagreement_type", "adjudication_decision", "consensus_score", "confidence_tier",
           "looks_like_qualification", "review_status", "is_valid_task", "experiment_arm", "phase",
           "source_statement_id", "built_at"]


def _exists(con, table: str) -> bool:
    return bool(con.execute("SELECT count(*) FROM duckdb_tables() WHERE schema_name='prototype' AND table_name=?",
                            [table]).fetchone()[0])


def _status(r) -> str:
    if r["evidence_match_type"] == "none" or r.get("gate_status") == "rejected_evidence_not_found":
        return "rejected_invalid_evidence"
    if r.get("gate_status") == "rejected_unsupported_statement":
        return "rejected_unsupported_statement"
    if r["experiment_arm"] == "claude_third_provider":
        return "review_required" if (r["evidence_match_type"] == "fuzzy" or r["looks_like_qualification"]) \
            else "third_provider_experiment"
    if (r["evidence_match_type"] == "fuzzy" or r["disagreement_type"] in ("kind_conflict", "boundary")
            or r["adjudication_decision"] == "reject" or r["looks_like_qualification"]):
        return "review_required"
    if r["agreement_status"] in ("lexical_agree", "semantic_agree"):
        return "accepted_cross_model_agreement"
    if str(r["agreement_status"]).endswith("_only"):
        return "single_model_unconfirmed"
    return "single_model_not_compared"


def build(con) -> pd.DataFrame:
    if not _exists(con, "prototype_statements"):
        raise RuntimeError("prototype.prototype_statements missing - run python -m src.prototype.run_extraction first")
    ts_col = "extracted_at" if "extracted_at" in con.execute(
        "SELECT * FROM prototype.prototype_model_outputs LIMIT 0").df().columns else "NULL"
    st = con.execute(f"""
        WITH mo AS (SELECT posting_id, provider, statement_text, min({ts_col}) AS extracted_at
                    FROM prototype.prototype_model_outputs WHERE kind='task' GROUP BY ALL)
        SELECT s.statement_id AS source_statement_id, s.posting_id, s.statement_text AS atomic_task,
               s.evidence_section AS source_section, s.evidence_text AS evidence_span,
               s.evidence_start, s.evidence_end, s.evidence_offset_basis, s.evidence_match_type, s.evidence_valid,
               s.support_overlap, s.extractor_provider AS provider, s.extractor_model AS model, s.prompt_version,
               s.schema_version, mo.extracted_at AS extraction_timestamp,
               CASE WHEN s.agreement_status IN ('lexical_agree','semantic_agree') THEN s.validator_provider END
                   AS confirming_provider,
               s.agreement_status, s.disagreement_type, CAST(s.adjudication_decision AS VARCHAR) AS adjudication_decision,
               s.consensus_score, s.confidence_tier, s.phase, 'accepted' AS gate_status
        FROM prototype.prototype_statements s
        LEFT JOIN mo ON mo.posting_id = s.posting_id AND mo.provider = s.extractor_provider
                    AND mo.statement_text = s.statement_text
        WHERE s.kind = 'task'""").df()
    st["experiment_arm"] = "baseline_gemini_groq"
    rej = con.execute(f"""
        SELECT o.output_id AS source_statement_id, o.posting_id, o.statement_text AS atomic_task,
               o.evidence_section AS source_section, o.evidence_text AS evidence_span, o.evidence_start, o.evidence_end,
               CASE WHEN o.evidence_start IS NULL THEN 'offset unavailable' ELSE 'job_postings.posting_text' END
                   AS evidence_offset_basis,
               o.evidence_match_type, false AS evidence_valid, o.support_overlap, o.provider, o.model, o.prompt_version,
               NULL AS schema_version, {ts_col} AS extraction_timestamp, NULL AS confirming_provider,
               NULL AS agreement_status, NULL AS disagreement_type, NULL AS adjudication_decision,
               NULL AS consensus_score, NULL AS confidence_tier, e.phase, o.gate_status
        FROM prototype.prototype_model_outputs o
        LEFT JOIN prototype.prototype_extraction_sample e USING (posting_id)
        WHERE o.kind = 'task' AND o.gate_status IN ('rejected_evidence_not_found','rejected_unsupported_statement')""").df()
    rej["experiment_arm"] = "baseline_gemini_groq"
    parts = [st, rej]
    if _exists(con, "prototype_model_outputs_anthropic"):
        an = con.execute("""
            SELECT o.output_id AS source_statement_id, o.posting_id, o.statement_text AS atomic_task,
                   o.evidence_section AS source_section, o.evidence_text AS evidence_span, o.evidence_start,
                   o.evidence_end,
                   CASE WHEN o.evidence_start IS NULL THEN 'offset unavailable' ELSE 'job_postings.posting_text' END
                       AS evidence_offset_basis,
                   o.evidence_match_type, o.evidence_match_type IN ('exact','normalized') AS evidence_valid,
                   o.support_overlap, o.provider, o.model, o.prompt_version, NULL AS schema_version,
                   o.extracted_at AS extraction_timestamp, NULL AS confirming_provider, NULL AS agreement_status,
                   NULL AS disagreement_type, NULL AS adjudication_decision, NULL AS consensus_score,
                   NULL AS confidence_tier, e.phase, o.gate_status
            FROM prototype.prototype_model_outputs_anthropic o
            LEFT JOIN prototype.prototype_extraction_sample e USING (posting_id)
            WHERE o.kind = 'task' AND o.gate_status <> 'rejected_duplicate_in_output'""").df()
        an["experiment_arm"] = "claude_third_provider"
        parts.append(an)
    df = pd.concat([p for p in parts if len(p)], ignore_index=True)
    df["looks_like_qualification"] = df.atomic_task.fillna("").str.match(QUALIFICATION_LIKE)
    df["review_status"] = [_status(r) for r in df.to_dict("records")]
    df["is_valid_task"] = df.evidence_valid.fillna(False).astype(bool) & ~df.review_status.str.startswith("rejected")
    df["task_statement_id"] = "ts_" + df.experiment_arm.str[:1] + "_" + df.source_statement_id.astype(str)
    df["built_at"] = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    df = df.drop_duplicates("task_statement_id")[COLUMNS]
    con.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
    con.register("ts", df)
    con.execute(f"""CREATE OR REPLACE TABLE {SCHEMA}.task_statements AS
        SELECT task_statement_id::VARCHAR task_statement_id, posting_id::VARCHAR posting_id,
               atomic_task::VARCHAR atomic_task, source_section::VARCHAR source_section,
               evidence_span::VARCHAR evidence_span, evidence_start::INTEGER evidence_start,
               evidence_end::INTEGER evidence_end, evidence_offset_basis::VARCHAR evidence_offset_basis,
               evidence_match_type::VARCHAR evidence_match_type, evidence_valid::BOOLEAN evidence_valid,
               support_overlap::DOUBLE support_overlap, provider::VARCHAR provider, model::VARCHAR model,
               prompt_version::VARCHAR prompt_version, schema_version::VARCHAR schema_version,
               extraction_timestamp::VARCHAR extraction_timestamp, confirming_provider::VARCHAR confirming_provider,
               agreement_status::VARCHAR agreement_status, disagreement_type::VARCHAR disagreement_type,
               adjudication_decision::VARCHAR adjudication_decision, consensus_score::DOUBLE consensus_score,
               confidence_tier::VARCHAR confidence_tier, looks_like_qualification::BOOLEAN looks_like_qualification,
               review_status::VARCHAR review_status, is_valid_task::BOOLEAN is_valid_task,
               experiment_arm::VARCHAR experiment_arm, phase::VARCHAR phase,
               source_statement_id::VARCHAR source_statement_id, built_at::VARCHAR built_at
        FROM ts ORDER BY posting_id, experiment_arm, evidence_start NULLS LAST""")
    con.unregister("ts")
    log.info("task_statements: %d rows (%s)", len(df), df.groupby(["experiment_arm", "review_status"]).size().to_dict())
    return df


if __name__ == "__main__":
    c = duckdb.connect(str(DB_PATH))
    try:
        d = build(c)
    finally:
        c.close()
    print(d.groupby(["experiment_arm", "review_status"]).size().to_string())
