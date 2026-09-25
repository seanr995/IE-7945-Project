"""Raw records -> canonical job-posting corpus (data contract v1.0).

Rules (see docs/data_contract.md):
 * NULL means "not available" - nothing is fabricated.
 * Source-provided values keep their meaning; anything inferred lives in a *_derived
   column or carries a *_basis column saying how it was obtained.
 * posting_id = first 20 hex chars of sha256("<source>|<source_job_id>").
"""
from __future__ import annotations

import gzip
import hashlib
import html as htmllib
import json
import re
import unicodedata
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.utils.common import RAW_JOBS, get_logger, rel

log = get_logger("cleaning.canonical")
warnings.filterwarnings("ignore", category=UserWarning, module="bs4")

CONTRACT_VERSION = "1.0"

CANONICAL_COLUMNS = [
    "posting_id", "source", "source_job_id", "source_record_ref", "retrieval_id",
    "job_title", "normalized_job_title", "company_name", "industry", "industry_derived",
    "job_category_source", "location_raw", "country", "state", "city", "geo_basis",
    "remote_flag", "employment_type", "seniority_source", "seniority_derived", "seniority",
    "seniority_basis", "description", "responsibilities", "requirements", "preferred_skills",
    "salary_min", "salary_max", "salary_currency", "salary_period", "date_posted",
    "date_collected", "language", "language_confidence", "source_url", "posting_text",
    "description_char_count", "description_word_count", "posting_text_word_count",
    "exact_dup_key", "normalized_dup_key", "duplicate_group_id", "is_exact_duplicate",
    "is_normalized_duplicate", "is_source_id_repeat", "contract_version",
]

# ------------------------------------------------------------------ text helpers

_ws = re.compile(r"[ \t  -​]+")
_nl = re.compile(r"\n{3,}")


def clean_text(s) -> str | None:
    """Safe HTML->text + whitespace normalisation. Returns None for empty."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    s = str(s)
    # Some sources deliver entity-encoded HTML (&lt;p&gt;...). Unescape and strip tags repeatedly until
    # no markup remains (bugfix 2026-09-25, see docs/sprint1_bugfix_log.md).
    from bs4 import BeautifulSoup
    tag = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]*(\s[^<>]*)?/?>")
    for _ in range(6):  # some records are double/triple entity-encoded
        if tag.search(s):
            s = BeautifulSoup(s, "html.parser").get_text("\n")
        u = htmllib.unescape(s)
        if u == s and not tag.search(s):
            break
        s = u
    s = unicodedata.normalize("NFC", s).replace("\r\n", "\n").replace("\r", "\n")
    s = "\n".join(_ws.sub(" ", line).strip() for line in s.split("\n"))
    s = _nl.sub("\n\n", s).strip()
    return s or None


def _s(x) -> str | None:
    """Only non-empty strings count as values (pandas may hand us NaN)."""
    return x if isinstance(x, str) and x else None


def normalize_title(t: str | None) -> str | None:
    t = _s(t)
    if not t:
        return None
    t = unicodedata.normalize("NFKC", t).lower()
    # German gender tags such as (m/w/d), (w/m/x), (all genders), (gn*)
    t = re.sub(r"\(\s*(?:[mwdfxi]\s*[/|,]\s*){1,3}[mwdfxi]\s*\)|\(\s*all genders?\s*\)|\(\s*gn\*?\s*\)", " ", t)
    t = re.sub(r"(?<!\w)(?:m/w/d|w/m/d|m/f/d|f/m/d|f/m/x|m/w/x|all genders)(?!\w)", " ", t)
    t = re.sub(r"[^\w\s\-/&+#.]", " ", t)
    t = re.sub(r"\s+", " ", t).strip(" -/")
    return t or None


def dup_norm(s: str | None) -> str:
    s = _s(s)
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s.lower())
    return re.sub(r"[^a-z0-9]+", "", s)


def sha(s: str, n: int = 20) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:n]


def words(s: str | None) -> int | None:
    s = _s(s)
    return len(s.split()) if s else None


# ------------------------------------------------------------------ seniority

SENIORITY_LEVELS = ["intern_student", "entry", "experienced", "senior", "manager", "executive"]

NYC_CAREER = {"Student": "intern_student", "Entry-Level": "entry",
              "Experienced (non-manager)": "experienced", "Manager": "manager", "Executive": "executive"}

ARB_SENIORITY = [  # (regex on a single Arbeitnow job_types value, level)
    (r"intern|praktik|working student|werkstudent|^student$|hilfst.tigkeit", "intern_student"),
    (r"^entry|berufseinstieg|junior|graduate|trainee|ausbildung", "entry"),
    (r"^senior", "senior"),
    (r"teamleitung|^lead|manager|f.hrungskraft", "manager"),
    (r"executive|director|gesch.ftsf.hrung", "executive"),
    (r"experienced|berufserfahren|^mid|professional", "experienced"),
]

TITLE_RULES = [  # derived seniority from job title keywords (ordered)
    (r"\b(intern|internship|praktikum|praktikant\w*|werkstudent\w*|working student|student)\b", "intern_student"),
    (r"\b(chief|vp|vice president|cxo|ceo|cto|cfo|coo|commissioner|executive director|geschäftsführ\w*)\b", "executive"),
    (r"\b(director|head of|leiter\w*|leitung)\b", "manager"),
    (r"\b(manager|supervisor|team lead|teamleiter\w*)\b", "manager"),
    (r"\b(senior|sr\.?|principal|staff|lead|expert)\b", "senior"),
    (r"\b(junior|jr\.?|entry[- ]level|graduate|trainee|associate|assistant|einsteiger\w*)\b", "entry"),
]


def seniority_from_title(title: str | None) -> str | None:
    title = _s(title)
    if not title:
        return None
    t = title.lower()
    for pat, lvl in TITLE_RULES:
        if re.search(pat, t):
            return lvl
    return None


def arb_seniority(job_types: list[str]) -> tuple[str | None, str | None]:
    """Return (raw seniority-bearing job_types values, mapped level) - source provided."""
    hits, levels = [], []
    for jt in job_types or []:
        v = jt.strip().lower()
        for pat, lvl in ARB_SENIORITY:
            if re.search(pat, v):
                hits.append(jt)
                levels.append(lvl)
                break
    if not hits:
        return None, None
    # most junior wins for student/intern, otherwise take the highest level mentioned
    lvl = "intern_student" if "intern_student" in levels else max(levels, key=SENIORITY_LEVELS.index)
    return "; ".join(hits), lvl


# ------------------------------------------------------------------ dates

def iso_date(s) -> str | None:
    if s is None or (isinstance(s, float) and pd.isna(s)) or str(s).strip() == "":
        return None
    for fmt in ("%m/%d/%Y", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(s).strip()[:26], fmt).date().isoformat()
        except ValueError:
            continue
    return None


def num(s) -> float | None:
    try:
        v = float(str(s).replace(",", "").replace("$", ""))
        return v if v == v else None
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------ source readers

def latest_retrieval(source: str) -> Path | None:
    d = RAW_JOBS / source
    done = sorted(d.glob("retrieval_*/.complete")) if d.exists() else []
    return done[-1].parent if done else None


def _collected(ret: Path) -> str | None:
    m = json.loads((ret / "retrieval_metadata.json").read_text(encoding="utf-8"))
    return (m.get("started_utc") or "")[:10] or None


def read_nyc(ret: Path) -> tuple[list[dict], list[dict]]:
    f = ret / "jobs_nyc_postings.csv"
    df = pd.read_csv(f, dtype=str, keep_default_na=False, encoding="utf-8")
    collected = _collected(ret)
    raws, rows = [], []
    for i, r in enumerate(df.to_dict("records")):
        ref = f"{rel(f)}#row={i + 2}"
        raws.append({"source": "nyc_jobs", "retrieval_id": ret.name, "source_record_ref": ref,
                     "raw_record_json": json.dumps(r, ensure_ascii=False)})
        g = lambda k: (r.get(k) or "").strip() or None
        title = clean_text(g("Business Title"))
        desc = clean_text(g("Job Description"))
        freq = g("Salary Frequency")
        rows.append({
            "source": "nyc_jobs",
            # Job ID is shared by the Internal and External posting of the same vacancy
            "source_job_id": f"{g('Job ID')}-{(g('Posting Type') or 'NA').lower()}",
            "source_record_ref": ref, "retrieval_id": ret.name,
            "job_title": title, "company_name": clean_text(g("Agency")),
            "industry": None,
            "industry_derived": "Public administration (City of New York agency)",
            "job_category_source": g("Job Category"),
            "location_raw": clean_text(g("Work Location")),
            "country": "US", "state": "NY", "city": None,
            "geo_basis": "country/state derived from source scope (NYC municipal employer); city not parsed",
            "remote_flag": None,
            "employment_type": {"F": "full_time", "P": "part_time"}.get(g("Full-Time/Part-Time indicator") or ""),
            "seniority_source": g("Career Level"),
            "seniority_source_mapped": NYC_CAREER.get(g("Career Level") or ""),
            "description": desc, "responsibilities": None,
            "requirements": clean_text(g("Minimum Qual Requirements")),
            "preferred_skills": clean_text(g("Preferred Skills")),
            "salary_min": num(g("Salary Range From")), "salary_max": num(g("Salary Range To")),
            "salary_currency": "USD" if (g("Salary Range From") or g("Salary Range To")) else None,
            "salary_period": freq.lower() if freq else None,
            "date_posted": iso_date(g("Posting Date")), "date_collected": collected,
            "source_url": None,
        })
    return raws, rows


def read_arbeitnow(ret: Path) -> tuple[list[dict], list[dict]]:
    collected = _collected(ret)
    raws, rows = [], []
    for p in sorted(ret.glob("page_*.json")):
        for i, x in enumerate(json.loads(p.read_text(encoding="utf-8")).get("data", [])):
            ref = f"{rel(p)}#data[{i}]"
            raws.append({"source": "arbeitnow", "retrieval_id": ret.name, "source_record_ref": ref,
                         "raw_record_json": json.dumps(x, ensure_ascii=False)})
            sen_raw, sen_lvl = arb_seniority(x.get("job_types") or [])
            emp = [t for t in (x.get("job_types") or []) if re.search(r"full|part|permanent|temporary|contract|fixed|freelance|minijob|teilzeit|vollzeit", t, re.I)]
            ts = x.get("created_at")
            rows.append({
                "source": "arbeitnow", "source_job_id": x.get("slug"),
                "source_record_ref": ref, "retrieval_id": ret.name,
                "job_title": clean_text(x.get("title")), "company_name": clean_text(x.get("company_name")),
                "industry": None, "industry_derived": None,
                "job_category_source": "; ".join(x.get("tags") or []) or None,
                "location_raw": clean_text(x.get("location")),
                "country": None, "state": None, "city": None,
                "geo_basis": "location_raw only; country not provided by source (not inferred)",
                "remote_flag": bool(x.get("remote")) if x.get("remote") is not None else None,
                "employment_type": "; ".join(emp) or None,
                "seniority_source": sen_raw, "seniority_source_mapped": sen_lvl,
                "description": clean_text(x.get("description")), "responsibilities": None,
                "requirements": None, "preferred_skills": None,
                "salary_min": None, "salary_max": None, "salary_currency": None, "salary_period": None,
                "date_posted": datetime.fromtimestamp(int(ts), tz=timezone.utc).date().isoformat() if ts else None,
                "date_collected": collected, "source_url": x.get("url"),
            })
    return raws, rows


READERS = {"nyc_jobs": read_nyc, "arbeitnow": read_arbeitnow}


# ------------------------------------------------------------------ language

def detect_language(texts: list[str | None]) -> tuple[list, list]:
    from langdetect import DetectorFactory, detect_langs
    DetectorFactory.seed = 0  # deterministic
    langs, confs = [], []
    for t in texts:
        if not t or len(t) < 20:
            langs.append(None)
            confs.append(None)
            continue
        try:
            best = detect_langs(t[:3000])[0]
            langs.append(best.lang)
            confs.append(round(best.prob, 4))
        except Exception:
            langs.append(None)
            confs.append(None)
    return langs, confs


# ------------------------------------------------------------------ build

def build() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    raws, rows, stats = [], [], {}
    for source, reader in READERS.items():
        ret = latest_retrieval(source)
        if not ret:
            log.warning("No complete retrieval for %s", source)
            continue
        r, c = reader(ret)
        raws += r
        rows += c
        stats[source] = {"retrieval_id": ret.name, "raw_records": len(r)}
        log.info("%s: %d raw records from %s", source, len(r), ret.name)
    raw_df = pd.DataFrame(raws)
    raw_df.insert(0, "raw_record_id", [sha(f"{a}|{b}") for a, b in zip(raw_df.source, raw_df.source_record_ref)])
    df = pd.DataFrame(rows)

    df["posting_id"] = [sha(f"{s}|{j}") if _s(j) else sha(f"{s}|{_s(t) or ''}|{_s(c) or ''}|{_s(d) or ''}")
                        for s, j, t, c, d in zip(df.source, df.source_job_id, df.job_title,
                                                 df.company_name, df.description)]
    # identical source IDs inside one snapshot (exact re-listed rows): keep first, count the rest
    df["is_source_id_repeat"] = df.duplicated("posting_id", keep="first")
    stats["source_id_repeats_collapsed"] = int(df.is_source_id_repeat.sum())
    df = df[~df.is_source_id_repeat].copy()

    df["normalized_job_title"] = df.job_title.map(normalize_title)
    df["seniority_derived"] = df.job_title.map(seniority_from_title)
    df["seniority"] = df.seniority_source_mapped.where(df.seniority_source_mapped.notna(), df.seniority_derived)
    df["seniority_basis"] = [("source" if _s(s) else ("title_rule" if _s(d) else None))
                             for s, d in zip(df.seniority_source_mapped, df.seniority_derived)]
    df = df.drop(columns=["seniority_source_mapped"])

    parts = zip(df.job_title, df.description, df.requirements, df.preferred_skills)
    df["posting_text"] = ["\n\n".join(p for p in ps if _s(p)) or None for ps in parts]
    df["description_char_count"] = df.description.map(lambda s: len(s) if _s(s) else None).astype("Int64")
    df["description_word_count"] = df.description.map(words).astype("Int64")
    df["posting_text_word_count"] = df.posting_text.map(words).astype("Int64")

    log.info("Detecting language for %d postings (langdetect, seed=0)...", len(df))
    df["language"], df["language_confidence"] = detect_language(
        [_s(d) or _s(t) for d, t in zip(df.description, df.job_title)])

    df["exact_dup_key"] = [sha(f"{_s(t) or ''}|{_s(c) or ''}|{_s(d) or ''}", 32)
                           for t, c, d in zip(df.job_title, df.company_name, df.description)]
    df["normalized_dup_key"] = [sha(f"{dup_norm(t)}|{dup_norm(c)}|{dup_norm(d)}", 32)
                                for t, c, d in zip(df.normalized_job_title, df.company_name, df.description)]
    df = df.sort_values(["source", "date_posted", "posting_id"], na_position="last").reset_index(drop=True)
    first = df.groupby("normalized_dup_key")["posting_id"].transform("first")
    df["duplicate_group_id"] = first
    df["is_exact_duplicate"] = df.duplicated("exact_dup_key", keep="first")
    df["is_normalized_duplicate"] = df.duplicated("normalized_dup_key", keep="first")
    df["contract_version"] = CONTRACT_VERSION
    for c in ("salary_min", "salary_max", "language_confidence"):
        df[c] = pd.to_numeric(df[c])
    df = df.astype(object).where(df.notna(), None)
    return raw_df, df[CANONICAL_COLUMNS], stats
