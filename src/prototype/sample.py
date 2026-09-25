"""Deterministic stratified experimental sample. PROTOTYPE.

Population : job_postings_unique (no normalized duplicates), language='en',
             description_word_count >= min_description_words.
Strata     : source x seniority (6 levels + unknown) x description-length tercile (per source).
Selection  : sources get an equal share of the sample. Inside a source, strata are visited
             round-robin; within a stratum postings are ordered by sha256(seed|posting_id). A posting is
             skipped if its normalized_job_title was already taken (title diversity).
Phases     : sample_rank alternates sources; the first `dual_model_postings` ranks form the
             dual-model phase, the rest the Gemini-primary phase. `gold_100` = 100 postings chosen by
             hash order (50 from each phase) for human annotation.
Same seed + same database -> identical sample.
"""
from __future__ import annotations

import hashlib

import pandas as pd


def _h(seed: str, pid: str) -> str:
    return hashlib.sha256(f"{seed}|{pid}".encode()).hexdigest()


def build_sample(con, seed: str, size: int, dual_n: int, min_words: int) -> pd.DataFrame:
    pop = con.execute(f"""
        SELECT posting_id, source, job_title, normalized_job_title, coalesce(seniority,'unknown') AS seniority,
               description_word_count, language
        FROM job_postings_unique
        WHERE language = 'en' AND description_word_count >= {int(min_words)}""").df()
    pop["h"] = [_h(seed, p) for p in pop.posting_id]
    pop["length_tercile"] = pop.groupby("source").description_word_count.transform(
        lambda s: pd.qcut(s.rank(method="first"), 3, labels=["short", "medium", "long"])).astype(str)
    sources = sorted(pop.source.unique())
    per_source = {s: size // len(sources) + (1 if i < size % len(sources) else 0) for i, s in enumerate(sources)}
    picked = {}
    for s in sources:
        sub = pop[pop.source == s].sort_values("h")
        strata = {k: list(g.itertuples()) for k, g in sub.groupby(["seniority", "length_tercile"], sort=True)}
        order = sorted(strata, key=lambda k: _h(seed, "|".join(k)))
        seen_titles, chosen = set(), []
        while len(chosen) < per_source[s] and any(strata[k] for k in order):
            for k in order:
                while strata[k]:
                    r = strata[k].pop(0)
                    if r.normalized_job_title in seen_titles:
                        continue
                    seen_titles.add(r.normalized_job_title)
                    chosen.append((r, k))
                    break
                if len(chosen) >= per_source[s]:
                    break
        picked[s] = chosen
    rows, i = [], 0
    iters = {s: iter(picked[s]) for s in sources}
    while any(iters.values()):
        for s in sources:
            nxt = next(iters[s], None) if iters[s] else None
            if nxt is None:
                iters[s] = None
                continue
            r, k = nxt
            rows.append({"sample_rank": i + 1, "posting_id": r.posting_id, "source": r.source, "job_title": r.job_title,
                         "normalized_job_title": r.normalized_job_title, "seniority": r.seniority,
                         "length_tercile": r.length_tercile, "description_word_count": int(r.description_word_count),
                         "stratum": f"{r.source}|{k[0]}|{k[1]}"})
            i += 1
    df = pd.DataFrame(rows)
    df["phase"] = ["dual_model" if r <= dual_n else "gemini_primary" for r in df.sample_rank]
    df["hash_order"] = [_h(seed + "|gold", p) for p in df.posting_id]
    gold = set(df[df.phase == "dual_model"].sort_values("hash_order").posting_id.head(50)) | \
        set(df[df.phase == "gemini_primary"].sort_values("hash_order").posting_id.head(50))
    df["in_gold_100"] = df.posting_id.isin(gold)
    df["sampling_seed"] = seed
    df["sampling_rules"] = (f"job_postings_unique; language='en'; description_word_count>={min_words}; "
                            f"equal per-source share; round-robin over seniority x length-tercile strata; "
                            f"sha256(seed|posting_id) order; one posting per normalized_job_title")
    return df.drop(columns=["hash_order"])
