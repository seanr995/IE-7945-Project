"""EDA metrics, tables and figures computed from the DuckDB database only."""
from __future__ import annotations

import json

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.utils.common import DB_PATH, OUTPUTS, get_logger

log = get_logger("analysis.eda")

BLUE, ORANGE = "#2a78d6", "#eb6834"          # categorical slots 1, 2 (validated default palette)
SOURCE_COLORS = {"nyc_jobs": BLUE, "arbeitnow": ORANGE}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
TABLES, FIGS = OUTPUTS / "tables", OUTPUTS / "figures"

CANONICAL_FIELDS = ["job_title", "company_name", "industry", "industry_derived", "location_raw", "country",
                    "state", "city", "employment_type", "seniority_source", "seniority", "description",
                    "responsibilities", "requirements", "preferred_skills", "salary_min", "salary_max",
                    "salary_currency", "date_posted", "language", "source_url"]


def style(ax, title: str, xlabel: str = "", ylabel: str = ""):
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2)
    ax.grid(axis="x" if ax.get_ylabel() == "" else "y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def q(con, sql: str) -> pd.DataFrame:
    return con.execute(sql).df()


def missingness(con) -> pd.DataFrame:
    parts = [f"SELECT source, '{c}' AS field, avg(CASE WHEN {c} IS NULL THEN 1 ELSE 0 END) AS null_rate "
             f"FROM job_postings GROUP BY source" for c in CANONICAL_FIELDS]
    df = q(con, " UNION ALL ".join(parts))
    return df.pivot(index="field", columns="source", values="null_rate").reindex(CANONICAL_FIELDS).round(3)


def desc_stats(con) -> pd.DataFrame:
    return q(con, """
        SELECT source, count(*) n,
          round(avg(description_char_count),1) chars_mean, median(description_char_count) chars_median,
          round(stddev(description_char_count),1) chars_std, min(description_char_count) chars_min,
          max(description_char_count) chars_max,
          round(avg(description_word_count),1) words_mean, median(description_word_count) words_median,
          round(stddev(description_word_count),1) words_std, min(description_word_count) words_min,
          round(quantile_cont(description_word_count, 0.05),1) words_p05, round(quantile_cont(description_word_count, 0.25),1) words_p25,
          round(quantile_cont(description_word_count, 0.75),1) words_p75, round(quantile_cont(description_word_count, 0.95),1) words_p95,
          max(description_word_count) words_max
        FROM job_postings GROUP BY ROLLUP(source) ORDER BY source NULLS LAST""").fillna({"source": "ALL"})


def compute(con=None) -> dict:
    own = con is None
    con = con or duckdb.connect(str(DB_PATH), read_only=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    m: dict = {}
    m["raw_postings"] = q(con, "SELECT count(*) n FROM job_postings_raw").n[0]
    m["raw_by_source"] = q(con, "SELECT source, count(*) n FROM job_postings_raw GROUP BY 1 ORDER BY 1").set_index("source").n.to_dict()
    m["canonical_postings"] = q(con, "SELECT count(*) n FROM job_postings").n[0]
    m["source_id_repeats"] = int(m["raw_postings"] - m["canonical_postings"])
    d = q(con, """SELECT count(DISTINCT exact_dup_key) exact_unique, count(DISTINCT normalized_dup_key) norm_unique,
                  sum(is_exact_duplicate::INT) exact_dups, sum(is_normalized_duplicate::INT) norm_dups FROM job_postings""")
    m["unique_postings_exact"] = int(d.exact_unique[0])
    m["unique_postings"] = int(d.norm_unique[0])
    m["exact_duplicates"] = int(d.exact_dups[0])
    m["duplicates"] = int(d.norm_dups[0])
    m["duplicate_rate"] = round(m["duplicates"] / m["canonical_postings"], 4)
    m["raw_duplicate_rate"] = round((m["raw_postings"] - m["unique_postings"]) / m["raw_postings"], 4)
    by_src = q(con, """SELECT source, count(*) canonical, count(DISTINCT normalized_dup_key) unique_postings,
                       sum(is_normalized_duplicate::INT)::INTEGER duplicates,
                       round(avg(is_normalized_duplicate::INT),4) duplicate_rate FROM job_postings GROUP BY 1 ORDER BY 1""")
    raw = pd.Series(m["raw_by_source"], name="raw")
    by_src = by_src.merge(raw, left_on="source", right_index=True)
    by_src.to_csv(TABLES / "postings_by_source.csv", index=False)
    m["by_source"] = by_src.to_dict("records")
    m["nyc_internal_external_pairs"] = int(q(con, """
        SELECT count(*) n FROM (SELECT split_part(source_job_id,'-',1) jid FROM job_postings WHERE source='nyc_jobs'
        GROUP BY 1 HAVING count(DISTINCT split_part(source_job_id,'-',2))=2)""").n[0])

    lang = q(con, """SELECT coalesce(language,'(undetected)') AS language, source, count(*) n FROM job_postings
                     GROUP BY ALL ORDER BY n DESC""")
    lang.to_csv(TABLES / "language_mix.csv", index=False)
    tot = lang.groupby("language").n.sum().sort_values(ascending=False)
    m["language_mix"] = {k: {"n": int(v), "pct": round(v / m["canonical_postings"], 4)} for k, v in tot.items()}

    sen = q(con, """SELECT coalesce(seniority,'(unknown)') AS seniority, coalesce(seniority_basis,'none') AS basis,
                    source, count(*) n FROM job_postings GROUP BY ALL ORDER BY source, n DESC""")
    sen.to_csv(TABLES / "seniority.csv", index=False)
    s2 = sen.groupby("seniority").n.sum().sort_values(ascending=False)
    m["seniority"] = {k: int(v) for k, v in s2.items()}
    m["seniority_basis"] = {k: int(v) for k, v in sen.groupby("basis").n.sum().items()}
    m["nyc_career_level"] = q(con, """SELECT seniority_source, count(*) n FROM job_postings WHERE source='nyc_jobs'
                                      GROUP BY 1 ORDER BY 2 DESC""").set_index("seniority_source").n.to_dict()

    ds = desc_stats(con)
    ds.to_csv(TABLES / "description_length_stats.csv", index=False)
    m["description_stats"] = ds.to_dict("records")

    miss = missingness(con)
    miss.to_csv(TABLES / "missingness_by_source.csv")
    m["missingness"] = miss.reset_index().to_dict("records")

    ind = q(con, """SELECT source, count(industry) industry_source_nonnull, count(industry_derived) industry_derived_nonnull,
                    count(job_category_source) job_category_nonnull FROM job_postings GROUP BY 1""")
    m["industry_availability"] = ind.to_dict("records")
    cat = q(con, """SELECT job_category_source, count(*) n FROM job_postings WHERE source='nyc_jobs'
                    GROUP BY 1 ORDER BY 2 DESC""")
    cat.to_csv(TABLES / "nyc_job_category.csv", index=False)
    m["nyc_job_category_top"] = cat.head(10).set_index("job_category_source").n.to_dict()

    titles = q(con, """SELECT normalized_job_title, source, count(*) n FROM job_postings_unique
                       GROUP BY ALL ORDER BY n DESC LIMIT 25""")
    titles.to_csv(TABLES / "top_titles.csv", index=False)
    m["top_titles"] = titles.head(10).to_dict("records")
    m["distinct_normalized_titles"] = int(q(con, "SELECT count(DISTINCT normalized_job_title) n FROM job_postings").n[0])

    geo = q(con, """SELECT source, coalesce(location_raw,'(missing)') AS location_raw, count(*) n FROM job_postings
                    GROUP BY ALL ORDER BY n DESC""")
    geo.to_csv(TABLES / "locations.csv", index=False)
    m["top_locations"] = {s: g.head(8).set_index("location_raw").n.to_dict() for s, g in geo.groupby("source")}
    m["distinct_locations"] = geo.groupby("source").size().to_dict()
    m["remote_arbeitnow"] = int(q(con, "SELECT count(*) n FROM job_postings WHERE remote_flag").n[0])

    dates = q(con, """SELECT source, min(date_posted) min_posted, max(date_posted) max_posted,
                      median(date_posted) median_posted, count(date_posted) n_dated,
                      min(date_collected) collected FROM job_postings GROUP BY 1""")
    dates.to_csv(TABLES / "date_coverage.csv", index=False)
    m["date_coverage"] = {r.source: {k: str(getattr(r, k))[:10] for k in ("min_posted", "max_posted", "median_posted", "collected")}
                          for r in dates.itertuples()}

    sal = q(con, """SELECT salary_period, count(*) n, median(salary_min) median_min, median(salary_max) median_max
                    FROM job_postings WHERE salary_min IS NOT NULL GROUP BY 1 ORDER BY 2 DESC""")
    sal.to_csv(TABLES / "salary_nyc.csv", index=False)
    m["salary"] = sal.to_dict("records")
    m["empty_or_short_descriptions"] = int(q(con, "SELECT count(*) n FROM job_postings WHERE coalesce(description_word_count,0) < 50").n[0])

    ref = q(con, """SELECT table_name, estimated_size FROM duckdb_tables() WHERE schema_name = 'main' ORDER BY table_name""")
    counts = {t: int(con.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0]) for t in ref.table_name}
    m["table_counts"] = counts
    m["onet_version"] = con.execute("SELECT any_value(onet_version) FROM onet_occupations").fetchone()[0] \
        if "onet_occupations" in counts else None
    esco = q(con, "SELECT version, status, final_or_temporary FROM source_metadata WHERE dataset_name='esco_status'")
    m["esco_version"] = esco.version[0] if len(esco) else None
    m["esco_status"] = esco.status[0] if len(esco) else "ESCO_STATUS = MANUAL_DOWNLOAD_REQUIRED"
    m["esco_mode"] = esco.final_or_temporary[0] if len(esco) else None
    m["sources"] = q(con, "SELECT * FROM source_metadata").to_dict("records")
    pd.Series(counts, name="rows").rename_axis("table").to_csv(TABLES / "table_counts.csv")

    figures(con)
    if own:
        con.close()
    out = json.loads(json.dumps(m, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    (TABLES / "sprint1_metrics.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("EDA metrics written to outputs/tables/")
    return out


def figures(con) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})

    # description length histogram, one panel per source (shared x)
    df = q(con, "SELECT source, description_word_count w FROM job_postings WHERE description_word_count IS NOT NULL")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharex=True)
    for ax, (src, g) in zip(axes, df.groupby("source")):
        ax.hist(g.w.clip(upper=2000), bins=40, color=SOURCE_COLORS.get(src, BLUE), edgecolor="#fcfcfb", linewidth=1)
        ax.axvline(g.w.median(), color=INK, linewidth=1, linestyle="--")
        ax.text(g.w.median(), ax.get_ylim()[1] * 0.92, f" median {int(g.w.median())} words", color=INK, fontsize=9)
        style(ax, f"{src}: description length", "words (clipped at 2,000)", "postings")
    fig.tight_layout()
    fig.savefig(FIGS / "description_length_hist.png", dpi=130)
    plt.close(fig)

    # language mix
    lang = q(con, "SELECT coalesce(language,'(undetected)') AS language, count(*) n FROM job_postings GROUP BY 1 ORDER BY 2")
    fig, ax = plt.subplots(figsize=(6, 0.45 * len(lang) + 1))
    ax.barh(lang.language, lang.n, color=BLUE, height=0.6)
    for y, v in enumerate(lang.n):
        ax.text(v, y, f" {v:,}", va="center", color=INK2, fontsize=9)
    style(ax, "Detected language (langdetect) - all postings")
    fig.tight_layout()
    fig.savefig(FIGS / "language_mix.png", dpi=130)
    plt.close(fig)

    # seniority by source (small multiples, one hue per source)
    sen = q(con, "SELECT source, coalesce(seniority,'(unknown)') AS s, count(*) n FROM job_postings GROUP BY ALL")
    order = ["intern_student", "entry", "experienced", "senior", "manager", "executive", "(unknown)"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, (src, g) in zip(axes, sen.groupby("source")):
        g = g.set_index("s").reindex(order).fillna(0)
        ax.barh(order, g.n, color=SOURCE_COLORS.get(src, BLUE), height=0.6)
        for y, v in enumerate(g.n):
            if v:
                ax.text(v, y, f" {int(v):,}", va="center", color=INK2, fontsize=9)
        style(ax, f"{src}: seniority")
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIGS / "seniority_by_source.png", dpi=130)
    plt.close(fig)

    # duplicates by source
    d = q(con, """SELECT source, count(DISTINCT normalized_dup_key) uniq, sum(is_normalized_duplicate::INT)::INTEGER dups
                  FROM job_postings GROUP BY 1 ORDER BY 1""")
    fig, ax = plt.subplots(figsize=(8, 2.6))
    ax.barh(d.source, d.uniq, color=BLUE, height=0.55, label="unique")
    ax.barh(d.source, d.dups, left=d.uniq + 8, color="#86b6ef", height=0.55, label="duplicate")
    for y, (u, du) in enumerate(zip(d.uniq, d.dups)):
        ax.text(u + du + 20, y, f"{du:,} dup / {u + du:,}", va="center", color=INK2, fontsize=9)
    ax.set_xlim(0, (d.uniq + d.dups).max() * 1.3)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=9)
    style(ax, "Unique vs duplicate postings (normalized key)")
    fig.tight_layout()
    fig.savefig(FIGS / "duplicates_by_source.png", dpi=130)
    plt.close(fig)
