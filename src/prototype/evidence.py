"""Evidence validation, cross-model agreement and deterministic consensus. PROTOTYPE.

Nothing here calls an LLM. All rules are deterministic and unit-tested.

EVIDENCE LOCATION (against job_postings.posting_text; offsets index into that string)
  exact       evidence_text is a verbatim substring
  normalized  matches after whitespace/quote/dash/case normalisation (offsets mapped back to original)
  fuzzy       best source sentence has difflib ratio >= 0.90 (offsets of that sentence; original evidence kept)
  none        not traceable -> statement is REJECTED (never survives)

STATEMENT SUPPORT
  support_overlap = share of the statement's content-word stems found in the evidence stems.
  < 0.34 -> REJECTED as unsupported (model introduced words not in the evidence)
  < 0.60 -> kept, flagged weak_support

CONSENSUS (weights sum to 1; every component stored separately)
  consensus_score = 0.35*evidence_score + 0.40*agreement_score + 0.15*support_overlap + 0.10*kind_score
    evidence_score : exact 1.0 | normalized 0.9 | fuzzy 0.5
    agreement_score: lexical match 1.0 | semantic match 0.8 | boundary (partial) 0.5 |
                     validator-confirmed single model 0.6 | single model 0.25
    kind_score     : 1 unless the other model labelled the same evidence with the other kind (0)
  tier: high >= 0.80 AND lexical/semantic cross-model agreement | medium >= 0.55 | low < 0.55
"""
from __future__ import annotations

import difflib
import re

STOP = set("""a an the and or of to for in on with by at from as is are be been being this that these those
our your their its it we you they will would can could should may must including include includes such
other others all any each within across into via using use used per etc e g i ie able ability strong
good excellent proven solid working work knowledge experience skills skill understanding related
responsible responsibility responsibilities duties tasks help ensure well both new""".split())

_Q = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", " ": " ", "•": " "})


def _norm_with_map(s: str) -> tuple[str, list[int]]:
    """Lowercase, unify quotes/dashes, collapse whitespace; keep map norm_index -> original index."""
    out, idx, prev_space = [], [], False
    for i, ch in enumerate(s.translate(_Q)):
        if ch.isspace():
            if prev_space or not out:
                continue
            out.append(" ")
            idx.append(i)
            prev_space = True
        else:
            out.append(ch.lower())
            idx.append(i)
            prev_space = False
    return "".join(out), idx


def _norm(s: str) -> str:
    return _norm_with_map(s)[0].strip(" .;:,")


_SENT = re.compile(r"[^\n.;!?•]+[.;!?]?")


def locate_evidence(source: str, evidence: str) -> dict:
    res = {"match_type": "none", "start": None, "end": None, "exact_text": None, "ratio": None}
    if not source or not evidence or not evidence.strip():
        return res
    ev = evidence.strip()
    i = source.find(ev)
    if i >= 0:
        return {"match_type": "exact", "start": i, "end": i + len(ev), "exact_text": ev, "ratio": 1.0}
    ns, mp = _norm_with_map(source)
    ne = _norm(ev)
    if len(ne) >= 3:
        j = ns.find(ne)
        if j >= 0:
            a, b = mp[j], mp[j + len(ne) - 1] + 1
            return {"match_type": "normalized", "start": a, "end": b, "exact_text": source[a:b], "ratio": 1.0}
    best, bm = 0.0, None
    for m in _SENT.finditer(source):
        seg = m.group(0)
        if abs(len(seg) - len(ev)) > max(40, len(ev)):
            continue
        r = difflib.SequenceMatcher(None, _norm(seg), ne).ratio()
        if r > best:
            best, bm = r, m
    if bm is not None and best >= 0.90:
        a = bm.start() + (len(bm.group(0)) - len(bm.group(0).lstrip()))
        return {"match_type": "fuzzy", "start": a, "end": bm.end(), "exact_text": source[a:bm.end()].strip(),
                "ratio": round(best, 3)}
    res["ratio"] = round(best, 3) if best else None
    return res


def _stem(w: str) -> str:
    for suf in ("ations", "ation", "ments", "ment", "ings", "ing", "ies", "ied", "ers", "er", "ed", "es", "s", "al", "ly"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def content_tokens(s: str) -> list[str]:
    toks = re.findall(r"[a-z0-9][a-z0-9+#./\-]*", (s or "").lower().translate(_Q))
    return [t.strip(".-/") for t in toks if t.strip(".-/") and t.strip(".-/") not in STOP]


def support_overlap(statement: str, evidence: str) -> float:
    st = [_stem(t) for t in content_tokens(statement)]
    if not st:
        return 0.0
    ev = [_stem(t) for t in content_tokens(evidence)]
    evs = set(ev)

    def found(t):
        return t in evs or any(len(t) >= 5 and (e.startswith(t[:5]) and t.startswith(e[:5])) for e in ev)

    return round(sum(found(t) for t in st) / len(st), 3)


def jaccard(a: str, b: str) -> float:
    x, y = {_stem(t) for t in content_tokens(a)}, {_stem(t) for t in content_tokens(b)}
    if not x or not y:
        return 0.0
    return len(x & y) / len(x | y)


def containment(small: str, big: str) -> float:
    x, y = {_stem(t) for t in content_tokens(small)}, {_stem(t) for t in content_tokens(big)}
    return len(x & y) / len(x) if x else 0.0


def span_iou(a: tuple, b: tuple) -> float:
    if None in a or None in b:
        return 0.0
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union else 0.0


def match_items(g: list[dict], q: list[dict], sim: dict | None = None, lex_t: float = 0.5, sem_t: float = 0.88) -> dict:
    """Greedy one-to-one matching of two providers' items for ONE posting.
    items need: kind, statement_text, start, end. sim[(i,j)] optional cosine similarity.
    Returns pairs [(i, j, match_type, score)], plus classification of unmatched items."""
    sim = sim or {}
    cands = []
    for i, a in enumerate(g):
        for j, b in enumerate(q):
            if a["kind"] != b["kind"]:
                continue
            jac = 1.0 if _norm(a["statement_text"]) == _norm(b["statement_text"]) else jaccard(a["statement_text"], b["statement_text"])
            s = sim.get((i, j))
            if jac >= lex_t:
                cands.append((jac + 1.0, i, j, "lexical", jac))
            elif s is not None and s >= sem_t:
                cands.append((s, i, j, "semantic", s))
    cands.sort(reverse=True)
    used_g, used_q, pairs = set(), set(), []
    for _, i, j, mt, sc in cands:
        if i in used_g or j in used_q:
            continue
        used_g.add(i)
        used_q.add(j)
        pairs.append((i, j, mt, round(sc, 3)))

    def classify(items, others, used_self, side):
        out = {}
        for i, a in enumerate(items):
            if i in used_self:
                continue
            span_a = (a.get("start"), a.get("end"))
            kind_conf = [j for j, b in enumerate(others) if b["kind"] != a["kind"] and
                         (jaccard(a["statement_text"], b["statement_text"]) >= 0.5 or span_iou(span_a, (b.get("start"), b.get("end"))) >= 0.6)]
            boundary = [j for j, b in enumerate(others) if b["kind"] == a["kind"] and
                        (max(containment(a["statement_text"], b["statement_text"]),
                             containment(b["statement_text"], a["statement_text"])) >= 0.5 or
                         span_iou(span_a, (b.get("start"), b.get("end"))) >= 0.5)]
            if kind_conf:
                out[i] = ("kind_conflict", kind_conf)
            elif boundary:
                out[i] = ("boundary", boundary)
            else:
                out[i] = ("missing_in_other", [])
        return out

    return {"pairs": pairs, "gemini_unmatched": classify(g, q, used_g, "g"),
            "groq_unmatched": classify(q, g, used_q, "q")}


def split_events(a_items: list[dict], b_items: list[dict], t: float = 0.7) -> int:
    """Number of B items that contain >= 2 same-kind A items (A split what B kept together)."""
    n = 0
    for b in b_items:
        inside = [a for a in a_items if a["kind"] == b["kind"] and
                  containment(a["statement_text"], b["statement_text"]) >= t and
                  _norm(a["statement_text"]) != _norm(b["statement_text"])]
        if len(inside) >= 2:
            n += 1
    return n


EVIDENCE_SCORE = {"exact": 1.0, "normalized": 0.9, "fuzzy": 0.5}
AGREEMENT_SCORE = {"lexical": 1.0, "semantic": 0.8, "boundary": 0.5, "validator_confirmed": 0.6, "single_model": 0.25}


def consensus(match_type: str, agreement: str, support: float, kind_conflict: bool) -> tuple[float, str]:
    ev = EVIDENCE_SCORE.get(match_type, 0.0)
    ag = AGREEMENT_SCORE.get(agreement, 0.25)
    score = round(0.35 * ev + 0.40 * ag + 0.15 * float(support) + 0.10 * (0.0 if kind_conflict else 1.0), 3)
    tier = "high" if (score >= 0.80 and agreement in ("lexical", "semantic")) else ("medium" if score >= 0.55 else "low")
    return score, tier
