"""Deterministic section detection. PROTOTYPE.

Order of precedence (recorded in section_detection_method):
  source_field   - the source already separates the section (NYC Minimum Qual Requirements / Preferred Skills)
  heading_rule   - a recognised heading (line heading or inline "Heading:") inside the description
  llm_in_extraction - no recognised heading: text is passed as general_description and the extraction
                   model assigns evidence_section per item (no separate LLM call).
The original description is never modified; sections are character spans into it.
"""
from __future__ import annotations

import re

HEADINGS = [  # (section, regex matched against a normalised heading string)
    ("excluded", r"^(benefits?|what we offer|why (us|join( us)?|you should work for us)|compensation|salary( range)?|"
                 r"about (us|the company|the agency|the office|the team|the division)|who we are|our (global )?structure|"
                 r"(to|how to) apply|interview process|note|please note|important note|hours(/shift|/schedule)?|"
                 r"work location|additional information|fees|residency|work from home policy|perks|your benefits|"
                 r"equal opportunity|diversity|location|for more information.*|all applicants.*|search for job id.*|"
                 r"click on the job.*|for nycha employees|special working conditions|how we work|what success looks like)$"),
    ("responsibilities", r"^(.*responsibilit.*|.*duties.*|what (you('ll| will)|you would) (do|be doing|own)|your (role|mission|tasks|impact|day)|"
                         r"tasks|the role|about the role|role overview|in this role( you will)?|job description|"
                         r"position summary|the opportunity|deine aufgaben|ihre aufgaben|aufgaben|what you.ll work on)$"),
    ("preferred_qualifications", r"^(.*nice[- ]?to[- ]?haves?.*|.*preferred.*|bonus( points)?|pluses|it.s a plus.*|wünschenswert)$"),
    ("skills", r"^(skills|key skills|technical skills|tech stack|our stack)$"),
    ("requirements", r"^(requirements|qualifications|minimum (required )?qualifications|what (you bring|we.re looking for|you.ll bring|makes you a great fit)|"
                     r"who you are|about you|your profile|you|experience|must[- ]haves?|dein profil|ihr profil)$"),
]
_COMPILED = [(s, re.compile(p)) for s, p in HEADINGS]


def _norm(h: str) -> str:
    h = h.lower().replace("’", "'").strip(" :\t-*#")
    return re.sub(r"\s+", " ", re.sub(r"[^a-zäöüß' /&\-]", " ", h)).strip()


def classify_heading(h: str) -> str | None:
    n = _norm(h)
    if not n or len(n) > 60:
        return None
    for sec, rx in _COMPILED:
        if rx.match(n):
            return sec
    return None


_LINE = re.compile(r"(?m)^[ \t]*([^\n]{2,70}?)[ \t]*:?[ \t]*$")
_INLINE = re.compile(r"(?:(?<=^)|(?<=\n)|(?<=\.\s)|(?<=\s{2}))([A-Z][A-Za-z/&' ,\-]{2,60}):")


def find_headings(text: str) -> list[tuple[int, int, str]]:
    """Return (heading_start, content_start, section) sorted by position."""
    hits = {}
    for m in _LINE.finditer(text):
        sec = classify_heading(m.group(1))
        if sec and len(m.group(1).split()) <= 8:
            hits[m.start(1)] = (m.start(1), m.end(), sec)
    for m in _INLINE.finditer(text):
        sec = classify_heading(m.group(1))
        if sec and m.start(1) not in hits:
            hits[m.start(1)] = (m.start(1), m.end(), sec)
    return sorted(hits.values())


def detect_sections(description: str | None, requirements: str | None = None,
                    preferred_skills: str | None = None) -> dict:
    """Return {'sections': [{section, text, origin, start, end, method}], 'method': str}."""
    out = []
    d = description or ""
    heads = find_headings(d)
    if heads:
        if heads[0][0] > 0 and d[:heads[0][0]].strip():
            out.append({"section": "general_description", "text": d[:heads[0][0]].strip(), "origin": "description",
                        "start": 0, "end": heads[0][0], "method": "heading_rule"})
        for i, (hs, cs, sec) in enumerate(heads):
            end = heads[i + 1][0] if i + 1 < len(heads) else len(d)
            body = d[cs:end].strip()
            if body:
                out.append({"section": sec, "text": body, "origin": "description", "start": cs, "end": end,
                            "method": "heading_rule"})
    elif d.strip():
        out.append({"section": "general_description", "text": d.strip(), "origin": "description", "start": 0,
                    "end": len(d), "method": "llm_in_extraction"})
    if requirements:
        out.append({"section": "requirements", "text": requirements, "origin": "requirements", "start": 0,
                    "end": len(requirements), "method": "source_field"})
    if preferred_skills:
        out.append({"section": "preferred_qualifications", "text": preferred_skills, "origin": "preferred_skills",
                    "start": 0, "end": len(preferred_skills), "method": "source_field"})
    has_core = any(s["section"] in ("responsibilities", "requirements") and s["method"] == "heading_rule" for s in out)
    methods = sorted({s["method"] for s in out})
    overall = "+".join(methods) if methods else "none"
    return {"sections": out, "method": overall, "has_core_heading": has_core}


PRIORITY = ["responsibilities", "requirements", "skills", "preferred_qualifications", "general_description"]


def model_input(sections: list[dict], max_chars: int) -> list[dict]:
    """Sections sent to the extractor: excluded dropped, highest-value sections first, total capped."""
    keep = sorted([s for s in sections if s["section"] != "excluded"], key=lambda s: PRIORITY.index(s["section"]))
    out, used = [], 0
    for s in keep:
        room = max_chars - used
        if room < 200:
            break
        text = s["text"] if len(s["text"]) <= room else s["text"][:room].rsplit(" ", 1)[0]
        out.append({"section": s["section"], "text": text})
        used += len(text)
    return out
