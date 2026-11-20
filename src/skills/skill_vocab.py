"""Hand-reviewed vocabulary decisions for the 188-posting skill file.

MERGE   raw normalized_skill -> final name. Judgement calls, reviewed one by one; extends the
        unambiguous spelling variants already applied in merge_skills.py.
GENERIC transferable skills that appear in almost every kind of job (soft skills, basic computer
        literacy). They are flagged, not removed: filter them out before computing task-skill
        co-occurrence if they swamp the signal.
LANGUAGE spoken-language requirements (not job skills in the usual sense).
"""

MERGE = {
    "organization": "organizational skills", "oral communication": "verbal communication",
    "analytical thinking": "analytical skills", "team leadership": "leadership",
    "team supervision": "supervision", "staff supervision": "supervision",
    "team management": "people management", "personnel management": "people management",
    "process optimization": "process improvement", "data analytics": "data analysis", "analytics": "data analysis",
    "budget tracking": "budget management", "budgeting": "budget management", "budget planning": "budget management",
    "recruitment": "recruiting", "talent acquisition": "recruiting", "candidate sourcing": "recruiting",
    "relationship management": "relationship building", "database management": "database",
    "cloud infrastructure": "cloud computing", "technical troubleshooting": "troubleshooting",
    "record keeping": "records management", "electronic records management": "records management",
    "calendar management": "scheduling", "compliance": "regulatory compliance",
    "pipeline generation": "lead generation", "mentoring": "coaching and mentoring", "coaching": "coaching and mentoring",
    "microsoft 365": "microsoft office", "stakeholder engagement": "stakeholder management",
    "stakeholder collaboration": "stakeholder management", "cross-functional coordination": "cross-functional collaboration",
    "reporting analytics": "analytics reporting", "report preparation": "report writing", "legal drafting": "legal writing",
    "customer relationship management": "crm", "crm management": "crm", "speaking": "public speaking",
    "public presentation": "public speaking", "computer literacy": "computer skills",
}

GENERIC = {
    "communication", "written communication", "verbal communication", "interpersonal skills", "attention to detail",
    "analytical skills", "critical thinking", "problem solving", "organizational skills", "prioritization",
    "multitasking", "time management", "teamwork", "collaboration", "decision making", "meeting deadlines",
    "confidentiality", "computer skills", "active listening",
}

LANGUAGE = {
    "english", "german", "french", "spanish", "italian", "dutch", "arabic", "korean", "chinese", "mandarin chinese",
    "cantonese", "bengali", "haitian creole", "hebrew", "hindi", "russian", "urdu", "yiddish", "foreign language",
    "multilingual",
}
