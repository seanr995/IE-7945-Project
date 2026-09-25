import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cleaning.canonical import arb_seniority, clean_text, iso_date, normalize_title, seniority_from_title, sha
from src.utils.common import looks_like_html


def test_clean_text_strips_html_and_whitespace():
    assert clean_text("<p>Hello&nbsp; <b>world</b></p>") == "Hello\nworld" or clean_text("<p>Hello&nbsp; <b>world</b></p>").replace("\n", " ") == "Hello world"
    assert clean_text("   ") is None
    assert clean_text(None) is None


def test_normalize_title_removes_gender_tags_only():
    assert normalize_title("Senior Data Engineer (m/w/d)") == "senior data engineer"
    assert normalize_title("Analyst (Data)") == "analyst data"


def test_seniority_rules():
    assert seniority_from_title("Working Student Marketing") == "intern_student"
    assert seniority_from_title("Senior Frontend Engineer") == "senior"
    assert seniority_from_title("Director of Finance") == "manager"
    assert seniority_from_title("Clerk") is None
    assert arb_seniority(["Full Time", "Experienced"]) == ("Experienced", "experienced")
    assert arb_seniority(["Full Time"]) == (None, None)


def test_dates_and_ids():
    assert iso_date("09/15/2026") == "2026-09-15"
    assert iso_date("") is None
    assert sha("nyc_jobs|123-external") == sha("nyc_jobs|123-external")
    assert len(sha("x")) == 20


def test_html_detection(tmp_path):
    p = tmp_path / "fake.zip"
    p.write_text("<!DOCTYPE html><html><body>consent</body></html>")
    assert looks_like_html(p)
