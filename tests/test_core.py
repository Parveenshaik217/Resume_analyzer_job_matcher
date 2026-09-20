from pathlib import Path

from src.analyzer import analyze_resume
from src.matcher import SemanticScorer, load_jobs, match_to_description, rank_jobs
from src.parser import extract_text
from src.skills import extract_skills, flatten

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = (ROOT / "data" / "sample_resume.txt").read_text(encoding="utf-8")


def test_skill_boundaries():
    skills = flatten(extract_skills("Worked with JavaScript and MySQL. Also C++ and C#."))
    assert "JavaScript" in skills
    assert "MySQL" in skills
    assert "C++" in skills and "C#" in skills
    assert "Java" not in skills   # not glued to JavaScript
    assert "SQL" not in skills    # not glued to MySQL


def test_skill_phrase_across_lines():
    assert "Machine Learning" in flatten(extract_skills("machine\nlearning engineer"))


def test_txt_parser():
    assert extract_text(b"Hello\r\n\r\n\r\nWorld", "cv.txt") == "Hello\n\nWorld"


def test_analyzer_on_sample():
    report = analyze_resume(SAMPLE)
    assert report.score >= 80
    assert {"education", "skills", "projects", "experience"} <= set(report.sections)
    assert report.contact["email"] and report.contact["linkedin"]
    assert "PyTorch" in report.skills


def test_analyzer_penalises_empty_resume():
    assert analyze_resume("just some words").score < 30


def test_ml_resume_ranks_ml_roles_first():
    scorer = SemanticScorer(use_embeddings=False)
    ranked = rank_jobs(SAMPLE, load_jobs(), scorer)
    top3 = [r.title for r in ranked[:3]]
    assert "Machine Learning Engineer" in top3
    assert "Computer Vision Engineer" in top3
    assert ranked[-1].title in {"Full Stack Developer", "Data Analyst", "Data Engineer"}


def test_custom_jd_reports_missing_skills():
    scorer = SemanticScorer(use_embeddings=False)
    result = match_to_description(SAMPLE, "We need Kubernetes, Terraform and Python experience.", scorer)
    assert "Python" in result.matched
    assert {"Kubernetes", "Terraform"} <= set(result.missing)
