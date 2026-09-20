"""Resume analysis: structure checks, contact info, skills and an ATS-style score."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .skills import extract_skills, flatten

SECTION_PATTERNS = {
    "summary": r"^(professional\s+)?(summary|profile|objective|about( me)?)$",
    "experience": r"^(work\s+|professional\s+|relevant\s+)?(experience|employment( history)?|internships?)$",
    "education": r"^(education|academic (background|qualifications))$",
    "skills": r"^(technical\s+|key\s+|core\s+)?(skills|competencies)$|^technologies$",
    "projects": r"^(academic\s+|personal\s+|key\s+)?projects$",
    "certifications": r"^(certifications?|licenses|courses|training)$",
}

ACTION_VERBS = {
    "built", "developed", "designed", "implemented", "led", "created", "deployed",
    "optimized", "optimised", "improved", "automated", "analyzed", "analysed",
    "trained", "reduced", "increased", "launched", "managed", "architected",
    "engineered", "integrated", "migrated", "delivered", "achieved", "collaborated",
    "mentored", "researched", "evaluated", "streamlined", "refactored", "tested",
    "scaled", "established", "generated", "published", "presented", "resolved",
    "spearheaded", "coordinated", "produced", "maintained", "configured",
    "documented", "boosted", "wrote", "containerized", "containerised", "wrapped",
}

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"\+?\d[\d\s().-]{8,}\d")
LINKEDIN_RE = re.compile(r"linkedin\.com/in/[\w-]+", re.IGNORECASE)
GITHUB_RE = re.compile(r"github\.com/[\w-]+", re.IGNORECASE)
QUANT_RE = re.compile(
    r"\d+(?:\.\d+)?\s?%"                      # 94%
    r"|\$\s?\d"                                # $5
    r"|\b\d+(?:[.,]\d+)?\s?(?:x|k|m|ms|hours?|users|customers|requests|records|"
    r"images|samples|models|classes|movies|engineers|projects|seconds)\b",
    re.IGNORECASE,
)


@dataclass
class Check:
    name: str
    points: int
    passed: bool
    tip: str


@dataclass
class ResumeReport:
    word_count: int
    score: int
    checks: list[Check]
    skills_by_category: dict[str, list[str]]
    skills: set[str]
    sections: list[str]
    contact: dict[str, bool]
    quantified_bullets: int
    action_verbs: list[str] = field(default_factory=list)

    @property
    def tips(self) -> list[str]:
        return [c.tip for c in self.checks if not c.passed]


def detect_sections(text: str) -> list[str]:
    found = []
    for line in text.splitlines():
        header = line.strip().strip(":-–—•*# ").strip()
        if not header or len(header) > 40:
            continue
        for name, pattern in SECTION_PATTERNS.items():
            if name not in found and re.match(pattern, header, re.IGNORECASE):
                found.append(name)
    return found


def detect_contact(text: str) -> dict[str, bool]:
    phones = [m for m in PHONE_RE.findall(text) if 10 <= len(re.sub(r"\D", "", m)) <= 13]
    return {
        "email": bool(EMAIL_RE.search(text)),
        "phone": bool(phones),
        "linkedin": bool(LINKEDIN_RE.search(text)),
        "github": bool(GITHUB_RE.search(text)),
    }


def count_quantified(text: str) -> int:
    return sum(1 for line in text.splitlines() if QUANT_RE.search(line))


def find_action_verbs(text: str) -> list[str]:
    words = set(re.findall(r"[a-z]+", text.lower()))
    return sorted(words & ACTION_VERBS)


def analyze_resume(text: str) -> ResumeReport:
    words = len(text.split())
    sections = detect_sections(text)
    contact = detect_contact(text)
    quantified = count_quantified(text)
    verbs = find_action_verbs(text)
    skills_by_cat = extract_skills(text)
    skills = flatten(skills_by_cat)

    checks = [
        Check("Email address", 10, contact["email"], "Add a professional email address near the top."),
        Check("Phone number", 5, contact["phone"], "Add a phone number recruiters can call."),
        Check("LinkedIn or GitHub link", 10, contact["linkedin"] or contact["github"],
              "Link your LinkedIn and/or GitHub profile."),
        Check("Experience or projects section", 15, "experience" in sections or "projects" in sections,
              "Add an Experience or Projects section with clear headings."),
        Check("Education section", 10, "education" in sections,
              "Add an Education section (degree, institute, year)."),
        Check("Skills section", 10, "skills" in sections,
              "Add a dedicated Skills section so ATS parsers can find your keywords."),
        Check("Length 250-900 words", 10, 250 <= words <= 900,
              "Aim for roughly one page (250-900 words); trim or expand accordingly."),
        Check("3+ quantified achievements", 15, quantified >= 3,
              "Add numbers to bullets (e.g. 'improved recall by 18%')."),
        Check("5+ strong action verbs", 10, len(verbs) >= 5,
              "Start bullets with action verbs such as Built, Deployed, Optimized."),
        Check("8+ recognised skills", 5, len(skills) >= 8,
              "List more relevant tools and technologies you actually use."),
    ]
    score = sum(c.points for c in checks if c.passed)

    return ResumeReport(
        word_count=words,
        score=score,
        checks=checks,
        skills_by_category=skills_by_cat,
        skills=skills,
        sections=sections,
        contact=contact,
        quantified_bullets=quantified,
        action_verbs=verbs,
    )
