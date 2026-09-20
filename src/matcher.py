"""Job matching: semantic similarity (embeddings or TF-IDF) blended with skill coverage."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .skills import extract_skills, flatten

JOBS_PATH = Path(__file__).resolve().parent.parent / "data" / "jobs.json"

# Final score = 100 * (SEMANTIC_WEIGHT * semantic + SKILL_WEIGHT * skill_coverage)
SEMANTIC_WEIGHT = 0.4
SKILL_WEIGHT = 0.6

# Raw cosine similarities are not spread over 0-1, so they are rescaled.
# These bounds are heuristics - tune them on your own data.
CALIBRATION = {
    "embeddings": (0.45, 0.85),  # bge-small cosine: unrelated ~0.45, near-duplicate ~0.85+
    "tf-idf": (0.0, 0.35),       # tf-idf cosine between a resume and a JD is usually low
}


class SemanticScorer:
    """Scores how semantically close a resume is to job texts (0-1)."""

    def __init__(self, use_embeddings: bool = True):
        self.model = None
        if use_embeddings:
            try:
                from fastembed import TextEmbedding

                self.model = TextEmbedding("BAAI/bge-small-en-v1.5")
            except Exception:  # not installed / model download failed -> fall back
                self.model = None

    @property
    def backend(self) -> str:
        return "embeddings" if self.model is not None else "tf-idf"

    def score(self, resume_text: str, docs: list[str]) -> np.ndarray:
        try:
            sims = self._embedding_sims(resume_text, docs) if self.model is not None else None
        except Exception:
            self.model = None
            sims = None
        if sims is None:
            sims = self._tfidf_sims(resume_text, docs)
        lo, hi = CALIBRATION[self.backend]
        return np.clip((sims - lo) / (hi - lo), 0.0, 1.0)

    def _embedding_sims(self, query: str, docs: list[str]) -> np.ndarray:
        vecs = np.array(list(self.model.embed([query] + docs)), dtype=float)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs[1:] @ vecs[0]

    @staticmethod
    def _tfidf_sims(query: str, docs: list[str]) -> np.ndarray:
        from sklearn.feature_extraction.text import TfidfVectorizer

        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        matrix = vec.fit_transform([query] + docs)
        return (matrix[1:] @ matrix[0].T).toarray().ravel()


@dataclass
class MatchResult:
    title: str
    company: str
    location: str
    score: float          # 0-100
    semantic: float       # 0-1
    coverage: float       # 0-1 share of the job's skills found in the resume
    matched: list[str]
    missing: list[str]

    @property
    def verdict(self) -> str:
        if self.score >= 75:
            return "Strong match"
        if self.score >= 50:
            return "Good match"
        if self.score >= 30:
            return "Partial match"
        return "Weak match"


def load_jobs(path: str | Path = JOBS_PATH) -> list[dict]:
    jobs = json.loads(Path(path).read_text(encoding="utf-8"))
    for job in jobs:
        job["skills"] = sorted(flatten(extract_skills(f"{job['title']}. {job['description']}")))
    return jobs


def rank_jobs(
    resume_text: str,
    jobs: list[dict],
    scorer: SemanticScorer,
    resume_skills: set[str] | None = None,
    top_k: int | None = None,
) -> list[MatchResult]:
    if resume_skills is None:
        resume_skills = flatten(extract_skills(resume_text))
    docs = [f"{j['title']}. {j['description']}" for j in jobs]
    semantic = scorer.score(resume_text, docs)

    results = []
    for job, sem in zip(jobs, semantic):
        required = set(job.get("skills") or flatten(extract_skills(job["description"])))
        matched = sorted(required & resume_skills)
        missing = sorted(required - resume_skills)
        coverage = len(matched) / len(required) if required else 0.0
        score = round(100 * (SEMANTIC_WEIGHT * float(sem) + SKILL_WEIGHT * coverage), 1)
        results.append(MatchResult(
            title=job["title"], company=job.get("company", ""), location=job.get("location", ""),
            score=score, semantic=float(sem), coverage=coverage, matched=matched, missing=missing,
        ))
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_k] if top_k else results


def match_to_description(resume_text: str, jd_text: str, scorer: SemanticScorer) -> MatchResult:
    """Score a resume against one pasted job description."""
    job = {"title": "Pasted job description", "company": "", "location": "", "description": jd_text}
    return rank_jobs(resume_text, [job], scorer)[0]
