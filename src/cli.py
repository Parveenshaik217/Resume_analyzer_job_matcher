"""Command-line interface.

Examples:
    python -m src.cli --resume data/sample_resume.txt
    python -m src.cli --resume my_resume.pdf --top 5
    python -m src.cli --resume my_resume.pdf --jd job_description.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .analyzer import analyze_resume
from .matcher import SemanticScorer, load_jobs, match_to_description, rank_jobs
from .parser import extract_text_from_path


def main() -> None:
    ap = argparse.ArgumentParser(description="AI Resume Analyzer & Job Matcher")
    ap.add_argument("--resume", required=True, help="Path to resume (.pdf, .docx, .txt)")
    ap.add_argument("--jd", help="Optional path to a job description (.txt) to match against")
    ap.add_argument("--top", type=int, default=5, help="How many sample jobs to show")
    ap.add_argument("--no-embeddings", action="store_true", help="Use TF-IDF only (fast, no model download)")
    args = ap.parse_args()

    text = extract_text_from_path(args.resume)
    report = analyze_resume(text)
    scorer = SemanticScorer(use_embeddings=not args.no_embeddings)

    print(f"\n=== Resume analysis ({Path(args.resume).name}) ===")
    print(f"ATS-style score : {report.score}/100")
    print(f"Words           : {report.word_count}")
    print(f"Sections found  : {', '.join(report.sections) or 'none'}")
    print("Skills detected :")
    for cat, names in report.skills_by_category.items():
        print(f"  {cat}: {', '.join(names)}")
    if report.tips:
        print("Improvements    :")
        for tip in report.tips:
            print(f"  - {tip}")

    if args.jd:
        jd_text = Path(args.jd).read_text(encoding="utf-8")
        m = match_to_description(text, jd_text, scorer)
        print(f"\n=== Match vs {args.jd} [{scorer.backend}] ===")
        print(f"{m.score}/100 - {m.verdict}")
        print(f"Matched skills: {', '.join(m.matched) or 'none'}")
        print(f"Missing skills: {', '.join(m.missing) or 'none'}")
        return

    print(f"\n=== Top {args.top} job matches [{scorer.backend}] ===")
    for i, m in enumerate(rank_jobs(text, load_jobs(), scorer, report.skills, top_k=args.top), 1):
        print(f"{i}. {m.title} - {m.company}  ->  {m.score}/100 ({m.verdict})")
        print(f"     have: {', '.join(m.matched) or '-'}")
        print(f"     gap : {', '.join(m.missing) or '-'}")


if __name__ == "__main__":
    main()
