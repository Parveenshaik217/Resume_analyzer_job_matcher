"""Streamlit UI for the AI Resume Analyzer & Job Matcher.

Run locally:  streamlit run app.py
"""
from pathlib import Path

import pandas as pd
import streamlit as st

from src.analyzer import analyze_resume
from src.matcher import SemanticScorer, load_jobs, match_to_description, rank_jobs
from src.parser import extract_text

SAMPLE_RESUME = Path(__file__).parent / "data" / "sample_resume.txt"

st.set_page_config(page_title="AI Resume Analyzer & Job Matcher", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Loading embedding model (first run downloads ~130 MB)...")
def get_scorer(use_embeddings: bool) -> SemanticScorer:
    return SemanticScorer(use_embeddings=use_embeddings)


@st.cache_data
def get_jobs() -> list[dict]:
    return load_jobs()


def show_match_details(match) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Overall match", f"{match.score:.0f}/100")
    c2.metric("Semantic similarity", f"{match.semantic * 100:.0f}%")
    c3.metric("Skill coverage", f"{match.coverage * 100:.0f}%")
    left, right = st.columns(2)
    left.markdown("**✅ Skills you have**")
    left.markdown(" ".join(f"`{s}`" for s in match.matched) or "_none_")
    right.markdown("**📚 Skills to add**")
    right.markdown(" ".join(f"`{s}`" for s in match.missing) or "_none — you cover everything listed_")


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("1. Your resume")
    uploaded = st.file_uploader("PDF, DOCX or TXT", type=["pdf", "docx", "txt"])
    use_sample = st.checkbox("Use the sample resume", value=False)

    st.header("2. Settings")
    use_embeddings = st.toggle(
        "Semantic embeddings", value=True,
        help="On: neural sentence embeddings (better). Off: TF-IDF only (instant, no model download).",
    )
    top_k = st.slider("Jobs to show", 3, 12, 6)

    st.caption("🔒 Files are processed in memory and never stored.")

st.title("📄 AI Resume Analyzer & Job Matcher")
st.caption("Upload a resume → get an ATS-style score, detected skills and ranked job matches.")

# ---------------------------------------------------------------- load resume
resume_text = None
if uploaded is not None:
    try:
        resume_text = extract_text(uploaded.getvalue(), uploaded.name)
    except Exception as exc:  # corrupt / unsupported file
        st.error(f"Could not read that file: {exc}")
        st.stop()
    if len(resume_text.split()) < 30:
        st.error("Very little text found. Scanned/image-only PDFs aren't supported - try a text-based PDF or DOCX.")
        st.stop()
elif use_sample:
    resume_text = SAMPLE_RESUME.read_text(encoding="utf-8")

if not resume_text:
    st.info("👈 Upload your resume, or tick **Use the sample resume** to try the demo.")
    st.stop()

report = analyze_resume(resume_text)
scorer = get_scorer(use_embeddings)
jobs = get_jobs()

tab_analysis, tab_jobs, tab_jd = st.tabs(["📊 Resume analysis", "🎯 Job matches", "📝 Match my own job description"])

# ---------------------------------------------------------------- tab 1
with tab_analysis:
    m1, m2, m3 = st.columns(3)
    m1.metric("ATS-style score", f"{report.score}/100")
    m2.metric("Skills detected", len(report.skills))
    m3.metric("Word count", report.word_count)
    st.progress(report.score / 100)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Checklist")
        for check in report.checks:
            icon = "✅" if check.passed else "❌"
            line = f"{icon} **{check.name}**"
            if not check.passed:
                line += f" — {check.tip}"
            st.markdown(line)
    with col_b:
        st.subheader("Detected skills")
        if report.skills_by_category:
            for category, names in report.skills_by_category.items():
                st.markdown(f"**{category}**  \n" + " ".join(f"`{n}`" for n in names))
        else:
            st.warning("No known skills detected. Add a Skills section with the tools you use.")

    with st.expander("Extracted text (what the parser sees)"):
        st.text(resume_text)

# ---------------------------------------------------------------- tab 2
with tab_jobs:
    matches = rank_jobs(resume_text, jobs, scorer, report.skills, top_k=top_k)
    st.caption(f"Similarity backend: **{scorer.backend}** · score = 40% semantic similarity + 60% skill coverage")

    # NOTE: st.dataframe (not st.bar_chart) so the app doesn't depend on altair
    summary_df = pd.DataFrame({
        "Job": [f"{m.title} ({m.company})" for m in matches],
        "Match score": [m.score for m in matches],
        "Verdict": [m.verdict for m in matches],
    })
    st.dataframe(
        summary_df,
        hide_index=True,
        column_config={
            "Match score": st.column_config.ProgressColumn(
                "Match score", min_value=0, max_value=100, format="%.0f"
            )
        },
    )

    for match in matches:
        label = f"{match.score:.0f}/100 · {match.title} — {match.company} · {match.location} · {match.verdict}"
        with st.expander(label):
            show_match_details(match)

# ---------------------------------------------------------------- tab 3
with tab_jd:
    st.write("Paste any job description to see how well your resume fits and which skills are missing.")
    jd_text = st.text_area("Job description", height=220, placeholder="Paste the full job description here...")
    if st.button("Analyze match", type="primary"):
        if len(jd_text.split()) < 15:
            st.warning("Please paste a longer job description (at least a couple of sentences).")
        else:
            result = match_to_description(resume_text, jd_text, scorer)
            st.subheader(f"{result.verdict} — {result.score:.0f}/100")
            show_match_details(result)
