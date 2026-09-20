# 📄 AI Resume Analyzer & Job Matcher

Upload a resume (PDF / DOCX / TXT) and get:

- **ATS-style score** (0-100) with an actionable checklist
- **Skill extraction** across ~100 skills in 7 categories
- **Ranked job matches** = 40% semantic similarity + 60% skill coverage
- **Skill-gap analysis** for every job, and for any job description you paste in

**Live demo:https://ai-resume-analyzer-job-matcher-idhnp9ffqxvgmi2xvcetpj.streamlit.app/   [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

## How it works

```
resume file ──► parser ──► analyzer (sections, contact, metrics, ATS score)
                   │
                   └──► skills (regex taxonomy) ──┐
                                                  ├──► matcher ──► ranked jobs + skill gaps
job descriptions ──► skills ─────────────────────┤
                                                  │
resume + job text ──► embeddings (fastembed, BAAI/bge-small-en-v1.5)
                      └─ falls back to TF-IDF cosine if the model is unavailable
```

## Project structure

```
resume-job-matcher/
├── app.py                  # Streamlit UI
├── requirements.txt
├── requirements-dev.txt    # + pytest
├── conftest.py             # makes `src` importable in tests
├── .streamlit/config.toml
├── data/
│   ├── skills.json         # skills taxonomy (edit to add skills/aliases)
│   ├── jobs.json           # sample job postings (replace with your own)
│   └── sample_resume.txt
├── src/
│   ├── parser.py           # PDF/DOCX/TXT -> text
│   ├── skills.py           # skill extraction
│   ├── analyzer.py         # ATS-style scoring
│   ├── matcher.py          # semantic + skill-coverage matching
│   └── cli.py              # terminal interface
└── tests/test_core.py
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py                 # opens http://localhost:8501
```

### Terminal mode (no UI)

```bash
python -m src.cli --resume data/sample_resume.txt
python -m src.cli --resume my_resume.pdf --top 5
python -m src.cli --resume my_resume.pdf --jd job_description.txt
python -m src.cli --resume my_resume.pdf --no-embeddings    # TF-IDF only, no model download
```

### Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Deploy (Streamlit Community Cloud, free)

1. Push this folder to a **public GitHub repo**.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, choose the repo, branch `main`, main file `app.py`.
4. Click **Deploy**. You get a public `https://resumeanalyzerjobmatcher-4fzwelppyexhhcenqqcbla.streamlit.app/` URL.

Updating: `git push` and the app redeploys automatically.

## Customising

- **More skills:** add entries to `data/skills.json` (canonical name + aliases).
- **Real jobs:** replace `data/jobs.json` with your own postings or a Kaggle job dataset (`title`, `company`, `location`, `description`).
- **Scoring:** tweak `SEMANTIC_WEIGHT`, `SKILL_WEIGHT` and `CALIBRATION` in `src/matcher.py`.

## Limitations

- Skill extraction is keyword/alias based, so it won't catch skills that aren't in the taxonomy.
- Scanned (image-only) PDFs aren't supported; there is no OCR.
- Embedding models read ~512 tokens, so very long resumes are truncated for the semantic score.
- The ATS score is a heuristic checklist, not a real ATS simulation.

## Ideas to extend

- LLM-written rewrite suggestions for weak bullets
- Resume ↔ JD keyword highlighting
- Job data from a live API, plus a saved-jobs shortlist
- Evaluate ranking quality against hand-labelled resume/job pairs
