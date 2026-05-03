"""
ML Resume Scorer — main.py
================================
Harini Jeyashree A | Rewritten with proper FastAPI engineering

WHAT THIS FILE TEACHES YOU (Module 5 — Backend Engineering):
  - What FastAPI actually is and why it is not Flask
  - What async/await means and when to use it
  - What Pydantic models are (request/response validation)
  - How HTTP status codes and error handling work
  - Why you never save temp files without try/finally
  - What CORS middleware does and why it exists
"""

import os
import re
import shutil
import joblib
import pandas as pd
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PyPDF2 import PdfReader
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


# ──────────────────────────────────────────────
# WHAT IS FASTAPI vs FLASK — the real difference
# ──────────────────────────────────────────────
# Flask: synchronous. One request blocks the server until done.
#   If PDF extraction takes 2 seconds, all other users wait.
#
# FastAPI: asynchronous. While waiting for slow I/O (file read,
#   database query, external API call), the server handles
#   other requests. Built on Starlette + uvicorn.
#
# FastAPI also auto-generates OpenAPI docs at /docs — free.
# Flask requires flask-restx or flasgger for docs.
#
# For your Resume Auditor: PDF extraction is I/O-bound (slow
# disk read). async lets the server stay responsive.


# ──────────────────────────────────────────────
# APP SETUP
# ──────────────────────────────────────────────

app = FastAPI(
    title="ML Resume Scorer",
    description="Scores resume-JD match using ML + rule-based skill extraction",
    version="2.0.0"
)
# 1. Mount static files so the browser can read style.css
# (Make sure style.css is placed inside a folder named "static")
app.mount("/static", StaticFiles(directory="static"), name="static")

# 2. Add a route for the root URL to serve the index.html file
@app.get("/", response_class=FileResponse)
async def read_index():
    return FileResponse("index.html")

# WHAT IS CORS:
# Browser security blocks requests from one domain to another
# by default. Your React frontend (localhost:3000) calling your
# FastAPI backend (localhost:8000) is a cross-origin request.
# CORSMiddleware tells the browser: "this server allows it."
# allow_origins=["*"] means any domain — fine for development,
# restrict to your frontend domain in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# PYDANTIC RESPONSE MODELS
# ──────────────────────────────────────────────
# WHY PYDANTIC: FastAPI uses Pydantic to validate and document
# request/response shapes. If your function returns a dict that
# doesn't match the schema, FastAPI raises an error immediately
# instead of sending broken JSON to the client.
# This is type safety at the API boundary.

class ResumeScoreResponse(BaseModel):
    match_score: float          # 0.0 to 100.0
    feedback: str               # human-readable verdict
    matched_skills: list[str]   # skills in both resume and JD
    missing_skills: list[str]   # skills in JD but not resume
    model_used: str             # which classifier was chosen


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# ──────────────────────────────────────────────
# LOAD RESOURCES AT STARTUP
# ──────────────────────────────────────────────
# WHY AT MODULE LEVEL (not inside the route function):
# If you load the model inside check_resume(), it reloads
# from disk on EVERY request. Loading a .pkl file takes ~200ms.
# At 100 requests/minute that is 20 seconds wasted every minute.
# Load once at startup, reuse forever.

MODEL_PATH = "models/resume_match_model.pkl"
SKILLS_PATH = "skills.csv"

# Load ML pipeline
try:
    model = joblib.load(MODEL_PATH)
    model_loaded = True
    print(f"Model loaded from {MODEL_PATH}")
except FileNotFoundError:
    model = None
    model_loaded = False
    print(f"WARNING: Model not found at {MODEL_PATH}. Run train.py first.")

# Load skills ontology
SKILL_SET = set()
SKILL_SYNONYMS = {}

try:
    skills_df = pd.read_csv(SKILLS_PATH)
    for _, row in skills_df.iterrows():
        primary = str(row["PREFERREDLABEL"]).lower().strip()
        SKILL_SET.add(primary)
        if pd.notna(row.get("ALTLABELS", None)):
            for alt in str(row["ALTLABELS"]).split("\n"):
                alt = alt.lower().strip()
                if alt:
                    SKILL_SYNONYMS[alt] = primary
    print(f"Skills loaded: {len(SKILL_SET)} canonical + "
          f"{len(SKILL_SYNONYMS)} synonyms")
except FileNotFoundError:
    print(f"WARNING: skills.csv not found at {SKILLS_PATH}.")


# ──────────────────────────────────────────────
# UTILITY FUNCTIONS
# ──────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Normalise text for both ML input and skill matching.
    Identical to train.py — CRITICAL that they match.

    WHY IDENTICAL TO TRAINING:
    If training cleaned text one way and inference cleans it
    another way, the same resume produces different tokens.
    The model gets features it was never trained on.
    Always use the same cleaning function in both files.
    Put it in a shared utils.py in production.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_text_from_pdf(path: str) -> str:
    """
    Extract all text from a PDF file.
    Returns empty string if extraction fails (corrupt PDF etc.)
    """
    try:
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + " "
        return clean_text(text)
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Could not extract text from PDF: {str(e)}"
        )


def extract_skills(text: str) -> set[str]:
    """
    Rule-based skill extraction using skills ontology.
    Two passes: canonical skill names, then synonym mapping.

    WHY TWO PASSES:
    Resume might say "ML" instead of "machine learning".
    First pass misses it. Second pass catches it via SYNONYMS.
    Both map to the same canonical skill name so set operations
    (intersection, difference) work correctly.
    """
    found = set()

    # Pass 1: direct canonical match
    for skill in SKILL_SET:
        # WHY f" {skill} ": word boundary check without regex.
        # Prevents "r" matching inside "docker".
        # Wrapping text in spaces ensures whole-word matching.
        if f" {skill} " in f" {text} ":
            found.add(skill)

    # Pass 2: synonym mapping
    for alt, canonical in SKILL_SYNONYMS.items():
        if f" {alt} " in f" {text} ":
            found.add(canonical)

    return found


def ml_match_score(resume_text: str, jd_text: str) -> float:
    """
    Uses the trained ML pipeline to score resume-JD match.
    Returns probability (0.0 to 100.0) of being a good match.

    WHY predict_proba NOT predict:
    predict() returns 0 or 1 — binary, not useful as a score.
    predict_proba() returns [P(class=0), P(class=1)].
    [0][1] = probability of class 1 (good match).
    Multiply by 100 for a percentage score.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run train.py first."
        )
    # 1. Get the ML probability
    try:
        combined = resume_text + " " + jd_text
        ml_prob = model.predict_proba([combined])[0][1]
    except Exception:
        ml_prob = 0.5  # Fallback if model fails

    # 2. Calculate actual skill overlap
    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(jd_text)
    
    if not jd_skills:
        skill_score = 1.0
    else:
        # What percentage of the JD skills does the candidate have?
        skill_score = len(resume_skills & jd_skills) / len(jd_skills)

    # 3. Hybrid Score: Combine both (50% ML + 50% Skill Overlap)
    final_score = (ml_prob * 0.5) + (skill_score * 0.5)
    
    return min(100, max(0, int(final_score * 100)))


def generate_feedback(score: float) -> str:
    """Human-readable feedback based on match score."""
    if score >= 80:
        return "Excellent match — strong candidate for this role"
    elif score >= 65:
        return "Good match — address the missing skills to strengthen"
    elif score >= 45:
        return "Moderate match — significant skill gaps present"
    else:
        return "Low match — consider upskilling before applying"


# ──────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    WHY: production systems need a /health route so load
    balancers and monitoring tools can check if the app is up.
    Render and AWS use this to know when deployment succeeded.
    """
    return HealthResponse(
        status="ok",
        model_loaded=model_loaded
    )


@app.post("/check_resume", response_model=ResumeScoreResponse)
async def check_resume(
    resume: UploadFile = File(...),
    job_description: str = Form(...)
):
    try:
        # Reset the pointer to the beginning of the file in case it was already read
        await resume.seek(0)
        
        # Read the contents into memory
        contents = await resume.read()
        
        # VERY IMPORTANT: If you are about to use the file again, seek back to 0
        await resume.seek(0)

        # Process the contents with PyPDF2
        # (Pass a BytesIO stream of contents or handle it directly)
        import io
        pdf_file = io.BytesIO(contents)
        reader = PdfReader(pdf_file)
        
        resume_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                resume_text += text + " "

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")
    """
    Score a resume against a job description.

    Accepts:
      - resume: PDF file upload
      - job_description: plain text from a form field

    Returns:
      - match_score: 0-100
      - feedback: string verdict
      - matched_skills: list of skills found in both
      - missing_skills: list of skills in JD but not resume
      - model_used: which classifier was selected at training
    """

    # ── Input validation ──────────────────────────────────
    # WHY VALIDATE FILE TYPE:
    # Without this, anyone can upload a .exe, .js, or .py file.
    # You save it to disk and try to "extract text" from it.
    # This is a security vulnerability — arbitrary file write.
    if not resume.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted"
        )

    if not job_description.strip():
        raise HTTPException(
            status_code=400,
            detail="Job description cannot be empty"
        )

    if len(job_description) > 10_000:
        raise HTTPException(
            status_code=400,
            detail="Job description too long (max 10,000 characters)"
        )

    # ── Save PDF temporarily ──────────────────────────────
    # WHY TEMP FILE:
    # UploadFile is a streaming object — you can't seek backwards.
    # PdfReader needs a file path or seekable stream.
    # We save it temporarily, read it, then delete it.
    #
    # WHY TRY/FINALLY:
    # If extract_text_from_pdf() raises an exception, the code
    # after it never runs — os.remove() is skipped.
    # The temp file stays on disk forever. Disk fills up. Server dies.
    # try/finally GUARANTEES cleanup even when exceptions occur.

    temp_path = f"temp_{resume.filename}"

    try:
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(resume.file, f)

        resume_text = extract_text_from_pdf(temp_path)

    finally:
        # This runs whether extract succeeded or raised an error
        if os.path.exists(temp_path):
            os.remove(temp_path)

    # ── Validate extracted text ───────────────────────────
    if not resume_text.strip():
        raise HTTPException(
            status_code=422,
            detail="Could not extract text from PDF. Is it a scanned image?"
        )

    # ── Score and analyse ─────────────────────────────────
    jd_text = clean_text(job_description)

    score = ml_match_score(resume_text, jd_text)

    resume_skills = extract_skills(resume_text)
    jd_skills     = extract_skills(jd_text)

    matched  = sorted(resume_skills & jd_skills)   # set intersection
    missing  = sorted(jd_skills - resume_skills)    # set difference

    # Get model name from eval report if available
    model_name = "unknown"
    try:
        import json
        with open("models/eval_report.json") as f:
            report = json.load(f)
            model_name = report.get("best_model", "unknown")
    except Exception:
        pass

    return ResumeScoreResponse(
        match_score=score,
        feedback=generate_feedback(score),
        matched_skills=matched,
        missing_skills=missing,
        model_used=model_name
    )


# ──────────────────────────────────────────────
# ENTRYPOINT
# ──────────────────────────────────────────────
# WHY uvicorn:
# FastAPI is a framework, not a server.
# Uvicorn is the ASGI server that actually handles HTTP connections
# and calls your FastAPI app. It is to FastAPI what gunicorn
# is to Flask.
#
# Run with: uvicorn main:app --reload --port 8000
# --reload: auto-restarts when you save a file (development only)
# --port 8000: listen on port 8000

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)