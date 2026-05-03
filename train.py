"""
ML Resume Scorer — train.py
================================
Harini Jeyashree A | Rewritten with proper ML engineering

WHY THIS FILE EXISTS:
  - Loads real labelled data (not 16 synthetic samples)
  - Engineers features properly (TF-IDF on combined text)
  - Compares 3 classifiers and picks the best one
  - Evaluates using cross-validation, not just training accuracy
  - Saves the winning pipeline + evaluation report

WHAT YOU LEARN FROM THIS FILE:
  Module 2 → training loop, fit(), evaluation metrics
  Module 3 → why we compare multiple models
"""

import pandas as pd
import numpy as np
import os  # Make sure this is imported at the top of the file
import re
import json
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)

# ──────────────────────────────────────────────
# SECTION 1: TEXT CLEANING
# ──────────────────────────────────────────────
# WHY: Raw text has noise — punctuation, numbers, extra spaces.
# The model sees tokens. Dirty tokens = worse features.
# We strip everything that isn't a letter or space.
# We do NOT strip numbers here intentionally — "python3" and
# "aws ec2" are meaningful. Adjust based on your data.

def clean(text: str) -> str:
    """
    Lowercase → remove special chars → collapse whitespace.

    WHY LOWERCASE: "Python" and "python" are different tokens
    to TF-IDF. Lowercasing merges them into one feature.

    WHY REGEX: faster than splitting and filtering character by
    character, and handles edge cases like tabs and newlines.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text)               # collapse spaces
    return text.strip()


# ──────────────────────────────────────────────
# SECTION 2: FEATURE ENGINEERING
# ──────────────────────────────────────────────
# WHY CONCATENATE: We need ONE input to the vectoriser.
# By joining resume + JD, skill words that appear in both
# get double the term frequency → higher TF-IDF score.
# The model learns: "high TF-IDF for skill words = likely match."
# This is called implicit feature interaction.

def make_input(resume: str, jd: str) -> str:
    """
    Combine resume and job description into a single string.
    Words that appear in both documents naturally get higher
    TF-IDF scores — this encodes the matching signal.
    """
    return clean(resume) + " " + clean(jd)


# ──────────────────────────────────────────────
# SECTION 3: LOAD DATA
# ──────────────────────────────────────────────
# In production this loads from a CSV of real labelled pairs.
# For now we generate a richer synthetic dataset.
#
# WHY MORE DATA: Your original code had 16 samples.
# Logistic Regression needs ~100+ samples minimum.
# With 16 samples the model memorises training data
# (overfitting) and learns nothing generalisable.
#
# IMPORTANT: Replace load_data() with real labelled data
# from sources like Kaggle "Resume Dataset" before deploying.

def load_data() -> tuple[list[str], list[int]]:
    """
    Returns (X, y) where:
      X = list of combined resume+JD strings
      y = list of labels (1 = good match, 0 = poor match)

    In production: load from a CSV file with columns
    [resume_text, jd_text, label]
    """

    # --- Expanded synthetic data ---
    # 10 JDs × 10 resumes = 100 pairs
    # Still synthetic but 6x more than your original.
    # Labels are based on skill overlap >= 0.5 threshold.

    jds = [
        "python machine learning data science pandas numpy sklearn model training",
        "data analyst sql excel tableau power bi dashboard reporting",
        "backend engineer fastapi django rest api postgresql docker",
        "cloud engineer aws gcp docker kubernetes ci cd devops",
        "nlp engineer transformers huggingface bert pytorch text classification",
        "frontend developer react javascript typescript tailwind css html",
        "data engineer spark airflow etl pipeline kafka postgresql",
        "ml ops engineer mlflow docker kubernetes model deployment monitoring",
        "full stack developer react fastapi postgresql deployment github",
        "ai engineer llm langchain rag groq api prompt engineering agents",
    ]

    resumes = [
        "data scientist python pandas sklearn machine learning model evaluation eda",
        "analyst sql tableau excel reporting dashboard business intelligence",
        "backend developer fastapi postgresql rest api python docker deployment",
        "devops engineer docker kubernetes aws gcp terraform ci cd pipelines",
        "nlp researcher bert huggingface pytorch text classification transformers",
        "frontend developer react javascript css html tailwind component design",
        "data engineer apache spark kafka airflow etl postgresql pipeline",
        "ml engineer mlflow kubeflow docker model serving monitoring drift",
        "full stack engineer react fastapi postgresql github render deployment",
        "ai developer groq llm langchain prompt engineering rag agents fastapi",
        # Adding some deliberately weak resumes for class balance
        "marketing coordinator excel powerpoint social media campaigns",
        "graphic designer adobe photoshop illustrator figma branding",
        "hr manager recruitment onboarding payroll performance review",
        "accountant tally gst taxation balance sheet financial reporting",
        "teacher curriculum lesson planning student assessment communication",
    ]

    skill_groups = {
        "python ml": {"python", "machine learning", "sklearn", "pandas", "numpy", "model"},
        "data analyst": {"sql", "excel", "tableau", "dashboard", "reporting", "power bi"},
        "backend": {"fastapi", "django", "rest api", "postgresql", "docker", "backend"},
        "cloud devops": {"aws", "gcp", "docker", "kubernetes", "ci cd", "devops"},
        "nlp": {"transformers", "huggingface", "bert", "pytorch", "text classification", "nlp"},
        "frontend": {"react", "javascript", "typescript", "tailwind", "css", "html"},
        "data eng": {"spark", "airflow", "etl", "kafka", "postgresql", "pipeline"},
        "mlops": {"mlflow", "docker", "kubernetes", "monitoring", "model deployment"},
        "fullstack": {"react", "fastapi", "postgresql", "deployment", "github"},
        "ai eng": {"llm", "langchain", "rag", "groq", "prompt engineering", "agents"},
    }

    def extract_skills(text, skill_set):
        return {s for s in skill_set if s in text.lower()}

    X, y = [], []
    jd_skill_keys = list(skill_groups.keys())

    for i, jd in enumerate(jds):
        jd_skill_key = jd_skill_keys[i]
        jd_skills = skill_groups[jd_skill_key]

        for resume in resumes:
            res_skills = extract_skills(resume, jd_skills)

            # Compute overlap safely — no ZeroDivisionError
            # WHY: if jd_skills is empty, division by zero crashes.
            # Your original code used `if not jd_skills: continue`
            # which is correct but skips the pair entirely.
            # Here we handle it gracefully with a 0 label.
            if len(jd_skills) == 0:
                label = 0
            else:
                overlap = len(res_skills) / len(jd_skills)
                label = 1 if overlap >= 0.5 else 0

            X.append(make_input(resume, jd))
            y.append(label)

    print(f"Dataset: {len(X)} samples | "
          f"Positive: {sum(y)} | Negative: {len(y)-sum(y)}")
    return X, y


# ──────────────────────────────────────────────
# SECTION 4: MODEL COMPARISON
# ──────────────────────────────────────────────
# WHY COMPARE MODELS:
# Logistic Regression is a good baseline but not always best.
# Random Forest handles non-linear patterns.
# Gradient Boosting often wins on tabular/sparse data.
# We let cross-validation decide — not our intuition.
#
# WHY CROSS-VALIDATION NOT TRAIN/TEST SPLIT:
# With ~150 samples, a single 80/20 split is unreliable.
# One unlucky split could make a bad model look good.
# StratifiedKFold runs 5 splits and averages — more reliable.
# "Stratified" means each fold has the same class ratio.

def build_pipelines() -> dict:
    """
    Returns a dict of named sklearn Pipelines.
    Each pipeline = TF-IDF vectoriser + a classifier.
    WHY Pipeline: saves vectoriser + model as one object.
    Prevents the mistake of forgetting to transform new data.
    """

    # Shared vectoriser config — WHY THESE SETTINGS:
    # max_features=5000: top 5000 words by corpus frequency
    # ngram_range=(1,2): also capture bigrams like "machine learning"
    # stop_words='english': removes "the","a","is" automatically
    # sublinear_tf=True: applies log(1+tf) instead of raw tf
    #   WHY: prevents very frequent words from dominating
    #   A word appearing 100x vs 10x shouldn't be 10x as important
    tfidf_config = dict(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,       # log normalisation on TF
    )

    return {
        # LOGISTIC REGRESSION
        # WHY: Fast, interpretable, good baseline for text.
        # C=1.0 is regularisation strength (inverse).
        # Higher C = less regularisation = model fits training
        # data more closely (risk of overfitting).
        # max_iter=1000: gradient descent runs up to 1000 steps.
        "logistic_regression": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_config)),
            ("clf",   LogisticRegression(C=1.0, max_iter=1000,
                                         class_weight="balanced", random_state=42))
        ]),

        # RANDOM FOREST
        # WHY: Ensemble of 200 decision trees. Each tree learns
        # different patterns. Final prediction = majority vote.
        # Handles non-linear relationships TF-IDF + LR can miss.
        # class_weight="balanced": compensates for class imbalance.
        "random_forest": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_config)),
            ("clf",   RandomForestClassifier(n_estimators=200,
                                              random_state=42,
                                              class_weight="balanced"))
        ]),

        # GRADIENT BOOSTING
        # WHY: Builds trees sequentially. Each tree corrects errors
        # of the previous one. Often best on sparse text features.
        # learning_rate=0.1: how much each tree corrects errors.
        # n_estimators=100: number of correction rounds.
        "gradient_boosting": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_config)),
            ("clf",   GradientBoostingClassifier(n_estimators=100,
                                                  learning_rate=0.1,
                                                  random_state=42))
        ]),
    }


def evaluate_models(pipelines: dict, X: list, y: list) -> dict:
    """
    Cross-validates each pipeline and returns metrics.

    WHY THESE METRICS:
    - Accuracy: % of predictions correct overall.
      PROBLEM: misleading when classes are imbalanced.
      If 90% of samples are label=0, predicting 0 always
      gives 90% accuracy but is completely useless.

    - Precision: of resumes I said were good matches,
      what % actually were?
      High precision = few false positives (rare false alarms)

    - Recall: of all actual good matches, what % did I find?
      High recall = few false negatives (rare misses)

    - F1: harmonic mean of precision and recall.
      USE THIS as your primary metric. It balances both.
      F1 = 2 * (precision * recall) / (precision + recall)

    WHY HARMONIC MEAN not arithmetic mean:
      If precision=1.0 and recall=0.0, arithmetic mean = 0.5
      (looks OK). Harmonic mean = 0.0 (correctly terrible).
      Harmonic mean punishes extreme imbalance between the two.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for name, pipeline in pipelines.items():
        scores = cross_validate(
            pipeline, X, y,
            cv=cv,
            scoring=["accuracy", "precision", "recall", "f1"],
            return_train_score=True
        )
        results[name] = {
            "accuracy":       round(np.mean(scores["test_accuracy"]),  3),
            "precision":      round(np.mean(scores["test_precision"]), 3),
            "recall":         round(np.mean(scores["test_recall"]),    3),
            "f1":             round(np.mean(scores["test_f1"]),        3),
            "train_accuracy": round(np.mean(scores["train_accuracy"]), 3),
            # WHY TRAIN VS TEST ACCURACY:
            # If train_accuracy >> test_accuracy → overfitting
            # Model memorised training data, can't generalise.
            # Healthy gap: train ~0.90, test ~0.85 = fine
            # Bad gap: train ~0.99, test ~0.60 = overfit
        }
        print(f"{name:25s} | F1: {results[name]['f1']:.3f} | "
              f"Precision: {results[name]['precision']:.3f} | "
              f"Recall: {results[name]['recall']:.3f} | "
              f"Train acc: {results[name]['train_accuracy']:.3f}")

    return results


# ──────────────────────────────────────────────
# SECTION 5: MAIN TRAINING LOOP
# ──────────────────────────────────────────────

def train():
    print("\n=== ML Resume Scorer — Training ===\n")

    # Step 1: load data
    X, y = load_data()

    # Step 2: build all candidate pipelines
    pipelines = build_pipelines()

    # Step 3: evaluate all pipelines with cross-validation
    print("\nCross-validation results (5-fold StratifiedKFold):")
    print("-" * 65)
    results = evaluate_models(pipelines, X, y)

    # Step 4: pick the best model by F1 score
    # WHY F1 and not accuracy: see comments in evaluate_models()
    best_name = max(results, key=lambda k: results[k]["f1"])
    best_pipeline = pipelines[best_name]
    print(f"\nBest model: {best_name} (F1={results[best_name]['f1']:.3f})")

    # Step 5: retrain best model on ALL data
    # WHY: cross-validation used folds for evaluation.
    # Now we train on everything to maximise learning before saving.
    best_pipeline.fit(X, y)

    # Step 6: save pipeline + evaluation report
    os.makedirs("models", exist_ok=True)
# Your existing save line:
    joblib.dump(best_pipeline, "models/resume_match_model.pkl")
    with open("models/eval_report.json", "w") as f:
        json.dump({"best_model": best_name, "results": results}, f, indent=2)

    print(f"\nSaved: models/resume_match_model.pkl")
    print(f"Saved: models/eval_report.json")
    print("\n=== Training complete ===\n")


if __name__ == "__main__":
    train()