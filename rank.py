"""
rank.py — Two-stage candidate ranking pipeline for the Redrob Hackathon.

Stage 1: Stream all 100k candidates through hard filters (honeypot detection,
         trap-title removal, location blocks) then rank by lightweight heuristic
         score to select the top 1,500 for deep scoring.

Stage 2: Dense semantic embedding with all-MiniLM-L6-v2 (cached offline),
         combined with experience, skills (recency-decayed), employer quality,
         location, and a multiplicative behavioral modifier from Redrob signals.

Compute constraints (must satisfy all):
  - Runtime  <=  5 minutes wall-clock (CPU only)
  - Memory   <= 16 GB RAM
  - No GPU   (torch CPU threads capped at 4)
  - No network during ranking (model cached in ./model_cache)

Usage:
  python rank.py --candidates ./candidates.jsonl --out ./submission.csv
"""

import os
import sys
import json
import math
import datetime
import argparse

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# CPU thread cap
torch.set_num_threads(4)

# Reference date
CURRENT_DATE = datetime.date(2026, 6, 18)


# =============================================================================
# CONSTANTS
# =============================================================================

FOUNDING_RULES = {
    "Krutrim":     {"min_start_year": 2023, "max_duration_months": 30},
    "Sarvam AI":   {"min_start_year": 2023, "max_duration_months": 35},
    "CRED":        {"min_start_year": 2018, "max_duration_months": 91},
    "Aganitha":    {"min_start_year": 2020, "max_duration_months": 78},
    "Glance":      {"min_start_year": 2019, "max_duration_months": 87},
    "Rephrase.ai": {"min_start_year": 2019, "max_duration_months": 90},
}

TRAP_TITLES = {
    "hr manager", "talent acquisition", "recruiter",
    "accountant", "finance analyst", "cfo",
    "marketing manager", "brand manager", "growth manager",
    "graphic designer", "ui designer",
    "customer support", "operations manager", "sales executive",
    "logistics coordinator", "supply chain manager",
}

CONSULTING_FIRMS = {
    "tcs", "infosys", "wipro", "accenture", "capgemini",
    "cognizant", "tech mahindra", "hcl", "genpact", "mphasis",
}

PRODUCT_TIERS = {
    "google": 1.00, "meta": 1.00, "openai": 1.00, "cohere": 1.00,
    "anthropic": 1.00, "deepmind": 1.00, "juspay": 1.00,
    "cred": 1.00, "phonepe": 1.00,
    "flipkart": 0.80, "swiggy": 0.80, "zomato": 0.80, "razorpay": 0.80,
    "postman": 0.80, "browserstack": 0.80, "freshworks": 0.80,
    "dream11": 0.80, "meesho": 0.80, "unacademy": 0.80,
    "nykaa": 0.75, "paytm": 0.75, "policybazaar": 0.75,
    "vedantu": 0.70, "upgrad": 0.70, "observe.ai": 0.75,
    "verloop": 0.70, "adobe": 0.80, "microsoft": 0.85,
    "netflix": 0.85, "amazon": 0.80,
}

SKILL_CLUSTERS = {
    "vector_db":        {"pinecone", "weaviate", "qdrant", "milvus", "faiss",
                         "chroma", "pgvector", "opensearch"},
    "embedding_models": {"sentence-transformers", "text-embedding-ada", "e5",
                         "bge", "all-minilm", "sentence transformers"},
    "nlp_ir":           {"information retrieval", "bm25", "elasticsearch",
                         "solr", "rag", "colbert", "nlp", "search"},
    "ml_training":      {"pytorch", "tensorflow", "jax", "keras",
                         "scikit-learn", "xgboost", "deep learning"},
    "mlops_deploy":     {"docker", "kubernetes", "fastapi", "ray serve",
                         "triton", "mlflow", "dvc"},
}

CV_SPEECH_SKILLS = {
    "computervision", "imageclassification", "objectdetection",
    "speechrecognition", "tts", "stt", "robotics",
}
NLP_IR_SKILLS = {
    "nlp", "informationretrieval", "search", "rag",
    "embeddings", "sentencetransformers", "vectorsearch",
}
CORE_ML_SKILLS   = {"pytorch", "tensorflow", "jax", "keras", "scikitlearn", "deeplearning"}
SHALLOW_AI_SKILLS = {"langchain", "openai", "chatgpt", "promptengineering"}


def _clean(s: str) -> str:
    return "".join(c for c in s.lower() if c.isalnum())


CLEAN_CLUSTERS = {
    cluster: {_clean(s) for s in skills}
    for cluster, skills in SKILL_CLUSTERS.items()
}
ALL_CRITICAL_CLEAN = set().union(*CLEAN_CLUSTERS.values())


# =============================================================================
# UTILITY HELPERS
# =============================================================================

def parse_date(s):
    if not s:
        return None
    try:
        return datetime.datetime.strptime(s, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def job_duration_months(job: dict) -> int:
    dur = job.get("duration_months")
    if dur is not None:
        return int(dur)
    start = parse_date(job.get("start_date"))
    end   = parse_date(job.get("end_date")) or CURRENT_DATE
    if start:
        return max(0, (end.year - start.year) * 12 + (end.month - start.month))
    return 0


def seniority_level(title: str) -> int:
    t = (title or "").lower()
    if any(w in t for w in ["director", "vp", "vice president", "head", "chief"]):
        return 6
    if any(w in t for w in ["principal", "lead", "architect"]):
        return 5
    if "staff" in t:
        return 4
    if any(w in t for w in ["senior", "sr.", "sr "]):
        return 3
    if any(w in t for w in ["junior", "associate", "jr.", "jr "]):
        return 1
    if any(w in t for w in ["intern", "trainee", "student"]):
        return 0
    return 2


def employer_tier_score(company: str, size: str, industry: str) -> float:
    comp = (company or "").lower()
    for key, val in PRODUCT_TIERS.items():
        if key in comp:
            return val
    for key in CONSULTING_FIRMS:
        if key in comp:
            return 0.25
    if any(w in comp for w in ["university", "lab", "research", "iit",
                                "iisc", "institute", "academia"]):
        return 0.20
    ind = (industry or "").lower()
    is_tech = any(w in ind for w in ["software", "fintech", "saas", "ai",
                                      "ml", "ecommerce", "internet", "technology"])
    if is_tech:
        return 0.60 if size in ("51-200", "201-500", "501-1000") else 0.50
    return 0.45


# =============================================================================
# STAGE 1 — HARD FILTERS + HEURISTIC SELECTION
# =============================================================================

def stage1_filter_and_score(candidates_file: str) -> list:
    survivors = []
    total = 0

    with open(candidates_file, "r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            total += 1
            c = json.loads(raw)

            profile  = c.get("profile", {})
            career   = c.get("career_history", [])
            skills   = c.get("skills", [])
            signals  = c.get("redrob_signals", {})
            exp = float(profile.get("years_of_experience", 0) or 0)

            # 1A. HONEYPOT — duration > total experience
            hp = False
            for job in career:
                if job_duration_months(job) / 12.0 > exp + 0.15:
                    hp = True
                    break
            if hp:
                continue

            # 1B. HONEYPOT — company founding date violations
            for job in career:
                comp  = job.get("company", "")
                start = job.get("start_date", "")
                dur   = job_duration_months(job)
                if comp in FOUNDING_RULES:
                    rule = FOUNDING_RULES[comp]
                    try:
                        sy = int(start[:4])
                        if sy < rule["min_start_year"] or dur > rule["max_duration_months"]:
                            hp = True
                            break
                    except (ValueError, IndexError):
                        pass
            if hp:
                continue

            # 1C. HONEYPOT — expert skill inflation
            expert_zero = sum(
                1 for s in skills
                if s.get("proficiency") == "expert" and s.get("duration_months", 0) == 0
            )
            if expert_zero >= 5:
                continue

            # 1D. HONEYPOT — multiple current roles
            if sum(1 for j in career if j.get("is_current")) > 1:
                continue

            # 1E. HONEYPOT — future start dates
            for job in career:
                sd = parse_date(job.get("start_date"))
                if sd and sd > CURRENT_DATE:
                    hp = True
                    break
                ed = parse_date(job.get("end_date"))
                if sd and ed and sd > ed:
                    hp = True
                    break
            if hp:
                continue

            # 1F. TRAP TITLE filter
            cur_title = profile.get("current_title", "").lower()
            if any(t in cur_title for t in TRAP_TITLES):
                continue

            # 1G. LOCATION hard block
            country          = profile.get("country", "").lower()
            willing_relocate = signals.get("willing_to_relocate", None)
            if "india" not in country:
                if not willing_relocate:
                    continue

            # 1H. HEURISTIC SCORE
            if 6.0 <= exp <= 8.0:
                h_exp = 1.00
            elif 5.0 <= exp < 6.0 or 8.0 < exp <= 9.5:
                h_exp = 0.85
            elif 4.0 <= exp < 5.0 or 9.5 < exp <= 12.0:
                h_exp = 0.60
            else:
                h_exp = 0.20

            cand_skill_names = {_clean(s.get("name", "")) for s in skills}
            h_skills = len(cand_skill_names & ALL_CRITICAL_CLEAN)

            h_title = 0.0
            if any(w in cur_title for w in [
                "machine learning", " ml ", "ai engineer", "artificial intelligence",
                "nlp", "data scientist", "deep learning", "search engineer",
                "recommendation", "applied scientist",
            ]):
                h_title = 1.0
            elif any(w in cur_title for w in ["software engineer", "backend", "developer"]):
                h_title = 0.5

            rr = float(signals.get("recruiter_response_rate", 0.5) or 0.5)
            h_score = h_exp * 2.0 + h_skills * 1.5 + h_title * 2.0 + rr * 1.0
            survivors.append((h_score, c))

    print(f"  Stage 1: {total} read → {len(survivors)} passed hard filters")
    survivors.sort(key=lambda x: x[0], reverse=True)
    top = [x[1] for x in survivors[:1500]]
    print(f"  Stage 1: top {len(top)} selected for deep scoring")
    return top