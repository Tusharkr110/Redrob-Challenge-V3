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

# =============================================================================
# STAGE 2 — DEEP SEMANTIC + MULTI-SIGNAL SCORING
# =============================================================================

def stage2_score(candidates: list, model: SentenceTransformer) -> list:

    jd_text = (
        "Senior AI Engineer Founding Team Series A talent intelligence platform "
        "Pune Noida India. Deep technical depth: embeddings, sentence-transformers, "
        "dense vector search, hybrid retrieval, information retrieval, RAG, "
        "vector databases (Pinecone, Weaviate, Qdrant, Milvus, FAISS), "
        "learning-to-rank, ranking evaluation NDCG MRR MAP, LLM fine-tuning, "
        "PyTorch, strong Python, product company, shipper mindset."
    )
    jd_emb = model.encode(jd_text, normalize_embeddings=True)

    texts = []
    for c in candidates:
        p  = c.get("profile", {})
        sk = c.get("skills", [])
        texts.append(
            f"{p.get('current_title','')} - {p.get('headline','')}. "
            f"{p.get('summary','')}. "
            f"Skills: {', '.join(s.get('name','') for s in sk)}."
        )

    print(f"  Stage 2: encoding {len(texts)} candidate texts ...")
    cand_embs = model.encode(
        texts,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    sem_scores = cand_embs @ jd_emb

    results = []

    for idx, c in enumerate(candidates):
        cid     = c.get("candidate_id")
        profile = c.get("profile", {})
        career  = c.get("career_history", [])
        skills  = c.get("skills", [])
        signals = c.get("redrob_signals", {})

        exp             = float(profile.get("years_of_experience", 0) or 0)
        cur_title_lower = profile.get("current_title", "").lower()
        cand_clean_skills = {_clean(s.get("name", "")) for s in skills}
        willing_relocate  = signals.get("willing_to_relocate", False)

        # S_exp
        if 6.0 <= exp <= 8.0:
            s_exp = 1.00
        elif 5.0 <= exp < 6.0 or 8.0 < exp <= 9.5:
            s_exp = 0.85
        elif 4.0 <= exp < 5.0 or 9.5 < exp <= 12.0:
            s_exp = 0.60
        elif 3.0 <= exp < 4.0 or 12.0 < exp <= 14.0:
            s_exp = 0.35
        else:
            s_exp = 0.10

        # S_loc
        country = profile.get("country", "").lower()
        city    = profile.get("location", "").lower()
        primary = any(c in city for c in [
            "pune", "noida", "delhi", "mumbai", "hyderabad",
            "gurgaon", "ncr", "bangalore", "bengaluru",
        ])
        secondary = any(c in city for c in ["chennai", "kolkata", "kochi"])

        if "india" in country:
            if primary:
                s_loc = 1.00
            elif secondary:
                s_loc = 1.00 if willing_relocate else 0.70
            else:
                s_loc = 0.85 if willing_relocate else 0.35
        else:
            s_loc = 0.55

        # S_emp
        total_months = sum(job_duration_months(j) for j in career)
        if total_months == 0:
            s_emp = 0.50
        else:
            s_emp = sum(
                employer_tier_score(
                    j.get("company", ""),
                    j.get("company_size", "1-10"),
                    j.get("industry", ""),
                ) * job_duration_months(j)
                for j in career
            ) / total_months

        # S_sem
        s_sem = float(max(0.0, sem_scores[idx]))

        # S_skill with recency decay
        skill_recency: dict = {}
        for s in skills:
            sclean = _clean(s.get("name", ""))
            latest = None
            for j in career:
                desc  = j.get("description", "").lower()
                title = j.get("title", "").lower()
                if sclean in _clean(title) or sclean in _clean(desc):
                    ed = parse_date(j.get("end_date"))
                    if j.get("is_current") or ed is None:
                        latest = CURRENT_DATE
                        break
                    elif latest is None or ed > latest:
                        latest = ed
            if latest:
                skill_recency[sclean] = max(0.0, (CURRENT_DATE - latest).days / 365.25)
            else:
                skill_recency[sclean] = 2.0

        prof_weight = {"expert": 1.0, "advanced": 0.8, "intermediate": 0.5, "beginner": 0.2}

        cluster_scores: dict = {}
        for cluster, clean_set in CLEAN_CLUSTERS.items():
            cluster_vals = []
            for s in skills:
                sc = _clean(s.get("name", ""))
                if sc not in clean_set:
                    continue
                pw  = prof_weight.get(s.get("proficiency", "beginner"), 0.2)
                dur = min(s.get("duration_months", 0), 36)
                recency_w = math.exp(-0.15 * skill_recency.get(sc, 2.0))
                cluster_vals.append(pw * recency_w * dur / 36.0)
            cluster_scores[cluster] = min(sum(cluster_vals), 1.0)

        s_skill = sum(cluster_scores.values()) / len(CLEAN_CLUSTERS)

        # ── PENALTIES ────────────────────────────────────────────────────────

        flags = []

        # Consulting fraction penalty
        consulting_months = sum(
            job_duration_months(j) for j in career
            if any(cf in j.get("company", "").lower() for cf in CONSULTING_FIRMS)
        )
        consulting_frac    = consulting_months / total_months if total_months else 0.0
        consulting_penalty = 1.0 - 0.5 * consulting_frac
        if consulting_frac > 0.0:
            flags.append("consulting")

        # Pure-academic penalty
        is_all_academic = all(
            any(w in j.get("company", "").lower()
                for w in ["university", "lab", "research", "iit", "iisc", "institute"])
            for j in career
        ) if career else False

        if is_all_academic:
            gh = float(signals.get("github_activity_score", 0) or 0)
            prod_kw   = ["production", "deploy", "scale", "shipped", "product company"]
            full_text = profile.get("summary", "").lower() + " " + " ".join(
                j.get("description", "").lower() for j in career
            )
            if gh < 50 and not any(w in full_text for w in prod_kw):
                continue

        # Shallow AI penalty
        ml_months = sum(
            job_duration_months(j) for j in career
            if any(w in j.get("title", "").lower() + j.get("description", "").lower()
                   for w in ["machine learning", " ml ", "ai ", "deep learning", "nlp"])
        )
        p_shallow = 1.0
        if ml_months < 12 and not (cand_clean_skills & CORE_ML_SKILLS) and \
                (cand_clean_skills & SHALLOW_AI_SKILLS):
            p_shallow = 0.05
            flags.append("shallow_ai")

        # Non-coding manager penalty
        is_mgr = any(w in cur_title_lower for w in
                     ["manager", "lead", "architect", "head", "director"])
        has_coding = False
        months_checked = 0
        for j in career:
            desc = j.get("description", "").lower()
            if any(w in desc for w in
                   ["code", "develop", "implement", "python", "pytorch",
                    "tensorflow", "engineer", "build", "program"]):
                has_coding = True
                break
            months_checked += job_duration_months(j)
            if months_checked >= 18:
                break
        p_noncoding = 1.0
        if is_mgr and not has_coding:
            p_noncoding = 0.10
            flags.append("noncoding_lead")

        # Title-chaser penalty
        eligible_jobs = [
            j for j in career
            if j.get("company_size", "") not in ("1-10", "11-50")
            and "layoff" not in j.get("description", "").lower()
        ]
        p_title_chaser = 1.0
        if len(eligible_jobs) >= 3:
            avg_tenure = sum(job_duration_months(j) for j in eligible_jobs) / len(eligible_jobs)
            if avg_tenure < 12.0:
                p_title_chaser = 0.20
                flags.append("title_chaser")

        # CV/Speech-only penalty
        p_cv_speech = 1.0
        if (cand_clean_skills & CV_SPEECH_SKILLS) and not (cand_clean_skills & NLP_IR_SKILLS):
            p_cv_speech = 0.05
            flags.append("cv_speech_only")

        # ── BONUSES ──────────────────────────────────────────────────────────

        gh_score = float(signals.get("github_activity_score", -1) or -1)
        gh_bonus = 0.0
        if gh_score >= 0:
            full_text = (profile.get("summary", "") + " " +
                         " ".join(j.get("description", "") for j in career)).lower()
            ir_kw = ["rag", "vector search", "embedding", "information retrieval",
                     "nlp", "large language model", "huggingface",
                     "sentence-transformers", "faiss", "langchain"]
            gh_bonus += 0.04 if any(w in full_text for w in ir_kw) else 0
            gh_bonus += 0.03 if gh_score > 50 else 0
            gh_bonus += 0.02 if gh_score > 30 else 0
            gh_bonus  = min(gh_bonus, 0.10)

        founding_bonus = 0.0
        full_lower = (profile.get("summary", "") + " " +
                      " ".join(j.get("description", "") for j in career)).lower()
        if any(j.get("company_size", "") in ("1-10", "11-50") for j in career):
            founding_bonus += 0.05
        if any(w in full_lower for w in
               ["founding member", "founding engineer", "first engineer"]):
            founding_bonus += 0.08
        if any(w in full_lower for w in
               ["built from scratch", "built ml infra", "built search from scratch"]):
            founding_bonus += 0.03
        founding_bonus = min(founding_bonus, 0.10)

        bonuses = gh_bonus + founding_bonus

        # Coverage multiplier
        n_clusters = sum(
            1 for _, cs in CLEAN_CLUSTERS.items()
            if cand_clean_skills & cs
        )
        coverage_mult = 0.85 + 0.15 * (n_clusters / len(CLEAN_CLUSTERS))

        # Trajectory multiplier
        trajectory_mult = 1.0
        dated_jobs = sorted(
            [(parse_date(j.get("start_date")), j) for j in career if j.get("start_date")],
            key=lambda x: x[0] or datetime.date.min,
        )
        if len(dated_jobs) >= 2 and exp > 0:
            sen_delta = (seniority_level(dated_jobs[-1][1].get("title", ""))
                         - seniority_level(dated_jobs[0][1].get("title", "")))
            trajectory_mult = max(0.90, min(1.15, 0.90 + 0.05 * (sen_delta / exp)))

        # Behavioral modifier
        rr = float(signals.get("recruiter_response_rate", 0.5) or 0.5)

        last_active   = parse_date(signals.get("last_active_date"))
        days_inactive = (CURRENT_DATE - last_active).days if last_active else 540
        if days_inactive <= 30:    recency_mult = 1.00
        elif days_inactive <= 90:  recency_mult = 0.85
        elif days_inactive <= 180: recency_mult = 0.70
        elif days_inactive <= 365: recency_mult = 0.50
        else:                      recency_mult = 0.20

        notice = signals.get("notice_period_days")
        if notice is None:    notice_mult = 0.60
        elif notice <= 30:    notice_mult = 1.00
        elif notice <= 60:    notice_mult = 0.70
        elif notice <= 90:    notice_mult = 0.40
        else:                 notice_mult = 0.20

        verified    = (signals.get("verified_email", False)
                       and signals.get("verified_phone", False))
        verify_mult = 1.0 if verified else 0.90
        otw_boost   = 1.10 if signals.get("open_to_work_flag") else 1.0
        m_beh       = rr * recency_mult * notice_mult * verify_mult * otw_boost

        # Final score
        base = (
            0.35 * s_sem
            + 0.30 * s_skill
            + 0.15 * s_exp
            + 0.12 * s_emp
            + 0.08 * s_loc
        )
        final = (
            (base + bonuses)
            * m_beh
            * coverage_mult
            * trajectory_mult
            * consulting_penalty
            * p_shallow
            * p_noncoding
            * p_title_chaser
            * p_cv_speech
        )

        # Reasoning string
        last_job  = career[-1] if career else {}
        company   = last_job.get("company", "—")
        job_title = last_job.get("title", "AI/ML Engineer")

        top_skills = sorted(
            [(s.get("name", ""), s.get("duration_months", 0))
             for s in skills if _clean(s.get("name", "")) in ALL_CRITICAL_CLEAN],
            key=lambda x: x[1], reverse=True,
        )[:3]
        skill_str = ", ".join(x[0] for x in top_skills) if top_skills else "Python, ML"

        loc_note   = (f"Willing to relocate from {profile.get('location','India')}"
                      if willing_relocate
                      else f"Based in {profile.get('location','India')}")
        notice_str = f"{notice}d" if notice is not None else "unknown"
        gh_note    = " · active GitHub contributor" if gh_bonus > 0.05 else ""

        concern = ""
        if exp > 12:
            concern = f" Note: {int(exp)}yr exp is above the JD 5-9yr range."
        elif exp < 4:
            concern = f" Note: {int(exp)}yr exp is below the JD target."
        if consulting_frac > 0.5:
            concern += " Consulting-heavy career."

        reasoning = (
            f"{profile.get('anonymized_name','Candidate')} · {int(exp)}yr · "
            f"{job_title} @ {company}. "
            f"Strong signal on {skill_str}. "
            f"{loc_note}. Notice: {notice_str}{gh_note}.{concern}"
        )

        
        

        results.append({
            "candidate_id": cid,
            "final_score":  round(final, 4),
            "s_sem":   round(s_sem,   4),
            "s_skill": round(s_skill, 4),
            "s_exp":   round(s_exp,   4),
            "s_emp":   round(s_emp,   4),
            "s_loc":   round(s_loc,   4),
            "m_beh":   round(m_beh,   4),
            "flags":   "|".join(flags) if flags else "none",
            "reasoning": reasoning,
        })

    print(f"  Stage 2: {len(results)} candidates scored")
    results.sort(key=lambda x: (-x["final_score"], x["candidate_id"]))
    return results

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_ranking(candidates_file: str, output_file: str) -> None:
    print(f"\n[rank.py] Reading candidates from: {candidates_file}")

    # Stage 1
    print("\n[Stage 1] Hard filters + heuristic selection")
    top_candidates = stage1_filter_and_score(candidates_file)

    # Load model from local cache — no network needed
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache")
    print(f"\n[Model] Loading SentenceTransformer from {model_path}")
    model = SentenceTransformer(model_path)

    # Stage 2
    print("\n[Stage 2] Semantic + multi-signal scoring")
    scored = stage2_score(top_candidates, model)

    # Write submission CSV — exactly 100 rows
    top100  = scored[:100]
    sub_rows = [
        {
            "candidate_id": r["candidate_id"],
            "rank":         i + 1,
            "score":        r["final_score"],
            "reasoning":    r["reasoning"],
        }
        for i, r in enumerate(top100)
    ]
    sub_df = pd.DataFrame(sub_rows)[["candidate_id", "rank", "score", "reasoning"]]
    sub_df.to_csv(output_file, index=False)
    print(f"\n[Output] Submission CSV -> {output_file}  ({len(sub_df)} rows)")

    # Write detailed CSV for debugging
    detail_file = output_file.replace(".csv", "_detailed.csv")
    detail_cols = ["candidate_id", "final_score", "s_sem", "s_skill",
                   "s_exp", "s_emp", "s_loc", "m_beh", "flags", "reasoning"]
    pd.DataFrame(scored[:100])[detail_cols].to_csv(detail_file, index=False)
    print(f"[Output] Detailed CSV  -> {detail_file}")
    print("\n[rank.py] Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rank 100k candidates against the Redrob Senior AI Engineer JD."
    )
    parser.add_argument("--candidates", required=True,
                        help="Path to candidates.jsonl (uncompressed)")
    parser.add_argument("--out", required=True,
                        help="Path for the output submission.csv")
    args = parser.parse_args()
    run_ranking(args.candidates, args.out)