# Redrob Challenge — Team Purushartha

> Intelligent Candidate Discovery & Ranking · Senior AI Engineer JD · 100,000 candidates

## What this repo contains

| File | Purpose |
|---|---|
| `rank.py` | Main ranking pipeline (Stage 1 + Stage 2) |
| `app.py` | Streamlit sandbox UI |
| `validate_submission.py` | Official format validator |
| `submission_metadata.yaml` | Team metadata for portal submission |
| `requirements.txt` | Python dependencies |
| `model_cache/` | Offline all-MiniLM-L6-v2 weights (no network needed) |

---

## Quick start

### 1. Clone and set up

```bash
git clone https://github.com/tusharkr110/Redrob-Challenge.git
cd Redrob-Challenge
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Run the ranker

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

Expected output: