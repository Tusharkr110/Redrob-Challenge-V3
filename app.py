"""
app.py — Streamlit sandbox for the Redrob Hackathon ranker.

Upload a small candidates JSONL file (e.g. sample_candidates.json),
click Run, and see the top ranked candidates live.

Satisfies hackathon sandbox requirement (Section 10.5).
"""

import io
import json
import tempfile
import os

import streamlit as st
import pandas as pd

from rank import run_ranking

st.set_page_config(
    page_title="Purushartha — Redrob Ranker",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Redrob Candidate Ranker — Team Purushartha")
st.markdown(
    "Upload a **candidates JSONL** file (one JSON object per line). "
    "The pipeline runs Stage 1 hard filters + Stage 2 semantic scoring "
    "and returns the **top 100** ranked candidates."
)

st.info(
    "This sandbox uses the same rank.py pipeline that produced the "
    "competition submission. Model weights are cached locally — "
    "no network calls happen during ranking.",
    icon="ℹ️",
)

uploaded = st.file_uploader(
    "Upload candidates.jsonl or sample_candidates.json",
    type=["jsonl", "json"],
)

if uploaded:
    raw_bytes = uploaded.read()

    # Support both JSONL (one object per line) and JSON array
    lines = []
    try:
        data = json.loads(raw_bytes.decode("utf-8"))
        if isinstance(data, list):
            lines = [json.dumps(obj) for obj in data]
        else:
            lines = [raw_bytes.decode("utf-8").strip()]
    except json.JSONDecodeError:
        lines = [l for l in raw_bytes.decode("utf-8").splitlines() if l.strip()]

    st.success(f"Loaded **{len(lines):,}** candidate records.")

    if st.button("🚀 Run Ranking", type="primary"):
        with st.spinner("Running pipeline ... Stage 1 → Stage 2 → Output"):

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
            ) as tmp_in:
                tmp_in.write("\n".join(lines))
                tmp_in_path = tmp_in.name

            tmp_out_path = tmp_in_path.replace(".jsonl", "_out.csv")

            try:
                run_ranking(tmp_in_path, tmp_out_path)
                df = pd.read_csv(tmp_out_path)

                st.subheader(f"Top {len(df)} candidates")
                st.dataframe(df, use_container_width=True, height=500)

                csv_bytes = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️  Download submission.csv",
                    data=csv_bytes,
                    file_name="submission.csv",
                    mime="text/csv",
                )

            except Exception as e:
                st.error(f"Pipeline error: {e}")
            finally:
                os.unlink(tmp_in_path)
                if os.path.exists(tmp_out_path):
                    os.unlink(tmp_out_path)
                detail = tmp_out_path.replace("_out.csv", "_out_detailed.csv")
                if os.path.exists(detail):
                    os.unlink(detail)

else:
    st.markdown(
        """
        **Quick start:**
        1. Use `sample_candidates.json` from the hackathon bundle (first 50 candidates).
        2. Upload it above.
        3. Click **Run Ranking**.

        For the full 100k run, use the CLI:
        """
    )

st.divider()
st.caption("Team Purushartha · Redrob Intelligent Candidate Discovery & Ranking Challenge")
