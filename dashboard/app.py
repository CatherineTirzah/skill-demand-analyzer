"""
Streamlit dashboard for the Skill Demand Analyzer.
Talks to the FastAPI service (default http://localhost:8000).
Run: streamlit run dashboard/app.py
"""
import os

import plotly.express as px
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Skill Demand Analyzer", layout="wide")
st.title("📊 AI-Driven Job Market Skill Demand Analyzer")

# --- Health / model status -------------------------------------------------
try:
    health = requests.get(f"{API_URL}/health", timeout=5).json()
    cols = st.columns(3)
    cols[0].metric("API status", health["status"])
    cols[1].metric("Model status", "✅ loaded" if health["model_id"] else "⚠️ not trained")
    cols[2].metric("Skills tracked", health.get("n_skills_tracked") or "—")
except requests.exceptions.ConnectionError:
    st.error(f"Can't reach the API at {API_URL}. Start it with: "
             f"`uvicorn api.main:app --reload --port 8000`")
    st.stop()

st.divider()

# --- Top in-demand skills ---------------------------------------------------
st.subheader("🔥 Top in-demand skills (latest period)")
n = st.slider("How many skills to show", 5, 30, 15)
try:
    top = requests.get(f"{API_URL}/skills/top", params={"n": n}, timeout=5).json()
    import pandas as pd
    top_df = pd.DataFrame(top["skills"])
    if not top_df.empty:
        fig = px.bar(top_df, x="demand_pct", y="skills", orientation="h",
                     title=f"Demand % of postings — {top['period']}",
                     labels={"demand_pct": "% of postings", "skills": "skill"})
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No skill data yet — run the training pipeline.")
except Exception as e:
    st.warning(f"Couldn't load top skills: {e}")

st.divider()

# --- Skill trend lookup ------------------------------------------------------
st.subheader("📈 Skill demand trend over time")
skill = st.text_input("Enter a skill (e.g. python, machine learning, docker)", "python")
if st.button("Show trend"):
    try:
        resp = requests.get(f"{API_URL}/skills/{skill}/trend", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            hist_df = pd.DataFrame(data["history"])
            fig = px.line(hist_df, x="period", y="demand_pct", markers=True,
                          title=f"'{data['skill']}' demand over time")
            st.plotly_chart(fig, use_container_width=True)
            if data["predicted_next_period_pct"] is not None:
                st.success(f"Predicted next-period demand: **{data['predicted_next_period_pct']}%** of postings")
        else:
            st.warning(resp.json().get("detail", "No data for this skill."))
    except Exception as e:
        st.warning(f"Couldn't load trend: {e}")

st.divider()

# --- Skill gap analysis -------------------------------------------------------
st.subheader("🎯 Your skill gap vs. the market")
user_skills_input = st.text_input("Your skills, comma-separated", "python, sql, excel")
if st.button("Check my skill gap"):
    skills_list = [s.strip() for s in user_skills_input.split(",") if s.strip()]
    try:
        resp = requests.post(f"{API_URL}/skill-gap",
                              json={"user_skills": skills_list, "top_n": 15}, timeout=5)
        gap = resp.json()
        c1, c2 = st.columns(2)
        c1.metric("Coverage of top skills", f"{gap['coverage_pct']}%")
        c2.write("**You have:** " + ", ".join(gap["have"]) if gap["have"] else "—")
        st.warning("**Missing / worth learning:** " + ", ".join(gap["missing"]))
    except Exception as e:
        st.warning(f"Couldn't compute skill gap: {e}")

st.divider()

# --- Monitoring / drift -------------------------------------------------------
st.subheader("🛰️ Monitoring — latest drift check")
try:
    drift = requests.get(f"{API_URL}/monitoring/latest", timeout=5).json()
    if drift.get("drift_detected"):
        st.error(f"⚠️ Drift detected — PSI={drift['psi_score']} (threshold {drift['threshold']})")
    else:
        st.success(f"✅ No significant drift — PSI={drift.get('psi_score', 'n/a')}")
    st.json(drift)
except Exception:
    st.info("No monitoring report yet — run `python -m src.monitoring`.")
