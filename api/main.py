"""
FastAPI serving layer for the skill-demand model + dashboard data.

Endpoints:
  GET  /health                  -> liveness + which model is in production
  GET  /skills/top              -> current top in-demand skills
  GET  /skills/{skill}/trend    -> historical demand + next-period prediction
  POST /skill-gap               -> compare a user's skills vs. top in-demand
  GET  /monitoring/latest       -> most recent drift report
"""
import json
import pickle
from pathlib import Path

import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException

from api.schemas import (HealthResponse, SkillGapRequest, SkillGapResponse,
                          SkillTrendResponse, TopSkillsResponse, TrendPoint)
from src.trend_model import predict_next_period, skill_gap, top_skills

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load(open(ROOT / "config.yaml"))
MODELS_DIR = ROOT / CONFIG["paths"]["models_dir"]
MONITORING_DIR = ROOT / CONFIG["paths"]["monitoring_dir"]
PROCESSED_DIR = ROOT / CONFIG["paths"]["processed_dir"]

app = FastAPI(title="Skill Demand Analyzer API", version="1.0.0")

_state = {"models": None, "ts": None, "model_id": None}


def _get_production_record() -> dict | None:
    registry_path = MODELS_DIR / "registry.json"
    if not registry_path.exists():
        return None
    registry = json.loads(registry_path.read_text())
    return next((r for r in reversed(registry) if r["status"] == "production"), None)


def _load_production_model():
    record = _get_production_record()
    if record is None:
        return None, None, None

    model_dir = MODELS_DIR / record["model_id"]
    with open(model_dir / "trend_models.pkl", "rb") as f:
        models = pickle.load(f)
    ts = pd.read_csv(model_dir / "skill_timeseries.csv")
    return models, ts, record["model_id"]


@app.on_event("startup")
def load_model():
    models, ts, model_id = _load_production_model()
    _state["models"], _state["ts"], _state["model_id"] = models, ts, model_id
    print(f"[api] loaded production model: {model_id}")


@app.get("/health", response_model=HealthResponse)
def health():
    record = _get_production_record()
    return HealthResponse(
        status="ok",
        model_status="loaded" if _state["models"] else "no model trained yet — run `python -m src.train`",
        model_id=_state["model_id"],
        n_skills_tracked=len(_state["models"]) if _state["models"] else None,
    )


@app.get("/skills/top", response_model=TopSkillsResponse)
def get_top_skills(n: int = 15):
    if _state["ts"] is None:
        raise HTTPException(503, "Model not loaded — train the model first.")
    ts = _state["ts"]
    latest_period = ts["period"].max()
    latest = ts[ts["period"] == latest_period].sort_values("demand_pct", ascending=False).head(n)
    return TopSkillsResponse(
        period=latest_period,
        skills=latest[["skills", "demand_pct", "posting_count"]].to_dict(orient="records"),
    )


@app.get("/skills/{skill}/trend", response_model=SkillTrendResponse)
def get_skill_trend(skill: str):
    if _state["ts"] is None:
        raise HTTPException(503, "Model not loaded — train the model first.")
    ts = _state["ts"]
    sub = ts[ts["skills"] == skill.lower()].sort_values("period")
    if sub.empty:
        raise HTTPException(404, f"No trend data for skill '{skill}'")

    history = [TrendPoint(period=r.period, demand_pct=round(r.demand_pct, 2)) for r in sub.itertuples()]
    prediction = predict_next_period(_state["models"], ts, skill.lower())
    return SkillTrendResponse(
        skill=skill.lower(),
        history=history,
        predicted_next_period_pct=round(prediction, 2) if prediction is not None else None,
    )


@app.post("/skill-gap", response_model=SkillGapResponse)
def get_skill_gap(request: SkillGapRequest):
    if _state["ts"] is None:
        raise HTTPException(503, "Model not loaded — train the model first.")
    top = top_skills(_state["ts"], n=request.top_n)
    gap = skill_gap(request.user_skills, top)
    return SkillGapResponse(**gap, top_in_demand=top)


@app.get("/monitoring/latest")
def latest_drift_report():
    if not MONITORING_DIR.exists():
        raise HTTPException(404, "No monitoring reports yet.")
    reports = sorted(MONITORING_DIR.glob("drift_*.json"))
    if not reports:
        raise HTTPException(404, "No monitoring reports yet.")
    return json.loads(reports[-1].read_text())
