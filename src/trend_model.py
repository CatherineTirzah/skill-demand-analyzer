"""
Skill-demand trend model (Iteration 3 from Project Activity 6).

Builds a monthly skill-frequency time series from extracted skills, then fits
a lightweight linear-regression-per-skill model to predict next-period demand
(% of postings mentioning that skill). Also exposes skill-gap comparison:
a user's current skills vs. the top in-demand skills right now.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load(open(ROOT / "config.yaml"))
PROCESSED_DIR = ROOT / CONFIG["paths"]["processed_dir"]
TOP_N = CONFIG["trend_model"]["top_n_skills"]


def build_skill_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """Long-format table: period, skill, posting_count, total_postings, demand_pct."""
    df = df.copy()
    df["skills"] = df["skills_json"].apply(json.loads)
    df["posted_date"] = pd.to_datetime(df["posted_date"], errors="coerce")
    df = df.dropna(subset=["posted_date"])
    df["period"] = df["posted_date"].dt.to_period("M").astype(str)

    exploded = df.explode("skills").dropna(subset=["skills"])
    counts = exploded.groupby(["period", "skills"]).size().reset_index(name="posting_count")
    totals = df.groupby("period").size().reset_index(name="total_postings")

    ts = counts.merge(totals, on="period")
    ts["demand_pct"] = ts["posting_count"] / ts["total_postings"] * 100
    return ts.sort_values(["skills", "period"]).reset_index(drop=True)


def top_skills(ts: pd.DataFrame, n: int = TOP_N) -> list[str]:
    latest_period = ts["period"].max()
    latest = ts[ts["period"] == latest_period]
    return latest.sort_values("demand_pct", ascending=False)["skills"].head(n).tolist()


def fit_trend_models(ts: pd.DataFrame, skills: list[str]) -> dict:
    """Fit one linear regression per skill: period_index -> demand_pct."""
    models = {}
    metrics = {}
    for skill in skills:
        sub = ts[ts["skills"] == skill].sort_values("period")
        if len(sub) < 3:
            continue
        X = np.arange(len(sub)).reshape(-1, 1)
        y = sub["demand_pct"].values

        split = max(1, int(len(sub) * (1 - CONFIG["trend_model"]["test_size"])))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = LinearRegression().fit(X_train, y_train)
        if len(X_test) > 0:
            mae = mean_absolute_error(y_test, model.predict(X_test))
        else:
            mae = mean_absolute_error(y_train, model.predict(X_train))

        models[skill] = model
        metrics[skill] = {"mae": float(mae), "slope": float(model.coef_[0]), "n_periods": len(sub)}
    return {"models": models, "metrics": metrics}


def predict_next_period(models: dict, ts: pd.DataFrame, skill: str) -> float | None:
    if skill not in models:
        return None
    sub = ts[ts["skills"] == skill]
    next_index = len(sub)
    pred = models[skill].predict([[next_index]])[0]
    return float(np.clip(pred, 0, 100))


def skill_gap(user_skills: list[str], top_in_demand: list[str]) -> dict:
    user_set = {s.lower() for s in user_skills}
    top_set = set(top_in_demand)
    return {
        "have": sorted(user_set & top_set),
        "missing": sorted(top_set - user_set),
        "coverage_pct": round(100 * len(user_set & top_set) / max(1, len(top_set)), 1),
    }


if __name__ == "__main__":
    df = pd.read_csv(PROCESSED_DIR / "postings_with_skills.csv")
    ts = build_skill_timeseries(df)
    ts.to_csv(PROCESSED_DIR / "skill_timeseries.csv", index=False)
    skills = top_skills(ts)
    result = fit_trend_models(ts, skills)
    print(f"[trend_model] trained {len(result['models'])} per-skill trend models")
    for skill, m in list(result["metrics"].items())[:5]:
        print(f"  {skill}: slope={m['slope']:.3f} mae={m['mae']:.3f}")
