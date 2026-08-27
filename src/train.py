"""
Full training run: ingestion -> preprocessing -> skill extraction -> trend
model -> log to MLflow -> compare against current production model -> only
promote if it's not worse (Level 3 "automated rollback if new model
underperforms", per Project Activity 5).
"""
import argparse
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml

from src import data_ingestion, preprocessing, skill_extraction
from src.trend_model import build_skill_timeseries, fit_trend_models, top_skills

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load(open(ROOT / "config.yaml"))
MODELS_DIR = ROOT / CONFIG["paths"]["models_dir"]
REGISTRY_PATH = MODELS_DIR / "registry.json"


def _load_registry() -> list:
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text())
    return []


def _save_registry(registry: list) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2))


def _avg_mae(metrics: dict) -> float:
    values = [m["mae"] for m in metrics.values()]
    return float(np.mean(values)) if values else float("inf")


def run(skip_ingestion: bool = False) -> dict:
    mlflow.set_tracking_uri(CONFIG["mlflow"]["tracking_uri"])
    mlflow.set_experiment(CONFIG["mlflow"]["experiment_name"])

    with mlflow.start_run():
        if not skip_ingestion:
            data_ingestion.ingest_all()
        preprocessing.run()
        df = skill_extraction.run()

        ts = build_skill_timeseries(df)
        skills = top_skills(ts)
        result = fit_trend_models(ts, skills)
        avg_mae = _avg_mae(result["metrics"])

        mlflow.log_param("n_skills_tracked", len(result["models"]))
        mlflow.log_param("top_n_config", CONFIG["trend_model"]["top_n_skills"])
        mlflow.log_metric("avg_mae", avg_mae)
        mlflow.log_metric("n_postings", len(df))

        model_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        model_dir = MODELS_DIR / model_id
        model_dir.mkdir(parents=True, exist_ok=True)

        with open(model_dir / "trend_models.pkl", "wb") as f:
            pickle.dump(result["models"], f)
        (model_dir / "metrics.json").write_text(json.dumps(result["metrics"], indent=2))
        ts.to_csv(model_dir / "skill_timeseries.csv", index=False)

        registry = _load_registry()
        current_prod = next((r for r in reversed(registry) if r["status"] == "production"), None)

        candidate = {
            "model_id": model_id,
            "avg_mae": avg_mae,
            "n_skills": len(result["models"]),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mlflow_run_id": mlflow.active_run().info.run_id,
            "status": "candidate",
        }

        # Promotion gate: only replace production model if it's not worse.
        if current_prod is None or avg_mae <= current_prod["avg_mae"]:
            candidate["status"] = "production"
            if current_prod:
                current_prod["status"] = "retired"
            mlflow.log_param("promoted", True)
            if current_prod:
                print(f"[train] new model {model_id} PROMOTED to production "
                      f"(avg_mae={avg_mae:.3f} vs previous {current_prod['avg_mae']:.3f})")
            else:
                print(f"[train] first model {model_id} promoted (avg_mae={avg_mae:.3f})")
        else:
            mlflow.log_param("promoted", False)
            print(f"[train] new model {model_id} REJECTED — worse than production "
                  f"(avg_mae={avg_mae:.3f} > {current_prod['avg_mae']:.3f}). "
                  f"Automated rollback: production model unchanged.")

        registry.append(candidate)
        _save_registry(registry)

    return candidate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-ingestion", action="store_true",
                         help="reuse existing data/processed files instead of re-scraping")
    args = parser.parse_args()
    run(skip_ingestion=args.skip_ingestion)
