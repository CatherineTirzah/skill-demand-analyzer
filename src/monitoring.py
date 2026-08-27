"""
Continuous Monitoring (Project Activity 7's "CM"): tracks data quality and
skill-distribution drift between the current and previous data version, and
can trigger automated retraining when drift crosses the configured threshold
(Level 3 maturity, Project Activity 5).

Uses PSI (Population Stability Index) over the skill-frequency distribution —
a standard drift metric: PSI < 0.1 stable, 0.1-0.2 moderate shift, > 0.2 major
shift (e.g. "prompt engineering" suddenly appearing en masse, per Activity 7).
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data_versioning import get_latest_version, get_previous_version, load_version

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load(open(ROOT / "config.yaml"))
MONITORING_DIR = ROOT / CONFIG["paths"]["monitoring_dir"]
DRIFT_THRESHOLD = CONFIG["monitoring"]["drift_threshold"]


def _psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Population Stability Index between two distributions."""
    breakpoints = np.linspace(0, 100, buckets + 1)
    e_perc = np.histogram(expected, bins=np.percentile(expected, breakpoints))[0] / max(1, len(expected))
    a_perc = np.histogram(actual, bins=np.percentile(expected, breakpoints))[0] / max(1, len(actual))
    e_perc = np.where(e_perc == 0, 1e-4, e_perc)
    a_perc = np.where(a_perc == 0, 1e-4, a_perc)
    return float(np.sum((a_perc - e_perc) * np.log(a_perc / e_perc)))


def check_skill_drift() -> dict:
    """Compare skill_count distribution between the two most recent data versions."""
    latest = get_latest_version("skills_extracted", "unified")
    previous = get_previous_version("skills_extracted", "unified")

    if not latest or not previous:
        return {"status": "insufficient_history", "drift_detected": False}

    latest_df = load_version(latest)
    previous_df = load_version(previous)

    psi_score = _psi(
        previous_df["skill_count"].values.astype(float),
        latest_df["skill_count"].values.astype(float),
    )
    drift_detected = psi_score > DRIFT_THRESHOLD

    report = {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metric": "psi_skill_count",
        "psi_score": round(psi_score, 4),
        "threshold": DRIFT_THRESHOLD,
        "drift_detected": drift_detected,
        "previous_version": previous["hash"],
        "latest_version": latest["hash"],
    }
    return report


def check_data_quality(df: pd.DataFrame) -> dict:
    """Basic data-quality checks: missing fields, empty descriptions, dup rate."""
    return {
        "n_rows": len(df),
        "pct_missing_description": round(100 * df["job_description"].isna().mean(), 2),
        "pct_zero_skills": round(100 * (df.get("skill_count", pd.Series(dtype=float)) == 0).mean(), 2)
        if "skill_count" in df.columns else None,
    }


def run(auto_retrain: bool = False) -> dict:
    MONITORING_DIR.mkdir(parents=True, exist_ok=True)
    drift_report = check_skill_drift()

    report_path = MONITORING_DIR / f"drift_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.json"
    report_path.write_text(json.dumps(drift_report, indent=2))
    print(f"[monitoring] {drift_report}")

    if drift_report.get("drift_detected"):
        print(f"[monitoring] ALERT: skill-distribution drift detected "
              f"(PSI={drift_report['psi_score']} > {DRIFT_THRESHOLD})")
        if auto_retrain:
            print("[monitoring] auto-retrain enabled -> triggering src.train")
            from src import train
            train.run(skip_ingestion=False)
        else:
            print("[monitoring] auto-retrain disabled -> flagged for manual review "
                  "(run with --auto-retrain to retrain automatically)")

    return drift_report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto-retrain", action="store_true")
    args = parser.parse_args()
    run(auto_retrain=args.auto_retrain)
