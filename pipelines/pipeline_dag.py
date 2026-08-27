"""
Airflow DAG: ingestion -> preprocessing -> skill extraction -> train -> monitor.

The --run-now mode intentionally avoids importing Airflow so the local
pipeline can be tested on Windows. The actual Airflow DAG remains available
for Linux/WSL2/Docker environments.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _run_steps():
    """Run the complete pipeline sequentially for local testing."""
    from src import data_ingestion, monitoring, preprocessing, skill_extraction, train

    print("[pipeline_dag] STEP 1/5 — ingestion")
    data_ingestion.ingest_all()

    print("[pipeline_dag] STEP 2/5 — preprocessing")
    preprocessing.run()

    print("[pipeline_dag] STEP 3/5 — skill extraction")
    skill_extraction.run()

    print("[pipeline_dag] STEP 4/5 — training")
    train.run(skip_ingestion=True)

    print("[pipeline_dag] STEP 5/5 — monitoring")
    monitoring.run(auto_retrain=True)

    print("[pipeline_dag] COMPLETE — all pipeline steps finished successfully")


def _build_airflow_dag():
    """Create the real Airflow DAG when running in a supported environment."""
    from datetime import datetime, timedelta

    from airflow import DAG
    from airflow.operators.python import PythonOperator

    default_args = {
        "owner": "skill-demand-analyzer",
        "retries": 2,
        "retry_delay": timedelta(minutes=10),
    }

    with DAG(
        dag_id="skill_demand_pipeline",
        default_args=default_args,
        description="Weekly ingest -> extract -> train -> monitor pipeline",
        schedule="@weekly",
        start_date=datetime(2026, 1, 1),
        catchup=False,
        tags=["mlops", "skill-demand-analyzer"],
    ) as dag:

        def _ingest():
            from src import data_ingestion
            data_ingestion.ingest_all()

        def _preprocess():
            from src import preprocessing
            preprocessing.run()

        def _extract():
            from src import skill_extraction
            skill_extraction.run()

        def _train():
            from src import train
            train.run(skip_ingestion=True)

        def _monitor():
            from src import monitoring
            monitoring.run(auto_retrain=True)

        t1 = PythonOperator(
            task_id="ingest_data",
            python_callable=_ingest,
        )

        t2 = PythonOperator(
            task_id="preprocess_data",
            python_callable=_preprocess,
        )

        t3 = PythonOperator(
            task_id="extract_skills",
            python_callable=_extract,
        )

        t4 = PythonOperator(
            task_id="train_model",
            python_callable=_train,
        )

        t5 = PythonOperator(
            task_id="monitor_drift",
            python_callable=_monitor,
        )

        t1 >> t2 >> t3 >> t4 >> t5

    return dag


# Only construct the Airflow DAG when explicitly requested.
dag = None

if __name__ != "__main__":
    try:
        dag = _build_airflow_dag()
    except ImportError:
        dag = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-now",
        action="store_true",
        help="Run all pipeline steps sequentially without importing Airflow.",
    )
    args = parser.parse_args()

    if args.run_now:
        _run_steps()
    else:
        print(
            "[pipeline_dag] Use --run-now for local execution, "
            "or load this module through Airflow on Linux/WSL2/Docker."
        )