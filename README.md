# AI-Driven Job Market Skill Demand Analyzer
### MLOps Course Project (AD4V71)

An end-to-end MLOps system that ingests job postings (LinkedIn, Naukri, Indeed),
extracts required skills via NLP, tracks skill-demand trends over time, flags
skill gaps, and continuously retrains as the job market shifts.

This matches the lifecycle, architecture, and maturity levels documented in the
project notebook (Tasks 1–7): iterative SDLC, DevOps vs MLOps pipeline, CI/CD/CT/CM,
and Level 0 → Level 3 automation maturity.

---

## 1. Project Structure

```
skill-demand-analyzer/
├── data/
│   ├── raw/            # drop the 3 Kaggle CSVs here (see Datasets below)
│   ├── processed/       # cleaned + unified schema output
│   └── versions/        # data version snapshots (DVC-lite, hash-tracked)
├── src/
│   ├── data_ingestion.py    # loads & unifies LinkedIn/Naukri/Indeed datasets
│   ├── preprocessing.py     # cleaning, dedup, text normalization
│   ├── skill_taxonomy.py    # canonical skill list + synonym map
│   ├── skill_extraction.py  # NER + keyword-based skill extraction
│   ├── data_versioning.py   # hash-based dataset version registry
│   ├── trend_model.py       # skill-demand frequency/trend model
│   ├── train.py             # orchestrates the full training run, logs to MLflow
│   └── monitoring.py        # drift detection (data + prediction drift)
├── api/
│   ├── main.py           # FastAPI serving layer (predict, skill-gap, health)
│   └── schemas.py        # request/response models
├── dashboard/
│   └── app.py             # Streamlit dashboard (trends, skill gaps, monitoring)
├── pipelines/
│   └── pipeline_dag.py    # Airflow DAG: ingestion → extraction → train → monitor
├── tests/
│   └── test_skill_extraction.py
├── models/                # saved model artifacts + registry.json
├── monitoring/            # drift reports + alert logs (generated at runtime)
├── .github/workflows/ci-cd.yml  # CI (test) + CD (build/deploy) pipeline
├── Dockerfile
├── docker-compose.yml     # api + dashboard + mlflow, one command
├── requirements.txt
└── config.yaml
```

## 2. Datasets (drop into `data/raw/`)

| File | Source | Kaggle |
|---|---|---|
| `linkedin_job_postings.csv` | LinkedIn Job Postings | kaggle.com/datasets/arshkon/linkedin-job-postings |
| `naukri_job_postings.csv` | Naukri.com postings | kaggle.com/datasets/iqbal303/naukri-job-postings |
| `data_science_job_postings.csv` | Indeed / DS-focused | kaggle.com/datasets/asaniczka/data-science-job-postings-and-skills |

If any file is missing, `data_ingestion.py` raises a clear error telling you
which file to add — the pipeline only trains on real data now, no synthetic
fallback.

## 3. Quick Start

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 1. Ingest + version raw data
python -m src.data_ingestion

# 2. Clean + extract skills
python -m src.preprocessing
python -m src.skill_extraction

# 3. Train the skill-demand trend model (logs to MLflow, saves to models/)
python -m src.train

# 4. Serve
uvicorn api.main:app --reload --port 8000

# 5. Dashboard
streamlit run dashboard/app.py

# 6. Check for drift (run manually or via cron/Airflow)
python -m src.monitoring
```

Or run everything containerized:

```bash
docker-compose up --build
```
- API → http://localhost:8000/docs
- Dashboard → http://localhost:8501
- MLflow UI → http://localhost:5000

## 4. MLOps maturity implemented here

| Level | What this repo gives you |
|---|---|
| 0 — Manual | `src/*.py` runnable standalone, notebooks-style |
| 1 — ML Pipeline Automation | `pipelines/pipeline_dag.py` chains ingestion → extraction → train on a schedule; `data_versioning.py` hash-tracks every dataset version |
| 2 — CI/CD Automation | `.github/workflows/ci-cd.yml` runs tests on every push, builds Docker image, and can auto-deploy on merge |
| 3 — Full MLOps | `src/monitoring.py` detects drift and can trigger `src/train.py` automatically (see `--auto-retrain` flag); MLflow model registry compares new vs current model and blocks promotion if it underperforms |

## 5. Team roles (from Task 6)

- **Data & ML Lead** — `src/data_ingestion.py`, `preprocessing.py`, `skill_extraction.py`, `trend_model.py`
- **MLOps & Product Lead** — `pipelines/`, `api/`, `dashboard/`, `monitoring.py`, `.github/workflows/`
