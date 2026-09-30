# AI-Driven Job Market Skill Demand Analyzer

### MLOps Course Project — AD4V71

An end-to-end **MLOps system for analyzing job-market skill demand** from job postings. The system processes job-posting data, extracts relevant technical skills using NLP, analyzes skill-demand trends, identifies skill gaps, and supports continuous model monitoring and retraining.

**Course:** MLOps
**Program:** B.Tech Artificial Intelligence and Data Science
**Institution:** Chennai Institute of Technology

### Team

* **Catherine Tirzah**
* **Preethikgha**

---

## 1. Project Overview

The **AI-Driven Job Market Skill Demand Analyzer** is designed to analyze job postings and identify the technical skills currently demanded in the job market.

The project follows an end-to-end MLOps workflow:

```text
Job Posting Data
       ↓
Data Ingestion
       ↓
Data Preprocessing
       ↓
Skill Extraction using NLP
       ↓
Skill Demand Analysis
       ↓
Trend Detection
       ↓
Skill Gap Analysis
       ↓
Model Training
       ↓
Monitoring & Drift Detection
       ↓
Continuous Retraining
```

The project demonstrates the application of MLOps concepts including data versioning, automated pipelines, model training, monitoring, CI/CD, and continuous retraining.

---

## 2. Key Features

* Job-posting data ingestion from multiple sources
* Data cleaning and preprocessing
* NLP-based skill extraction
* Skill taxonomy and synonym mapping
* Skill-demand frequency analysis
* Skill-demand trend tracking
* Skill-gap identification
* Dataset version tracking
* Machine learning model training
* MLflow experiment tracking
* Data and prediction drift monitoring
* FastAPI-based API
* Streamlit dashboard
* Apache Airflow pipeline orchestration
* Docker-based deployment
* Automated testing using GitHub Actions

---

## 3. Project Structure

```text
skill-demand-analyzer/

├── data/
│   ├── raw/                  # Raw job-posting datasets
│   ├── processed/            # Cleaned and processed data
│   └── versions/             # Dataset version snapshots
│
├── src/
│   ├── data_ingestion.py     # Data loading and schema unification
│   ├── preprocessing.py      # Data cleaning and preprocessing
│   ├── skill_taxonomy.py     # Skill list and synonym mapping
│   ├── skill_extraction.py   # NLP-based skill extraction
│   ├── data_versioning.py    # Dataset version tracking
│   ├── trend_model.py        # Skill-demand trend analysis
│   ├── train.py              # Model training pipeline
│   └── monitoring.py         # Drift detection and monitoring
│
├── api/
│   ├── main.py               # FastAPI application
│   └── schemas.py             # API request/response schemas
│
├── dashboard/
│   └── app.py                # Streamlit dashboard
│
├── pipelines/
│   └── pipeline_dag.py       # Airflow pipeline
│
├── tests/
│   └── test_skill_extraction.py
│
├── models/                   # Saved model artifacts
├── monitoring/               # Monitoring reports and logs
│
├── .github/
│   └── workflows/
│       └── ci-cd.yml         # CI/CD workflow
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── config.yaml
```

---

## 4. Technologies Used

| Category               | Technologies   |
| ---------------------- | -------------- |
| Programming Language   | Python         |
| Data Processing        | Pandas, NumPy  |
| NLP                    | spaCy          |
| Machine Learning       | Scikit-learn   |
| API                    | FastAPI        |
| Dashboard              | Streamlit      |
| Experiment Tracking    | MLflow         |
| Workflow Orchestration | Apache Airflow |
| Containerization       | Docker         |
| Testing                | Pytest         |
| CI/CD                  | GitHub Actions |

---

## 5. Dataset

The system works with job-posting datasets containing information such as:

* Job title
* Job description
* Required skills
* Job location
* Company information
* Other job-related attributes

The raw datasets are placed inside:

```text
data/raw/
```

The data ingestion module validates the required input files before processing.

---

## 6. Quick Start

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Download the spaCy Model

```bash
python -m spacy download en_core_web_sm
```

### Run Data Ingestion

```bash
python -m src.data_ingestion
```

### Run Preprocessing

```bash
python -m src.preprocessing
```

### Extract Skills

```bash
python -m src.skill_extraction
```

### Train the Model

```bash
python -m src.train
```

### Start the FastAPI Server

```bash
uvicorn api.main:app --reload --port 8000
```

### Start the Streamlit Dashboard

```bash
streamlit run dashboard/app.py
```

### Run Monitoring

```bash
python -m src.monitoring
```

---

## 7. Docker

The complete application can also be run using Docker:

```bash
docker-compose up --build
```

The main services can be accessed at:

| Service               | URL                        |
| --------------------- | -------------------------- |
| FastAPI               | http://localhost:8000      |
| FastAPI Documentation | http://localhost:8000/docs |
| Streamlit Dashboard   | http://localhost:8501      |
| MLflow                | http://localhost:5000      |

---

## 8. MLOps Workflow

The project demonstrates multiple levels of MLOps automation.

### Level 0 — Manual Execution

Individual Python modules can be executed independently for data ingestion, preprocessing, skill extraction, model training, and monitoring.

### Level 1 — Pipeline Automation

The Airflow DAG connects the major stages of the machine learning workflow:

```text
Ingestion
    ↓
Preprocessing
    ↓
Skill Extraction
    ↓
Training
    ↓
Monitoring
```

Dataset versions are also tracked using hash-based versioning.

### Level 2 — CI/CD Automation

GitHub Actions is used to automate testing and build-related processes when changes are pushed to the repository.

### Level 3 — Continuous MLOps

The monitoring component detects changes in data and model behavior. Based on configured conditions, the system can support model retraining when the job-market data changes significantly.

---

## 9. API

After starting the FastAPI server, interactive API documentation is available at:

```text
http://localhost:8000/docs
```

The API provides endpoints for functionality such as:

* Skill-demand analysis
* Skill-gap analysis
* Health checks
* Model-related operations

---

## 10. Dashboard

The Streamlit dashboard provides a visual interface for exploring:

* Skill-demand trends
* Frequently demanded skills
* Skill gaps
* Monitoring information
* Model-related insights

Run the dashboard using:

```bash
streamlit run dashboard/app.py
```

---

## 11. Project Objectives

The main objectives of this project are to:

1. Analyze job-market data using machine learning and NLP.
2. Identify technical skills demanded by employers.
3. Track changes in skill demand over time.
4. Identify potential skill gaps.
5. Build an automated ML pipeline.
6. Apply MLOps practices for model monitoring and retraining.
7. Provide accessible results through an API and interactive dashboard.

---

## 12. Team

### Catherine Tirzah

B.Tech Artificial Intelligence and Data Science
Chennai Institute of Technology

### Preethikgha

B.Tech Artificial Intelligence and Data Science
Chennai Institute of Technology

---

## 13. Course Information

**Course:** MLOps
**Course Code:** AD4V71
**Program:** B.Tech Artificial Intelligence and Data Science
**Institution:** Chennai Institute of Technology
