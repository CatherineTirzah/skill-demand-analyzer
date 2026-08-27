"""
Canonical skill taxonomy + synonym map, addressing Task 1's "label ambiguity"
challenge (no universal standard for what counts as a distinct "skill").

Extend SKILL_TAXONOMY as your NLP pipeline discovers new terms (e.g. "prompt
engineering" showing up as a sudden new in-demand skill, per Project Activity 7).
"""

SKILL_TAXONOMY = {
    # canonical_name: [synonyms / surface forms found in postings]
    "python": ["python", "python3", "py"],
    "sql": ["sql", "mysql", "postgresql", "postgres", "t-sql"],
    "machine learning": ["machine learning", "ml", "statistical learning"],
    "deep learning": ["deep learning", "dl", "neural networks"],
    "tensorflow": ["tensorflow", "tf"],
    "pytorch": ["pytorch", "torch"],
    "nlp": ["nlp", "natural language processing", "text mining"],
    "computer vision": ["computer vision", "cv", "opencv"],
    "aws": ["aws", "amazon web services"],
    "gcp": ["gcp", "google cloud"],
    "azure": ["azure", "microsoft azure"],
    "docker": ["docker", "containerization"],
    "kubernetes": ["kubernetes", "k8s"],
    "airflow": ["airflow", "apache airflow"],
    "mlflow": ["mlflow"],
    "spark": ["spark", "pyspark", "apache spark"],
    "tableau": ["tableau"],
    "power bi": ["power bi", "powerbi"],
    "excel": ["excel", "ms excel"],
    "git": ["git", "github", "gitlab", "version control"],
    "ci/cd": ["ci/cd", "cicd", "continuous integration", "continuous deployment"],
    "statistics": ["statistics", "statistical analysis", "hypothesis testing"],
    "prompt engineering": ["prompt engineering", "prompt design"],
    "generative ai": ["generative ai", "genai", "gen ai"],
    "llm": ["llm", "large language model", "large language models"],
    "data engineering": ["data engineering", "etl", "elt"],
    "r": ["r programming", " r,", " r "],
    "java": ["java"],
    "scala": ["scala"],
}

_SURFACE_TO_CANONICAL = {
    surface: canonical
    for canonical, surfaces in SKILL_TAXONOMY.items()
    for surface in surfaces
}


def canonicalize(surface_form: str) -> str:
    return _SURFACE_TO_CANONICAL.get(surface_form.strip().lower(), surface_form.strip().lower())


def all_surface_forms() -> list[str]:
    return list(_SURFACE_TO_CANONICAL.keys())
