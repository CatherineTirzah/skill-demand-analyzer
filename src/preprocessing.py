"""
Preprocessing: cleans and normalizes the unified job postings dataset
(dedup, whitespace/HTML cleanup, dropping unusable rows), matching Task 1's
data-quality challenges (duplicates, formatting, noise, multiple sources
with different shapes).

Dates arrive already normalized to YYYY-MM-DD (or empty) from
data_ingestion.py, so no re-parsing happens here. Because several sources
have no free-text description (they ship pre-extracted skills instead), a
row is only dropped if it has NEITHER usable description text NOR any
pre-extracted skills — dropping on description-length alone (as a
single-source pipeline would) would wipe out most of the real data here.
"""
import re
from pathlib import Path

import pandas as pd
import yaml

from src.data_versioning import register_version

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load(open(ROOT / "config.yaml"))
PROCESSED_DIR = ROOT / CONFIG["paths"]["processed_dir"]

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _clean_text(text) -> str:
    if not isinstance(text, str):
        return ""
    text = _TAG_RE.sub(" ", text)          # strip HTML
    text = text.lower()
    text = _WS_RE.sub(" ", text).strip()    # collapse whitespace
    return text


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["job_title"] = df["job_title"].apply(_clean_text)
    df["job_description"] = df["job_description"].apply(_clean_text)
    df["location"] = df["location"].fillna("unknown").apply(_clean_text)
    df["skills_raw"] = df["skills_raw"].fillna("").astype(str)
    # posted_date already YYYY-MM-DD or NaN from ingestion — just normalize NaN->None
    df["posted_date"] = df["posted_date"].where(df["posted_date"].notna(), None)

    before = len(df)
    has_description = df["job_description"].str.len() > 20
    has_skills = df["skills_raw"].str.len() > 0
    has_title = df["job_title"].str.len() > 0
    df = df[has_title & (has_description | has_skills)]

    dedup_keys = ["job_title", "company", "location", "posted_date", "skills_raw"]
    df = df.drop_duplicates(subset=dedup_keys)
    after = len(df)
    print(f"[preprocessing] dropped {before - after} rows (unusable/duplicate); {after} remain")

    df = df.reset_index(drop=True)
    return df


def run() -> pd.DataFrame:
    raw_path = PROCESSED_DIR / "unified_raw.csv"
    if not raw_path.exists():
        raise FileNotFoundError("Run `python -m src.data_ingestion` first.")

    raw = pd.read_csv(raw_path)
    cleaned = clean(raw)

    register_version(cleaned, stage="processed", source="unified", notes="cleaned + deduped")

    out_path = PROCESSED_DIR / "cleaned_postings.csv"
    cleaned.to_csv(out_path, index=False)
    print(f"[preprocessing] wrote cleaned dataset -> {out_path}")

    by_source = cleaned["source"].value_counts().to_dict()
    print(f"[preprocessing] rows by source: {by_source}")
    return cleaned


if __name__ == "__main__":
    run()
