"""
Data ingestion: loads each real source dataset according to its own schema
(configured per-source in config.yaml) and unifies them into one common
table. Handles three real-world quirks found in the actual datasets:

  1. Relative date strings ("17 days ago", "a month ago", "3+ weeks ago")
     -> converted to real dates, either anchored to a known scrape
     timestamp (indian_tech) or a configured reference date (data_science).
  2. Pre-extracted skills columns (python-list-repr or comma-separated)
     -> parsed directly instead of re-running NLP on text that doesn't exist.
  3. LinkedIn's companion job_skills.csv join, and a role-keyword filter
     since the raw LinkedIn dump is a general jobs file, not tech-specific.

Raises FileNotFoundError with setup instructions if a required raw file is
missing. Sources with `enabled: false` in config.yaml are skipped entirely.
"""
import ast
import re
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml

from src.data_versioning import register_version

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load(open(ROOT / "config.yaml"))
RAW_DIR = ROOT / CONFIG["paths"]["raw_dir"]
ROLE_KEYWORDS = [k.lower() for k in CONFIG.get("role_filter_keywords", [])]

UNIFIED_COLUMNS = ["source", "job_title", "job_description", "skills_raw",
                    "company", "location", "posted_date"]

_RELATIVE_RE = re.compile(
    r"(a|an|\d+)\+?\s*(day|week|month|year)s?\s+ago", re.IGNORECASE
)


def _parse_relative_date(text: str, anchor: datetime) -> str | None:
    """'17 days ago' / 'a month ago' / '3+ weeks ago' -> 'YYYY-MM-DD', else None."""
    if not isinstance(text, str):
        return None
    match = _RELATIVE_RE.search(text)
    if not match:
        return None
    qty_raw, unit = match.groups()
    qty = 1 if qty_raw.lower() in ("a", "an") else int(qty_raw)
    unit = unit.lower()
    days = {"day": 1, "week": 7, "month": 30, "year": 365}[unit] * qty
    return (anchor - timedelta(days=days)).strftime("%Y-%m-%d")


def _parse_skills(value, fmt: str) -> list[str]:
    """Parse a pre-extracted skills field into a clean list of lowercase strings."""
    if not isinstance(value, str) or not value.strip() or value.strip().lower() in ("not available", "[]", "none"):
        return []
    if fmt == "python_list":
        try:
            parsed = ast.literal_eval(value)
            return [str(s).strip().lower() for s in parsed if str(s).strip()]
        except (ValueError, SyntaxError):
            return []
    if fmt == "comma":
        return [s.strip().lower() for s in value.split(",") if s.strip()]
    return []


def _matches_role_filter(title: str) -> bool:
    if not ROLE_KEYWORDS:
        return True
    title = (title or "").lower()
    return any(kw in title for kw in ROLE_KEYWORDS)


def _load_linkedin(src_cfg: dict) -> pd.DataFrame:
    filepath = RAW_DIR / src_cfg["file"]
    if not filepath.exists():
        raise FileNotFoundError(f"\n[data_ingestion] Missing required dataset: {filepath}")

    n = src_cfg.get("sample_size")
    df = pd.read_csv(filepath, low_memory=False)
    if n and len(df) > n:
        df = df.sample(n=n, random_state=42)

    if src_cfg.get("role_filter"):
        before = len(df)
        df = df[df[src_cfg["title_column"]].apply(_matches_role_filter)]
        print(f"[data_ingestion] linkedin: role filter kept {len(df)}/{before} rows")

    skills_map = {}
    join_cfg = src_cfg.get("join")
    if join_cfg:
        join_path = RAW_DIR / join_cfg["file"]
        if join_path.exists():
            skills_df = pd.read_csv(join_path, low_memory=False,
                                     usecols=[join_cfg["right_key"], join_cfg["skills_column"]])
            skills_map = dict(zip(skills_df[join_cfg["right_key"]], skills_df[join_cfg["skills_column"]]))
            print(f"[data_ingestion] linkedin: loaded {len(skills_map)} skill mappings from {join_cfg['file']}")
        else:
            print(f"[data_ingestion] linkedin: {join_cfg['file']} not found — "
                  f"proceeding with title-only signal (weak). Add it to data/raw/ for real skill coverage.")

    def _row_skills(row):
        raw = skills_map.get(row[join_cfg["left_key"]]) if join_cfg else None
        return _parse_skills(raw, join_cfg["skills_format"]) if join_cfg else []

    unified = pd.DataFrame({
        "source": "linkedin",
        "job_title": df[src_cfg["title_column"]],
        "job_description": "",
        "skills_raw": df.apply(_row_skills, axis=1) if join_cfg else [[] for _ in range(len(df))],
        "company": df.get(src_cfg["company_column"], ""),
        "location": df.get(src_cfg["location_column"], ""),
        "posted_date": pd.to_datetime(df.get(src_cfg["date_column"]), errors="coerce").dt.strftime("%Y-%m-%d"),
    })
    return unified


def _load_generic(source_key: str, src_cfg: dict) -> pd.DataFrame:
    filepath = RAW_DIR / src_cfg["file"]
    if not filepath.exists():
        raise FileNotFoundError(
            f"\n[data_ingestion] Missing required dataset: {filepath}\n"
            f"  Place '{src_cfg['file']}' in data/raw/ before running."
        )
    df = pd.read_csv(filepath, low_memory=False)

    # --- dates ---
    if src_cfg["date_mode"] == "relative":
        anchor = datetime.fromisoformat(src_cfg["reference_date"])
        posted_date = df[src_cfg["relative_date_column"]].apply(lambda v: _parse_relative_date(v, anchor))
    elif src_cfg["date_mode"] == "computed_from_scrape":
        scraped = pd.to_datetime(df[src_cfg["scrape_date_column"]], errors="coerce")
        days_since = pd.to_numeric(df[src_cfg["days_since_column"]], errors="coerce")
        posted_date = (scraped - pd.to_timedelta(days_since, unit="D")).dt.strftime("%Y-%m-%d")
    else:  # absolute
        posted_date = pd.to_datetime(df[src_cfg["date_column"]], errors="coerce").dt.strftime("%Y-%m-%d")

    # --- skills ---
    skills_col = src_cfg.get("skills_column")
    if skills_col and skills_col in df.columns:
        skills_raw = df[skills_col].apply(lambda v: _parse_skills(v, src_cfg["skills_format"]))
    else:
        skills_raw = [[] for _ in range(len(df))]

    unified = pd.DataFrame({
        "source": source_key,
        "job_title": df.get(src_cfg["title_column"], ""),
        "job_description": df.get(src_cfg.get("description_column"), "") if src_cfg.get("description_column") else "",
        "skills_raw": skills_raw,
        "company": df.get(src_cfg["company_column"], ""),
        "location": df.get(src_cfg["location_column"], ""),
        "posted_date": posted_date,
    })
    return unified


def ingest_all() -> pd.DataFrame:
    frames = []
    for source_key, src_cfg in CONFIG["sources"].items():
        if not src_cfg.get("enabled", True):
            print(f"[data_ingestion] skipping '{source_key}' (disabled in config.yaml)")
            continue

        df = _load_linkedin(src_cfg) if source_key == "linkedin" else _load_generic(source_key, src_cfg)
        register_version(df.assign(skills_raw=df["skills_raw"].astype(str)),
                          stage="raw", source=source_key, notes="ingested this run")
        frames.append(df)

    if not frames:
        raise RuntimeError("[data_ingestion] No sources enabled — check config.yaml `sources.*.enabled`.")

    unified = pd.concat(frames, ignore_index=True)[UNIFIED_COLUMNS]
    register_version(unified.assign(skills_raw=unified["skills_raw"].astype(str)),
                      stage="raw", source="unified", notes="concat of all enabled sources")

    out_path = ROOT / CONFIG["paths"]["processed_dir"] / "unified_raw.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    unified.assign(skills_raw=unified["skills_raw"].apply(lambda s: "|".join(s))).to_csv(out_path, index=False)
    print(f"[data_ingestion] wrote {len(unified)} unified rows -> {out_path}")
    return unified


if __name__ == "__main__":
    ingest_all()
