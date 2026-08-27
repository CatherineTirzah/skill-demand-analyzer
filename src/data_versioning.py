"""
Lightweight data versioning (DVC-lite).

Every time a dataset is ingested or processed, we hash its content, copy a
snapshot into data/versions/<hash>.csv, and append a record to
data/versions/registry.json. This gives us:
  - reproducibility (know exactly which data trained which model)
  - drift comparison (diff current vs previous version's skill distribution)
  - audit trail (who/when/what changed)

For a real production setup, swap this for DVC + remote storage (S3/GCS) —
the interface (register_version / get_latest_version / diff_versions) stays
the same so nothing downstream needs to change.
"""
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

CONFIG = yaml.safe_load(open(Path(__file__).resolve().parents[1] / "config.yaml"))
VERSIONS_DIR = Path(__file__).resolve().parents[1] / CONFIG["paths"]["versions_dir"]
REGISTRY_PATH = VERSIONS_DIR / "registry.json"


def _load_registry() -> list:
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text())
    return []


def _save_registry(registry: list) -> None:
    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2))


def _hash_df(df: pd.DataFrame) -> str:
    payload = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.sha256(payload).hexdigest()[:16]


def register_version(df: pd.DataFrame, stage: str, source: str, notes: str = "") -> dict:
    """
    Snapshot a dataframe and record it in the version registry.
    stage: 'raw' | 'processed' | 'skills_extracted'
    source: e.g. 'linkedin', 'naukri', 'indeed_ds', 'unified'
    """
    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    version_hash = _hash_df(df)
    filename = f"{stage}__{source}__{version_hash}.csv"
    filepath = VERSIONS_DIR / filename

    is_new = not filepath.exists()
    if is_new:
        df.to_csv(filepath, index=False)

    registry = _load_registry()
    record = {
        "hash": version_hash,
        "stage": stage,
        "source": source,
        "rows": len(df),
        "columns": list(df.columns),
        "file": filename,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "notes": notes,
        "is_new_content": is_new,
    }
    registry.append(record)
    _save_registry(registry)
    print(f"[data_versioning] registered {stage}/{source} -> {version_hash} "
          f"({'new' if is_new else 'duplicate content, reused snapshot'}, {len(df)} rows)")
    return record


def get_latest_version(stage: str, source: str) -> dict | None:
    registry = _load_registry()
    matches = [r for r in registry if r["stage"] == stage and r["source"] == source]
    return matches[-1] if matches else None


def get_previous_version(stage: str, source: str) -> dict | None:
    registry = _load_registry()
    matches = [r for r in registry if r["stage"] == stage and r["source"] == source]
    return matches[-2] if len(matches) >= 2 else None


def load_version(record: dict) -> pd.DataFrame:
    return pd.read_csv(VERSIONS_DIR / record["file"])


def diff_versions(stage: str, source: str) -> dict:
    """Compare row counts between the latest and previous version of a dataset."""
    latest = get_latest_version(stage, source)
    previous = get_previous_version(stage, source)
    if not latest or not previous:
        return {"status": "insufficient_history"}
    return {
        "status": "ok",
        "previous_rows": previous["rows"],
        "latest_rows": latest["rows"],
        "row_delta": latest["rows"] - previous["rows"],
        "previous_hash": previous["hash"],
        "latest_hash": latest["hash"],
    }


if __name__ == "__main__":
    sample = pd.DataFrame({"a": [1, 2, 3]})
    register_version(sample, stage="raw", source="demo", notes="self-test")
    print(diff_versions("raw", "demo"))
