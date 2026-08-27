"""
Skill extraction: uses whichever signal each row actually has, in order of
reliability:
  1. Pre-extracted skills (from the dataset author, in `skills_raw`) —
     canonicalized through the taxonomy's synonym map. Most reliable.
  2. NLP (spaCy PhraseMatcher, falls back to regex) over `job_description`,
     when real free text is available.
  3. NLP over `job_title` only, as a last resort (weak signal — used for
     rows like LinkedIn's without a companion skills file).

This is Iteration 2 from Project Activity 6, adapted for datasets that don't
all provide the same kind of signal.
"""
import json
import re
from pathlib import Path

import pandas as pd
import yaml

from src.data_versioning import register_version
from src.skill_taxonomy import all_surface_forms, canonicalize

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load(open(ROOT / "config.yaml"))
PROCESSED_DIR = ROOT / CONFIG["paths"]["processed_dir"]

_SPACY_MODEL_NAME = CONFIG["skill_extraction"]["spacy_model"]


def _try_load_spacy():
    try:
        import spacy
        from spacy.matcher import PhraseMatcher
        nlp = spacy.load(_SPACY_MODEL_NAME)
        matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
        patterns = [nlp.make_doc(term) for term in all_surface_forms()]
        matcher.add("SKILLS", patterns)
        return nlp, matcher
    except Exception as e:
        print(f"[skill_extraction] spaCy unavailable ({e}); falling back to regex keyword matching")
        return None, None


_NLP, _MATCHER = _try_load_spacy()


def _extract_regex(text: str) -> set[str]:
    found = set()
    for surface in all_surface_forms():
        pattern = r"(?<![a-z0-9])" + re.escape(surface.strip()) + r"(?![a-z0-9])"
        if re.search(pattern, text):
            found.add(canonicalize(surface))
    return found


def extract_from_text(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    if _NLP is not None:
        doc = _NLP(text)
        matches = _MATCHER(doc)
        found = {canonicalize(doc[start:end].text) for _, start, end in matches}
    else:
        found = _extract_regex(text)
    return sorted(found)


def extract_skills_for_row(row) -> tuple[list[str], str]:
    """Returns (skills, method) where method is 'pre_extracted' | 'nlp_description' | 'nlp_title' | 'none'."""
    pre_extracted = [s for s in str(row.get("skills_raw", "")).split("|") if s.strip()]
    if pre_extracted:
        canon = sorted({canonicalize(s) for s in pre_extracted})
        return canon, "pre_extracted"

    from_description = extract_from_text(row.get("job_description", ""))
    if from_description:
        return from_description, "nlp_description"

    from_title = extract_from_text(row.get("job_title", ""))
    if from_title:
        return from_title, "nlp_title"

    return [], "none"


def run() -> pd.DataFrame:
    cleaned_path = PROCESSED_DIR / "cleaned_postings.csv"
    if not cleaned_path.exists():
        raise FileNotFoundError("Run `python -m src.preprocessing` first.")

    df = pd.read_csv(cleaned_path)
    results = df.apply(extract_skills_for_row, axis=1, result_type="expand")
    df["skills"] = results[0]
    df["extraction_method"] = results[1]
    df["skill_count"] = df["skills"].apply(len)
    df["skills_json"] = df["skills"].apply(json.dumps)

    before = len(df)
    df = df[df["skill_count"] > 0]
    print(f"[skill_extraction] dropped {before - len(df)} rows with zero extractable skills; {len(df)} remain")

    register_version(df.drop(columns=["skills"]), stage="skills_extracted",
                      source="unified", notes="skills extracted (pre-extracted > NLP description > NLP title)")

    out_path = PROCESSED_DIR / "postings_with_skills.csv"
    df.drop(columns=["skills"]).to_csv(out_path, index=False)
    print(f"[skill_extraction] extracted skills for {len(df)} postings -> {out_path}")
    print(f"[skill_extraction] avg skills/posting: {df['skill_count'].mean():.2f}")
    print(f"[skill_extraction] extraction method breakdown:\n{df['extraction_method'].value_counts()}")
    return df


if __name__ == "__main__":
    run()
