import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.skill_extraction import extract_from_text
from src.skill_taxonomy import canonicalize
from src.trend_model import skill_gap


def test_extract_skills_finds_known_terms():
    text = "looking for a candidate skilled in python, sql and machine learning"
    skills = extract_from_text(text)
    assert "python" in skills
    assert "sql" in skills
    assert "machine learning" in skills


def test_extract_skills_handles_synonyms():
    text = "must know k8s and pytorch for this ml role"
    skills = extract_from_text(text)
    assert "kubernetes" in skills
    assert "pytorch" in skills


def test_extract_skills_empty_text():
    assert extract_from_text("") == []
    assert extract_from_text(None) == []


def test_canonicalize_maps_synonym():
    assert canonicalize("k8s") == "kubernetes"
    assert canonicalize("unknown_tool_xyz") == "unknown_tool_xyz"


def test_skill_gap_computes_coverage():
    result = skill_gap(["python", "sql"], ["python", "sql", "docker", "aws"])
    assert result["have"] == ["python", "sql"]
    assert set(result["missing"]) == {"docker", "aws"}
    assert result["coverage_pct"] == 50.0
