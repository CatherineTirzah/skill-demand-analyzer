from typing import List, Optional

from pydantic import BaseModel, Field


class SkillGapRequest(BaseModel):
    user_skills: List[str] = Field(..., example=["python", "sql", "excel"])
    top_n: int = Field(default=15, ge=1, le=50)


class SkillGapResponse(BaseModel):
    have: List[str]
    missing: List[str]
    coverage_pct: float
    top_in_demand: List[str]


class TrendPoint(BaseModel):
    period: str
    demand_pct: float


class SkillTrendResponse(BaseModel):
    skill: str
    history: List[TrendPoint]
    predicted_next_period_pct: Optional[float]


class TopSkillsResponse(BaseModel):
    period: str
    skills: List[dict]


class HealthResponse(BaseModel):
    status: str
    model_status: str
    model_id: Optional[str]
    n_skills_tracked: Optional[int]
