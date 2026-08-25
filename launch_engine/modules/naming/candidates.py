from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from launch_engine.core.contracts import BaseModuleOutput
from datetime import datetime
from .brief import NameTypology


class InternalAssessment(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    rationale: str
    source: Literal["llm_self_assessment"] = "llm_self_assessment"


class NameCandidate(BaseModel):
    candidate_id: str
    name: str
    typology: NameTypology
    rationale: str
    phonetic_notes: Optional[str] = None
    tagline_options: List[str] = Field(default_factory=list)
    brand_story_seed: Optional[str] = None
    internal_assessment: Optional[InternalAssessment] = None


class NameCandidateList(BaseModuleOutput):
    brief_ref: str
    candidates: List[NameCandidate]
    llm_model_used: str
    llm_provider: str
    generated_at: datetime
