from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field
from launch_engine.core.contracts import BaseModuleInput


class NameTypology(str, Enum):
    INVENTED = "INVENTED"
    DESCRIPTIVE = "DESCRIPTIVE"
    SUGGESTIVE = "SUGGESTIVE"
    METAPHORICAL = "METAPHORICAL"
    ACRONYM = "ACRONYM"
    PORTMANTEAU = "PORTMANTEAU"
    FOUNDER = "FOUNDER"
    COMPOUND = "COMPOUND"


class PhoneticConstraints(BaseModel):
    max_syllables: Optional[int] = None
    max_length: Optional[int] = None
    avoid_sounds: List[str] = Field(default_factory=list)
    prefer_sounds: List[str] = Field(default_factory=list)


class NamingBrief(BaseModuleInput):
    project_codename: str
    description: str
    target_markets: List[str]
    industry: str
    brand_personality: Optional[str] = None
    phonetic_constraints: Optional[PhoneticConstraints] = None
    avoid_terms: List[str] = Field(default_factory=list)
    preferred_typologies: List[NameTypology] = Field(default_factory=list)
    candidate_count: int = 15
    language: str = "auto"
