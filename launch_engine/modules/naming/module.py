"""
Brand & Naming module - core logic for generating and evaluating brand names.
"""

from __future__ import annotations

import json
from typing import List
from datetime import datetime


from launch_engine.core.contracts import BaseModuleInput, LaunchModule
from launch_engine.llm import LLMAdapter
from .brief import NamingBrief, NameTypology
from .candidates import NameCandidate, NameCandidateList, InternalAssessment
from .phonetics import (
    check_phonetic_constraints,
    PhoneticConstraints as PhoneticsConstraintsDataclass,
)


class BrandNamingModule(LaunchModule):
    """Brand naming module that generates and evaluates brand name candidates."""

    name: str = "brand_naming"

    def __init__(self, llm_adapter: LLMAdapter):
        """Initialize the brand naming module.

        Args:
            llm_adapter: LLM adapter for generating and scoring names
        """
        self.llm_adapter = llm_adapter

    async def run(self, input_data: BaseModuleInput) -> NameCandidateList:
        """Run the brand naming module to generate and rank name candidates.

        Args:
            input_data: Naming brief containing project details and constraints

        Returns:
            NameCandidateList containing generated and scored candidates

        Raises:
            TypeError: If input_data is not a NamingBrief
        """
        if not isinstance(input_data, NamingBrief):
            raise TypeError(f"Expected NamingBrief, got {type(input_data)}")
        brief = input_data

        # Prefilter: validate brief and clean data
        self._prefilter(brief)

        # Stage 1: Divergent generation - create diverse candidates
        candidates = await self._divergent_generate(brief)

        # Midfilter: apply phonetic constraints and remove avoided terms
        candidates = self._midfilter(candidates, brief)

        # Stage 2: Convergent scoring - score and rank candidates
        candidates = await self._convergent_score(candidates, brief)

        # Create and return the result
        return NameCandidateList(
            brief_ref=brief.project_codename,
            candidates=candidates,
            llm_model_used=self.llm_adapter.model_id,
            llm_provider=self.llm_adapter.provider,
            generated_at=datetime.utcnow(),
        )

    def _prefilter(self, brief: NamingBrief) -> None:
        """Validate brief and clean avoid list.

        Args:
            brief: Naming brief to validate and clean
        """
        # Ensure avoid_terms is a list and remove duplicates/empty strings
        if brief.avoid_terms:
            brief.avoid_terms = list(
                set(
                    term.strip().lower()
                    for term in brief.avoid_terms
                    if term and term.strip()
                )
            )

        # Ensure preferred_typologies is a list and remove duplicates
        if brief.preferred_typologies:
            brief.preferred_typologies = list(set(brief.preferred_typologies))

    async def _divergent_generate(self, brief: NamingBrief) -> List[NameCandidate]:
        """Generate diverse name candidates across all typologies.

        Args:
            brief: Naming brief

        Returns:
            List of generated name candidates
        """
        # Create prompt for divergent generation
        prompt = self._create_divergent_prompt(brief)

        try:
            # Call LLM to generate candidates
            response = await self.llm_adapter.generate(prompt)

            # Parse JSON response
            candidates_data = self._parse_candidates_json(response)

            # Convert to NameCandidate objects
            candidates = []
            for i, candidate_data in enumerate(candidates_data):
                # Generate unique candidate ID
                candidate_id = f"cand_{i+1:03d}"

                # Create NameCandidate
                candidate = NameCandidate(
                    candidate_id=candidate_id,
                    name=candidate_data.get("name", ""),
                    typology=NameTypology(candidate_data.get("typology", "INVENTED")),
                    rationale=candidate_data.get("rationale", ""),
                    phonetic_notes=candidate_data.get("phonetic_notes"),
                    tagline_options=candidate_data.get("tagline_options", []),
                    brand_story_seed=candidate_data.get("brand_story_seed"),
                )
                candidates.append(candidate)

            return candidates

        except Exception as e:
            # Handle LLM failures gracefully - return empty list
            print(f"Warning: LLM generation failed: {e}")
            return []

    def _midfilter(
        self, candidates: List[NameCandidate], brief: NamingBrief
    ) -> List[NameCandidate]:
        """Apply phonetic constraints and remove avoided terms.

        Args:
            candidates: List of candidates to filter
            brief: Naming brief containing constraints

        Returns:
            Filtered list of candidates
        """
        filtered_candidates = []

        for candidate in candidates:
            # Check if candidate name contains any avoided terms
            name_lower = candidate.name.lower()
            if any(avoided_term in name_lower for avoided_term in brief.avoid_terms):
                continue

            # Apply phonetic constraints if specified
            if brief.phonetic_constraints:
                # Convert Pydantic model to dataclass for phonetics function
                phonetic_constraints = PhoneticsConstraintsDataclass(
                    max_syllables=brief.phonetic_constraints.max_syllables,
                    max_length=brief.phonetic_constraints.max_length,
                    avoid_sounds=brief.phonetic_constraints.avoid_sounds,
                )

                assessment = check_phonetic_constraints(
                    candidate.name, phonetic_constraints
                )
                if not assessment.is_valid:
                    # Add phonetic notes to candidate
                    candidate.phonetic_notes = assessment.notes
                    continue

            filtered_candidates.append(candidate)

        return filtered_candidates

    async def _convergent_score(
        self, candidates: List[NameCandidate], brief: NamingBrief
    ) -> List[NameCandidate]:
        """Score and rank candidates using LLM.

        Args:
            candidates: List of candidates to score
            brief: Naming brief

        Returns:
            List of scored candidates sorted by score (descending)
        """
        if not candidates:
            return []

        # Create prompt for convergent scoring
        prompt = self._create_convergent_prompt(candidates, brief)

        try:
            # Call LLM to score candidates
            response = await self.llm_adapter.generate(prompt)

            # Parse JSON response
            scored_data = self._parse_scoring_json(response)

            # Update candidates with scores
            scored_candidates = []
            for candidate in candidates:
                # Find score for this candidate
                score_info = next(
                    (
                        item
                        for item in scored_data
                        if item.get("candidate_id") == candidate.candidate_id
                    ),
                    None,
                )

                if score_info:
                    # Create internal assessment
                    assessment = InternalAssessment(
                        score=float(score_info.get("score", 0.0)),
                        rationale=score_info.get("rationale", ""),
                    )
                    candidate.internal_assessment = assessment
                else:
                    # Default low score if not found
                    candidate.internal_assessment = InternalAssessment(
                        score=0.0, rationale="No score provided by LLM"
                    )

                scored_candidates.append(candidate)

            # Sort by score descending
            scored_candidates.sort(
                key=lambda c: (
                    c.internal_assessment.score if c.internal_assessment else 0.0
                ),
                reverse=True,
            )

            # Limit to candidate_count
            return scored_candidates[: brief.candidate_count]

        except Exception as e:
            # Handle LLM failures gracefully - return candidates with no scores
            print(f"Warning: LLM scoring failed: {e}")
            for candidate in candidates:
                candidate.internal_assessment = InternalAssessment(
                    score=0.0, rationale=f"Scoring failed: {str(e)}"
                )
            return candidates[: brief.candidate_count]

    def _create_divergent_prompt(self, brief: NamingBrief) -> str:
        """Create prompt for divergent name generation.

        Args:
            brief: Naming brief

        Returns:
            Formatted prompt for LLM
        """
        typologies = [t.value for t in NameTypology]
        preferred_typologies = (
            [t.value for t in brief.preferred_typologies]
            if brief.preferred_typologies
            else typologies
        )

        prompt = f"""
You are an expert brand naming consultant. Generate creative brand name candidates based on the following brief:

PROJECT OVERVIEW:
- Project Codename: {brief.project_codename}
- Description: {brief.description}
- Target Markets: {', '.join(brief.target_markets)}
- Industry: {brief.industry}
- Brand Personality: {brief.brand_personality or 'Not specified'}

REQUIREMENTS:
- Generate exactly 20 diverse name candidates
- Cover ALL naming typologies: {', '.join(typologies)}
- Focus particularly on these preferred typologies: {', '.join(preferred_typologies)}
- Avoid these terms: {', '.join(brief.avoid_terms) if brief.avoid_terms else 'None'}
{f"- Phonetic constraints: max length {brief.phonetic_constraints.max_length}, max syllables {brief.phonetic_constraints.max_syllables}, avoid sounds {', '.join(brief.phonetic_constraints.avoid_sounds) if brief.phonetic_constraints and brief.phonetic_constraints.avoid_sounds else 'None'}" if brief.phonetic_constraints else ""}
- Language: {brief.language}

For each candidate, provide:
1. name: The brand name
2. typology: One of {', '.join(typologies)}
3. rationale: Explanation why this name fits the brief
4. phonetic_notes: Any observations about pronunciation, sound, or rhythm (optional)
5. tagline_options: Up to 3 optional tagline suggestions
6. brand_story_seed: Optional seed for brand story development

Return ONLY a JSON array of objects with these exact fields:
[
  {{
    "name": "example",
    "typology": "INVENTED",
    "rationale": "Explanation here",
    "phonetic_notes": "Pronunciation notes",
    "tagline_options": ["Tagline 1", "Tagline 2"],
    "brand_story_seed": "Brand story idea"
  }}
]

Do not include any other text, explanation, or formatting - just the JSON array.
""".strip()

        return prompt

    def _create_convergent_prompt(
        self, candidates: List[NameCandidate], brief: NamingBrief
    ) -> str:
        """Create prompt for convergent scoring.

        Args:
            candidates: List of candidates to score
            brief: Naming brief

        Returns:
            Formatted prompt for LLM
        """
        candidates_json = []
        for candidate in candidates:
            candidates_json.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "name": candidate.name,
                    "typology": candidate.typology.value,
                    "rationale": candidate.rationale,
                }
            )

        prompt = f"""
You are an expert brand naming evaluator. Score and rank the following brand name candidates based on how well they fit the brief:

PROJECT BRIEF:
- Project Codename: {brief.project_codename}
- Description: {brief.description}
- Target Markets: {', '.join(brief.target_markets)}
- Industry: {brief.industry}
- Brand Personality: {brief.brand_personality or 'Not specified'}
- Avoid Terms: {', '.join(brief.avoid_terms) if brief.avoid_terms else 'None'}

CANDIDATES TO EVALUATE:
{json.dumps(candidates_json, indent=2)}

For each candidate, provide a score (0.0 to 1.0) and rationale based on:
- Fit with brand personality and industry
- Memorability and pronounceability
- Uniqueness and distinctiveness
- Appropriateness for target markets
- Avoidance of negative connotations
- Alignment with project description

Return ONLY a JSON array of objects with these exact fields:
[
  {{
    "candidate_id": "cand_001",
    "score": 0.85,
    "rationale": "Explanation of the score"
  }}
]

Do not include any other text, explanation, or formatting - just the JSON array.
""".strip()

        return prompt

    def _parse_candidates_json(self, response: str) -> List[dict]:
        """Parse JSON response from divergent generation.

        Args:
            response: LLM response string

        Returns:
            List of candidate dictionaries
        """
        # Try to find JSON array in response
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()

        # Parse JSON
        data = json.loads(response)
        if not isinstance(data, list):
            raise ValueError("Expected JSON array")

        return data

    def _parse_scoring_json(self, response: str) -> List[dict]:
        """Parse JSON response from convergent scoring.

        Args:
            response: LLM response string

        Returns:
            List of scoring dictionaries
        """
        # Try to find JSON array in response
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()

        # Parse JSON
        data = json.loads(response)
        if not isinstance(data, list):
            raise ValueError("Expected JSON array")

        return data
