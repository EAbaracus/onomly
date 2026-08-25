from launch_engine.modules.naming.brief import (
    NameTypology,
    PhoneticConstraints,
    NamingBrief,
)


def test_name_typology_enum():
    """Test that NameTypology enum has all 8 values."""
    expected = {
        "INVENTED",
        "DESCRIPTIVE",
        "SUGGESTIVE",
        "METAPHORICAL",
        "ACRONYM",
        "PORTMANTEAU",
        "FOUNDER",
        "COMPOUND",
    }
    actual = {item.value for item in NameTypology}
    assert actual == expected


def test_phonetic_constraints_default():
    """Test PhoneticConstraints with default values."""
    constraints = PhoneticConstraints()
    assert constraints.max_syllables is None
    assert constraints.max_length is None
    assert constraints.avoid_sounds == []
    assert constraints.prefer_sounds == []


def test_phonetic_constraints_custom():
    """Test PhoneticConstraints with custom values."""
    constraints = PhoneticConstraints(
        max_syllables=3,
        max_length=10,
        avoid_sounds=["z", "x"],
        prefer_sounds=["a", "o"],
    )
    assert constraints.max_syllables == 3
    assert constraints.max_length == 10
    assert constraints.avoid_sounds == ["z", "x"]
    assert constraints.prefer_sounds == ["a", "o"]


def test_naming_brief_required_fields():
    """Test NamingBrief with required fields only."""
    brief = NamingBrief(
        project_codename="ProjectX",
        description="A test project",
        target_markets=["US", "EU"],
        industry="Technology",
    )
    assert brief.project_codename == "ProjectX"
    assert brief.description == "A test project"
    assert brief.target_markets == ["US", "EU"]
    assert brief.industry == "Technology"
    assert brief.brand_personality is None
    assert brief.phonetic_constraints is None
    assert brief.avoid_terms == []
    assert brief.preferred_typologies == []
    assert brief.candidate_count == 15
    assert brief.language == "auto"


def test_naming_brief_all_fields():
    """Test NamingBrief with all optional fields."""
    phonetic_constraints = PhoneticConstraints(max_syllables=2, avoid_sounds=["z"])
    brief = NamingBrief(
        project_codename="ProjectY",
        description="Another test project",
        target_markets=["Asia"],
        industry="Healthcare",
        brand_personality="Innovative",
        phonetic_constraints=phonetic_constraints,
        avoid_terms=["bad"],
        preferred_typologies=[NameTypology.INVENTED, NameTypology.METAPHORICAL],
        candidate_count=10,
        language="en",
    )
    assert brief.project_codename == "ProjectY"
    assert brief.description == "Another test project"
    assert brief.target_markets == ["Asia"]
    assert brief.industry == "Healthcare"
    assert brief.brand_personality == "Innovative"
    assert brief.phonetic_constraints == phonetic_constraints
    assert brief.avoid_terms == ["bad"]
    assert brief.preferred_typologies == [
        NameTypology.INVENTED,
        NameTypology.METAPHORICAL,
    ]
    assert brief.candidate_count == 10
    assert brief.language == "en"
