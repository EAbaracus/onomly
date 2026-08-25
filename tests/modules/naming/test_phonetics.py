import unittest
from launch_engine.modules.naming.phonetics import (
    PhoneticConstraints,
    check_phonetic_constraints,
    estimate_syllables,
)


class TestPhonetics(unittest.TestCase):
    def test_valid_name_passes_all_constraints(self):
        """Test a valid name passes all constraints."""
        constraints = PhoneticConstraints(
            max_syllables=3, max_length=10, avoid_sounds=["zz", "xx"]
        )
        name = "hello"  # 2 syllables, length 5, no avoided sounds
        result = check_phonetic_constraints(name, constraints)
        self.assertTrue(result.is_valid)
        self.assertIsNone(result.notes)

    def test_name_exceeding_max_syllables_fails(self):
        """Test name with too many syllables fails."""
        constraints = PhoneticConstraints(max_syllables=2)
        name = "beautiful"  # 3 syllables
        result = check_phonetic_constraints(name, constraints)
        self.assertFalse(result.is_valid)
        self.assertIn("syllable count", result.notes)

    def test_name_exceeding_max_length_fails(self):
        """Test name too long fails."""
        constraints = PhoneticConstraints(max_length=5)
        name = "abcdef"  # length 6
        result = check_phonetic_constraints(name, constraints)
        self.assertFalse(result.is_valid)
        self.assertIn("length", result.notes)

    def test_name_containing_avoided_sounds_fails(self):
        """Test name containing avoided sound fails."""
        constraints = PhoneticConstraints(avoid_sounds=["zz", "xx"])
        name = "buzz"  # contains "zz"
        result = check_phonetic_constraints(name, constraints)
        self.assertFalse(result.is_valid)
        self.assertIn("avoided sound", result.notes)

    def test_notes_explain_validation_failures(self):
        """Test that notes explain why validation failed."""
        constraints = PhoneticConstraints(
            max_syllables=1, max_length=3, avoid_sounds=["a"]
        )
        name = "astra"  # 2 syllables (as-tra), length 5, contains 'a'
        result = check_phonetic_constraints(name, constraints)
        self.assertFalse(result.is_valid)
        # Notes should mention at least one failure
        self.assertIsNotNone(result.notes)
        # Check that notes contain relevant info
        self.assertTrue(
            "syllable" in result.notes.lower()
            or "length" in result.notes.lower()
            or "avoided sound" in result.notes.lower()
        )

    def test_empty_constraints_passes_any_name(self):
        """Test that empty constraints pass any name."""
        constraints = PhoneticConstraints()  # all None
        for name in ["", "a", "very long name", "buzz"]:
            result = check_phonetic_constraints(name, constraints)
            self.assertTrue(result.is_valid, f"Failed for name: {name}")
            self.assertIsNone(result.notes)

    def test_syllable_estimation(self):
        """Test the syllable estimation heuristic."""
        self.assertEqual(estimate_syllables("hello"), 2)  # hel-lo
        self.assertEqual(estimate_syllables("beautiful"), 3)  # beau-ti-ful
        self.assertEqual(estimate_syllables("a"), 1)
        self.assertEqual(estimate_syllables("aa"), 1)  # vowel group
        self.assertEqual(estimate_syllables("ae"), 1)  # vowel group
        self.assertEqual(estimate_syllables("aei"), 1)  # vowel group
        self.assertEqual(
            estimate_syllables("aeib"), 1
        )  # aei -> one group, then b -> no new vowel group
        self.assertEqual(estimate_syllables("coach"), 1)  # coa -> one vowel group
        self.assertEqual(estimate_syllables("boat"), 1)  # oa -> one vowel group
        self.assertEqual(estimate_syllables("create"), 2)  # cre-ate -> two vowel groups


if __name__ == "__main__":
    unittest.main()
