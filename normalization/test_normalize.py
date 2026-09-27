"""Small unit-style correctness checks for Stage 3.3 normalization."""

import unittest

from normalize import (
    address_token_set,
    conservative_normalize,
    normalize_address,
    normalize_address_abbreviations,
    normalize_name,
    tokenize_address,
)


class NormalizeTests(unittest.TestCase):
    def test_whitespace_punctuation_and_case(self):
        self.assertEqual(
            normalize_name("  ACME,  Co.\tLLC  "),
            "acme co llc",
        )

    def test_unicode_nfkc_and_casefold(self):
        self.assertEqual(normalize_name("ＡＣＭＥ Straße"), "acme strasse")

    def test_empty_and_missing_values(self):
        self.assertEqual(normalize_address("  \t "), "")
        self.assertIsNone(normalize_address(None))
        self.assertIsNone(normalize_name(None))
        self.assertIsNone(normalize_address_abbreviations(None))
        self.assertIsNone(tokenize_address(""))
        self.assertIsNone(tokenize_address(None))
        self.assertIsNone(address_token_set(""))
        self.assertIsNone(address_token_set(None))

    def test_abbreviation_representation_is_separate_and_limited(self):
        raw = "12 Main St, Oak Ave, 3rd Ln, Pine Dr"
        self.assertEqual(
            normalize_address_abbreviations(raw),
            "12 main street oak avenue 3rd lane pine drive",
        )
        self.assertEqual(normalize_address(raw), "12 main st oak ave 3rd ln pine dr")
        self.assertEqual(
            normalize_address_abbreviations("Hwy Rd Blvd Ct"),
            "hwy rd blvd ct",
        )

    def test_tokens_preserve_order_and_set_is_unordered(self):
        self.assertEqual(
            tokenize_address("42 Main St, Unit 3"),
            ("42", "main", "st", "unit", "3"),
        )
        self.assertEqual(
            address_token_set("42 Main St, Unit 3"),
            frozenset({"42", "main", "st", "unit", "3"}),
        )

    def test_idempotence_and_already_normalized_values(self):
        values = [
            "already normalized",
            "Café & Sons, LLC",
            "１２ Main St",
            "कंपनी, लिमिटेड",
            "",
        ]
        for value in values:
            once = conservative_normalize(value)
            self.assertEqual(conservative_normalize(once), once)
            self.assertEqual(normalize_address_abbreviations(
                normalize_address_abbreviations(value)
            ), normalize_address_abbreviations(value))
        self.assertIsNone(conservative_normalize(None))

    def test_no_transliteration(self):
        self.assertEqual(normalize_name("தமிழ்"), "தமிழ்")
        self.assertNotEqual(normalize_name("Tamil"), normalize_name("தமிழ்"))


if __name__ == "__main__":
    unittest.main()
