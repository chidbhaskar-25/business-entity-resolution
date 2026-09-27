"""Unit tests for preprocessing layer with realistic noisy entity resolution examples."""

import pytest
from src.business_entity_resolution.preprocessing.name_normalizer import BusinessNameNormalizer
from src.business_entity_resolution.preprocessing.address_normalizer import AddressNormalizer
from src.business_entity_resolution.preprocessing.text_utils import (
    extract_numeric_tokens,
    strip_accents,
    strip_domain_artifacts,
    tokenize_words,
)


@pytest.fixture
def name_normalizer():
    return BusinessNameNormalizer()


@pytest.fixture
def address_normalizer():
    return AddressNormalizer()


# ---------------------------------------------------------
# 1. Business Name Normalization Tests
# ---------------------------------------------------------

def test_name_normalizer_preserves_original(name_normalizer):
    raw = "Maure Williams Colombier Inc"
    norm = name_normalizer.normalize(raw)
    assert norm.original == raw
    assert norm.normalized == "maure williams colombier inc"
    assert norm.clean_without_legal == "maure williams colombier"
    assert norm.detected_legal_suffix == "inc"


def test_name_normalizer_legal_suffixes(name_normalizer):
    cases = [
        ("Acme Pvt Ltd", "pvt ltd", "acme"),
        ("Tata Private Limited", "pvt ltd", "tata"),
        ("Reliance Industries LLP", "llp", "reliance industries"),
        ("Google LLC", "llc", "google"),
        ("Boeing Corporation", "corp", "boeing"),
        ("ZNB Club SARL", "sarl", "znb club"),
        ("Thermal & Fils SASU", "sasu", "thermal and fils"),
        ("Elephant Centre EURL", "eurl", "elephant centre"),
        ("SCI Ptit Amicale", "sci", "ptit amicale"),
    ]
    for raw, expected_suffix, expected_clean in cases:
        res = name_normalizer.normalize(raw)
        assert res.detected_legal_suffix == expected_suffix, f"Failed on {raw}"
        assert res.clean_without_legal == expected_clean, f"Clean failed on {raw}: got '{res.clean_without_legal}'"


def test_name_normalizer_domain_formats(name_normalizer):
    # Tests matching web domain names to entity names
    raw = "www.maurewilliamscolombier.com"
    norm = name_normalizer.normalize(raw)
    assert norm.normalized == "maurewilliamscolombier"
    assert "maurewilliamscolombier" in norm.tokens


def test_name_normalizer_unicode_accents(name_normalizer):
    # French accent decomposition
    raw = "Maison de Santé Génération"
    norm = name_normalizer.normalize(raw)
    assert norm.normalized == "maison de sante generation"
    assert "sante" in norm.tokens
    assert "generation" in norm.tokens


def test_name_normalizer_special_characters(name_normalizer):
    # Brackets, arrows, punctuation artifacts
    raw = "<< Team Ecole >> & Co. [EURL]"
    norm = name_normalizer.normalize(raw)
    assert "team ecole and co" in norm.normalized
    assert norm.detected_legal_suffix == "eurl"


def test_name_normalizer_empty_and_whitespace(name_normalizer):
    empty_norm = name_normalizer.normalize("   ")
    assert empty_norm.normalized == ""
    assert empty_norm.clean_without_legal == ""
    assert len(empty_norm.tokens) == 0
    assert empty_norm.detected_legal_suffix is None


# ---------------------------------------------------------
# 2. Address Normalization Tests
# ---------------------------------------------------------

def test_address_normalizer_preserves_original(address_normalizer):
    raw = "1795 Westchester Drive, High Point, NC"
    norm = address_normalizer.normalize(raw)
    assert norm.original == raw
    assert norm.is_empty is False
    assert "1795" in norm.numeric_tokens
    assert "drive" in norm.tokens


def test_address_normalizer_abbreviations(address_normalizer):
    cases = [
        ("123 Main St, 2nd Fl", ["street", "floor"]),
        ("456 Oak Rd, Suite 300", ["road", "suite"]),
        ("789 Grand Ave, Apt 4B", ["avenue", "apartment"]),
        ("154 BD du President Wilson", ["boulevard", "president"]),
        ("63 R. DE DIEPPE", ["rue", "dieppe"]),
        ("Near Mother India Public School, Opp. RTA Office", ["near", "opposite"]),
        ("B/H Natraj Cinema", ["behind", "natraj"]),
    ]
    for raw, expected_tokens in cases:
        norm = address_normalizer.normalize(raw)
        for exp in expected_tokens:
            assert exp in norm.tokens, f"Expected '{exp}' in tokens for '{raw}': got {norm.tokens}"


def test_address_normalizer_numeric_token_alignment(address_normalizer):
    # Critical insight: S1 'Af-684' vs S2 'AF-0684'
    raw_s1 = "Af-684, Nandgram, Ghaziabad 9487203"
    raw_s2 = "AF-0684, NANDGRAM, GHAZIABAD, 9487203"

    norm1 = address_normalizer.normalize(raw_s1)
    norm2 = address_normalizer.normalize(raw_s2)

    # Both must extract '684' without leading zero
    assert "684" in norm1.numeric_tokens
    assert "684" in norm2.numeric_tokens
    assert "9487203" in norm1.numeric_tokens
    assert "9487203" in norm2.numeric_tokens
    # Numeric tokens match identically
    assert set(norm1.numeric_tokens) == set(norm2.numeric_tokens)


def test_address_normalizer_empty_and_null_values(address_normalizer):
    for empty_val in ["", "   ", "null", "NULL", "none", "None", "nan", "empty"]:
        norm = address_normalizer.normalize(empty_val)
        assert norm.is_empty is True
        assert norm.normalized == ""
        assert len(norm.tokens) == 0
        assert len(norm.numeric_tokens) == 0


def test_address_normalizer_french_characters(address_normalizer):
    raw = "NO. 5 ALLÉE DES HÊTRES, Pornic"
    norm = address_normalizer.normalize(raw)
    assert "allee" in norm.tokens
    assert "hetres" in norm.tokens
    assert "5" in norm.numeric_tokens
