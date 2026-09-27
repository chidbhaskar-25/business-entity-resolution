"""Conservative normalization and tokenization for business addresses."""

from dataclasses import dataclass
import re
from typing import List, Optional, Set

from .text_utils import (
    expand_common_characters,
    extract_numeric_tokens,
    normalize_whitespace,
    strip_accents,
    strip_punctuation,
    tokenize_char_ngrams,
    tokenize_words,
)


@dataclass
class NormalizedAddress:
    """Container holding original and normalized address representations."""
    original: str
    normalized: str
    tokens: List[str]
    numeric_tokens: List[str]
    char_ngrams: Set[str]
    is_empty: bool


class AddressNormalizer:
    """Normalizes addresses across US, India, and France while preserving original values."""

    # Address token expansion dictionary across English, Indian, and French conventions
    ADDRESS_ABBREVIATIONS = {
        # Street types (US / International)
        "st": "street",
        "rd": "road",
        "ave": "avenue",
        "av": "avenue",
        "blvd": "boulevard",
        "ln": "lane",
        "dr": "drive",
        "hwy": "highway",
        "pl": "place",
        "sq": "square",
        "ct": "court",
        "apt": "apartment",
        "ste": "suite",
        "fl": "floor",
        "flr": "floor",
        "bldg": "building",
        "ter": "terrace",
        # Indian address patterns
        "opp": "opposite",
        "nr": "near",
        "bh": "behind",
        "b/h": "behind",
        "hno": "house number",
        "h.no": "house number",
        "flt": "flat",
        "ph": "phone",
        # French address patterns (France open-set)
        "r": "rue",
        "r.": "rue",
        "bd": "boulevard",
        "bvd": "boulevard",
        "imp": "impasse",
        "chem": "chemin",
        "che": "chemin",
        "all": "allee",
    }

    # Common placeholders for missing address data
    EMPTY_STRINGS = {"", "null", "none", "nan", "empty", "undefined", "n/a"}

    def __init__(self, expand_abbreviations: bool = True):
        self.expand_abbreviations = expand_abbreviations

    def normalize(self, raw_address: str) -> NormalizedAddress:
        """Normalizes an address string.

        Pipeline:
        1. Check for empty or placeholder strings
        2. Unicode accent stripping (NFKD)
        3. Lowercase
        4. Symbol expansion ('&' -> 'and', slashes/hyphens to space)
        5. Expand common address abbreviations
        6. Strip punctuation and collapse whitespace
        7. Extract word tokens, numeric tokens (with zero-strip), and char n-grams

        Returns:
            NormalizedAddress containing original and processed representations.
        """
        original = str(raw_address) if raw_address is not None else ""
        cleaned_check = original.strip().lower()

        if cleaned_check in self.EMPTY_STRINGS:
            return NormalizedAddress(
                original=original,
                normalized="",
                tokens=[],
                numeric_tokens=[],
                char_ngrams=set(),
                is_empty=True,
            )

        # 1. Accent stripping
        s = strip_accents(original)

        # 2. Lowercase
        s = s.lower()

        # 3. Handle landmark notations like 'b/h' -> 'behind', 'h.no' -> 'hno'
        s = re.sub(r"\bb\/h\b", "behind", s)
        s = re.sub(r"\bh\.no\.?\b", "hno", s)
        s = re.sub(r"\bno\.\s*", "no ", s)

        # 4. Expand symbols
        s = expand_common_characters(s)

        # 5. Expand abbreviations
        tokens = re.split(r"[\s,]+", s)
        if self.expand_abbreviations:
            expanded_tokens = []
            for t in tokens:
                if not t:
                    continue
                t_clean = t.strip(".,;:!?")
                if t_clean:
                    expanded_tokens.append(self.ADDRESS_ABBREVIATIONS.get(t_clean, t_clean))
        else:
            expanded_tokens = [t.strip(".,;:!?") for t in tokens if t.strip(".,;:!?")]
        s = " ".join(expanded_tokens)

        # 6. Strip punctuation while preserving letters, numbers, marks, and spaces
        normalized_str = strip_punctuation(s)

        if not normalized_str:
            return NormalizedAddress(
                original=original,
                normalized="",
                tokens=[],
                numeric_tokens=[],
                char_ngrams=set(),
                is_empty=True,
            )

        # 7. Token extraction
        words = tokenize_words(normalized_str)
        numeric_tokens = extract_numeric_tokens(normalized_str)
        char_ngrams = tokenize_char_ngrams(normalized_str, n=3)

        return NormalizedAddress(
            original=original,
            normalized=normalized_str,
            tokens=words,
            numeric_tokens=numeric_tokens,
            char_ngrams=char_ngrams,
            is_empty=False,
        )
