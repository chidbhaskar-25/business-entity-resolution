"""Conservative normalization and tokenization for business names."""

from dataclasses import dataclass
import re
from typing import List, Optional, Set, Tuple

from .text_utils import (
    expand_common_characters,
    normalize_whitespace,
    strip_accents,
    strip_domain_artifacts,
    strip_punctuation,
    tokenize_char_ngrams,
    tokenize_words,
)


@dataclass
class NormalizedName:
    """Container holding both original and processed business name components."""
    original: str
    normalized: str
    clean_without_legal: str
    tokens: List[str]
    char_ngrams: Set[str]
    detected_legal_suffix: Optional[str] = None


class BusinessNameNormalizer:
    """Normalizes business names while preserving original values."""

    # Common business name abbreviations
    NAME_ABBREVIATIONS = {
        "intl": "international",
        "natl": "national",
        "tech": "technology",
        "mfg": "manufacturing",
        "svcs": "services",
        "svc": "services",
        "assn": "association",
    }

    # Multi-jurisdictional legal entity suffixes (corporate entities prioritized)
    LEGAL_SUFFIX_PATTERNS = [
        # Multi-word corporate
        (r"\bprivate\s+limited\b", "pvt ltd"),
        (r"\bpvt\s+ltd\b", "pvt ltd"),
        (r"\bpvt\s+limited\b", "pvt ltd"),
        (r"\bmicro[- ]entreprise\b", "micro-entreprise"),
        # French corporate entity types (France open-set)
        (r"\bsasu\b", "sasu"),
        (r"\bsarl\b", "sarl"),
        (r"\beurl\b", "eurl"),
        (r"\bsci\b", "sci"),
        (r"\bsnc\b", "snc"),
        (r"\bsas\b", "sas"),
        (r"\bsa\b", "sa"),
        (r"\bei\b", "ei"),
        # Single-word US / International corporate
        (r"\bincorporated\b", "inc"),
        (r"\bcorporation\b", "corp"),
        (r"\bcorp\b", "corp"),
        (r"\binc\b", "inc"),
        (r"\bllc\b", "llc"),
        (r"\bllp\b", "llp"),
        (r"\bltd\b", "ltd"),
        (r"\blimited\b", "ltd"),
        (r"\bgmbh\b", "gmbh"),
        (r"\bplc\b", "plc"),
        # Multi-word partnership / familial terms
        (r"\band\s+company\b", "co"),
        (r"\bet\s+cie\b", "et cie"),
        (r"\bet\s+fils\b", "et fils"),
        (r"\band\s+fils\b", "et fils"),
    ]

    _compiled_legal = [(re.compile(p, re.IGNORECASE), label) for p, label in LEGAL_SUFFIX_PATTERNS]

    def __init__(self, expand_abbreviations: bool = True):
        self.expand_abbreviations = expand_abbreviations

    def normalize(self, raw_name: str) -> NormalizedName:
        """Applies conservative normalization to a business name.

        Pipeline:
        1. Strip URL/domain artifacts (e.g. 'company.com' -> 'company')
        2. Strip unicode accents via NFKD (e.g. 'Président' -> 'President')
        3. Convert to lowercase
        4. Expand symbols ('&' -> 'and', '+' -> 'plus', acronym dots)
        5. Expand common business abbreviations
        6. Identify and extract legal entity suffix
        7. Strip punctuation and collapse whitespace
        8. Tokenize into words and character n-grams

        Returns:
            NormalizedName object containing original, normalized, clean, and tokenized fields.
        """
        original = str(raw_name) if raw_name is not None else ""
        if not original.strip():
            return NormalizedName(
                original=original,
                normalized="",
                clean_without_legal="",
                tokens=[],
                char_ngrams=set(),
                detected_legal_suffix=None,
            )

        # 1. Domain cleaning
        s = strip_domain_artifacts(original)

        # 2. Accent stripping
        s = strip_accents(s)

        # 3. Lowercase
        s = s.lower()

        # 4. Expand symbols and clean acronym dots
        s = expand_common_characters(s)

        # 5. Expand business abbreviations
        tokens = s.split()
        if self.expand_abbreviations:
            tokens = [self.NAME_ABBREVIATIONS.get(t, t) for t in tokens]
        s = " ".join(tokens)

        # 6. Detect and extract legal suffix
        detected_suffix = None
        s_without_legal = s
        for pattern, label in self._compiled_legal:
            if pattern.search(s):
                detected_suffix = label
                s_without_legal = pattern.sub(" ", s_without_legal)
                break

        # 7. Strip punctuation while preserving letters, numbers, marks, and spaces
        normalized_str = strip_punctuation(s)
        clean_without_legal = strip_punctuation(s_without_legal)

        # 8. Tokenize
        words = tokenize_words(normalized_str)
        char_ngrams = tokenize_char_ngrams(normalized_str, n=3)

        return NormalizedName(
            original=original,
            normalized=normalized_str,
            clean_without_legal=clean_without_legal,
            tokens=words,
            char_ngrams=char_ngrams,
            detected_legal_suffix=detected_suffix,
        )
