"""Text normalization for business names, addresses, and countries."""

import re
import unicodedata
from typing import Dict, List, Optional


class TextNormalizer:
    """Provides conservative, robust normalization for business entity resolution."""

    # Multi-jurisdictional legal entity suffixes (US, India, France, International)
    LEGAL_PATTERNS = [
        # Multi-word legal terms (process first)
        r"\bprivate limited\b",
        r"\bpvt\s+ltd\b",
        r"\bpvt\s+limited\b",
        r"\band company\b",
        r"\b& co\b",
        r"\bet cie\b",
        r"\bmicro[- ]entreprise\b",
        # Single-word legal terms
        r"\bincorporated\b",
        r"\bcorporation\b",
        r"\bcorp\b",
        r"\binc\b",
        r"\bllc\b",
        r"\bllp\b",
        r"\bltd\b",
        r"\blimited\b",
        r"\bgmbh\b",
        r"\bsarl\b",
        r"\bsas\b",
        r"\bsasu\b",
        r"\bsa\b",
        r"\beurl\b",
        r"\bsci\b",
        r"\bsnc\b",
        r"\bei\b",
        r"\bplc\b",
    ]

    # Address token expansion dictionary
    ADDRESS_ABBREVIATIONS: Dict[str, str] = {
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
        "bldg": "building",
        "opp": "opposite",
        "nr": "near",
        "r.": "rue",
        "bd": "boulevard",
    }

    # Name abbreviations
    NAME_ABBREVIATIONS: Dict[str, str] = {
        "intl": "international",
        "natl": "national",
        "tech": "technology",
        "mfg": "manufacturing",
        "svcs": "services",
        "svc": "services",
        "assn": "association",
    }

    # Domain suffixes to strip if a name is formatted as a URL
    DOMAIN_PATTERN = re.compile(
        r"^(?:https?:\/\/)?(?:www\.)?([a-zA-Z0-9_\-]+)\.(?:com|in|org|net|fr|co\.in|co|io|edu|gov)$",
        re.IGNORECASE,
    )

    _compiled_legal_regex = re.compile(
        "|".join(LEGAL_PATTERNS),
        re.IGNORECASE,
    )

    @classmethod
    def strip_accents(cls, text: str) -> str:
        """Decomposes unicode accents and returns ASCII equivalent."""
        if not text:
            return ""
        nfkd = unicodedata.normalize("NFKD", text)
        return "".join([c for c in nfkd if not unicodedata.combining(c)])

    @classmethod
    def normalize_country(cls, text: str) -> str:
        """Normalizes country label without restricting to closed set."""
        if not text:
            return ""
        return cls.strip_accents(str(text)).strip().upper()

    @classmethod
    def clean_domain(cls, text: str) -> str:
        """Strips URL prefixes and domain extensions if text is a web domain."""
        if not text:
            return ""
        m = cls.DOMAIN_PATTERN.match(text.strip())
        if m:
            return m.group(1)
        return text

    @classmethod
    def normalize_name(
        cls,
        text: str,
        strip_legal: bool = False,
    ) -> str:
        """Normalizes a business name.

        Steps:
        1. Handle domain format (e.g. example.com -> example)
        2. Unicode accent stripping (NFKD)
        3. Lowercase
        4. Expand & to 'and'
        5. Expand standard name abbreviations
        6. Optional: strip legal suffixes (Inc, Ltd, Pvt Ltd, SARL, etc.)
        7. Remove special punctuation
        8. Collapse whitespace

        Args:
            text: Raw business name.
            strip_legal: If True, strips legal entity suffixes.

        Returns:
            Normalized business name string.
        """
        if not text:
            return ""

        s = str(text).strip()
        s = cls.clean_domain(s)
        s = cls.strip_accents(s).lower()

        # Replace ampersands
        s = re.sub(r"&", " and ", s)

        # Replace hyphens and slashes with space
        s = re.sub(r"[\-\/\\_]", " ", s)

        # Remove dots between abbreviation letters (e.g. U.S. -> us, C.I.T. -> cit)
        s = re.sub(r"\b([a-z])\.(?=[a-z]\b|\s|$)", r"\1", s)

        if strip_legal:
            s = cls._compiled_legal_regex.sub(" ", s)

        # Expand name abbreviations
        tokens = s.split()
        expanded = [cls.NAME_ABBREVIATIONS.get(t, t) for t in tokens]
        s = " ".join(expanded)

        # Keep alphanumeric, spaces
        s = re.sub(r"[^a-z0-9\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    @classmethod
    def normalize_address(cls, text: str) -> str:
        """Normalizes a business address.

        Steps:
        1. Unicode accent stripping (NFKD)
        2. Lowercase
        3. Expand & to 'and'
        4. Normalize common address abbreviations (st -> street, rd -> road)
        5. Remove special punctuation while keeping digits and alphanumeric tokens
        6. Collapse whitespace

        Args:
            text: Raw address string.

        Returns:
            Normalized address string.
        """
        if not text:
            return ""

        s = str(text).strip()
        if s.lower() in ("null", "none", "nan", "empty"):
            return ""

        s = cls.strip_accents(s).lower()
        s = re.sub(r"&", " and ", s)
        s = re.sub(r"[\-\/\\_]", " ", s)
        s = re.sub(r"\b([a-z])\.(?=[a-z]\b|\s|$)", r"\1", s)

        tokens = re.split(r"[\s,]+", s)
        expanded = [cls.ADDRESS_ABBREVIATIONS.get(t, t) for t in tokens if t]
        s = " ".join(expanded)

        s = re.sub(r"[^a-z0-9\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s
