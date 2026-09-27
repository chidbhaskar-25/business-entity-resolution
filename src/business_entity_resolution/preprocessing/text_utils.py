"""Shared low-level text utilities for entity resolution preprocessing."""

import re
import unicodedata
from typing import List, Set


def strip_accents(text: str) -> str:
    """Decomposes Latin unicode characters into NFKD form and removes accent marks.

    Leaves non-Latin scripts (Devanagari, Tamil, etc.) and their combining marks intact.

    Example:
        'Président Thénard Àmicale' -> 'President Thenard Amicale'
        'राम मार्केटिंग' -> 'राम मार्केटिंग'
    """
    if not text:
        return ""
    chars = []
    for c in str(text):
        if ord(c) < 0x0590:
            # Latin / Greek / Cyrillic decomposition
            nfkd = unicodedata.normalize("NFKD", c)
            base = "".join(ch for ch in nfkd if not unicodedata.combining(ch))
            chars.append(base)
        else:
            chars.append(c)
    return "".join(chars)


def normalize_whitespace(text: str) -> str:
    """Collapses multiple spaces, tabs, and newlines into a single space and strips ends."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def strip_domain_artifacts(text: str) -> str:
    """Strips web protocols (http, https, www) and top-level domain extensions (.com, .org, etc.).

    Example:
        'www.maurewilliamscolombier.com' -> 'maurewilliamscolombier'
        'http://example.fr' -> 'example'
    """
    if not text:
        return ""
    s = str(text).strip()
    s = re.sub(r"^(?:https?:\/\/)?(?:www\.)?", "", s, flags=re.IGNORECASE)
    s = re.sub(
        r"\.(?:com|in|org|net|fr|co\.in|co|io|biz|info|edu|gov)(?:\/.*)?$",
        "",
        s,
        flags=re.IGNORECASE,
    )
    return s


def expand_common_characters(text: str) -> str:
    """Standardizes symbols like '&', '+', '/', and hyphens into clean word-delimited text."""
    if not text:
        return ""
    s = str(text)
    # Expand ampersand
    s = re.sub(r"&", " and ", s)
    # Expand plus sign
    s = re.sub(r"\+", " plus ", s)
    # Replace slashes, hyphens, underscores with space
    s = re.sub(r"[\/\-_\\|]", " ", s)
    # Remove dots between single letters in acronyms: 'C.I.T.' -> 'cit', 'U.S.' -> 'us'
    s = re.sub(r"\b([a-zA-Z])\.(?=[a-zA-Z]\b|\s|$)", r"\1", s)
    # Remove quotes, brackets, angle brackets: '<<', '>>', '[', ']', '(', ')'
    s = re.sub(r"[«»<>\[\]\(\)\{\}\"\'`]", " ", s)
    return s


def strip_punctuation(text: str) -> str:
    """Replaces punctuation (P) and symbols (S) with spaces while preserving letters, numbers, and marks."""
    if not text:
        return ""
    # In Unicode: categories starting with 'P' (Punctuation) or 'S' (Symbol)
    # Categories 'L' (Letter), 'N' (Number), 'M' (Mark) are preserved
    cleaned_chars = [
        " " if (unicodedata.category(c).startswith("P") or unicodedata.category(c).startswith("S"))
        else c
        for c in text
    ]
    return normalize_whitespace("".join(cleaned_chars))


def tokenize_words(text: str, min_len: int = 1) -> List[str]:
    """Extracts word tokens from normalized text across any language/script."""
    if not text:
        return []
    cleaned = strip_punctuation(text.lower())
    tokens = [t for t in cleaned.split() if len(t) >= min_len]
    return tokens


def tokenize_char_ngrams(text: str, n: int = 3) -> Set[str]:
    """Generates character n-grams from text for fuzzy matching."""
    cleaned = normalize_whitespace(text.lower())
    if not cleaned:
        return set()
    if len(cleaned) < n:
        return {cleaned}
    return {cleaned[i : i + n] for i in range(len(cleaned) - n + 1)}


def extract_numeric_tokens(text: str) -> List[str]:
    """Extracts numeric digit sequences and standardizes leading zeroes.

    Example:
        'AF-0684' -> ['684']
        'Plot 007, Road 12' -> ['7', '12']
    """
    if not text:
        return []
    raw_nums = re.findall(r"\b\d+\b", text)
    standardized = []
    for n in raw_nums:
        clean_num = n.lstrip("0") or "0"
        standardized.append(clean_num)
    return standardized
