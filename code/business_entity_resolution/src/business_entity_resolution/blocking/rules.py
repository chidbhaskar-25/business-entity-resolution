"""Blocking key generation rules for multi-strategy candidate generation."""

import re
from typing import Dict, List, Set
from ..preprocessing.normalizer import TextNormalizer
from ..preprocessing.tokenizer import TextTokenizer


class BlockingKeyGenerator:
    """Generates multi-strategy blocking keys for business records."""

    COMMON_STOPWORDS = {
        "and", "the", "for", "with", "near", "opp", "road", "street",
        "behind", "floor", "building", "complex", "center", "centre",
        "market", "nagar", "colony", "plaza", "avenue", "lane", "city",
        "india", "usa", "france", "north", "south", "east", "west"
    }

    def __init__(
        self,
        min_token_len: int = 3,
        prefix_len: int = 6,
        use_exact_name: bool = True,
        use_name_tokens: bool = True,
        use_name_prefix: bool = True,
        use_address_numbers: bool = True,
        use_name_bigram: bool = True,
    ):
        self.min_token_len = min_token_len
        self.prefix_len = prefix_len
        self.use_exact_name = use_exact_name
        self.use_name_tokens = use_name_tokens
        self.use_name_prefix = use_name_prefix
        self.use_address_numbers = use_address_numbers
        self.use_name_bigram = use_name_bigram

    def generate_keys(self, business_name: str, business_address: str) -> Set[str]:
        """Extracts all active blocking keys for a single record."""
        keys: Set[str] = set()

        # 1. Cleaned name without legal suffixes
        clean_name = TextNormalizer.normalize_name(business_name, strip_legal=True)

        if self.use_exact_name and clean_name:
            compact_name = re.sub(r"\s+", "", clean_name)
            if len(compact_name) >= 3:
                keys.add(f"exact:{compact_name}")

        # 2. Name prefix blocking
        if self.use_name_prefix and clean_name:
            compact_name = re.sub(r"\s+", "", clean_name)
            if len(compact_name) >= self.prefix_len:
                keys.add(f"pfx:{compact_name[:self.prefix_len]}")

        # 3. Name token blocking
        tokens = TextTokenizer.tokenize_words(clean_name, min_len=self.min_token_len)
        sig_tokens = [t for t in tokens if t not in self.COMMON_STOPWORDS]

        if self.use_name_tokens:
            for tok in sig_tokens:
                keys.add(f"tok:{tok}")

        # 4. First two words bigram
        if self.use_name_bigram and len(sig_tokens) >= 2:
            keys.add(f"bi:{sig_tokens[0]}_{sig_tokens[1]}")

        # 5. Address number blocking (numbers >= 3 digits or house numbers)
        if self.use_address_numbers and business_address:
            clean_addr = TextNormalizer.normalize_address(business_address)
            numbers = TextTokenizer.extract_numbers(clean_addr)
            for num in numbers:
                if len(num) >= 2 and num not in ("1", "2", "3", "4", "5"):
                    keys.add(f"addr_num:{num}")

        return keys
