"""Name-based candidate blocking strategies."""

import re
from typing import List, Optional, Set
from ..preprocessing.name_normalizer import BusinessNameNormalizer, NormalizedName


class NameBlocker:
    """Generates blocking keys based on business names."""

    COMMON_NAME_STOPWORDS = {
        "and", "the", "for", "with", "all", "our", "new", "global", "best",
        "group", "enterprises", "enterprise", "services", "service", "solutions",
        "solution", "company", "india", "usa", "france", "industries", "industry",
    }

    def __init__(
        self,
        normalizer: Optional[BusinessNameNormalizer] = None,
        min_token_len: int = 3,
        prefix_len: int = 6,
        use_exact: bool = True,
        use_tokens: bool = True,
        use_prefix: bool = True,
        use_bigram: bool = True,
    ):
        self.normalizer = normalizer or BusinessNameNormalizer()
        self.min_token_len = min_token_len
        self.prefix_len = prefix_len
        self.use_exact = use_exact
        self.use_tokens = use_tokens
        self.use_prefix = use_prefix
        self.use_bigram = use_bigram

    def generate_keys(self, raw_name: str) -> Set[str]:
        """Extracts name blocking keys from raw business name."""
        if not raw_name or not str(raw_name).strip():
            return set()

        norm: NormalizedName = self.normalizer.normalize(raw_name)
        keys: Set[str] = set()

        # 1. Exact normalized name (without legal suffixes and spaces)
        clean_name = norm.clean_without_legal
        compact_name = re.sub(r"\s+", "", clean_name)
        if self.use_exact and compact_name and len(compact_name) >= 3:
            keys.add(f"name_exact:{compact_name}")

        # 2. Prefix blocking (approximate prefix matching)
        if self.use_prefix and compact_name and len(compact_name) >= self.prefix_len:
            keys.add(f"name_pfx:{compact_name[:self.prefix_len]}")

        # 3. Name token blocking
        sig_tokens = [
            t for t in norm.tokens
            if len(t) >= self.min_token_len and t not in self.COMMON_NAME_STOPWORDS
        ]
        if self.use_tokens:
            for tok in sig_tokens:
                keys.add(f"name_tok:{tok}")

        # 4. First two significant tokens bigram
        if self.use_bigram and len(sig_tokens) >= 2:
            keys.add(f"name_bi:{sig_tokens[0]}_{sig_tokens[1]}")

        return keys
