"""Address-based candidate blocking strategies."""

from typing import List, Optional, Set
from ..preprocessing.address_normalizer import AddressNormalizer, NormalizedAddress


class AddressBlocker:
    """Generates blocking keys based on address components and numeric identifiers."""

    COMMON_ADDRESS_STOPWORDS = {
        "road", "street", "avenue", "lane", "drive", "highway", "court",
        "near", "opposite", "behind", "floor", "building", "flat", "house",
        "number", "block", "sector", "city", "town", "state", "india", "usa", "france",
    }

    def __init__(
        self,
        normalizer: Optional[AddressNormalizer] = None,
        min_token_len: int = 4,
        use_numeric: bool = True,
        use_locality_tokens: bool = True,
    ):
        self.normalizer = normalizer or AddressNormalizer()
        self.min_token_len = min_token_len
        self.use_numeric = use_numeric
        self.use_locality_tokens = use_locality_tokens

    def generate_keys(self, raw_address: str) -> Set[str]:
        """Extracts address blocking keys from raw address."""
        if not raw_address or not str(raw_address).strip():
            return set()

        norm: NormalizedAddress = self.normalizer.normalize(raw_address)
        if norm.is_empty:
            return set()

        keys: Set[str] = set()

        # 1. Numeric tokens (house numbers, postal codes, phone segments)
        if self.use_numeric:
            for num in norm.numeric_tokens:
                # Discard trivial single digits 1-5 to prevent broad buckets
                if len(num) >= 2 or (len(num) == 1 and num not in ("1", "2", "3", "4", "5")):
                    keys.add(f"addr_num:{num}")

        # 2. Locality and significant street tokens
        if self.use_locality_tokens:
            for tok in norm.tokens:
                if (
                    len(tok) >= self.min_token_len
                    and not tok.isdigit()
                    and tok not in self.COMMON_ADDRESS_STOPWORDS
                ):
                    keys.add(f"addr_tok:{tok}")

        return keys
