"""Composite multi-strategy country-aware blocker."""

from collections import defaultdict
from typing import Dict, List, Optional, Set
from .address_blocking import AddressBlocker
from .name_blocking import NameBlocker
from ..preprocessing.text_utils import strip_accents


class CompositeBlocker:
    """Combines name and address blocking strategies within country partitions."""

    def __init__(
        self,
        name_blocker: Optional[NameBlocker] = None,
        address_blocker: Optional[AddressBlocker] = None,
        max_posting_size: int = 500,
    ):
        self.name_blocker = name_blocker or NameBlocker()
        self.address_blocker = address_blocker or AddressBlocker()
        self.max_posting_size = max_posting_size

        # Inverted index structure:
        # country -> blocking_key -> list of target_entity_id
        self.inverted_index: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.total_target_records = 0

    @staticmethod
    def normalize_country_label(country: str) -> str:
        """Normalizes country label for open-set partitioning."""
        if not country or not str(country).strip():
            return "UNKNOWN"
        return strip_accents(str(country)).strip().upper()

    def generate_record_keys(self, name: str, address: str) -> Set[str]:
        """Extracts combined multi-strategy blocking keys for a record."""
        name_keys = self.name_blocker.generate_keys(name)
        addr_keys = self.address_blocker.generate_keys(address)
        return name_keys | addr_keys

    def index_target_record(
        self,
        entity_id: str,
        name: str,
        address: str,
        country: str,
    ) -> None:
        """Indexes a single target record (S2/S3) into country partition."""
        c_label = self.normalize_country_label(country)
        country_index = self.inverted_index[c_label]
        keys = self.generate_record_keys(name, address)

        for k in keys:
            posting = country_index[k]
            if len(posting) < self.max_posting_size:
                posting.append(entity_id)

        self.total_target_records += 1

    def retrieve_candidates(
        self,
        name: str,
        address: str,
        country: str,
    ) -> Set[str]:
        """Retrieves candidate target IDs matching any blocking key in the same country."""
        c_label = self.normalize_country_label(country)
        country_index = self.inverted_index.get(c_label)
        if not country_index:
            return set()

        keys = self.generate_record_keys(name, address)
        candidates: Set[str] = set()

        for k in keys:
            if k in country_index:
                candidates.update(country_index[k])

        return candidates
