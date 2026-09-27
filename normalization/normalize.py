"""Separate, conservative text representations for the entity-resolution task.

These functions never mutate their input. Callers should retain supplied raw
values alongside any returned representation.
"""

from __future__ import annotations

import unicodedata
from typing import Optional


def conservative_normalize(value: Optional[str]) -> Optional[str]:
    """Apply NFKC, casefolding, punctuation-to-space, and whitespace cleanup.

    ``None`` remains ``None`` and an empty or whitespace-only string becomes
    ``""``. This function does not transliterate, correct spelling, remove
    suffixes, reorder tokens, or consult dictionaries.
    """
    if value is None:
        return None
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = "".join(
        " " if unicodedata.category(char).startswith("P") else char
        for char in normalized
    )
    return " ".join(normalized.split())


def normalize_name(value: Optional[str]) -> Optional[str]:
    """Return the conservative name representation, preserving missingness."""
    return conservative_normalize(value)


def normalize_address(value: Optional[str]) -> Optional[str]:
    """Return the conservative address representation, preserving missingness."""
    return conservative_normalize(value)


_OBSERVED_ADDRESS_ABBREVIATIONS = {
    "st": "street",
    "street": "street",
    "dr": "drive",
    "drive": "drive",
    "ave": "avenue",
    "avenue": "avenue",
    "ln": "lane",
    "lane": "lane",
}


def normalize_address_abbreviations(value: Optional[str]) -> Optional[str]:
    """Return a separate, sample-derived abbreviation representation.

    Only the approved Stage 3.2 pairs are mapped. The source value and the
    conservative normalized address are not modified.
    """
    normalized = normalize_address(value)
    if normalized is None or normalized == "":
        return normalized
    return " ".join(
        _OBSERVED_ADDRESS_ABBREVIATIONS.get(token, token)
        for token in normalized.split()
    )


def tokenize_address(value: Optional[str]) -> Optional[tuple[str, ...]]:
    """Return ordered tokens from the conservative address representation.

    Empty/missing addresses return ``None`` so they cannot be mistaken for a
    matching empty token set. The raw and normalized strings remain available
    separately; tokenization does not reorder either string.
    """
    normalized = normalize_address(value)
    if normalized is None or normalized == "":
        return None
    return tuple(normalized.split())


def address_token_set(value: Optional[str]) -> Optional[frozenset[str]]:
    """Return an unordered token-set representation, or ``None`` if missing."""
    tokens = tokenize_address(value)
    if tokens is None:
        return None
    return frozenset(tokens)
