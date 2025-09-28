"""Utilities for normalising champion names across different data sources."""

from __future__ import annotations

import unicodedata
from typing import Optional


class ChampionSanitizer:
    """Apply consistent transformations to champion names used throughout the app."""

    def sanitize(self, champion_name: Optional[str]) -> str:
        """Return a normalised champion name suitable for dictionary lookups."""

        if champion_name is None:
            return ""

        cleaned = str(champion_name).strip()
        if not cleaned:
            return ""

        cleaned = cleaned.upper()
        cleaned = "".join(ch for ch in cleaned if not self._is_punctuation(ch))
        # Collapse repeated whitespace that can arise from punctuation removal.
        cleaned = " ".join(cleaned.split())
        return cleaned

    @staticmethod
    def _is_punctuation(character: str) -> bool:
        """Return True when the character is classified as punctuation in Unicode."""

        return unicodedata.category(character).startswith("P")

