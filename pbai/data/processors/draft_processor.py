"""
Draft processor for handling draft sequence data (picks and bans).
"""

import pandas as pd
import torch
from typing import Dict, List, Optional
from .base_processor import BaseProcessor
from pbai.utils import champ_enum
from pbai.utils.draft_order import DRAFT_ORDER


class DraftProcessor(BaseProcessor):
    """Processor for extracting draft sequence, target, and already picked/banned champions from a row."""
    
    def process(self, row: pd.Series) -> tuple:
        """
        Given a preprocessed row, extract:
        - draft_sequence: list/array of champion indices (length 20)
        - target: the champion index to predict for this event
        - already_picked_or_banned: set of champion indices unavailable for this event
        """
        draft_sequence = self._extract_draft_sequence_from_row(row)
        target = self._extract_target_from_row(row)
        already_picked_or_banned = self._extract_already_picked_or_banned_from_row(row)
        return draft_sequence, target, already_picked_or_banned

    def _extract_draft_sequence_from_row(self, row):
        """
        Returns the draft sequence (list of champion indices, length 20) from the row dict.
        """
        return row['draft_sequence']

    def _extract_target_from_row(self, row: pd.Series):
        """
        Extracts the target champion index for this event from the row.
        Assumes the target column is already an index.
        """
        return row['target']

    def _extract_already_picked_or_banned_from_row(self, row: pd.Series):
        """
        Extracts the set of already picked or banned champion indices for this event from the row.
        Assumes a column 'already_picked_or_banned' exists as a list or set of indices.
        """
        return set(row['already_picked_or_banned']) 