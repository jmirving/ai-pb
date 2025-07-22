import logging
from typing import Tuple, Dict, List
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from pbai.utils import champ_enum
from pbai.utils.draft_order import DRAFT_ORDER
from pbai.data.processors import DraftProcessor
from .data_ingestion_service import DataIngestionService

class DraftDataset(Dataset):
    """
    Dataset for transforming Oracle's Elixir draft data (via DataIngestionService) into model-ready tensors.
    This class performs feature engineering, encoding, and sample expansion for model training.
    It does NOT perform any file reading, raw data cleaning, or caching directly.
    """
    
    def __init__(self, ingestion_service: DataIngestionService):
        # Aggregate and filter the raw data
        self.data = self.aggregate_training_data(ingestion_service)
        # Champion enum and mapping setup
        # This mapping is used to convert champion names to sequential indices for model input/output.
        self.champ_enum = champ_enum.create_champ_enum()
        self.champion2idx = {}
        self.idx2champion = {}
        # Add MISSING as index 0 (valid input, never valid output)
        self.champion2idx['MISSING'] = 0
        self.idx2champion[0] = 'MISSING'
        # Map real champions to indices 1-170 (excluding 'MISSING')
        real_champions = [name for name, member in self.champ_enum.__members__.items() if name != 'MISSING']
        for i, champ_name in enumerate(real_champions, start=1):
            self.champion2idx[champ_name] = i
            self.idx2champion[i] = champ_name
        self.num_champions = len(self.champion2idx)
        # TODO: If you want the model to learn series-level strategy (across all games in a series),
        #       increase draft_features (e.g., to 100) and provide the full series draft history as input.
        self.draft_features = 20  # Used for model/processor input shape
        # Initialize the draft processor for feature engineering and normalization
        self._initialize_processors()

        # --- Preprocessing: Build samples for each draft event ---
        self.samples = self._preprocess_samples()
    
    @staticmethod
    def aggregate_training_data(service: DataIngestionService) -> pd.DataFrame:
        """
        Aggregate and filter Oracle's Elixir data for model training:
        - Only use the latest patch
        - Only use team rows (participantid 100, 200)
        - Ensure data is ordered by seriesid, gameid, and draft event order
        """
        logging.info("Aggregating training data")
        # 1. Get all data
        all_data = service.get_all_data_df()
        # 2. Find the latest patch
        if all_data['patch'].dtype == object:
            # Try to convert to float for proper sorting if possible
            try:
                all_data['patch_float'] = all_data['patch'].astype(float)
                latest_patch = all_data.sort_values('patch_float')['patch'].iloc[-1]
            except Exception:
                latest_patch = all_data.sort_values('patch')['patch'].iloc[-1]
        else:
            latest_patch = all_data.sort_values('patch')['patch'].iloc[-1]
        # 3. Filter to only rows from the latest patch
        patch_data = all_data[all_data['patch'] == latest_patch].copy()
        # 4. Filter to only team rows (participantid 100 or 200)
        team_data = patch_data[patch_data['participantid'].isin([100, 200])].copy()
        # 5. Infer series IDs based on teamid, league, split, year, and game number reset
        team_data = DraftDataset._infer_series_ids(team_data)
        # 6. Order by seriesid, gameid, and draft event order (if available)
        sort_cols = [col for col in ['seriesid', 'gameid', 'eventid', 'pickbannumber', 'order'] if col in team_data.columns]
        if sort_cols:
            team_data = team_data.sort_values(sort_cols).reset_index(drop=True)
        return team_data

    @staticmethod
    def _infer_series_ids(df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign a unique seriesid to each row based on league, split, year, teamid (blue), teamid (red),
        and game number reset. Uses teamid for both blue and red teams. Treats a reset of game number to 1
        as the start of a new series. Adds a 'seriesid' column to the DataFrame.
        """
        # Sort to ensure games are grouped by matchup and order
        df = df.sort_values(['league', 'split', 'year', 'gameid', 'teamid', 'game']).reset_index(drop=True)
        series_ids = []
        series_counters = {}  # key: matchup key, value: current series number
        last_game_numbers = {}  # key: matchup key, value: last game number

        for idx, row in df.iterrows():
            teamid = row['teamid']
            gameid = row['gameid']
            league = row['league']
            split = row['split']
            year = row['year']
            other_teamid = df[(df['gameid'] == gameid) & (df['teamid'] != teamid)]['teamid'].values[0]
            matchup = tuple(sorted([str(teamid), str(other_teamid)]))
            key = (league, split, year, matchup)
            game_number = row['game']

            # Start a new series if this is the first time we've seen this matchup, or if game_number resets to 1
            if key not in series_counters or game_number == 1:
                series_counters[key] = series_counters.get(key, 0) + 1

            series_id = f"{league}_{split}_{year}_{matchup[0]}_{matchup[1]}_S{series_counters[key]}"
            series_ids.append(series_id)
            last_game_numbers[key] = game_number

        df = df.copy()
        df['seriesid'] = series_ids
        return df
    
    def _initialize_processors(self):
        """Initialize only the draft processor."""
        self.draft_processor = DraftProcessor()
    
    def _normalize_champion_id(self, champion_name: str) -> int:
        """Convert champion name to sequential index (0, 1, 2, ...)."""
        if pd.isna(champion_name) or champion_name == 'nan':
            champion_name = "MISSING"
        
        # Convert to uppercase for consistency
        champion_name = str(champion_name).upper()
        
        # Return sequential index
        return self.champion2idx.get(champion_name, self.champion2idx['MISSING'])
    
    def get_output_mask(self, already_picked_or_banned):
        """
        Create a mask for available champions for a draft event.

        Args:
            already_picked_or_banned (set): Champion names or IDs that are no longer available.

        Returns:
            np.ndarray: 1D array of shape [num_champions - 1], 1 if available, 0 if not.
        """
        mask = np.ones(self.num_champions - 1, dtype=np.float32)  # Exclude MISSING (index 0)
        for champ in already_picked_or_banned:
            idx = self.champion2idx.get(champ)
            if idx is not None and idx > 0:  # Exclude MISSING
                mask[idx - 1] = 0  # Output indices are 0-based for real champions
        return mask

    def _preprocess_samples(self):
        """
        Build a list of samples for each draft event (pick/ban) using the correct draft order.
        Each sample is a dict with draft_sequence, target, already_picked_or_banned.
        Handles Fearless Draft by including picks from previous games in the series.
        """
        logging.info("Preprocessing samples")
        samples = []
        # Define the correct draft order (20 steps)
        DRAFT_ORDER = [
            ('blue', 'ban', 1), ('red', 'ban', 1),
            ('blue', 'ban', 2), ('red', 'ban', 2),
            ('blue', 'ban', 3), ('red', 'ban', 3),
            ('blue', 'pick', 1), ('red', 'pick', 1),
            ('red', 'pick', 2), ('blue', 'pick', 2),
            ('blue', 'pick', 3), ('red', 'pick', 3),
            ('red', 'ban', 4), ('blue', 'ban', 4),
            ('red', 'ban', 5), ('blue', 'ban', 5),
            ('red', 'pick', 4), ('blue', 'pick', 4),
            ('blue', 'pick', 5), ('red', 'pick', 5)
        ]
        grouped = self.data.groupby(['seriesid', 'gameid', 'side'])
        for (seriesid, gameid, side), group in grouped:
            logging.info(f"Series {seriesid}, game {gameid}, side {side}")
            # Only process a single side per game, since we are returning a total
            # if side != 'blue':
            #     continue
            # Track picks from previous games in the series (for Fearless Draft)
            series_picked = set()
            # Get all previous games in the series and add their picks to the series_picked set
            if gameid != 1:
                prev_games = self.data[(self.data['seriesid'] == seriesid) & (self.data['gameid'] < gameid)]
                for _, prev_row in prev_games.iterrows():
                    for i in range(1, 6):
                        pick_col = f'pick{i}'
                        if pick_col in prev_row and not pd.isna(prev_row[pick_col]):
                            series_picked.add(prev_row[pick_col])
            logging.info(f"Already picked in series: {series_picked}")
            # Create the draft sequence
            draft_sequence = [0] * self.draft_features
            # Get the blue team row
            blue_row = self.data[(self.data['seriesid'] == seriesid) & (self.data['gameid'] == gameid) & (self.data['side'] == 'Blue')]
            # Find the associated red team row
            red_row = self.data[(self.data['seriesid'] == seriesid) & (self.data['gameid'] == gameid) & (self.data['side'] == 'Red')]
            # Interleave the blue and red rows into the draft sequence via the DRAFT_ORDER
            draft_sequence = []
            for side, action_type, action_number in DRAFT_ORDER:
                if side == 'blue':
                    if action_type == 'ban':
                        champ_name = blue_row[f'ban{action_number}'].values[0]
                        champ_idx = self._normalize_champion_id(champ_name)
                        draft_sequence.append(champ_idx)
                    else:
                        champ_name = blue_row[f'pick{action_number}'].values[0]
                        champ_idx = self._normalize_champion_id(champ_name)
                        draft_sequence.append(champ_idx)
                else:
                    if action_type == 'ban':
                        champ_name = red_row[f'ban{action_number}'].values[0]
                        champ_idx = self._normalize_champion_id(champ_name)
                        draft_sequence.append(champ_idx)
                    else:
                        champ_name = red_row[f'pick{action_number}'].values[0]
                        champ_idx = self._normalize_champion_id(champ_name)
                        draft_sequence.append(champ_idx)
            logging.info(f"Draft sequence: {draft_sequence}")

            # Store the sample
            samples.append({
                'draft_sequence': draft_sequence.copy(),
                'target': target,
                'already_picked_or_banned': series_picked
            })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]
        #logging.info(f"Row: {row}")
        draft_sequence, target, already_picked_or_banned = self.draft_processor.process(row)
        output_mask = self.get_output_mask(already_picked_or_banned)
        # Shift target down by 1 if not MISSING (0)
        target = target - 1 if target > 0 else 0
        return {
            'draft_sequence': torch.tensor(draft_sequence, dtype=torch.long),
            'target': torch.tensor(target, dtype=torch.long),
            'output_mask': torch.tensor(output_mask, dtype=torch.float32),
        }