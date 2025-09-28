import logging
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Sequence, Union, Iterable
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from pbai.utils import champ_enum
from pbai.utils.champion_sanitizer import ChampionSanitizer
from pbai.utils.draft_order import DRAFT_ORDER
from pbai.data.processors import DraftProcessor
from .data_ingestion_service import DataIngestionService

class DraftDataset(Dataset):
    """
    Dataset for transforming Oracle's Elixir draft data (via DataIngestionService) into model-ready tensors.
    This class performs feature engineering, encoding, and sample expansion for model training.
    It does NOT perform any file reading, raw data cleaning, or caching directly.
    """
    
    def __init__(
        self,
        ingestion_service: DataIngestionService,
        export_dir: Optional[Union[str, Path]] = None,
        export_formats: Optional[Sequence[str]] = None,
    ):
        # Aggregate and filter the raw data
        self.data = self.aggregate_training_data(
            ingestion_service,
            export_dir=export_dir,
            export_formats=export_formats,
        )
        # Champion enum and mapping setup
        # This mapping is used to convert champion names to sequential indices for model input/output.
        self.champ_enum = champ_enum.create_champ_enum()
        self.champion_sanitizer = ChampionSanitizer()
        self.champion2idx = {}
        self.idx2champion = {}
        # Add MISSING as index 0 (valid input, never valid output)
        self.missing_key = self.champion_sanitizer.sanitize('MISSING')
        self.champion2idx[self.missing_key] = 0
        self.idx2champion[0] = 'MISSING'
        # Map real champions to indices 1-170 (excluding 'MISSING')
        real_champions = [name for name, member in self.champ_enum.__members__.items() if name != 'MISSING']
        for i, champ_name in enumerate(real_champions, start=1):
            sanitized_name = self.champion_sanitizer.sanitize(champ_name)
            self.champion2idx[sanitized_name] = i
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
    def _normalize_export_formats(
        export_formats: Optional[Sequence[str]],
        default_to_csv: bool = False,
    ) -> List[str]:
        if export_formats is None:
            return ["csv"] if default_to_csv else []
        if isinstance(export_formats, str):
            normalized_iterable: Iterable = [export_formats]
        else:
            normalized_iterable = export_formats
        normalized: List[str] = []
        for fmt in normalized_iterable:
            if not fmt:
                continue
            normalized.append(str(fmt).lower())
        if not normalized and default_to_csv:
            normalized.append("csv")
        return normalized

    @staticmethod
    def _export_dataframe(
        dataframe: pd.DataFrame,
        export_dir: Optional[Union[str, Path]],
        stem: str,
        export_formats: Sequence[str],
    ) -> None:
        if not export_dir or not export_formats:
            return
        export_path = Path(export_dir)
        export_path.mkdir(parents=True, exist_ok=True)
        normalized = {fmt.lower() for fmt in export_formats}
        if "csv" in normalized:
            target_path = export_path / f"{stem}.csv"
            dataframe.to_csv(target_path, index=False)

    @staticmethod
    def aggregate_training_data(
        service: DataIngestionService,
        export_dir: Optional[Union[str, Path]] = None,
        export_formats: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """
        Aggregate and filter Oracle's Elixir data for model training:
        - Only use the latest patch
        - Only use team rows (participantid 100, 200)
        - Ensure data is ordered by seriesid, gameid, and draft event order
        """
        logging.info("Aggregating training data")
        normalized_formats = DraftDataset._normalize_export_formats(
            export_formats,
            default_to_csv=export_dir is not None,
        )
        service_export_formats = normalized_formats or None

        # 1. Get all data
        all_data = service.get_all_data_df(export_formats=service_export_formats)
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

        # Export intermediate tables when requested
        DraftDataset._export_dataframe(
            all_data,
            export_dir,
            "all_data",
            normalized_formats,
        )
        DraftDataset._export_dataframe(
            patch_data,
            export_dir,
            "patch_data",
            normalized_formats,
        )
        DraftDataset._export_dataframe(
            team_data,
            export_dir,
            "team_data",
            normalized_formats,
        )
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
        print("[infer_series_ids] starting with", len(df), "rows")
        print("[infer_series_ids] incoming columns:", list(df.columns))
        # Sort to ensure games are grouped by matchup and order
        sort_columns = ['league', 'split', 'year']
        if 'date' in df.columns:
            sort_columns.append('date')
        sort_columns.extend(['gameid', 'teamid', 'game'])
        print("[infer_series_ids] sorting by columns:", sort_columns)
        df = df.sort_values(sort_columns).reset_index(drop=True)
        print("[infer_series_ids] after sort head:\n", df.head())
        series_counters = {}  # key: matchup key, value: current series number
        last_game_numbers = {}  # key: matchup key, value: last observed game number
        last_dates = {}  # key: matchup key, value: last observed date (no series spans multiple days)

        parsed_dates = pd.to_datetime(df.get('date'), errors='coerce') if 'date' in df.columns else None
        if parsed_dates is not None:
            print("[infer_series_ids] parsed date example:", parsed_dates.head())

        rows_to_keep = []
        series_ids = []
        drop_gameids = set()

        for idx, row in df.iterrows():
            print("[infer_series_ids] processing index", idx, "gameid", row['gameid'], "teamid", row['teamid'])
            teamid = row['teamid']
            gameid = row['gameid']
            league = row['league']
            split = row['split']
            year = row['year']
            opponent_rows = df[(df['gameid'] == gameid) & (df['teamid'] != teamid)]

            if opponent_rows.empty:
                print(
                    "[infer_series_ids] no opponent rows found for gameid",
                    gameid,
                    "teamid",
                    teamid,
                )
                if gameid not in drop_gameids:
                    logging.warning(
                        "Missing opponent data for league=%s split=%s year=%s gameid=%s; dropping game",
                        league,
                        split,
                        year,
                        gameid,
                    )
                drop_gameids.add(gameid)
                continue

            other_teamid = opponent_rows['teamid'].iloc[0]
            matchup = tuple(sorted([str(teamid), str(other_teamid)]))
            key = (league, split, year, matchup)
            game_number = row['game']
            print(
                "[infer_series_ids] matchup",
                matchup,
                "key",
                key,
                "game_number",
                game_number,
            )
            if parsed_dates is not None:
                parsed_value = parsed_dates.iloc[idx]
                current_date = parsed_value.date() if not pd.isna(parsed_value) else None
            else:
                current_date = None

            # Start a new series if this is the first time we've seen this matchup
            # or if the game counter reset (i.e., fearless draft should restart).
            if (
                key not in series_counters
                or game_number < last_game_numbers.get(key, 0)
                or (
                    current_date is not None
                    and last_dates.get(key) is not None
                    and current_date != last_dates.get(key)
                )
            ):
                print(
                    "[infer_series_ids] starting new series for key",
                    key,
                    "previous game",
                    last_game_numbers.get(key),
                    "current",
                    game_number,
                    "current_date",
                    current_date,
                    "last_date",
                    last_dates.get(key),
                )
                series_counters[key] = series_counters.get(key, 0) + 1

            series_id = f"{league}_{split}_{year}_{matchup[0]}_{matchup[1]}_S{series_counters[key]}"
            print(
                "[infer_series_ids] assigned series_id",
                series_id,
                "for index",
                idx,
            )
            rows_to_keep.append(idx)
            series_ids.append(series_id)
            last_game_numbers[key] = game_number
            if current_date is not None:
                print(
                    "[infer_series_ids] updating last date for key",
                    key,
                    "to",
                    current_date,
                )
                last_dates[key] = current_date

        df = df.loc[rows_to_keep].copy()
        print("[infer_series_ids] kept", len(df), "rows after opponent filtering")
        df['seriesid'] = series_ids
        print("[infer_series_ids] assigned series ids head:\n", df[['gameid', 'teamid', 'seriesid']].head())
        df = df[~df['gameid'].isin(drop_gameids)].reset_index(drop=True)
        if drop_gameids:
            print("[infer_series_ids] dropped gameids due to missing opponent:", drop_gameids)
        print("[infer_series_ids] final row count", len(df))
        return df
    
    def _initialize_processors(self):
        """Initialize only the draft processor."""
        self.draft_processor = DraftProcessor()
    
    def _normalize_champion_id(self, champion_name: str) -> int:
        """Convert champion name to sequential index (0, 1, 2, ...)."""
        if pd.isna(champion_name) or champion_name == 'nan':
            champion_name = "MISSING"

        champion_name = self.champion_sanitizer.sanitize(champion_name)
        if not champion_name:
            champion_name = self.missing_key

        return self.champion2idx.get(champion_name, self.champion2idx[self.missing_key])
    
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
        # Group by series/game so that a single iteration has access to both blue and red
        # rows; we need information from *both* sides to reconstruct the pick/ban order.
        grouped_games = self.data.groupby(['seriesid', 'gameid'])

        for (seriesid, gameid), game_rows in grouped_games:
            logging.info(f"Processing series %s game %s", seriesid, gameid)

            blue_rows = game_rows[game_rows['side'].str.lower() == 'blue']
            red_rows = game_rows[game_rows['side'].str.lower() == 'red']

            if blue_rows.empty or red_rows.empty:
                logging.warning(
                    "Missing blue or red side data for series %s game %s; skipping", seriesid, gameid
                )
                continue

            blue_row = blue_rows.iloc[0]
            red_row = red_rows.iloc[0]

            # Seed fearless draft state with picks from previous games in the series
            fearless_picks = set()
            previous_games = self.data[
                (self.data['seriesid'] == seriesid) & (self.data['gameid'] < gameid)
            ]
            for _, prev_row in previous_games.iterrows():
                for pick_number in range(1, 6):
                    pick_col = f'pick{pick_number}'
                    if pick_col in prev_row and not pd.isna(prev_row[pick_col]):
                        # Fearless draft only restricts previously *picked* champions.
                        sanitized_pick = self.champion_sanitizer.sanitize(prev_row[pick_col])
                        if sanitized_pick:
                            fearless_picks.add(sanitized_pick)

            used_champions = set(fearless_picks)
            draft_sequence = [0] * self.draft_features

            # DRAFT_ORDER comes from pbai.utils.draft_order and encodes (side, action, slot)
            for event_index, (side, action_type, action_number) in enumerate(DRAFT_ORDER):
                row = blue_row if side == 'blue' else red_row
                column_prefix = 'ban' if action_type == 'ban' else 'pick'
                column_name = f'{column_prefix}{action_number}'
                champion_name = row.get(column_name)
                champion_index = self._normalize_champion_id(champion_name)

                if champion_index == 0:
                    logging.warning(
                        "Encountered missing champion for %s %s %s in series %s game %s; skipping sample",
                        side,
                        action_type,
                        action_number,
                        seriesid,
                        gameid,
                    )
                    continue

                samples.append(
                    {
                        'draft_sequence': draft_sequence.copy(),
                        'target': champion_index,
                        'already_picked_or_banned': set(used_champions),
                    }
                )

                if pd.notna(champion_name):
                    sanitized_name = self.champion_sanitizer.sanitize(champion_name)
                    if sanitized_name:
                        used_champions.add(sanitized_name)

                draft_sequence[event_index] = champion_index

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