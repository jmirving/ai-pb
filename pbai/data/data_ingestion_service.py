"""
Data service layer for raw data ingestion and cleaning from Oracle's Elixir data.
"""

import os
import pandas as pd
from typing import Optional, Dict
from .storage import DataRepository

class DataIngestionService:
    """Service layer for raw data ingestion and cleaning"""
    
    def __init__(self, repository: DataRepository, source_path: str):
        self.repository = repository
        self.source_path = source_path
    
    def _ensure_dataframe(self, result, context: str = "") -> pd.DataFrame:
        if isinstance(result, pd.DataFrame):
            return result
        elif isinstance(result, pd.Series):
            return result.to_frame().T
        else:
            raise TypeError(f"Expected DataFrame or Series from {context}")

    def get_all_data_df(self, force_refresh: bool = False) -> pd.DataFrame:
        """Get cleaned all_data DataFrame, refreshing if needed"""
        source_modified = os.path.getmtime(self.source_path)
        if force_refresh or self.repository.is_data_stale("all", source_modified):
            # Read and clean raw data
            source_data = pd.read_csv(self.source_path)
            # Optionally: add basic cleaning/validation here
            self.repository.save_all_raw_data(source_data, pd.Timestamp.now().strftime("%Y%m%d_%H%M%S"))
            return source_data
        else:
            result = self.repository.load_all_raw_data()
            return self._ensure_dataframe(result, "repository.load_all_data")

    def get_players_df(self, force_refresh: bool = False) -> pd.DataFrame:
        """Get cleaned player DataFrame (participantid 1-10)"""
        all_data_df = self.get_all_data_df(force_refresh=force_refresh)
        result = all_data_df[all_data_df['participantid'].between(1, 10)].copy()
        return self._ensure_dataframe(result, "player data selection")

    def get_teams_df(self, force_refresh: bool = False) -> pd.DataFrame:
        """Get cleaned team DataFrame (participantid 100, 200)"""
        all_data_df = self.get_all_data_df(force_refresh=force_refresh)
        result = all_data_df[all_data_df['participantid'].isin([100, 200])].copy()
        return self._ensure_dataframe(result, "team data selection")

    def get_series_df(self, series_id: str, force_refresh: bool = False) -> pd.DataFrame:
        """Get all team rows for a given series_id (participantid 100, 200)"""
        all_data_df = self.get_all_data_df(force_refresh=force_refresh)
        result = all_data_df[
            (all_data_df['participantid'].isin([100, 200])) &
            (all_data_df['seriesid'] == series_id)
        ].copy()
        return self._ensure_dataframe(result, f"series data selection for series_id={series_id}")

    def get_games_by_patch_df(self, patch: str, force_refresh: bool = False) -> pd.DataFrame:
        """Get all rows for a given patch version"""
        all_data_df = self.get_all_data_df(force_refresh=force_refresh)
        result = all_data_df[all_data_df['patch'] == patch].copy()
        return self._ensure_dataframe(result, f"games data selection for patch={patch}")