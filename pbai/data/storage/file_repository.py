"""
File-based implementation of DataRepository for data persistence.
"""

import os
import pickle
import glob
from pathlib import Path
from typing import Optional, Dict, List, Any
import pandas as pd
import logging
from .repository import DataRepository


class FileRepository(DataRepository):
    """File-based implementation of DataRepository using parquet files"""
    
    def __init__(self, base_path: str = "data/processed"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Directory structure
        self.all_dir = self.base_path / "all"
        self.players_dir = self.base_path / "players"
        self.teams_dir = self.base_path / "teams"
        self.series_dir = self.base_path / "series"
        self.features_dir = self.base_path / "features"
        self.cache_dir = self.base_path / "cache"
        
        # Create all directories
        for dir_path in [self.all_dir, self.players_dir, self.teams_dir, 
                        self.series_dir, self.features_dir, self.cache_dir]:
            dir_path.mkdir(exist_ok=True)
        
        logging.info(f"FileRepository initialized with base path: {self.base_path}")
    
    def save_all_raw_data(self, all_data: pd.DataFrame, version: str) -> None:
        """Save all data with version tracking"""
        file_path = self.all_dir / f"all_{version}.parquet"
        all_data.to_parquet(file_path)
        metadata = {
            'version': version,
            'row_count': len(all_data),
            'columns': list(all_data.columns),
            'created_at': pd.Timestamp.now(),
            'data_type': 'all'
        }
        with open(self.all_dir / f"all_{version}_metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        logging.info(f"Saved all data: {len(all_data)} rows to {file_path}")
    
    def load_all_raw_data(self) -> pd.DataFrame:
        """Load most recent all data"""
        latest_file = self._get_latest_file(self.all_dir, "all_*.parquet")
        if not latest_file:
            raise FileNotFoundError("No all data found")
        all_data = pd.read_parquet(latest_file)
        logging.info(f"Loaded all data: {len(all_data)} rows from {latest_file}")
        return all_data
    
    def save_raw_series_data(self, series: pd.DataFrame) -> None:
        """Save series grouping and fearless detection results"""
        series.to_parquet(self.series_dir / "series_data.parquet")
        metadata = {
            'row_count': len(series),
            'columns': list(series.columns),
            'created_at': pd.Timestamp.now(),
            'data_type': 'series'
        }
        with open(self.series_dir / "series_metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        logging.info(f"Saved series data: {len(series)} rows")
    
    def load_raw_series_data(self, series_id: str) -> Optional[Dict]:
        """Get series context for fearless draft calculations"""
        try:
            series_data = pd.read_parquet(self.series_dir / "series_data.parquet")
            series_row = series_data[series_data['series_id'] == series_id]
            if len(series_row) == 0:
                return None
            return series_row.iloc[0].to_dict()
        except FileNotFoundError:
            return None
    
    # def save_player_features(self, features: pd.DataFrame) -> None:
    #     """Save computed player features (champion rates, recent form)"""
    #     features.to_parquet(self.features_dir / "player_features.parquet")
    #     metadata = {
    #         'row_count': len(features),
    #         'columns': list(features.columns),
    #         'created_at': pd.Timestamp.now(),
    #         'data_type': 'player_features'
    #     }
    #     with open(self.features_dir / "player_features_metadata.pkl", 'wb') as f:
    #         pickle.dump(metadata, f)
    #     logging.info(f"Saved player features: {len(features)} rows")
    
    # def load_player_features(self, player_ids: List[str]) -> pd.DataFrame:
    #     """Load player features for specific players"""
    #     try:
    #         features = pd.read_parquet(self.features_dir / "player_features.parquet")
    #         if player_ids:
    #             filtered = features[features['playerid'].isin(player_ids)]
    #             if isinstance(filtered, pd.Series):
    #                 filtered = filtered.to_frame().T
    #             return filtered.reset_index(drop=True)
    #         if isinstance(features, pd.Series):
    #             features = features.to_frame().T
    #         return features.reset_index(drop=True)
    #     except FileNotFoundError:
    #         return pd.DataFrame()
    
    def save_raw_player_data(self, player_data: pd.DataFrame, version: str) -> None:
        """Save player data (participantid 1-10)"""
        file_path = self.players_dir / f"players_{version}.parquet"
        player_data.to_parquet(file_path)
        metadata = {
            'version': version,
            'row_count': len(player_data),
            'columns': list(player_data.columns),
            'created_at': pd.Timestamp.now(),
            'data_type': 'player'
        }
        with open(self.players_dir / f"players_{version}_metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        logging.info(f"Saved player data: {len(player_data)} rows to {file_path}")
    
    def load_raw_player_data(self) -> pd.DataFrame:
        """Load player data"""
        latest_file = self._get_latest_file(self.players_dir, "players_*.parquet")
        if not latest_file:
            raise FileNotFoundError("No processed player data found")
        player_data = pd.read_parquet(latest_file)
        logging.info(f"Loaded player data: {len(player_data)} rows from {latest_file}")
        return player_data
    
    def save_raw_team_data(self, team_data: pd.DataFrame, version: str) -> None:
        """Save team data (participantid 100, 200)"""
        file_path = self.teams_dir / f"teams_{version}.parquet"
        team_data.to_parquet(file_path)
        metadata = {
            'version': version,
            'row_count': len(team_data),
            'columns': list(team_data.columns),
            'created_at': pd.Timestamp.now(),
            'data_type': 'team'
        }
        with open(self.teams_dir / f"teams_{version}_metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        logging.info(f"Saved team data: {len(team_data)} rows to {file_path}")
    
    def load_raw_team_data(self) -> pd.DataFrame:
        """Load team data"""
        latest_file = self._get_latest_file(self.teams_dir, "teams_*.parquet")
        if not latest_file:
            raise FileNotFoundError("No processed team data found")
        team_data = pd.read_parquet(latest_file)
        logging.info(f"Loaded team data: {len(team_data)} rows from {latest_file}")
        return team_data
    
    def is_data_stale(self, data_type: str, source_modified_time: float) -> bool:
        """Check if cached data is stale compared to source"""
        if data_type == "all":
            latest_file = self._get_latest_file(self.all_dir, "all_*.parquet")
            if not latest_file:
                return True
            cache_file = latest_file
        elif data_type == "players":
            latest_file = self._get_latest_file(self.players_dir, "players_*.parquet")
            if not latest_file:
                return True
            cache_file = latest_file
        elif data_type == "teams":
            latest_file = self._get_latest_file(self.teams_dir, "teams_*.parquet")
            if not latest_file:
                return True
            cache_file = latest_file
        elif data_type == "series":
            cache_file = self.series_dir / "series_data.parquet"
        elif data_type == "player_features":
            cache_file = self.features_dir / "player_features.parquet"
        else:
            cache_file = self.cache_dir / f"{data_type}.parquet"
        if not cache_file.exists():
            return True
        cache_modified_time = cache_file.stat().st_mtime
        return cache_modified_time < source_modified_time
    
    def cleanup_old_versions(self, keep_versions: int = 3) -> None:
        """Clean up old data versions, keeping only the most recent"""
        for data_type in ["all", "players", "teams"]:
            if data_type == "all":
                data_files = list(self.all_dir.glob("all_*.parquet"))
                metadata_files = list(self.all_dir.glob("all_*_metadata.pkl"))
            elif data_type == "players":
                data_files = list(self.players_dir.glob("players_*.parquet"))
                metadata_files = list(self.players_dir.glob("players_*_metadata.pkl"))
            elif data_type == "teams":
                data_files = list(self.teams_dir.glob("teams_*.parquet"))
                metadata_files = list(self.teams_dir.glob("teams_*_metadata.pkl"))
            
            self._cleanup_files(data_files, metadata_files, keep_versions)
    
    def _get_latest_file(self, directory: Path, pattern: str) -> Optional[Path]:
        """Get the most recent file matching pattern"""
        files = list(directory.glob(pattern))
        if not files:
            return None
        
        # Sort by modification time, newest first
        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return files[0]
    
    def _cleanup_files(self, data_files: List[Path], metadata_files: List[Path], keep_versions: int) -> None:
        """Clean up old files, keeping only the most recent versions"""
        if len(data_files) <= keep_versions:
            return
        
        # Sort by modification time, newest first
        data_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        metadata_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Remove old files
        for file in data_files[keep_versions:]:
            try:
                file.unlink()
                logging.info(f"Removed old data file: {file}")
            except OSError as e:
                logging.error(f"Error removing {file}: {e}")
        
        for file in metadata_files[keep_versions:]:
            try:
                file.unlink()
                logging.info(f"Removed old metadata file: {file}")
            except OSError as e:
                logging.error(f"Error removing {file}: {e}") 