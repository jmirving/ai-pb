"""
Abstract repository interface for data persistence.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Sequence
import pandas as pd


class DataRepository(ABC):
    """Abstract interface for data persistence"""
    
    @abstractmethod
    def save_all_raw_data(
        self,
        games: pd.DataFrame,
        version: str,
        export_formats: Optional[Sequence[str]] = None,
    ) -> None:
        """Save processed game data with version tracking"""
        pass
    
    @abstractmethod
    def load_all_raw_data(self, filters: Optional[Dict] = None) -> pd.DataFrame:
        """Load processed games with optional filters"""
        pass
    
    @abstractmethod
    def save_raw_series_data(
        self,
        series: pd.DataFrame,
        export_formats: Optional[Sequence[str]] = None,
    ) -> None:
        """Save series grouping and fearless detection results"""
        pass
    
    @abstractmethod
    def load_raw_series_data(self, series_id: str) -> Optional[Dict]:
        """Get series context for fearless draft calculations"""
        pass
    
    @abstractmethod
    def save_raw_player_data(
        self,
        players: pd.DataFrame,
        version: str,
        export_formats: Optional[Sequence[str]] = None,
    ) -> None:
        """Save player data (participantid 1-10)"""
        pass
    
    @abstractmethod
    def load_raw_player_data(self, filters: Optional[Dict] = None) -> pd.DataFrame:
        """Load player data"""
        pass

    @abstractmethod
    def save_raw_team_data(
        self,
        teams: pd.DataFrame,
        version: str,
        export_formats: Optional[Sequence[str]] = None,
    ) -> None:
        """Save team data (participantid 100, 200)"""
        pass
    
    @abstractmethod
    def load_raw_team_data(self, filters: Optional[Dict] = None) -> pd.DataFrame:
        """Load team data"""
        pass 

    @abstractmethod
    def is_data_stale(self, data_type: str, source_modified_time: float) -> bool:
        """Check if cached data is stale compared to source"""
        pass