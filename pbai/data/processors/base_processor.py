"""
Base processor class for data processing operations.
"""

import pandas as pd
import torch
import numpy as np
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod


class BaseProcessor(ABC):
    """Base class for all data processors."""
    
    def __init__(self, feature_means: Optional[Dict[str, float]] = None, 
                 feature_stds: Optional[Dict[str, float]] = None):
        """
        Initialize the processor.
        
        Args:
            feature_means: Dictionary of feature means for normalization
            feature_stds: Dictionary of feature standard deviations for normalization
        """
        self.feature_means = feature_means or {}
        self.feature_stds = feature_stds or {}
    
    @abstractmethod
    def process(self, row: pd.Series, **kwargs) -> torch.Tensor:
        """
        Process a row of data and return a tensor.
        
        Args:
            row: Pandas Series containing the data row
            **kwargs: Additional arguments specific to the processor
            
        Returns:
            torch.Tensor: Processed features
        """
        pass
    
    def normalize_feature(self, value: float, feature_name: str, default_mean: float = 0.0, 
                         default_std: float = 1.0) -> float:
        """
        Normalize a feature value using stored statistics.
        
        Args:
            value: The feature value to normalize
            feature_name: Name of the feature for lookup
            default_mean: Default mean if not found in stored stats
            default_std: Default std if not found in stored stats
            
        Returns:
            float: Normalized feature value
        """
        if value is None or pd.isna(value):
            return 0.0
        
        mean = self.feature_means.get(feature_name, default_mean)
        std = self.feature_stds.get(feature_name, default_std)
        
        # Avoid division by zero
        if std < 0.01:
            std = 0.01
            
        return (value - mean) / std
    
    def normalize_win_rate(self, win_rate: float) -> float:
        """
        Normalize win rate to center around 0.5.
        
        Args:
            win_rate: Win rate value (0.0 to 1.0)
            
        Returns:
            float: Normalized win rate (-1.0 to 1.0)
        """
        if win_rate is None or pd.isna(win_rate):
            return 0.0
        return (win_rate - 0.5) / 0.5
    
    def normalize_series_value(self, value: float, max_value: float = 5.0) -> float:
        """
        Normalize series-related values (game number, series length, etc.).
        
        Args:
            value: Value to normalize
            max_value: Maximum expected value for normalization
            
        Returns:
            float: Normalized value (0.0 to 1.0)
        """
        if value is None or pd.isna(value):
            return 0.0
        return value / max_value
    
    def normalize_percentage(self, value: float, scale_factor: float = 10.0) -> float:
        """
        Normalize percentage values by scaling them up.
        
        Args:
            value: Percentage value (0.0 to 1.0)
            scale_factor: Factor to scale the value by
            
        Returns:
            float: Scaled percentage value
        """
        if value is None or pd.isna(value):
            return 0.0
        return value * scale_factor
    
    def is_fearless_draft(self, row: pd.Series) -> bool:
        """
        Check if the current game is in fearless draft mode.
        
        Args:
            row: Pandas Series containing the data row
            
        Returns:
            bool: True if fearless draft, False otherwise
        """
        # Check for is_fearless_draft column first
        is_fearless = self.safe_get(row, 'is_fearless_draft', False)
        if is_fearless:
            return True
        
        # Check for is_fearless column as fallback
        is_fearless = self.safe_get(row, 'is_fearless', False)
        return bool(is_fearless)
    
    def get_fearless_draft_value(self, row: pd.Series) -> float:
        """
        Get fearless draft value as a float for feature extraction.
        
        Args:
            row: Pandas Series containing the data row
            
        Returns:
            float: 1.0 if fearless draft, 0.0 otherwise
        """
        return float(self.is_fearless_draft(row))
    
    def safe_get(self, row: pd.Series, key: str, default: Any = None) -> Any:
        """
        Safely get a value from a pandas Series with fallback.
        
        Args:
            row: Pandas Series
            key: Key to look up
            default: Default value if key not found or value is None/NaN
            
        Returns:
            The value or default
        """
        try:
            value = row.get(key, default)
            # Handle pandas objects properly
            if value is None or (hasattr(value, '__len__') and len(value) == 0):
                return default
            # Convert to scalar if it's a pandas object
            if hasattr(value, 'item'):
                value = value.item()
            return value
        except (KeyError, TypeError):
            return default 