"""
Data module for LoL Draft Predictor.
"""

from .data_ingestion_service import DataIngestionService

# Import storage components
try:
    from .storage import (
        DataRepository,
        FileRepository
    )
    
    __all__ = [
        'DataRepository',
        'FileRepository',
        'DataIngestionService'
    ]
except ImportError as e:
    __all__ = [
        'DataIngestionService'
    ]