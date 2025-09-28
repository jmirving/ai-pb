import pandas as pd

from pbai.data.data_ingestion_service import DataIngestionService
from pbai.data.storage.repository import DataRepository


class DummyRepository(DataRepository):
    """Minimal repository implementation for unit tests."""

    def save_all_raw_data(self, games, version, export_formats=None):
        pass

    def load_all_raw_data(self, filters=None):
        return pd.DataFrame()

    def save_raw_series_data(self, series, export_formats=None):
        pass

    def load_raw_series_data(self, series_id):
        return None

    def save_raw_player_data(self, players, version, export_formats=None):
        pass

    def load_raw_player_data(self, filters=None):
        return pd.DataFrame()

    def save_raw_team_data(self, teams, version, export_formats=None):
        pass

    def load_raw_team_data(self, filters=None):
        return pd.DataFrame()

    def is_data_stale(self, data_type, source_modified_time):
        return False


def test_clean_all_data_drops_team_rows_with_missing_picks():
    raw = pd.DataFrame([
        {
            "participantid": 100,
            "pick1": "AATROX",
            "pick2": "",
            "pick3": "LEONA",
            "pick4": "JINX",
            "pick5": "VI",
        },
        {
            "participantid": 200,
            "pick1": "SERAPHINE",
            "pick2": "ASHE",
            "pick3": "BLITZCRANK",
            "pick4": "MORGANA",
            "pick5": "THRESH",
        },
        {
            "participantid": 1,
            "pick1": "",
            "pick2": None,
            "pick3": None,
            "pick4": None,
            "pick5": None,
        },
    ])

    service = DataIngestionService(DummyRepository(), "dummy.csv")

    cleaned = service._clean_all_data(raw)

    # The blue-side team row with an empty pick should be removed.
    assert cleaned[cleaned["participantid"] == 100].empty

    # The red-side team row with complete picks should remain.
    assert not cleaned[cleaned["participantid"] == 200].empty

    # Non-team rows should be preserved regardless of pick completeness.
    assert not cleaned[cleaned["participantid"] == 1].empty
