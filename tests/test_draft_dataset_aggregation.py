import unittest
import pandas as pd

from pbai.data.dataset import DraftDataset


class DraftDatasetAggregateTrainingDataTest(unittest.TestCase):
    def setUp(self):
        # Two games on different patches with full blue/red rows so the
        # aggregation logic can infer series identifiers.
        self.raw_dataframe = pd.DataFrame([
            {
                'patch': '13.1',
                'participantid': 100,
                'league': 'LCS',
                'split': 'Spring',
                'year': 2024,
                'gameid': 1,
                'teamid': 101,
                'game': 1,
                'side': 'Blue',
            },
            {
                'patch': '13.1',
                'participantid': 200,
                'league': 'LCS',
                'split': 'Spring',
                'year': 2024,
                'gameid': 1,
                'teamid': 202,
                'game': 1,
                'side': 'Red',
            },
            {
                'patch': '13.2',
                'participantid': 100,
                'league': 'LCS',
                'split': 'Spring',
                'year': 2024,
                'gameid': 2,
                'teamid': 101,
                'game': 1,
                'side': 'Blue',
            },
            # Row with unsupported participant id should be filtered out.
            {
                'patch': '13.2',
                'participantid': 101,
                'league': 'LCS',
                'split': 'Spring',
                'year': 2024,
                'gameid': 2,
                'teamid': 101,
                'game': 1,
                'side': 'Blue',
            },
            {
                'patch': '13.2',
                'participantid': 200,
                'league': 'LCS',
                'split': 'Spring',
                'year': 2024,
                'gameid': 2,
                'teamid': 202,
                'game': 1,
                'side': 'Red',
            },
        ])

    def test_filters_to_latest_patch_and_team_rows(self):
        """Latest patch should be selected and only valid team rows retained."""

        class _Service:
            def __init__(self, df):
                self._df = df

            def get_all_data_df(self):
                return self._df.copy()

        aggregated = DraftDataset.aggregate_training_data(_Service(self.raw_dataframe))

        self.assertTrue((aggregated['patch'] == '13.2').all())
        self.assertSetEqual(set(aggregated['participantid']), {100, 200})
        # Both rows should receive the same inferred seriesid for the matchup.
        self.assertEqual(len(aggregated['seriesid'].unique()), 1)


class DraftDatasetInferSeriesIdsTest(unittest.TestCase):
    def test_new_series_started_when_game_counter_resets(self):
        """A reset game counter should trigger a brand new inferred series."""

        dataframe = pd.DataFrame([
            {
                'league': 'LCS',
                'split': 'Spring',
                'year': 2024,
                'gameid': 1,
                'teamid': 10,
                'game': 1,
            },
            {
                'league': 'LCS',
                'split': 'Spring',
                'year': 2024,
                'gameid': 1,
                'teamid': 20,
                'game': 1,
            },
            {
                'league': 'LCS',
                'split': 'Spring',
                'year': 2024,
                'gameid': 2,
                'teamid': 10,
                'game': 2,
            },
            {
                'league': 'LCS',
                'split': 'Spring',
                'year': 2024,
                'gameid': 2,
                'teamid': 20,
                'game': 2,
            },
            # Game counter resets to 1, indicating the start of a new series.
            {
                'league': 'LCS',
                'split': 'Spring',
                'year': 2024,
                'gameid': 3,
                'teamid': 10,
                'game': 1,
            },
            {
                'league': 'LCS',
                'split': 'Spring',
                'year': 2024,
                'gameid': 3,
                'teamid': 20,
                'game': 1,
            },
        ])

        with_series = DraftDataset._infer_series_ids(dataframe)

        first_series = with_series[with_series['gameid'].isin([1, 2])]['seriesid']
        second_series = with_series[with_series['gameid'] == 3]['seriesid']

        self.assertEqual(len(first_series.unique()), 1)
        self.assertEqual(len(second_series.unique()), 1)
        self.assertNotEqual(first_series.iloc[0], second_series.iloc[0])


class DraftDatasetAggregateTrainingDataNonNumericPatchTest(unittest.TestCase):
    def test_latest_patch_detected_when_cast_to_float_fails(self):
        """String patches that cannot be cast to float should still compare correctly."""

        dataframe = pd.DataFrame([
            {
                'patch': '13.1b',
                'participantid': 100,
                'league': 'LCS',
                'split': 'Spring',
                'year': 2024,
                'gameid': 1,
                'teamid': 5,
                'game': 1,
                'side': 'Blue',
            },
            {
                'patch': '13.1b',
                'participantid': 200,
                'league': 'LCS',
                'split': 'Spring',
                'year': 2024,
                'gameid': 1,
                'teamid': 6,
                'game': 1,
                'side': 'Red',
            },
            {
                'patch': '13.1c',
                'participantid': 100,
                'league': 'LCS',
                'split': 'Spring',
                'year': 2024,
                'gameid': 2,
                'teamid': 5,
                'game': 1,
                'side': 'Blue',
            },
            {
                'patch': '13.1c',
                'participantid': 200,
                'league': 'LCS',
                'split': 'Spring',
                'year': 2024,
                'gameid': 2,
                'teamid': 6,
                'game': 1,
                'side': 'Red',
            },
        ])

        class _Service:
            def __init__(self, df):
                self._df = df

            def get_all_data_df(self):
                return self._df.copy()

        aggregated = DraftDataset.aggregate_training_data(_Service(dataframe))

        self.assertTrue((aggregated['patch'] == '13.1c').all())


if __name__ == '__main__':
    unittest.main()
