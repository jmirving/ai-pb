import unittest
import pandas as pd

from pbai.data.dataset import DraftDataset


class FakeIngestionService:
    def __init__(self, dataframe: pd.DataFrame):
        self._df = dataframe

    def get_all_data_df(self, force_refresh: bool = False) -> pd.DataFrame:
        return self._df.copy()


class DraftDatasetPreprocessSamplesTest(unittest.TestCase):
    def setUp(self):
        # Construct a minimal two-game series. The ``game`` column is the
        # per-series game counter used by the dataset to detect fearless-draft
        # history, so each pair of rows (blue/red) shares the same value.
        self.dataframe = pd.DataFrame([
            {
                'patch': '13.1',
                'participantid': 100,
                'league': 'LCS',
                'split': 'Spring',
                'year': 2024,
                'gameid': 1,
                'teamid': 1,
                'game': 1,
                'side': 'Blue',
                'ban1': 'AATROX',
                'ban2': 'AHRI',
                'ban3': 'AKALI',
                'ban4': 'ALISTAR',
                'ban5': 'AMUMU',
                'pick1': 'CAITLYN',
                'pick2': 'LUX',
                'pick3': 'LEONA',
                'pick4': 'JINX',
                'pick5': 'VI',
            },
            {
                'patch': '13.1',
                'participantid': 200,
                'league': 'LCS',
                'split': 'Spring',
                'year': 2024,
                'gameid': 1,
                'teamid': 2,
                'game': 1,
                'side': 'Red',
                'ban1': 'DARIUS',
                'ban2': 'DIANA',
                'ban3': 'DRAVEN',
                'ban4': 'EKKO',
                'ban5': 'ELISE',
                'pick1': 'SERAPHINE',
                'pick2': 'ASHE',
                'pick3': 'BLITZCRANK',
                'pick4': 'MORGANA',
                'pick5': 'THRESH',
            },
            {
                'patch': '13.1',
                'participantid': 100,
                'league': 'LCS',
                'split': 'Spring',
                'year': 2024,
                'gameid': 2,
                'teamid': 1,
                'game': 2,
                'side': 'Blue',
                'ban1': 'FIDDLESTICKS',
                'ban2': 'FIORA',
                'ban3': 'FIZZ',
                'ban4': 'GALIO',
                'ban5': 'GANGPLANK',
                'pick1': 'GAREN',
                'pick2': 'GNAR',
                'pick3': 'GRAGAS',
                'pick4': 'GRAVES',
                'pick5': 'HECARIM',
            },
            {
                'patch': '13.1',
                'participantid': 200,
                'league': 'LCS',
                'split': 'Spring',
                'year': 2024,
                'gameid': 2,
                'teamid': 2,
                'game': 2,
                'side': 'Red',
                'ban1': 'HEIMERDINGER',
                'ban2': 'ILLAOI',
                'ban3': 'IRELIA',
                'ban4': 'IVERN',
                'ban5': 'JANNA',
                'pick1': 'JARVAN IV',
                'pick2': 'JAYCE',
                'pick3': 'JHIN',
                'pick4': 'JINX',
                'pick5': "KAI'SA",
            },
        ])

    def test_samples_are_emitted_per_event_with_fearless_history(self):
        dataset = DraftDataset(FakeIngestionService(self.dataframe))
        self.assertEqual(len(dataset.samples), 40)

        # The very first sample should ask the model to predict the opening ban
        # with an otherwise empty draft sequence.
        first_sample = dataset.samples[0]
        first_target_index = dataset._normalize_champion_id('AATROX')
        self.assertEqual(first_sample['target'], first_target_index)
        # Future pick/ban slots are represented with the ``MISSING`` index (0).
        self.assertTrue(all(value == 0 for value in first_sample['draft_sequence']))

        # After the opening event the sequence should carry that ban forward.
        second_sample = dataset.samples[1]
        self.assertEqual(second_sample['draft_sequence'][0], first_target_index)

        # The second game should inherit every champion picked in game one via
        # fearless-draft rules.
        fearless_expected = {
            'CAITLYN', 'LUX', 'LEONA', 'JINX', 'VI',
            'SERAPHINE', 'ASHE', 'BLITZCRANK', 'MORGANA', 'THRESH',
        }
        second_game_first_sample = dataset.samples[20]
        self.assertSetEqual(second_game_first_sample['already_picked_or_banned'], fearless_expected)
        self.assertEqual(
            second_game_first_sample['target'],
            dataset._normalize_champion_id('FIDDLESTICKS'),
        )


if __name__ == '__main__':
    unittest.main()
