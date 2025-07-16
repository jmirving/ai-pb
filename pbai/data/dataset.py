import os
import logging
from typing import Tuple
import pandas as pd
import torch
from torch.utils.data import Dataset
from pbai.utils import champ_enum
from pbai.utils import config

class BansPickDataset(Dataset):
    """Custom Dataset for bans and pick data."""
    def __init__(self, file_name: str = config.DATA_FILE):
        file_path = os.path.join(config.DATA_DIR, file_name)
        try:
            self.data = pd.read_csv(file_path)
        except Exception as e:
            logging.error(f"Failed to read data file: {file_path}. Error: {e}")
            raise
        self.champ_enum = champ_enum.create_champ_enum()
        self.valid_picks = [int(champ.value) for champ in self.champ_enum.__members__.values()]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.data.iloc[index]
        bans = row[:-1].values.astype(str)
        normalized_bans = [
            self.valid_picks.index(int(self.champ_enum[str(champ).upper() if champ != 'nan' else "MISSING"].value))
            for champ in bans
        ]
        pick = row[-1]
        pick_int = int(self.champ_enum[str(pick).upper()].value)
        normalized_pick = self.valid_picks.index(pick_int)
        mat1 = torch.tensor(normalized_bans)
        mat2 = torch.tensor(normalized_pick)
        return mat1, mat2

    def __len__(self) -> int:
        return len(self.data) 