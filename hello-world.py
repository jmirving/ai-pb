import os
import logging
from typing import Tuple
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import ChampEnum

# Constants
DATA_DIR = "resources"
DATA_FILE = "fp-data2.csv"
INPUT_SIZE = 6
HIDDEN_SIZE = 128
OUTPUT_SIZE = 170
BATCH_SIZE = 6
EPOCHS = 70
LEARNING_RATE = 0.002

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_optimizer(parameters) -> optim.Optimizer:
    """Return Adam optimizer with predefined learning rate."""
    return optim.Adam(parameters, lr=LEARNING_RATE)


class MLP(nn.Module):
    """A simple Multi-Layer Perceptron for classification."""
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.hidden_layer = nn.Linear(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        x = torch.relu(self.hidden_layer(x))
        return self.output_layer(x)


class BansPickDataset(Dataset):
    """Custom Dataset for bans and pick data."""
    def __init__(self, file_name: str):
        file_path = os.path.join(DATA_DIR, file_name)
        try:
            self.data = pd.read_csv(file_path)
        except Exception as e:
            logging.error(f"Failed to read data file: {file_path}. Error: {e}")
            raise
        self.champ_enum = ChampEnum.create_champ_enum()
        self.valid_picks = [int(champ.value) for champ in self.champ_enum.__members__.values()]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.data.iloc[index]
        bans = row[:-1].values.astype(str)
        normalized_bans = [
            self.valid_picks.index(int(self.champ_enum[champ if champ != 'nan' else "MISSING"].value))
            for champ in bans
        ]
        pick = row[-1]
        pick_int = int(self.champ_enum[pick].value)
        normalized_pick = self.valid_picks.index(pick_int)
        mat1 = torch.tensor(normalized_bans)
        mat2 = torch.tensor(normalized_pick)
        return mat1, mat2

    def __len__(self) -> int:
        return len(self.data)


def train():
    champ_enum = ChampEnum.create_champ_enum()
    valid_picks = [int(champ.value) for champ in champ_enum.__members__.values()]

    model = MLP(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE)
    dataset = BansPickDataset(DATA_FILE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = get_optimizer(model.parameters())
    loss_function = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        total_loss = 0.0
        total_correct_picks = 0
        total_incorrect_picks = 0
        for bans, pick in dataloader:
            bans = bans.float()
            predictions = model(bans)
            predicted_classes = torch.argmax(predictions, dim=1)

            # Convert indices to names for logging
            prediction_names = [champ_enum(str(valid_picks[idx])).name for idx in predicted_classes]
            pick_names = [champ_enum(str(valid_picks[idx])).name for idx in pick]

            # Count correct/incorrect picks
            correct_picks = sum(pn == kn for pn, kn in zip(prediction_names, pick_names))
            incorrect_picks = len(prediction_names) - correct_picks

            loss = loss_function(predictions, pick)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_correct_picks += correct_picks
            total_incorrect_picks += incorrect_picks

        logging.info(f"Epoch {epoch}: Sum of Batch Losses = {total_loss:.5f}")
        logging.info(f"Total Correct Picks: {total_correct_picks}")
        logging.info(f"Total Incorrect Picks: {total_incorrect_picks}")


if __name__ == "__main__":
    train()
