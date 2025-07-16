import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pbai.models.mlp import MLP
from pbai.data.dataset import BansPickDataset
from pbai.utils import champ_enum
from pbai.utils import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_optimizer(parameters):
    """Return Adam optimizer with predefined learning rate."""
    import torch.optim as optim
    return optim.Adam(parameters, lr=config.LEARNING_RATE)

def train():
    champ_enum_obj = champ_enum.create_champ_enum()
    valid_picks = [int(champ.value) for champ in champ_enum_obj.__members__.values()]

    model = MLP(input_size=config.INPUT_SIZE, hidden_size=config.HIDDEN_SIZE, output_size=config.OUTPUT_SIZE)
    dataset = BansPickDataset(config.DATA_FILE)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    optimizer = get_optimizer(model.parameters())
    loss_function = nn.CrossEntropyLoss()

    for epoch in range(config.EPOCHS):
        total_loss = 0.0
        total_correct_picks = 0
        total_incorrect_picks = 0
        for bans, pick in dataloader:
            bans = bans.float()
            predictions = model(bans)
            predicted_classes = torch.argmax(predictions, dim=1)

            # Convert indices to names for logging
            prediction_names = [champ_enum_obj(str(valid_picks[idx])).name for idx in predicted_classes]
            pick_names = [champ_enum_obj(str(valid_picks[idx])).name for idx in pick]

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