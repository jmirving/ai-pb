import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pbai.models.mlp import MLP
from pbai.data.dataset import BansPickDataset
from pbai.utils import champ_enum
from pbai.utils import config
import numpy as np
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_optimizer(parameters):
    """Return Adam optimizer with predefined learning rate."""
    import torch.optim as optim
    return optim.Adam(parameters, lr=config.LEARNING_RATE)

def train():
    set_seed(42)
    champ_enum_obj = champ_enum.create_champ_enum()
    valid_picks = [int(champ.value) for champ in champ_enum_obj.__members__.values()]

    # Device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Load the full dataset and split indices
    full_dataset = BansPickDataset(config.DATA_FILE)
    indices = np.arange(len(full_dataset))
    train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42, shuffle=True)

    from torch.utils.data import Subset
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    model = MLP(input_size=config.INPUT_SIZE, hidden_size=config.HIDDEN_SIZE, output_size=config.OUTPUT_SIZE).to(device)
    optimizer = get_optimizer(model.parameters())
    loss_function = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0.0
        total_correct_picks = 0
        total_incorrect_picks = 0
        for bans, pick in train_loader:
            bans = bans.float().to(device)
            pick = pick.to(device)
            predictions = model(bans)
            predicted_classes = torch.argmax(predictions, dim=1)

            prediction_names = [champ_enum_obj(valid_picks[idx]).name for idx in predicted_classes]
            pick_names = [champ_enum_obj(valid_picks[idx]).name for idx in pick]

            correct_picks = sum(pn == kn for pn, kn in zip(prediction_names, pick_names))
            incorrect_picks = len(prediction_names) - correct_picks

            loss = loss_function(predictions, pick)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_correct_picks += correct_picks
            total_incorrect_picks += incorrect_picks

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for bans, pick in val_loader:
                bans = bans.float().to(device)
                pick = pick.to(device)
                predictions = model(bans)
                predicted_classes = torch.argmax(predictions, dim=1)
                loss = loss_function(predictions, pick)
                val_loss += loss.item()
                prediction_names = [champ_enum_obj(valid_picks[idx]).name for idx in predicted_classes]
                pick_names = [champ_enum_obj(valid_picks[idx]).name for idx in pick]
                val_correct += sum(pn == kn for pn, kn in zip(prediction_names, pick_names))
                val_total += len(prediction_names)
        val_accuracy = val_correct / val_total if val_total > 0 else 0.0
        logging.info(f"Epoch {epoch}: Train Loss = {total_loss:.5f}, Val Loss = {val_loss:.5f}, Val Acc = {val_accuracy:.3f}")
        logging.info(f"Total Correct Picks: {total_correct_picks}")
        logging.info(f"Total Incorrect Picks: {total_incorrect_picks}")

if __name__ == "__main__":
    train() 