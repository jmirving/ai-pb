import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import os

import ChampEnum

def optimizer(parameters):
    return optim.Adam(parameters,lr=0.002)

class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.hidden_layer = nn.Linear(input_size, 128)
        self.output_layer = nn.Linear(128, 170)

    def forward(self, x):
        x = x.float()
        x = torch.relu(self.hidden_layer(x))
        return self.output_layer(x)

model = MLP(input_size=6)
champEnum = ChampEnum.create_champ_enum()
valid_picks = [int(champ.value) for champ in champEnum.__members__.values()]

class BansPickDataset(Dataset):
    def __init__(self, file_name):
        file_path = os.path.join("resources", file_name)
        self.data = pd.read_csv(file_path)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        bans = row[:-1].values.astype(str)
        normalized_bans = []
        for champ in bans:
            if champ == 'nan':
                champ = "MISSING"
            key = int(champEnum[champ].value)
            normalized_key = valid_picks.index(key)
            normalized_bans.append(int(normalized_key))
        pick = row[-1]
        pick_int = int(champEnum[pick].value)
        normalized_pick = valid_picks.index(pick_int)
        mat1 = torch.tensor(normalized_bans)
        mat2 = torch.tensor(normalized_pick)
        return mat1, mat2

    def __len__(self):
        return len(self.data)


dataset = BansPickDataset("fp-data.csv")
dataloader = DataLoader(dataset, batch_size=6, shuffle=True)

optimizer = optimizer(model.parameters())
loss_function = nn.CrossEntropyLoss()

for epoch in range(70):
    total_loss = 0.0
    last_pick = []
    last_predict = []
    for bans, pick in dataloader: # Iterate over the batches
        bans = bans.float()
        # This is the forward pass
        predictions = model(bans) # Compute the model output
        predictions = predictions.float()
        predicted_classes = torch.argmax(predictions, dim=1)

        ban_names = []
        for game in bans:
            for single_ban in game:
                champ_key = valid_picks[int(single_ban)]
                ban_name = champEnum(str(champ_key)).name
                ban_names.append(ban_name)
        # print(f"Banned Champ: {ban_names}")

        prediction_names = []
        for tensor_data in predicted_classes:
            champ_key = valid_picks[tensor_data]
            prediction_name = champEnum(str(champ_key)).name
            prediction_names.append(prediction_name)
            last_predict = prediction_names
        # print(f"Predicted FP: {prediction_names}")

        pick_names = []
        for actual_pick in pick:
            champ_key = valid_picks[actual_pick]
            pick_name = champEnum(str(champ_key)).name
            pick_names.append(pick_name)
            last_pick = pick_names
        # print(f"Actual FP: {pick_names}")

        loss = loss_function(predictions, pick)
        loss = loss.float()
        # Backward pass and optimization
        optimizer.zero_grad()  # Clear previous gradients
        loss.backward()  # Compute gradients
        optimizer.step()  # Update model parameters

        total_loss += loss.item()  # Add the loss for all batches

    # Print the loss for this epoch
    print("Epoch {}: Sum of Batch Losses = {:.5f}".format(epoch, total_loss))
    print(f"Predicted FP: {last_predict}")
    print(f"Actual FP: {last_pick}")