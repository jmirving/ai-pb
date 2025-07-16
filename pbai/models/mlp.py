import torch
import torch.nn as nn

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