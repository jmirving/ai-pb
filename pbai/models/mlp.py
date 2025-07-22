import torch
import torch.nn as nn
from typing import Dict
from pbai.utils.draft_order import DRAFT_ORDER

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


class DraftMLP(nn.Module):
    """MLP for Oracle's Elixir draft data using only draft sequence as input."""
    def __init__(self, feature_dims: Dict[str, int], hidden_size: int = 256, output_size: int = 2):
        super().__init__()
        self.feature_dims = feature_dims
        # Embedding for champion IDs in draft sequence
        self.champion_embedding = nn.Embedding(
            feature_dims['num_champions'],
            embedding_dim=16
        )
        draft_input_size = feature_dims['draft_sequence'] * 16  # 20 * 16
        self.draft_encoder = nn.Sequential(
            nn.Linear(draft_input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Embed draft sequence (champion IDs)
        draft_sequence = features['draft_sequence']  # [batch, 20]
        draft_embedded = self.champion_embedding(draft_sequence)  # [batch, 20, 16]
        draft_flat = draft_embedded.view(draft_embedded.size(0), -1)  # [batch, 320]
        draft_encoded = self.draft_encoder(draft_flat)
        output = self.classifier(draft_encoded)
        return output
    
    def forward_legacy(self, x: torch.Tensor) -> torch.Tensor:
        """Legacy forward method for backward compatibility."""
        # Convert single tensor to feature dict format
        batch_size = x.size(0)
        
        # Split input tensor into feature types (simplified)
        draft_size = self.feature_dims['draft_sequence']
        
        features = {
            'draft_sequence': x[:, :draft_size].long(),
        }
        
        return self.forward(features) 