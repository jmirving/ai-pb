"""
Training logic for Oracle's Elixir data using the new DraftDataset and DraftMLP.
This module provides the train_oracle_elixir() function for use by scripts or admin tools.
"""

import logging
import torch
from torch._C import NoneType
import torch.nn as nn
from torch.utils.data import DataLoader
from pbai.models.mlp import DraftMLP
from pbai.data.dataset import DraftDataset
from pbai.data.storage import FileRepository
from pbai.data.data_ingestion_service import DataIngestionService
from pbai.utils import champ_enum
from pbai.utils import config
import numpy as np
from sklearn.model_selection import train_test_split
import os
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_optimizer(parameters, lr=0.001):
    """Return Adam optimizer with learning rate."""
    import torch.optim as optim
    return optim.Adam(parameters, lr=lr)

def initialize_data_ingestion_service(oracle_elixir_path=None, processed_data_dir="data/processed"):
    """
    Initialize and return a DataService object.
    
    Args:
        oracle_elixir_path (str): Path to Oracle's Elixir CSV file
        processed_data_dir (str): Directory containing processed data
    
    Returns:
        DataService: Initialized data service object, or None if initialization fails
    """
    repository = FileRepository(processed_data_dir)
    if oracle_elixir_path is None:
        oracle_elixir_path = "resources/2025_LoL_esports_match_data_from_OraclesElixir.csv"
    
    if not os.path.exists(oracle_elixir_path):
        logging.error(f"Oracle's Elixir data file not found: {oracle_elixir_path}")
        return None
    
    return DataIngestionService(repository, oracle_elixir_path)


def print_sample_predictions(masked_outputs, batch, dataset, is_train):
    """
    Print a few sample predictions for the first batch of an epoch.
    Shows predicted and actual champion names and the event index.
    """
    # Get predicted class indices
    _, predicted_classes = torch.max(masked_outputs, 1)
    for i in range(min(5, len(predicted_classes))):
        pred_idx = predicted_classes[i].item()
        target_idx = batch['target'][i].item()
        # Map from 0-169 back to 1-170 for champion names
        pred_champ = dataset.idx2champion.get(pred_idx + 1, f"UNKNOWN_{pred_idx}")
        target_champ = dataset.idx2champion.get(target_idx + 1, f"UNKNOWN_{target_idx}")
        event = batch['step_idx'][i].item() if 'step_idx' in batch else 'N/A'
        prefix = "[TRAIN]" if is_train else "[VAL]"
        print(f"{prefix} Sample {i}: Event {event} | Predicted: {pred_champ} | Actual: {target_champ}")


def run_epoch(model, data_loader, loss_function, optimizer=None, device='cpu', dataset=None, is_train=True):
    """
    Run one epoch of training or evaluation.
    - Enables/disables gradients as appropriate.
    - Iterates over all batches in the data loader.
    - For each batch:
        - Prepares input features and moves them to the correct device.
        - Runs the model to get output logits.
        - Applies the output mask to prevent prediction of unavailable champions.
        - Computes the loss between masked outputs and true targets.
        - (If training) Performs backpropagation and optimizer step.
        - Accumulates the loss for average calculation.
        - Prints a few sample predictions for the first batch.
    Returns the average loss for the epoch.
    """
    if is_train:
        model.train()  # Enable training mode (activates dropout, etc.)
    else:
        model.eval()   # Enable evaluation mode (disables dropout, etc.)
    total_loss = 0
    # Enable/disable gradient computation depending on mode
    with torch.set_grad_enabled(is_train):
        for batch_idx, batch in enumerate(data_loader):
            # Prepare input features and move to device
            features = {
                'draft_sequence': batch['draft_sequence'].to(device),
            }
            # Forward pass: get model outputs (logits)
            outputs = model(features)
            # Apply output mask: set logits for unavailable champions to a large negative value
            masked_outputs = outputs.masked_fill(batch['output_mask'].to(device) == 0, -1e9)
            # Compute loss between masked outputs and true targets
            loss = loss_function(masked_outputs, batch['target'].to(device))
            if is_train:
                # Backward pass and optimizer step (only in training)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # Accumulate loss for average calculation
            total_loss += loss.item()
            # Print a few sample predictions for the first batch
            if batch_idx == 0:
                print_sample_predictions(masked_outputs, batch, dataset, is_train)
    # Compute average loss for the epoch
    avg_loss = total_loss / len(data_loader)
    return avg_loss


def train_oracle_elixir(oracle_elixir_path=None, processed_data_dir="data/processed", model_path="models/draft_mlp_oracle_elixir.pth", epochs=20, batch_size=32, lr=0.001):
    set_seed(42)
    
    # Device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Initialize data service
    data_ingestion_service = initialize_data_ingestion_service(oracle_elixir_path=oracle_elixir_path, processed_data_dir=processed_data_dir)
    
    if data_ingestion_service is None:
        logging.error("Data Ingestion Service not initialized")
        return
    
    try:
        # --- DATASET CREATION ---
        dataset = DraftDataset(data_ingestion_service)
        
        if dataset is None:
            logging.error("Failed to create dataset")
            return
        
        # Split dataset
        indices = np.arange(len(dataset))
        train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42, shuffle=True)
        
        from torch.utils.data import Subset
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        
        logging.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        
        # Initialize model
        champ_enum_obj = champ_enum.create_champ_enum()
        num_real_champions = dataset.num_champions - 1  # Exclude MISSING (0) from output
        feature_dims = {
            'num_champions': dataset.num_champions,  # for embedding (includes MISSING)
            'draft_sequence': 20,  # 10 picks + 10 bans
        }
        model = DraftMLP(
            feature_dims=feature_dims,
            hidden_size=256,
            output_size=num_real_champions  # 170 for output (only real champions)
        ).to(device)
        
        optimizer = get_optimizer(model.parameters(), lr=lr)
        loss_function = nn.CrossEntropyLoss()
        
        # Data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # --- TRAINING LOOP ---
        best_val_loss = float('inf')
        model_save_path = model_path
        
        logging.info(f"Starting training for {epochs} epochs with batch size {batch_size} and learning rate {lr}")
        logging.info(f"Model architecture: {feature_dims}")
        
        total_train_batches = len(train_loader)
        total_val_batches = len(val_loader)
        logging.info(f"Total training batches per epoch: {total_train_batches}")
        logging.info(f"Total validation batches per epoch: {total_val_batches}")
        
        for epoch in range(epochs):
            train_loss = run_epoch(model, train_loader, loss_function, optimizer, device, dataset, is_train=True)
            print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}")
            val_loss = run_epoch(model, val_loader, loss_function, optimizer=None, device=device, dataset=dataset, is_train=False)
            print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), model_save_path)
                logging.info(f"New best model saved with validation loss: {best_val_loss:.4f}")
            else:
                logging.info(f"Validation loss {val_loss:.4f} did not improve on best {best_val_loss:.4f}")
            
            # Log epoch timing
            epoch_start_time = time.time() # Define epoch_start_time here
            epoch_time = time.time() - epoch_start_time
            logging.info(f"Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f} seconds")
        
        # Save model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)
        logging.info(f"Model saved to {model_path}")
        logging.info(f"Training completed successfully! Best validation loss: {best_val_loss:.4f}")
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise 