import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from logging_config import logging

# ----------------------------------
# 4. Training Function
# ----------------------------------

def train_autoencoder(model: nn.Module, 
                    dataloader: DataLoader, 
                    criterion: nn.Module, 
                    optimizer: optim.Optimizer, 
                    num_epochs: int = 200, 
                    log_interval: int = 20,
                    capture_latent=False) -> None:
    """
    Trains the autoencoder model.

    Parameters:
    - model: The autoencoder model.
    - dataloader: DataLoader for training data.
    - criterion: Loss function.
    - optimizer: Optimization algorithm.
    - num_epochs: Number of training epochs.
    - log_interval: Interval for logging loss.
    - capture_latent: Boolean indicating whether to capture and log latent vectors.
    """
    model.train()
    logging.info("Starting training...")
    
    latent_vectors = []
    
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        for batch_idx, (features, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            reconstructed, latent = model(features)
            loss = criterion(reconstructed, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            # Optionally capture latent vectors for analysis or GP
            if capture_latent and epoch % log_interval == 0:
                latent_vectors.append(latent.detach().cpu().numpy())
        
        avg_loss = epoch_loss / len(dataloader)
        if epoch % log_interval == 0 or epoch == 1:
            logging.info(f'Epoch [{epoch}/{num_epochs}], Loss: {avg_loss:.6f}')
    
    logging.info("Training completed.")
    
    if capture_latent:
        logging.info(f"Captured latent vectors during training for GP analysis.")
        return latent_vectors