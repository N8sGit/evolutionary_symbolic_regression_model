import torch
import torch.nn as nn
from logging_config import logging
from typing import Optional, Tuple

# ---------------------------------
# 3. Autoencoder Model Definition
# ---------------------------------

class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, latent_dim: int = 2, hidden_dims: Optional[list] = None):
        super(Autoencoder, self).__init__()
        if hidden_dims is None:
            hidden_dims = [16, 8]
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_layers.append(nn.Tanh())
            prev_dim = h_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(nn.Tanh())
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, output_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent