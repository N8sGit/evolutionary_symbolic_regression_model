import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sympy import symbols, lambdify
import logging
from typing import Tuple

# -----------------------------------
# 6. Hybrid Validation Function
# -----------------------------------

def validate_hybrid_model(model: nn.Module, 
                        symbolic_eqs: list, 
                        data: pd.DataFrame, 
                        feature_cols: list, 
                        latent_cols: list = ['latent_1', 'latent_2'],
                        scaler: MinMaxScaler = None) -> Tuple[float, float]:
    """
    Validates the hybrid model by comparing the symbolic and autoencoder reconstructions.

    Parameters:
    - model: Trained autoencoder model.
    - symbolic_eqs: List of symbolic equations for each latent variable.
    - data: Original DataFrame.
    - feature_cols: Columns used as features.
    - latent_cols: Names of the latent variable columns.
    - scaler: Scaler used for preprocessing (optional).

    Returns:
    - Tuple containing MSE for latent variables and actual reconstructions.
    """
    x_sym = symbols('x')
    
    # 1. Apply symbolic equations to predict latent variables
    latent_symbolic_funcs = [lambdify(x_sym, eq, modules=['numpy']) for eq in symbolic_eqs]
    
    # Predict latent variables using symbolic equations
    latent_symbolic_predictions = np.column_stack([func(data[feature_cols[0]].values) for func in latent_symbolic_funcs])
    
    # Convert to tensor for the autoencoder
    latent_symbolic_tensor = torch.tensor(latent_symbolic_predictions, dtype=torch.float32)
    
    # 2. Use the autoencoder's decoder to reconstruct data from latent variables
    model.eval()
    with torch.no_grad():
        reconstructed_from_symbolic = model.decoder(latent_symbolic_tensor)
    
    # Convert reconstructed data to numpy
    reconstructed_data_np = reconstructed_from_symbolic.numpy()
    
    # Inverse transform if a scaler was used
    if scaler is not None:
        reconstructed_data_np = scaler.inverse_transform(reconstructed_data_np)
    
    # Assign reconstructed values to the DataFrame
    data['symbolic_reconstructed'] = reconstructed_data_np[:, -1]  # Last column is the output (e.g., 'PRICE')

    # 3. Compare symbolic predictions with autoencoder's latent variables
    latent_autoencoder_tensor = torch.tensor(data[latent_cols].values, dtype=torch.float32)
    latent_diff = latent_autoencoder_tensor - latent_symbolic_tensor
    latent_mse = torch.mean(latent_diff**2).item()
    
    # 4. Measure the reconstruction accuracy (symbolic vs. actual)
    data['reconstruction_error'] = data['PRICE'] - data['symbolic_reconstructed']
    reconstruction_mse = np.mean(data['reconstruction_error']**2)
    
    logging.info(f'MSE between symbolic and autoencoder latent variables: {latent_mse:.6f}')
    logging.info(f'MSE between symbolic model reconstruction and actual data: {reconstruction_mse:.6f}')
    
    return latent_mse, reconstruction_mse