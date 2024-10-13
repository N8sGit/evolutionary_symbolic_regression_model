from sympy import sympify
import numpy as np
import pandas as pd
from symbolic_model import SymbolicModel
from data import CaliforniaHousingDatasetGenerator
from optimize_equations import optimize_equations, compare_equations
from helpers import get_next_batch_id, load_equations
import torch
from autoencoder import Autoencoder
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import csv
# ----------------------
# 1. Load Data
# ----------------------
# Initialize the data generator
data_generator = CaliforniaHousingDatasetGenerator()

# Generate the dataset (using scaled data for training consistency)
new_data = data_generator.generate(
    scaled=True,  # Ensure scaling matches what was used during training
    feature_cols=['HouseAge', 'AveRooms']  # Ensure only relevant features are selected
)

# Extract the features for prediction (ensure it matches the 3 required features)
new_data_features = new_data[['HouseAge', 'AveRooms', 'PRICE']].values

# ----------------------
# 2. Load and Optimize Symbolic Equations
# ----------------------
# Load the symbolic equations from the saved file 
# Load the last two equations from the CSV file

loaded_eqs = load_equations(batch_id=get_next_batch_id())

# Optimize the loaded symbolic equations
# Currently not working
# optimized_eqs = optimize_equations(loaded_eqs)

# Feature names (these should match the input feature names from training)
feature_names = ['HouseAge', 'AveRooms', 'PRICE']

# ----------------------
# 3. Compute Latent Dimensions using Autoencoder
# ----------------------
# Recreate the autoencoder model
model = Autoencoder(
    input_dim=2,  # Adjust based on your feature dimensions (e.g., 2 for 'HouseAge' and 'AveRooms')
    output_dim=1,  # Adjust based on your target dimension (e.g., PRICE)
    latent_dim=2,  # Number of latent dimensions
    hidden_dims=[16, 8]  # Hidden layer dimensions
)

# Load the saved state_dict (weights)
model.load_state_dict(torch.load('autoencoder_model.pth'))

# Ensure the model is in evaluation mode
model.eval()

# Prepare features for the autoencoder (ensure scaling matches training)
full_data_scaled_features = new_data[['HouseAge', 'AveRooms']].values  # Scale these features if necessary
full_data_tensor = torch.tensor(full_data_scaled_features, dtype=torch.float32)

# Measure inference time for the autoencoder
start_time = time.time()

# Run the autoencoder to get latent representations
with torch.no_grad():
    _, latent_representations = model(full_data_tensor)

autoencoder_inference_time = time.time() - start_time
print(f"Autoencoder inference time: {autoencoder_inference_time:.6f} seconds")

# Convert latent representations to NumPy
latent_numpy = latent_representations.detach().numpy()

# Add latent dimensions to the DataFrame for comparison
new_data['latent_dim_1'] = latent_numpy[:, 0]
new_data['latent_dim_2'] = latent_numpy[:, 1]

# Loop through each equation and compare it with the corresponding latent dimension
# for i, (orig_eq, opt_eq) in enumerate(zip(loaded_eqs, optimized_eqs)):
#     print(f"\nComparison for Equation {i + 1}:")
    
#     # Compare each equation against its corresponding latent dimension
#     metrics = compare_equations(
#         orig_eq, 
#         opt_eq, 
#         new_data_features, 
#         new_data[f'latent_dim_{i + 1}'].values
#     )

# ----------------------
# 4. Initialize Symbolic Model (Lambdification happens here)
# ----------------------
# Initialize the symbolic model with the optimized equations
symbolic_model = SymbolicModel(symbolic_eqs=loaded_eqs, feature_names=feature_names)

# Measure inference time for the symbolic model
start_time = time.time()
predictions = symbolic_model.predict(new_data_features)
predictions = np.nan_to_num(predictions, nan=0.0, posinf=np.inf, neginf=-np.inf)
symbolic_inference_time = time.time() - start_time

print(f"Symbolic model inference time: {symbolic_inference_time:.6f} seconds")

# Convert predictions to a DataFrame for interpretability
prediction_df = pd.DataFrame(predictions, columns=[f'latent_dim_{i+1}' for i in range(predictions.shape[1])])

# Handle invalid values (NaN, inf)
prediction_df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace infinities with NaNs
prediction_df.dropna(inplace=True)  # Drop rows with NaNs

# Print a summary of the predictions
print("Summary of predictions:")
print(prediction_df.describe())


# ----------------------
# 5. Evaluate Performance
# ----------------------
def evaluate_model(true_values, predicted_values):
    """
    Evaluates the performance of the model using various regression metrics.
    
    Parameters:
    - true_values: The true target values.
    - predicted_values: The predicted target values by the model.
    
    Returns:
    - A dictionary of computed performance metrics.
    """
    mse = mean_squared_error(true_values, predicted_values)
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    mae = mean_absolute_error(true_values, predicted_values)
    r2 = r2_score(true_values, predicted_values)

    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2
    }

# Extract true latent dimensions from the data
true_values = new_data[['latent_dim_1', 'latent_dim_2']].values

# Compute performance metrics between the symbolic model's predictions and the true latent dimensions
eval_metrics = evaluate_model(true_values, predictions)

# Print the performance metrics
print("\nPerformance Metrics:")
for metric, value in eval_metrics.items():
    print(f"{metric}: {value}")