import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import logging
import time
from data import CaliforniaHousingDatasetGenerator, GenericDataset
from autoencoder import Autoencoder
from symbolic_model import SymbolicModel
from training import train_autoencoder
from validation import validate_hybrid_model
from symbolic_regression import perform_symbolic_regression_deap

def main(config: dict) -> None:
    """
    Main function to execute the workflow.
    
    Parameters:
    - config: Dictionary containing configuration parameters.
    """
    # 1. Data Generation
    dataset_generator = CaliforniaHousingDatasetGenerator()
    data = dataset_generator.generate(feature_cols=config.get('feature_cols', ['HouseAge', 'AveRooms']))

    # 2. Dataset and DataLoader
    dataset = GenericDataset(
        dataframe=data,
        feature_cols=config.get('feature_cols', ['HouseAge', 'AveRooms']),
        target_cols=config.get('target_cols', ['PRICE'])
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.get('batch_size', 32),  # Use the batch size from the config
        shuffle=config.get('shuffle', True)       # Shuffle based on the config
    )
    
    # 3. Model Initialization
    model = Autoencoder(
        input_dim=len(config.get('feature_cols')),  # This should match the number of features, which is 2
        output_dim=len(config.get('target_cols')),  # Assuming you adjust your model to handle different output dimensions
        latent_dim=config.get('latent_dim', 2),
        hidden_dims=config.get('hidden_dims', [16, 8])
    )
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.get('learning_rate', 0.001))  # Adjusted learning rate for stability

        # 4. Training
    if config['training']:
        logging.info("Starting training phase...")
        train_autoencoder(
            model=model,
            dataloader=dataloader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=config.get('num_epochs', 500),
            log_interval=config.get('log_interval', 50)
        )
        logging.info("Training completed.")
        
        # Save the trained model
        model_save_path = config.get('model_save_path', 'autoencoder_model.pth')
        torch.save(model.state_dict(), model_save_path)  # Saving the model state dictionary
        logging.info(f"Model saved to {model_save_path}")
    else:
        # Load the model if not training
        model_load_path = config.get('model_load_path', 'autoencoder_model.pth')
        model.load_state_dict(torch.load(model_load_path))
        logging.info(f"Model loaded from {model_load_path}")

    # 5. Evaluation (whether training or loading an existing model)
    logging.info("Starting evaluation...")
    model.eval()
    with torch.no_grad():
        # Transform features using feature_scaler
        full_data_scaled_features = dataset.feature_scaler.transform(data[config['feature_cols']].values)
        full_data_tensor = torch.tensor(full_data_scaled_features, dtype=torch.float32)
        _, latent_representations = model(full_data_tensor)
        
        # Convert latent representations to NumPy array
        latent_numpy = latent_representations.detach().numpy()
        
        # Add latent dimensions to DataFrame
        data['latent_1'] = latent_numpy[:, 0]
        data['latent_2'] = latent_numpy[:, 1]

    # 6. Symbolic Regression using DEAP and Multiple Features
    # Use only input features for symbolic regression
    X = data[config['feature_cols']].values  # Exclude 'PRICE' from input features
    y_latent = np.column_stack((data['latent_1'].values, data['latent_2'].values))
    feature_names = config['feature_cols']  # ['HouseAge', 'AveRooms']

    symbolic_eqs = []
    for i in range(y_latent.shape[1]):
        logging.info(f"Performing symbolic regression for latent variable {i + 1}")
        eqs = perform_symbolic_regression_deap(
            X, 
            y_latent[:, i], 
            feature_names=feature_names, 
            generations=config.get('generations', 20)  # Increased generations for better evolution
        )
        symbolic_eqs.extend(eqs)

    for idx, eq in enumerate(symbolic_eqs):
        logging.info(f"Symbolic equation for latent dimension {idx + 1}: {eq}")

    # Initialize the symbolic model with all equations
    symbolic_model = SymbolicModel(symbolic_eqs=symbolic_eqs, feature_names=feature_names)

    # Predict latent variables using the symbolic model
    latent_symbolic_predictions = symbolic_model.predict(data[config['feature_cols']].values)

    # Determine the number of latent dimensions predicted
    latent_dim_count = latent_symbolic_predictions.shape[1] if len(latent_symbolic_predictions.shape) > 1 else 1

    # Handle single or multiple latent dimensions
    if latent_dim_count == 1:
        data['latent_1_symbolic'] = latent_symbolic_predictions
    else:
        data['latent_1_symbolic'] = latent_symbolic_predictions[:, 0]
        data['latent_2_symbolic'] = latent_symbolic_predictions[:, 1]

    # Compare the predictions of the symbolic model with the original autoencoder latent variables
    data['latent_1_diff'] = data['latent_1'] - data['latent_1_symbolic']
    
    if latent_dim_count > 1:
        data['latent_2_diff'] = data['latent_2'] - data['latent_2_symbolic']

    # Calculate MSE between the symbolic model's predictions and the autoencoder's latent variables
    latent_1_mse = np.mean(data['latent_1_diff']**2)
    
    if latent_dim_count > 1:
        latent_2_mse = np.mean(data['latent_2_diff']**2)
        logging.info(f'MSE for latent dimension 2 (symbolic vs autoencoder): {latent_2_mse}')
    
    logging.info(f'MSE for latent dimension 1 (symbolic vs autoencoder): {latent_1_mse}')

    # Measure inference time for the symbolic model
    start_time = time.time()
    latent_symbolic_predictions = symbolic_model.predict(data[config['feature_cols']].values)
    symbolic_inference_time = time.time() - start_time
    logging.info(f'Symbolic model inference time: {symbolic_inference_time:.6f} seconds')

    # Measure inference time for the autoencoder
    start_time = time.time()
    
    # Reshape input data and perform inference with the autoencoder
    input_data = torch.tensor(data[config['feature_cols']].values, dtype=torch.float32)
    input_data = input_data.view(-1, len(config['feature_cols']))
    
    autoencoder_inference_time = time.time() - start_time
    logging.info(f'Autoencoder inference time: {autoencoder_inference_time:.6f} seconds')

    logging.info(f"Evaluation completed with MSE for latent dimension 1: {latent_1_mse}" + 
                (f", MSE for latent dimension 2: {latent_2_mse}" if latent_dim_count > 1 else ""))

    with open("discovered_equations.txt", "w") as f:
        for eq in symbolic_eqs:
            f.write(str(eq) + "\n")

if __name__ == "__main__":
    def str2bool(v):
        """
        Convert string to boolean for command-line arguments.
        """
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    # Configure logging to capture detailed debug information during debugging
    logging.basicConfig(
        level=logging.DEBUG,  # Set to DEBUG to capture detailed logs
        format='%(asctime)s %(levelname)s:%(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

    # Argument parser for command-line arguments
    parser = argparse.ArgumentParser(description="Train and evaluate the autoencoder model.")
    parser.add_argument('--training', type=str2bool, default=True, help="Whether to run training or skip to evaluation.")
    args = parser.parse_args()
    
    # Configuration Parameters
    config = {
        'num_samples': 300,
        'noise_std': 0.2,
        'seed': 42,
        'feature_cols': ['HouseAge', 'AveRooms'],
        'target_cols': ['PRICE'],
        'batch_size': 32,
        'shuffle': True,
        'hidden_dims': [16, 8],
        'latent_dim': 2,
        'learning_rate': 0.001,
        'num_epochs': 500,
        'log_interval': 50,
        'generations': 200,
        'population_size': 500, 
        'output_csv': 'reconstructed_data.csv',
        'training': args.training # set training=False when running python main.py to skip training if already trained
    }
    
    main(config)