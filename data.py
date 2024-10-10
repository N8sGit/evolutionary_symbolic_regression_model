from logging_config import logging
from typing import Tuple
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import torch
from sklearn.datasets import fetch_california_housing

# ---------------------------
# 1. Data Generation Module
# ---------------------------

class CaliforniaHousingDatasetGenerator:
    def __init__(self):
        """
        Initializes and loads the California Housing dataset from sklearn.
        """
        logging.info("Loading California housing dataset.")
        self.data = fetch_california_housing()
    
    def generate(self, scaled=True, feature_cols=None) -> pd.DataFrame:
        """
        Generates the California housing dataset.
        
        Parameters:
        - scaled: If True, return scaled features; if False, return unscaled features.
        - feature_cols: List of features to include (optional).

        Returns:
        - DataFrame with features and target.
        """
        df = pd.DataFrame(self.data.data, columns=self.data.feature_names)
        df['PRICE'] = self.data.target  # Adding target (price) column

        # Filter for selected features
        if feature_cols:
            df = df[feature_cols + ['PRICE']]  # Always include 'PRICE' as target
        
        if not scaled:
            logging.info(f"Generated unscaled California housing dataset with {len(df)} samples.")
            return df

        # Scale only the feature columns
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(df[feature_cols])  # Scale only the features
        scaled_target = scaler.fit_transform(df[['PRICE']])  # Scale only the target (PRICE)

        # Reconstruct the DataFrame with scaled data
        df[feature_cols] = scaled_features
        df['PRICE'] = scaled_target
        
        logging.info(f"Generated scaled California housing dataset with {len(df)} samples.")
        return df

    
# --------------------------------
# 2. PyTorch Dataset and DataLoader
# --------------------------------

class GenericDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, feature_cols: list, target_cols: list, scaled=True):
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        
        if scaled:
            # Scale features and targets separately
            scaled_features = self.feature_scaler.fit_transform(dataframe[feature_cols].values)
            scaled_targets = self.target_scaler.fit_transform(dataframe[target_cols].values)
        else:
            scaled_features = dataframe[feature_cols].values
            scaled_targets = dataframe[target_cols].values
        
        # Assign the scaled features and targets
        self.features = torch.tensor(scaled_features, dtype=torch.float32)
        self.targets = torch.tensor(scaled_targets, dtype=torch.float32)
        logging.info(f"Dataset initialized with features {feature_cols} and targets {target_cols}.")
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]