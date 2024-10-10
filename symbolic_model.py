from sympy import lambdify
import numpy as np
from safe_operators import safe_div, safe_exp, safe_log, safe_sqrt
from logging_config import logging

class SymbolicModel:
    def __init__(self, symbolic_eqs, feature_names):
        """
        Initialize the symbolic model with a list of symbolic equations and feature names.
        
        Parameters:
        - symbolic_eqs: List of SymPy expressions representing the symbolic equations.
        - feature_names: List of feature names corresponding to the input data.
        """
        self.symbolic_eqs = symbolic_eqs
        self.feature_names = feature_names

        # Define the mapping from SymPy functions to NumPy functions for vectorized operations
        sympy_to_numpy = {
            'add': np.add,
            'subtract': np.subtract,
            'multiply': np.multiply,
            'sin': np.sin,
            'cos': np.cos,
            'exp': np.exp,
            'log': np.log,
            'sqrt': np.sqrt,
            'sub': np.subtract,
            'mul': np.multiply,
            'safe_div': safe_div,
            'safe_exp': safe_exp,
            'safe_log': safe_log,
            'safe_sqrt': safe_sqrt
        }

        # Convert symbolic equations to numeric functions using NumPy for vectorized evaluations
        self.latent_funcs = [
            lambdify(self.feature_names, eq, modules=[sympy_to_numpy, 'numpy']) 
            for eq in symbolic_eqs
        ]

    def predict(self, feature_values):
        """
        Predict latent variables using the symbolic model.
        
        Parameters:
        - feature_values: Input features as a NumPy array.
        
        Returns:
        - Predicted latent variables as a 2D NumPy array.
        """
        feature_values = np.array(feature_values)
        
        # Reshape if a single sample is passed
        if feature_values.ndim == 1:
            feature_values = feature_values.reshape(1, -1)
        elif feature_values.shape[1] != len(self.feature_names):
            raise ValueError(f"Expected {len(self.feature_names)} features, got {feature_values.shape[1]}")

        latent_preds = []
        
        for func in self.latent_funcs:
            try:
                # Vectorized evaluation 
                preds = func(*feature_values.T)  # Unpack columns as separate arguments

                # Handle scalar outputs by broadcasting
                if np.isscalar(preds):
                    preds = np.full(feature_values.shape[0], preds)
                else:
                    preds = np.array(preds)
                    if preds.shape == ():  # Handle zero-dimensional arrays
                        preds = np.full(feature_values.shape[0], preds)
                    elif preds.shape[0] != feature_values.shape[0]:
                        raise ValueError(f"Expected predictions of length {feature_values.shape[0]}, got {preds.shape[0]}")

                latent_preds.append(preds)
            except Exception as e:
                logging.error(f"Error evaluating function {func}: {e}")
                preds = np.full(feature_values.shape[0], np.nan)
                latent_preds.append(preds)
        
        return np.array(latent_preds).T