import numpy as np

def safe_div(x, y):
    """
    Safe division that handles division by zero by returning 1.0 where y is zero.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(y != 0, x / y, 1.0)

def safe_log(x):
    """
    Safe logarithm that returns 0.0 for non-positive inputs to avoid domain errors.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(x > 0, np.log(x), 0.0)

def safe_exp(x):
    """
    Safe exponential that caps the input to prevent overflow.
    """
    with np.errstate(over='ignore'):
        return np.where(x < 100, np.exp(x), np.exp(100))  # Prevent overflow

def safe_sqrt(x):
    """
    Safe square root that returns 0.0 for negative inputs to avoid domain errors.
    """
    return np.where(x >= 0, np.sqrt(x), 0.0)