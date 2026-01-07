"""
Missing data generation with different mechanisms (MCAR, MAR, MNAR).
"""

import numpy as np
from typing import Tuple


def create_mcar(X: np.ndarray, missing_rate: float, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create Missing Completely At Random (MCAR) pattern.
    
    Parameters
    ----------
    X : np.ndarray
        Complete data
    missing_rate : float
        Proportion of values to make missing (0 to 1)
    seed : int, optional
        Random seed
    
    Returns
    -------
    X_missing : np.ndarray
        Data with missing values
    mask : np.ndarray
        Boolean mask indicating missing positions (True = missing)
    """
    if seed is not None:
        np.random.seed(seed)
    
    X_missing = X.copy()
    n_samples, n_features = X.shape
    
    # Create random mask
    mask = np.random.rand(n_samples, n_features) < missing_rate
    
    # Apply mask
    X_missing[mask] = np.nan
    
    return X_missing, mask


def create_mar(X: np.ndarray, missing_rate: float, seed: int = None, 
               dependency_col: int = 0, threshold_quantile: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create Missing At Random (MAR) pattern.
    Missing values depend on observed values in other columns.
    
    Parameters
    ----------
    X : np.ndarray
        Complete data
    missing_rate : float
        Target proportion of missing values
    seed : int, optional
        Random seed
    dependency_col : int
        Column index that determines missingness
    threshold_quantile : float
        Quantile threshold for dependency (default: 0.5 = median)
    
    Returns
    -------
    X_missing : np.ndarray
        Data with missing values
    mask : np.ndarray
        Boolean mask indicating missing positions
    """
    if seed is not None:
        np.random.seed(seed)
    
    X_missing = X.copy()
    n_samples, n_features = X.shape
    
    # Calculate threshold based on dependency column
    threshold = np.quantile(X[:, dependency_col], threshold_quantile)
    
    # Higher probability of missingness for values above/below threshold
    mask = np.zeros((n_samples, n_features), dtype=bool)
    
    for col in range(n_features):
        if col == dependency_col:
            continue  # Don't make dependency column missing
        
        # Create probability based on dependency column
        prob = np.where(X[:, dependency_col] > threshold, 
                       missing_rate * 1.5,  # Higher rate for one group
                       missing_rate * 0.5)  # Lower rate for other group
        
        # Clip probabilities to [0, 1]
        prob = np.clip(prob, 0, 1)
        
        # Generate missing mask
        mask[:, col] = np.random.rand(n_samples) < prob
    
    # Adjust to match target missing rate
    current_rate = mask.sum() / mask.size
    if current_rate < missing_rate:
        # Add more missing values
        remaining_indices = np.where(~mask)
        n_add = int((missing_rate - current_rate) * mask.size)
        if n_add > 0:
            add_indices = np.random.choice(len(remaining_indices[0]), 
                                          min(n_add, len(remaining_indices[0])), 
                                          replace=False)
            mask[remaining_indices[0][add_indices], remaining_indices[1][add_indices]] = True
    
    X_missing[mask] = np.nan
    
    return X_missing, mask


def create_mnar(X: np.ndarray, missing_rate: float, seed: int = None,
                threshold_quantile: float = 0.7) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create Missing Not At Random (MNAR) pattern.
    Missing values depend on the values themselves (high/low values more likely to be missing).
    
    Parameters
    ----------
    X : np.ndarray
        Complete data
    missing_rate : float
        Target proportion of missing values
    seed : int, optional
        Random seed
    threshold_quantile : float
        Quantile threshold - values above this are more likely to be missing
    
    Returns
    -------
    X_missing : np.ndarray
        Data with missing values
    mask : np.ndarray
        Boolean mask indicating missing positions
    """
    if seed is not None:
        np.random.seed(seed)
    
    X_missing = X.copy()
    n_samples, n_features = X.shape
    
    mask = np.zeros((n_samples, n_features), dtype=bool)
    
    for col in range(n_features):
        # Calculate threshold for this column
        threshold = np.quantile(X[:, col], threshold_quantile)
        
        # Higher values have higher probability of being missing
        # Create logistic probability curve
        prob = 1 / (1 + np.exp(-5 * (X[:, col] - threshold) / np.std(X[:, col])))
        
        # Scale to achieve target missing rate for this column
        prob = prob * (missing_rate * 2)
        prob = np.clip(prob, 0, 1)
        
        mask[:, col] = np.random.rand(n_samples) < prob
    
    # Adjust to match target missing rate
    current_rate = mask.sum() / mask.size
    if current_rate > missing_rate * 1.2:  # Too many missing
        # Remove some
        missing_indices = np.where(mask)
        n_remove = int((current_rate - missing_rate) * mask.size)
        if n_remove > 0:
            remove_indices = np.random.choice(len(missing_indices[0]), 
                                             min(n_remove, len(missing_indices[0])), 
                                             replace=False)
            mask[missing_indices[0][remove_indices], missing_indices[1][remove_indices]] = False
    elif current_rate < missing_rate * 0.8:  # Too few missing
        # Add more
        remaining_indices = np.where(~mask)
        n_add = int((missing_rate - current_rate) * mask.size)
        if n_add > 0:
            add_indices = np.random.choice(len(remaining_indices[0]), 
                                          min(n_add, len(remaining_indices[0])), 
                                          replace=False)
            mask[remaining_indices[0][add_indices], remaining_indices[1][add_indices]] = True
    
    X_missing[mask] = np.nan
    
    return X_missing, mask


def generate_missing_data(X: np.ndarray, mechanism: str, missing_rate: float, 
                         seed: int = None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate missing data with specified mechanism.
    
    Parameters
    ----------
    X : np.ndarray
        Complete data
    mechanism : str
        Missing data mechanism ('MCAR', 'MAR', 'MNAR')
    missing_rate : float
        Proportion of missing values
    seed : int, optional
        Random seed
    **kwargs : dict
        Additional parameters for specific mechanisms
    
    Returns
    -------
    X_missing : np.ndarray
        Data with missing values
    mask : np.ndarray
        Boolean mask indicating missing positions
    """
    mechanism = mechanism.upper()
    
    if mechanism == 'MCAR':
        return create_mcar(X, missing_rate, seed)
    elif mechanism == 'MAR':
        return create_mar(X, missing_rate, seed, **kwargs)
    elif mechanism == 'MNAR':
        return create_mnar(X, missing_rate, seed, **kwargs)
    else:
        raise ValueError(f"Unknown mechanism: {mechanism}. Use 'MCAR', 'MAR', or 'MNAR'.")


def get_missing_statistics(X: np.ndarray) -> dict:
    """
    Calculate statistics about missing data.
    
    Parameters
    ----------
    X : np.ndarray
        Data with potential missing values
    
    Returns
    -------
    stats : dict
        Dictionary with missing data statistics
    """
    mask = np.isnan(X)
    
    stats = {
        'total_missing': mask.sum(),
        'missing_rate': mask.sum() / mask.size,
        'missing_per_column': mask.sum(axis=0),
        'missing_per_row': mask.sum(axis=1),
        'columns_with_missing': (mask.sum(axis=0) > 0).sum(),
        'rows_with_missing': (mask.sum(axis=1) > 0).sum(),
    }
    
    return stats
