"""
Metrics for evaluating imputation quality.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score as sklearn_r2
from typing import Dict


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    
    Returns
    -------
    float
        RMSE value
    """
    # Filter out NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return np.nan
    
    return np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    
    Returns
    -------
    float
        MAE value
    """
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return np.nan
    
    return mean_absolute_error(y_true[mask], y_pred[mask])


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Normalized RMSE (NRMSE).
    Normalized by the range of true values.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    
    Returns
    -------
    float
        NRMSE value
    """
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return np.nan
    
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]
    
    rmse_val = np.sqrt(mean_squared_error(y_true_filtered, y_pred_filtered))
    y_range = np.ptp(y_true_filtered)  # peak-to-peak (max - min)
    
    if y_range < 1e-10:
        return rmse_val  # Avoid division by zero
    
    return rmse_val / y_range


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate R² (coefficient of determination).
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    
    Returns
    -------
    float
        R² value
    """
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 2:
        return np.nan
    
    return sklearn_r2(y_true[mask], y_pred[mask])


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    
    Returns
    -------
    float
        MAPE value (as percentage)
    """
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]
    
    # Avoid division by zero
    nonzero_mask = np.abs(y_true_filtered) > 1e-10
    if nonzero_mask.sum() == 0:
        return np.nan
    
    return np.mean(np.abs((y_true_filtered[nonzero_mask] - y_pred_filtered[nonzero_mask]) / 
                          y_true_filtered[nonzero_mask])) * 100


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     metrics: list = None) -> Dict[str, float]:
    """
    Calculate multiple metrics at once.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    metrics : list, optional
        List of metric names to calculate
        Default: ['rmse', 'mae', 'nrmse', 'r2']
    
    Returns
    -------
    results : dict
        Dictionary with metric names as keys and values as floats
    """
    if metrics is None:
        metrics = ['rmse', 'mae']
    
    results = {}
    
    for metric in metrics:
        metric = metric.lower()
        if metric == 'rmse':
            results['rmse'] = rmse(y_true, y_pred)
        elif metric == 'mae':
            results['mae'] = mae(y_true, y_pred)
        elif metric == 'nrmse':
            results['nrmse'] = nrmse(y_true, y_pred)
        elif metric == 'r2':
            results['r2'] = r2_score(y_true, y_pred)
        elif metric == 'mape':
            results['mape'] = mape(y_true, y_pred)
        else:
            print(f"Warning: Unknown metric '{metric}'")
    
    return results


def calculate_imputation_error(X_true: np.ndarray, X_imputed: np.ndarray, 
                               missing_mask: np.ndarray, 
                               metrics: list = None) -> Dict[str, float]:
    """
    Calculate imputation error only on originally missing values.
    
    Parameters
    ----------
    X_true : np.ndarray
        Original complete data
    X_imputed : np.ndarray
        Imputed data
    missing_mask : np.ndarray
        Boolean mask indicating which values were missing (True = was missing)
    metrics : list, optional
        List of metrics to calculate
    
    Returns
    -------
    results : dict
        Dictionary with metric values
    """
    # Extract only the values that were missing
    y_true = X_true[missing_mask]
    y_pred = X_imputed[missing_mask]
    
    return calculate_metrics(y_true, y_pred, metrics)
