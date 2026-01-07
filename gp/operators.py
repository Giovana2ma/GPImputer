"""
Protected operators for GP to avoid numerical instabilities.
GPU-accelerated with automatic CPU fallback.
"""

import numpy as np
from typing import Union, Tuple, Any
from scipy import stats

# Type alias for numeric types (scalar or array)
Numeric = Union[float, np.ndarray]

def protected_div(x1: Numeric, x2: Numeric) -> Numeric:
    """
    Protected division to avoid division by zero.
    
    Parameters
    ----------
    x1 : Numeric
        Numerator.
    x2 : Numeric
        Denominator.
        
    Returns
    -------
    Numeric
        Result of division, with protection against division by zero.
    """
    # Convert to arrays to handle both scalars and arrays uniformly
    x1_arr = np.asarray(x1, dtype=float)
    x2_arr = np.asarray(x2, dtype=float)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        # Create mask for safe division
        mask = np.abs(x2_arr) > 1e-6
        # Initialize result with default value
        result = np.full_like(x1_arr, 1.0, dtype=float)
        # Perform division only where safe
        if np.any(mask):
            result = np.where(mask, x1_arr / x2_arr, 1.0)
    
    result = np.clip(result, -1e5, 1e5)
    final_result = np.nan_to_num(result, nan=1.0, posinf=1e5, neginf=-1e5)
    
    # If input was scalar, return scalar
    if final_result.shape == ():
        return float(final_result)
    return final_result


def protected_sqrt(x: Numeric) -> Numeric:
    """
    Protected square root (returns sqrt of absolute value).
    
    Parameters
    ----------
    x : Numeric
        Input value.
        
    Returns
    -------
    Numeric
        Square root of absolute value.
    """
    return np.sqrt(np.abs(x))


def protected_log(x: Numeric) -> Numeric:
    """
    Protected logarithm (log of absolute value + small constant).
    
    Parameters
    ----------
    x : Numeric
        Input value.
        
    Returns
    -------
    Numeric
        Logarithm of absolute value.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.log(np.abs(x) + 1e-6)
    return np.nan_to_num(result, nan=0.0, posinf=10.0, neginf=-10.0)


def protected_exp(x: Numeric) -> Numeric:
    """
    Protected exponential (clamped to avoid overflow).
    
    Parameters
    ----------
    x : Numeric
        Input value.
        
    Returns
    -------
    Numeric
        Exponential value.
    """
    x_clamped = np.clip(x, -10, 10)
    return np.exp(x_clamped)


def protected_pow(x1: Numeric, x2: Numeric) -> Numeric:
    """
    Protected power operation.
    
    Parameters
    ----------
    x1 : Numeric
        Base.
    x2 : Numeric
        Exponent.
        
    Returns
    -------
    Numeric
        Result of power operation.
    """
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        # Clamp exponent to reasonable range
        x2_clamped = np.clip(x2, -2, 2)
        result = np.power(np.abs(x1) + 1e-6, x2_clamped)
    result = np.clip(result, -1e5, 1e5)
    return np.nan_to_num(result, nan=1.0, posinf=1e5, neginf=-1e5)


def safe_min(x1: Numeric, x2: Numeric) -> Numeric:
    """
    Element-wise minimum.
    
    Parameters
    ----------
    x1 : Numeric
        First input.
    x2 : Numeric
        Second input.
        
    Returns
    -------
    Numeric
        Minimum value.
    """
    return np.minimum(x1, x2)


def safe_max(x1: Numeric, x2: Numeric) -> Numeric:
    """
    Element-wise maximum.
    
    Parameters
    ----------
    x1 : Numeric
        First input.
    x2 : Numeric
        Second input.
        
    Returns
    -------
    Numeric
        Maximum value.
    """
    return np.maximum(x1, x2)


def safe_abs(x: Numeric) -> Numeric:
    """
    Absolute value.
    
    Parameters
    ----------
    x : Numeric
        Input value.
        
    Returns
    -------
    Numeric
        Absolute value.
    """
    return np.abs(x)


def if_then_else(condition: Numeric, val_if_true: Numeric, val_if_false: Numeric) -> Numeric:
    """
    Ternary operator: if condition > 0 then val_if_true else val_if_false.
    
    Parameters
    ----------
    condition : Numeric
        Condition value.
    val_if_true : Numeric
        Value if condition > 0.
    val_if_false : Numeric
        Value if condition <= 0.
        
    Returns
    -------
    Numeric
        Result based on condition.
    """
    result = np.where(condition > 0, val_if_true, val_if_false)
    result = np.clip(result, -1e5, 1e5)
    return np.nan_to_num(result, nan=0.0, posinf=1e5, neginf=-1e5)


def safe_add(x1: Numeric, x2: Numeric) -> Numeric:
    """
    Protected addition.
    
    Parameters
    ----------
    x1 : Numeric
        First input.
    x2 : Numeric
        Second input.
        
    Returns
    -------
    Numeric
        Sum.
    """
    result = x1 + x2
    result = np.clip(result, -1e5, 1e5)
    return np.nan_to_num(result, nan=0.0, posinf=1e5, neginf=-1e5)


def safe_sub(x1: Numeric, x2: Numeric) -> Numeric:
    """
    Protected subtraction.
    
    Parameters
    ----------
    x1 : Numeric
        First input.
    x2 : Numeric
        Second input.
        
    Returns
    -------
    Numeric
        Difference.
    """
    result = x1 - x2
    result = np.clip(result, -1e5, 1e5)
    return np.nan_to_num(result, nan=0.0, posinf=1e5, neginf=-1e5)


def safe_mul(x1: Numeric, x2: Numeric) -> Numeric:
    """
    Protected multiplication.
    
    Parameters
    ----------
    x1 : Numeric
        First input.
    x2 : Numeric
        Second input.
        
    Returns
    -------
    Numeric
        Product.
    """
    result = x1 * x2
    result = np.clip(result, -1e5, 1e5)
    return np.nan_to_num(result, nan=0.0, posinf=1e5, neginf=-1e5)


def majority_vote(*values: Any) -> Any:
    """
    Majority vote for categorical features.
    Returns the most frequent value across all inputs.
    Works element-wise for arrays.
    
    Parameters
    ----------
    *values : array-like
        Multiple arrays with categorical values
        
    Returns
    -------
    result : array-like
        Array with majority vote for each position
    """
    if len(values) == 0:
        raise ValueError("majority_vote requires at least one input")
    
    if len(values) == 1:
        return values[0]
    
    # Stack all values
    stacked = np.array(values)
    
    # For each column (sample), find the most common value
    result, _ = stats.mode(stacked, axis=0, keepdims=False)
    
    return result
